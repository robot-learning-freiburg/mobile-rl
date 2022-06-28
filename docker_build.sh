#!/usr/bin/env bash
PARAMS=""
while (( "$#" )); do
  case "$1" in
    -t|--tag)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        TAG=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done


if [ -z ${TAG+x} ]; then echo "--tag not provided" && exit 1; else echo "tag is set to '$TAG'"; fi
REPO="dhonerkamp/mobile_rl"

#cuda_versions=("11.1" "10.2")
cuda_versions=("11.1")
for cv in "${cuda_versions[@]}"; do
  # DOCKER_BUILDKIT can fail to pull this if not available locally yet
  docker pull dhonerkamp/ros_torch:${cv}

  if [ ${TAG} == "no" ]; then
    DOCKER_BUILDKIT=1 docker build . --cache-from ${REPO}:${cv}-latest --build-arg BUILDKIT_INLINE_CACHE=1 --build-arg MY_CUDA_VERSION=${cv} -t ${REPO}:${cv}-latest || exit;
    echo "--tag set to no, not pushing image" && exit 1;
  fi
  DOCKER_BUILDKIT=1 docker build . --cache-from ${REPO}:${cv}-latest --build-arg BUILDKIT_INLINE_CACHE=1 --build-arg MY_CUDA_VERSION=${cv} -t ${REPO}:${cv}-${TAG} -t ${REPO}:${cv}-latest || exit \
    && docker push ${REPO}:${cv}-${TAG} \
    && if [ ${TAG} != "latest" ]; then docker push ${REPO}:${cv}-latest; fi;
done


#!/usr/bin/env bash
MY_CUDA_VERSION: "11.1",


docker build --pull --rm -f Dockerfile -t modulationrl_dev:latest --build-arg BUILDKIT_INLINE_CACHE=0 --build-arg MY_CUDA_VERSION=$MY_CUDA_VERSION "."

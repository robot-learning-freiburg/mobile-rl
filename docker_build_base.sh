#!/usr/bin/env bash

REPO="dhonerkamp/ros_torch"

DOCKER_BUILDKIT=1 docker build . -f DockerfileROSTorch11.1 -t ${REPO}:11.1 \
  && docker push ${REPO}:11.1 || exit

#DOCKER_BUILDKIT=1 docker build . -f DockerfileROSTorch10.2 -t ${REPO}:10.2 \
#  && docker push ${REPO}:10.2 || exit
#!/bin/bash

# build image
IMAGE_NAME=pde
docker build -t ${IMAGE_NAME} ./docker

# run container
WORKDIR_NATIVE=$(pwd)
WORKDIR_DOCKER=/opt/workdir  # doesn't really matter
docker run --rm -it \
    -v ${WORKDIR_NATIVE}:${WORKDIR_DOCKER} \
    -w ${WORKDIR_DOCKER} \
    $IMAGE_NAME /bin/bash

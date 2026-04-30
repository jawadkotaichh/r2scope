#!/bin/bash
HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
GPU=$1
USER_NAME=${USER:-${USERNAME:-user}}
USER_NAME=$(printf '%s' "$USER_NAME" | tr -c 'a-zA-Z0-9_.-' '_')
name=${USER_NAME}_pymarl_GPU_${GPU}_${HASH}

echo "Launching container named '${name}' on GPU '${GPU}'"
# Launches a docker container using our image, and runs the provided command

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

NV_GPU="$GPU" ${cmd} run -i \
    --name "$name" \
    --user $(id -u):$(id -g) \
    -v "$(pwd)":/pymarl \
    -t pymarl:1.0 \
    ${@:2}

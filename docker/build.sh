#!/bin/bash
set -e

IMAGE_NAME="rubahoussami/rode-runpod:latest"

echo "Building Docker image: ${IMAGE_NAME}"
docker build -f docker/Dockerfile -t "${IMAGE_NAME}" .

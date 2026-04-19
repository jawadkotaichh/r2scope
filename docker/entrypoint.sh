#!/usr/bin/env bash
set -e

mkdir -p /workspace/rode-results

# Make results persistent on the mounted volume
if [ -e /opt/rode/results ] && [ ! -L /opt/rode/results ]; then
  rm -rf /opt/rode/results
fi

if [ ! -L /opt/rode/results ]; then
  ln -s /workspace/rode-results /opt/rode/results
fi

exec "$@"

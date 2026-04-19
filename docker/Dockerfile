# Modern GPU-compatible base: PyTorch 2.3 + CUDA 12.1 + cuDNN 8.
# Required because the original CUDA 9 / PyTorch 0.4 stack does not support
# modern NVIDIA GPUs (Ampere / Ada / Hopper — SM 8.0+).
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

LABEL maintainer="RODE+ Runpod"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        wget \
        git \
        unzip \
        nano \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN pip install --upgrade pip setuptools wheel

# --- Python deps ---------------------------------------------------------
# Core scientific stack: versions compatible with Python 3.10 (default in
# pytorch/pytorch:2.3 image) and with each other.
RUN pip install \
        numpy==1.26.4 \
        scipy==1.11.4 \
        "PyYAML>=6.0" \
        matplotlib==3.8.4 \
        imageio==2.34.1 \
        pygame==2.5.2 \
        scikit-learn==1.4.2 \
        jsonpickle==3.0.4 \
        snakeviz==2.2.0 \
        pytest==7.4.4 \
        probscale==0.2.5 \
        tensorboard==2.16.2 \
        tensorboard-logger==0.1.0 \
        sacred==0.8.5 \
        wrapt==1.16.0

# SMAC / PySC2 stack — modern versions work on Python 3.10 + PyTorch 2.x.
RUN pip install \
        absl-py==2.1.0 \
        mock==5.1.0 \
        mpyq==0.2.5 \
        portpicker==1.6.0 \
        protobuf==3.20.3 \
        pysc2==4.0.0 \
        s2clientprotocol \
        sk-video==1.1.10 \
        websocket-client==1.7.0 \
        whichcraft==0.6.1

RUN pip install "git+https://github.com/oxwhirl/smac.git"

# --- Project code --------------------------------------------------------
# Put code OUTSIDE /workspace so a mounted Runpod volume does not hide it.
WORKDIR /opt/rode
COPY . /opt/rode

# SC2 lives on the persistent volume
ENV SC2PATH=/workspace/StarCraftII \
    PYTHONPATH=/opt/rode/src

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]

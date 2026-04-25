# Blackwell-compatible base: needs CUDA 12.8 + PyTorch built with sm_120.
# PyTorch 2.3/cu121 (prior base) only shipped kernels up to sm_90, which is
# why RTX PRO Blackwell cards fail with "no kernel image is available".
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

RUN apt-get update -y && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

# PyTorch nightly is currently the only build with sm_120 kernels.
# Switch back to a stable wheel (cu128) once one is released.
RUN pip install --pre --upgrade torch \
        --index-url https://download.pytorch.org/whl/nightly/cu128

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

FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel
LABEL maintainer="Tabish Rashid"

# CUDA includes
ENV CUDA_PATH=/usr/local/cuda
ENV CUDA_INCLUDE_PATH=/usr/local/cuda/include
ENV CUDA_LIBRARY_PATH=/usr/local/cuda/lib64

# Ubuntu Packages
# RUN apt-get update -y && apt-get install -y software-properties-common && \
#     add-apt-repository -y multiverse && apt-get update -y && \
#     apt-get install -y apt-utils nano vim man build-essential wget sudo && \
#     rm -rf /var/lib/apt/lists/*

# Ubuntu Packages
RUN rm -f /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \
    sed -i '/^deb-src /s/^/# /' /etc/apt/sources.list && \
    apt-get -o Acquire::Retries=5 -o Acquire::ForceIPv4=true update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=5 -o Acquire::ForceIPv4=true install -y --no-install-recommends \
    apt-utils \
    nano \
    vim \
    build-essential \
    wget \
    sudo \
    curl \
    ca-certificates \
    libssl-dev \
    openssl \
    libopenblas-dev \
    libhdf5-dev \
    hdf5-helpers \
    hdf5-tools \
    libhdf5-serial-dev \
    libprotobuf-dev \
    protobuf-compiler \
    git \
    unzip && \
    rm -rf /var/lib/apt/lists/*


# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git
RUN curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
    rm -rf /var/lib/apt/lists/*

# Use the Python/PyTorch environment from the base image.
RUN pip install --upgrade pip==20.3.4 setuptools==57.5.0 wheel==0.37.1

# Python packages we use (or used at one point...)
RUN pip install numpy==1.15.2 scipy==1.1.0 PyYAML==3.13 matplotlib==3.0.0
RUN pip install imageio==2.4.1
RUN pip install tensorboard-logger==0.1.0
RUN pip install pygame==1.9.4
RUN pip3 install --no-cache-dir wrapt==1.10.11 scikit-learn==0.24.2

RUN mkdir /install
WORKDIR /install

RUN pip install jsonpickle==0.9.6
# install Sacred (from OxWhirl fork)
RUN git clone https://github.com/oxwhirl/sacred.git /install/sacred && cd /install/sacred && python setup.py install
# Sacred's setup can pull a newer wrapt that is incompatible with Python 3.6.
RUN pip install --force-reinstall wrapt==1.10.11

#### -------------------------------------------------------------------
#### install pytorch
#### -------------------------------------------------------------------
RUN pip install snakeviz==1.0.0 pytest==3.8.2 probscale==0.2.3

## -- SMAC
RUN pip install absl-py==0.5.0 mock==2.0.0 mpyq==0.2.5 portpicker==1.2.0 \
    protobuf==3.6.1 pysc2==3.0.0 s2clientprotocol==4.10.1.75800.0 \
    sk-video==1.1.10 websocket-client==0.53.0 whichcraft==0.5.2
ENV SC2PATH=/pymarl/3rdparty/StarCraftII

WORKDIR /pymarl

# =============================================================================
# Intel OpenVINO + YOLO Development Environment
# Base: Ubuntu 24.04
# =============================================================================
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MODELS="yolo11n"

# =============================================================================
# System Dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        wget gpg git \
        pciutils pkg-config \
        libudev-dev libglib2.0-0 libtbb12 libopencv-dev libzbar0 \
        python3-pip python3-dev python3-setuptools \
        ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Intel Graphics Drivers
# =============================================================================
RUN add-apt-repository -y ppa:kobuk-team/intel-graphics && \
    apt-get update && apt-get install -y --no-install-recommends \
        libze-intel-gpu1 libze1 libze-dev \
        intel-metrics-discovery intel-opencl-icd intel-gsc intel-ocloc \
        intel-media-va-driver-non-free \
        libmfx-gen1 libvpl2 libvpl-tools libva-glx2 \
        va-driver-all vainfo clinfo \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Intel NPU Driver
# =============================================================================
ARG NPU_VERSION=v1.24.0
ARG NPU_BUILD=20251003-18218973328
RUN cd /tmp \
    && wget -q https://github.com/intel/linux-npu-driver/releases/download/${NPU_VERSION}/linux-npu-driver-${NPU_VERSION}.${NPU_BUILD}-ubuntu2404.tar.gz \
    && tar -xf linux-npu-driver-*.tar.gz && dpkg -i *.deb \
    && rm -rf /tmp/*

# =============================================================================
# Python Packages
# =============================================================================
RUN pip3 install --break-system-packages --no-cache-dir \
        imutils \
        protobuf \
        psutil \
        opencv-python \
        nncf==2.8.0 \
        ultralytics \
	openvino-dev
	
# =============================================================================
# Prepare Models
# =============================================================================
WORKDIR /opt/models
COPY ./utils/prepare_models.sh .
RUN chmod +x prepare_models.sh && ./prepare_models.sh ${MODELS}

WORKDIR /workspace

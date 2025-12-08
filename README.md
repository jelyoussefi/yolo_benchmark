# Models Benchmarking with Intel OpenVINO

Benchmark YOLO models using Intel OpenVINO Runtime. Runs on any Intel platform with support for CPU, GPU, and NPU accelerators.

## Features

- YOLO object detection and instance segmentation
- Multiple precision support: FP32, FP16, INT8
- Intel hardware acceleration: CPU, GPU, NPU
- Automatic model conversion to OpenVINO format
- Docker-based environment for easy deployment

## Prerequisites

### System Requirements

- **Operating System**: Ubuntu 24.04 LTS
- **Hardware**: Intel processor (Core, Xeon, Atom)
  - Optional: Intel integrated/discrete GPU
  - Optional: Intel NPU (Intel Core Ultra processors)

### Installation Steps

#### 1. Install Linux Kernel (Optional)

If you need a specific kernel version for your hardware:
```bash
make install_kernel
```

This will compile and install the latest stable kernel. **Reboot required after installation.**

#### 2. Install Intel GPU & NPU Drivers

Install Intel GPU and NPU drivers for hardware acceleration:
```bash
make install_gpu_npu_drivers
```

This script will:
- Remove any existing Intel driver packages
- Install Intel GPU drivers from Intel Graphics PPA
- Install Intel NPU drivers from GitHub releases
- Configure user permissions (render group)
- Set up udev rules for NPU device access

**Reboot required after driver installation.**

After reboot, verify the installation:
```bash
# Verify GPU drivers
clinfo -l

# Verify media acceleration
vainfo

# Verify NPU device
ls -l /dev/accel/accel0

# Check NPU kernel module
lsmod | grep intel_vpu
```

#### 3. Install Docker

Docker is required to build and run this project. Install Docker Engine on Ubuntu by following the official guide: **https://docs.docker.com/engine/install/ubuntu/**

Add your user to the docker group (optional, to run without sudo):
```bash
sudo usermod -aG docker $USER
```

**Note**: Log out and log back in for group changes to take effect.
## Usage

### Build Docker Image
```bash
make build
```

### Run Benchmark
```bash
# Default settings (yolo11n, FP16, GPU)
make benchmark

# Custom model and settings
make benchmark MODEL=yolo11s PRECISION=FP32 DEVICE=CPU INPUT=images/test.jpg
```

### Interactive Shell
```bash
make bash
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `yolo11n` | YOLO model name |
| `PRECISION` | `FP16` | Model precision: `FP32`, `FP16`, `INT8` |
| `DEVICE` | `GPU` | Target device: `CPU`, `GPU`, `NPU`, `AUTO` |
| `INPUT` | `images/person_horse_dog.jpg` | Input image path |
| `MODELS` | `yolo11n yolo11s yolo11m` | Models to include in Docker image |

### Examples
```bash
# Run on CPU with FP32 precision
make benchmark MODEL=yolo11n PRECISION=FP32 DEVICE=CPU

# Run on GPU with FP16 precision
make benchmark MODEL=yolo11s PRECISION=FP16 DEVICE=GPU

# Run on NPU (requires Intel NPU hardware)
make benchmark MODEL=yolo11n PRECISION=FP16 DEVICE=NPU

# Build with specific models only
make build MODELS="yolo11n yolo11s"

# Build with segmentation models
make build MODELS="yolo11n yolo11n-seg yolo11s-seg"
```

## Supported Models

The default Docker image includes the following models:

- `yolo11n` - Nano (fastest)
- `yolo11s` - Small
- `yolo11m` - Medium

### Adding More Models

Use the `MODELS` environment variable to customize which models are included in the Docker image:
```bash
# Build with specific models
make build MODELS="yolo11n yolo11s"

# Include segmentation models
make build MODELS="yolo11n yolo11n-seg yolo11s-seg"

# Include larger models
make build MODELS="yolo11n yolo11s yolo11m yolo11l yolo11x"
```

Available model variants:
- Detection: `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`
- Segmentation: `yolo11n-seg`, `yolo11s-seg`, `yolo11m-seg`, `yolo11l-seg`, `yolo11x-seg`

## Output

The benchmark outputs:
- **FPS**: Frames per second throughput
- **Latency**: Average inference time in milliseconds
- **Output image**: Annotated image with detections/segmentations saved to `output/`

Example output:
```
──────────────────────────────────────────────────────────────────────
                     Models Benchmarking - Intel OpenVINO
──────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────
                            Configuration
──────────────────────────────────────────────────────────────────────
    Model:     yolo11n
    Precision: FP16
    Device:    GPU
    Input:     images/person_horse_dog.jpg
──────────────────────────────────────────────────────────────────────
                               Result
──────────────────────────────────────────────────────────────────────
    FPS:      285.43
    Latency:    3.50 ms
    Output:   output/yolo11n.jpg
──────────────────────────────────────────────────────────────────────
```

## Hardware Requirements

Runs on any Intel platform:

- **CPU**: Any Intel processor (Core, Xeon, Atom)
- **GPU**: Intel integrated or discrete GPU
- **NPU**: Intel Neural Processing Unit (Intel Core Ultra processors)

## Troubleshooting

### Driver Installation Issues

**NPU device not found after reboot:**
```bash
# Check if kernel module is loaded
lsmod | grep intel_vpu

# Check device permissions
ls -l /dev/accel/accel0

# Verify udev rules
cat /etc/udev/rules.d/10-intel-vpu.rules
```

**OpenCL not detecting GPU:**
```bash
# Check ICD files
ls /etc/OpenCL/vendors/

# Run clinfo with sudo to check permissions
sudo clinfo
```

**Package conflicts during installation:**
```bash
# Force reinstall if conflicts persist
sudo dpkg -i --force-overwrite /tmp/intel_npu_install_*/*.deb
sudo apt-get install -f -y
```

### Docker Issues

**Permission denied when running docker:**
```bash
# Ensure you're in the docker group
groups | grep docker

# If not, add yourself and re-login
sudo usermod -aG docker $USER
newgrp docker
```

### Performance Issues

**Low FPS on GPU/NPU:**
- Ensure drivers are properly installed and loaded
- Check that the correct device is being used
- Verify model precision matches device capabilities
- Monitor system resources (CPU, memory, GPU usage)

## License

Copyright (C) 2024 Intel Corporation  
SPDX-License-Identifier: Apache-2.0

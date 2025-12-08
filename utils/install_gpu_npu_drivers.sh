#!/bin/bash

# ================================================================
# Intel GPU & NPU Driver Installation Script for Ubuntu 24.04
# Drop-in replacement for original script
# Based on working Dockerfile approach
# ================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }

# Configuration
NPU_VERSION="v1.26.0"
NPU_BUILD="20251125-19665715237"

if [[ $EUID -eq 0 ]]; then
    print_error "Don't run as root. Use sudo when needed."
    exit 1
fi

echo "================================================================"
echo "  Intel GPU & NPU Driver Installation"
echo "  Ubuntu 24.04"
echo "================================================================"
echo

# =============================================================================
# Step 1: Clean existing packages
# =============================================================================
print_status "Removing existing packages..."

sudo dpkg --purge --force-remove-reinstreq \
    intel-driver-compiler-npu intel-fw-npu intel-level-zero-npu \
    intel-level-zero-npu-dbgsym intel-validation-npu \
    intel-validation-npu-dbgsym intel-validation-models-npu \
    intel-validation-utils-npu intel-openvino-npu \
    intel-onnxruntime intel-kernel-module-npu-internal \
    2>/dev/null || true

sudo apt-get remove --purge -y \
    libze1 libze-intel-gpu1 libze-dev level-zero level-zero-dev \
    intel-opencl-icd intel-metrics-discovery \
    intel-media-va-driver-non-free libmfx-gen1 libvpl2 intel-gsc \
    2>/dev/null || true

sudo add-apt-repository --remove -y ppa:kobuk-team/intel-graphics 2>/dev/null || true
sudo apt-get autoremove -y
sudo apt-get update

print_success "Cleanup complete"
echo

# =============================================================================
# Step 2: Install GPU drivers
# =============================================================================
print_status "Installing GPU drivers..."

sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt-get update

sudo apt-get install -y --no-install-recommends \
    libze1 libze-intel-gpu1 libze-dev \
    intel-opencl-icd intel-metrics-discovery intel-gsc intel-ocloc \
    intel-media-va-driver-non-free \
    libmfx-gen1 libvpl2 libvpl-tools libva-glx2 \
    va-driver-all vainfo clinfo

print_success "GPU drivers installed"
echo

# =============================================================================
# Step 3: Install NPU drivers
# =============================================================================
print_status "Installing NPU drivers..."

sudo apt-get install -y libtbb12 ocl-icd-libopencl1 dkms

# Create working directory
WORK_DIR="/tmp/intel_npu_install_$$"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

NPU_PKG="linux-npu-driver-${NPU_VERSION}.${NPU_BUILD}-ubuntu2404.tar.gz"
print_status "Downloading ${NPU_PKG}..."

wget -q --show-progress \
    "https://github.com/intel/linux-npu-driver/releases/download/${NPU_VERSION}/${NPU_PKG}"

print_status "Extracting NPU drivers..."
tar -xf "$NPU_PKG"

print_status "Installing NPU packages..."
sudo dpkg -i *.deb || true
sudo apt-get install -f -y

# Cleanup
cd - >/dev/null
rm -rf "$WORK_DIR"

print_success "NPU drivers installed"
echo

# =============================================================================
# Step 4: Configure permissions
# =============================================================================
print_status "Configuring permissions..."

sudo gpasswd -a "${USER}" render

sudo bash -c 'cat > /etc/udev/rules.d/10-intel-vpu.rules' << 'EOF'
SUBSYSTEM=="accel", KERNEL=="accel*", GROUP="render", MODE="0660"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=accel 2>/dev/null || true

print_success "Permissions configured"
echo

# =============================================================================
# Verification
# =============================================================================
print_status "Installation summary:"
echo
dpkg -l | grep -E "intel.*npu|libze|level-zero" | grep "^ii" | awk '{print "  •", $2, $3}'
echo

echo "================================================================"
print_success "Installation complete!"
echo "================================================================"
echo
echo "REBOOT REQUIRED for all changes to take effect"
echo
echo "After reboot:"
echo "  • GPU:   clinfo -l"
echo "  • Media: vainfo"
echo "  • NPU:   ls -l /dev/accel/accel0"
echo

read -p "Reboot now? (y/N): " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] && sudo reboot || print_status "Reboot when ready"

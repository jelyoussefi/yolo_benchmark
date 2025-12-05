#!/bin/bash

# ================================================================
# Intel GPU & NPU Driver Installation Script for Ubuntu 24.04
# ================================================================
# This script automates the installation of Intel GPU and NPU 
# drivers for enhanced graphics and AI acceleration performance.
# 
# Usage: ./install_gpu_npu_drivers.sh [-f]
#   -f : Force purge of existing Intel drivers before installation
# ================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
NPU_DRIVER_URL="https://af01p-ir.devtools.intel.com/artifactory/drivers_vpu_linux_client-ir-local/engineering-drops/driver/main/release/25ww44.1.1/npu-linux-driver-ci-1.27.0.20251024-18786122221-ubuntu2404-release.tar.gz"
NPU_DRIVER_FILENAME="npu-linux-driver-ci-1.27.0.20251024-18786122221-ubuntu2404-release.tar.gz"
NPU_DRIVER_DIR="npu-linux-driver-ci-1.27.0.20251024-18786122221-ubuntu2404-release"
TEMP_DIR="/tmp/intel_drivers"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}$1${NC}"
}

# Function to check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root for safety reasons."
        print_error "It will prompt for sudo when needed."
        exit 1
    fi
}

# Function to purge existing drivers (only when -f flag is used)
purge_existing_drivers() {
    local force_purge=$1
    
    if [[ "$force_purge" != "true" ]]; then
        print_status "Skipping driver purge (use -f flag to force purge existing drivers)"
        return 0
    fi
    
    print_header "🗑️  Purging existing Intel drivers and conflicting packages..."
    
    # List of packages that commonly conflict
    local conflicting_packages=(
        "libmfxgen1"
        "libmfx-gen1"
        "libmfx1"
        "intel-media-va-driver"
        "intel-media-va-driver-non-free"
        "libze-intel-gpu1"
        "libze1"
        "intel-metrics-discovery"
        "intel-opencl-icd"
        "intel-gsc"
        "libvpl2"
        "libvpl-tools"
        "libva-glx2"
        "intel-media-driver"
        "gmmlib"
        "intel-igc-core"
        "intel-igc-cm"
        "intel-level-zero-gpu"
        "level-zero"
        "level-zero-dev"
    )
    
    print_status "Stopping any running applications that might use GPU drivers..."
    # Kill processes that might be using the drivers
    sudo pkill -f intel || true
    
    print_status "Removing conflicting packages..."
    for package in "${conflicting_packages[@]}"; do
        if dpkg -l | grep -q "^ii.*$package"; then
            print_status "Removing package: $package"
            sudo apt-get remove --purge -y "$package" 2>/dev/null || true
        fi
    done
    
    # Remove any orphaned packages
    print_status "Removing orphaned packages..."
    sudo apt-get autoremove -y
    
    # Clean package cache
    print_status "Cleaning package cache..."
    sudo apt-get autoclean
    sudo apt-get clean
    
    # Force remove any remaining problematic files
    print_status "Cleaning up residual driver files..."
    sudo rm -rf /usr/lib/x86_64-linux-gnu/libmfx* 2>/dev/null || true
    sudo rm -rf /usr/lib/x86_64-linux-gnu/intel-* 2>/dev/null || true
    sudo rm -rf /etc/OpenCL/vendors/intel* 2>/dev/null || true
    
    # Remove any existing Intel GPU PPA if present
    print_status "Removing existing Intel graphics PPA..."
    sudo add-apt-repository --remove -y ppa:kobuk-team/intel-graphics 2>/dev/null || true
    
    # Fix any broken packages
    print_status "Fixing any broken package dependencies..."
    sudo apt-get update
    sudo dpkg --configure -a
    sudo apt-get install -f -y
    
    print_success "Existing driver purge completed"
}

# Function to check system compatibility
check_system() {
    print_header "🔍 Checking system compatibility..."
    
    # Check Ubuntu version
    if ! grep -q "24.04" /etc/lsb-release 2>/dev/null; then
        print_warning "This script is designed for Ubuntu 24.04. Proceeding anyway..."
    fi
    
    # Check for Intel GPU
    if lspci | grep -i "vga\|3d\|display" | grep -qi intel; then
        print_success "Intel GPU detected"
    else
        print_warning "Intel GPU not clearly detected. Installation will continue but may not be effective."
    fi
    
    # Check for Intel NPU (if available in lspci)
    if lspci | grep -qi "processing.*intel\|ai.*intel\|npu"; then
        print_success "Intel NPU-compatible hardware detected"
    else
        print_status "NPU hardware detection inconclusive - proceeding with installation"
    fi
    
    print_success "System compatibility check completed"
}

# Function to prepare environment
prepare_environment() {
    print_header "🛠️  Preparing installation environment..."
    
    # Create temporary directory
    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"
    
    # Update package lists
    print_status "Updating package lists..."
    sudo apt-get update
    
    print_success "Environment preparation completed"
}

# Function to install GPU drivers
install_gpu_drivers() {
    print_header "🎮 Installing Intel GPU drivers..."
    
    # Install software-properties-common
    print_status "Installing prerequisite packages..."
    sudo apt-get install -y software-properties-common
    
    # Add Intel graphics PPA
    print_status "Adding Intel graphics PPA repository..."
    sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
    
    # Update package lists after adding PPA
    print_status "Updating package lists..."
    sudo apt-get update
    
    # Install Intel GPU libraries and tools
    print_status "Installing Intel GPU libraries and OpenCL support..."
    sudo apt-get install -y \
        libze-intel-gpu1 \
        libze1 \
        intel-metrics-discovery \
        intel-opencl-icd \
        clinfo \
        intel-gsc
    
    # Install Intel media drivers
    print_status "Installing Intel media acceleration drivers..."
    sudo apt-get install -y \
        intel-media-va-driver-non-free \
        libmfx-gen1 \
        libvpl2 \
        libvpl-tools \
        libva-glx2 \
        va-driver-all \
        vainfo
    
    # Add user to render group
    print_status "Adding user ${USER} to render group..."
    sudo gpasswd -a "${USER}" render
    
    print_success "Intel GPU drivers installation completed"
}

# Function to install NPU drivers
install_npu_drivers() {
    print_header "🧠 Installing Intel NPU drivers..."
    
    # Install NPU dependencies
    print_status "Installing NPU driver dependencies..."
    sudo apt update
    sudo apt install -y libtbb12 ocl-icd-libopencl1 dkms
    
    # Download NPU driver
    print_status "Downloading Intel NPU driver..."
    if [[ -f "$NPU_DRIVER_FILENAME" ]]; then
        print_status "NPU driver archive already exists, using existing file"
    else
        print_status "Attempting to download NPU driver from Intel's repository..."
        if wget --no-check-certificate "$NPU_DRIVER_URL"; then
            print_success "NPU driver downloaded successfully"
        else
            print_error "Failed to download NPU driver (Error $?)"
            print_warning "This could be due to:"
            print_warning "  • 403 Forbidden - Access restricted to Intel internal networks"
            print_warning "  • Network connectivity issues"
            print_warning "  • Server maintenance"
            echo
            print_status "Alternative options:"
            print_status "  1. Download manually from Intel's official support site"
            print_status "  2. Use Intel's driver installer from intel.com"
            print_status "  3. Skip NPU installation and continue with GPU drivers only"
            echo
            
            read -p "Do you want to continue without NPU drivers? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                print_status "Continuing with GPU driver installation only..."
                return 0
            else
                print_error "NPU driver installation cancelled. Exiting..."
                exit 1
            fi
        fi
    fi
    
    # Extract NPU driver
    print_status "Extracting NPU driver archive..."
    if [[ ! -f "$NPU_DRIVER_FILENAME" ]]; then
        print_error "NPU driver archive not found. Skipping NPU installation."
        return 0
    fi
    
    if [[ -d "$NPU_DRIVER_DIR" ]]; then
        print_status "Removing existing extraction directory..."
        rm -rf "$NPU_DRIVER_DIR"
    fi
    tar xf "$NPU_DRIVER_FILENAME"
    
    # Install NPU driver
    print_status "Installing Intel NPU driver..."
    if [[ ! -d "$NPU_DRIVER_DIR" ]]; then
        print_error "NPU driver directory not found. Skipping NPU installation."
        return 0
    fi
    
    cd "$NPU_DRIVER_DIR"
    chmod a+x ./npu-drv-installer
    sudo ./npu-drv-installer
    
    print_success "Intel NPU drivers installation completed"
}

# Function to verify installations
verify_installations() {
    print_header "✅ Verifying driver installations..."
    
    # Verify GPU drivers
    print_status "Checking GPU driver installation..."
    if command -v clinfo >/dev/null 2>&1; then
        print_success "OpenCL tools (clinfo) installed successfully"
        print_status "OpenCL devices detected:"
        clinfo -l 2>/dev/null || print_warning "No OpenCL devices found or error querying devices"
    else
        print_warning "clinfo not found - GPU drivers may not be properly installed"
    fi
    
    # Verify VA-API support
    if command -v vainfo >/dev/null 2>&1; then
        print_success "VA-API tools (vainfo) installed successfully"
        print_status "VA-API information:"
        vainfo 2>/dev/null || print_warning "VA-API driver issues detected"
    else
        print_warning "vainfo not found - media acceleration may not work"
    fi
    
    # Check render group membership
    if groups "${USER}" | grep -q render; then
        print_success "User ${USER} is member of render group"
    else
        print_warning "User ${USER} is not in render group - this may affect GPU access"
    fi
    
    # Verify NPU driver
    print_status "Checking NPU driver installation..."
    if lsmod | grep -q intel_vpu; then
        print_success "Intel VPU/NPU kernel module loaded"
    else
        print_warning "Intel VPU/NPU kernel module not loaded - may require reboot or unsupported hardware"
    fi
    
    # Check for NPU device nodes
    if ls /dev/accel* >/dev/null 2>&1; then
        print_success "NPU device nodes found: $(ls /dev/accel*)"
    else
        print_warning "No NPU device nodes found - may require reboot or unsupported hardware"
    fi
}

# Function to cleanup
cleanup() {
    print_header "🧹 Cleaning up temporary files..."
    
    # Return to home directory
    cd "$HOME"
    
    # Remove temporary directory (optional - keep for debugging)
    read -p "Remove temporary installation files? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$TEMP_DIR"
        print_success "Temporary files cleaned up"
    else
        print_status "Temporary files kept at: $TEMP_DIR"
    fi
}

# Function to display completion message
display_completion() {
    print_header "🎉 Driver Installation Complete!"
    echo
    print_success "Intel GPU and NPU drivers have been successfully installed."
    echo
    print_status "Next steps:"
    echo "  • Log out and log back in to apply group membership changes"
    echo "  • Or reboot your system for full driver activation"
    echo "  • Test GPU acceleration with: clinfo"
    echo "  • Test media acceleration with: vainfo"
    echo "  • Check NPU status with: lsmod | grep intel_vpu"
    echo
    print_warning "Some features may require a system reboot to function properly."
    echo
    
    read -p "Do you want to reboot now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Rebooting system..."
        sudo reboot
    else
        print_status "Remember to reboot or re-login when convenient."
        print_status "You can apply group changes immediately with: newgrp render"
    fi
}

# Main function
main() {
    local force_purge=false
    
    # Parse command line arguments
    while getopts "f" opt; do
        case $opt in
            f)
                force_purge=true
                ;;
            \?)
                echo "Usage: $0 [-f]"
                echo "  -f : Force purge of existing Intel drivers before installation"
                exit 1
                ;;
        esac
    done
    # Display welcome message
    clear
    echo "================================================================"
    print_header "🚀 Intel GPU & NPU Driver Installation Script"
    print_header "📋 For Ubuntu 24.04 LTS"
    echo "================================================================"
    echo
    print_status "This script will:"
    echo "  • Install Intel GPU drivers and OpenCL support"
    echo "  • Install Intel media acceleration drivers (VA-API)"
    echo "  • Install Intel NPU (Neural Processing Unit) drivers"
    echo "  • Configure user permissions and verify installations"
    echo
    print_status "Use -f flag to force purge existing drivers before installation"
    echo
    
    # Check if running as root
    check_root
    
    # Start the installation process
    print_header "🚀 Starting driver installation process..."
    echo
    
    # Step 1: Check system compatibility
    check_system
    echo
    
    # Step 2: Purge existing drivers (only with -f flag)
    purge_existing_drivers "$force_purge"
    echo
    echo
    
    # Step 3: Prepare environment
    prepare_environment
    echo
    
    # Step 4: Install GPU drivers
    install_gpu_drivers
    echo
    
    # Step 5: Install NPU drivers
    install_npu_drivers
    echo
    
    # Step 6: Verify installations
    verify_installations
    echo
    
    # Step 7: Cleanup
    cleanup
    echo
    
    # Step 8: Display completion message
    display_completion
}

# Trap to handle script interruption
trap 'print_error "Script interrupted by user"; exit 1' INT

# Run main function
main "$@"
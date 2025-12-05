#!/bin/bash

# ================================================================
# Linux Kernel Installation Script for Ubuntu 24.04
# ================================================================
# This script automates the process of cloning, building, and 
# installing a Linux kernel from source, along with firmware updates.
# 
# Usage: ./kernel_install.sh [-f]
#   -f : Force removal of existing kernel source directory
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
KERNEL_DIR="$HOME/linux-kernel"
KERNEL_REPO="https://github.com/torvalds/linux.git"
JOBS=$(nproc)

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

# Function to install dependencies
install_dependencies() {
    print_header "🔧 Installing build dependencies..."
    
    sudo apt update
    sudo apt install -y \
        build-essential \
        libncurses5-dev \
        libssl-dev \
        libelf-dev \
        bison \
        flex \
        git \
        fakeroot \
        ccache \
        dwarves \
        zstd \
        pahole \
        rsync \
        bc \
        kmod \
        cpio \
        initramfs-tools
    
    print_success "Dependencies installed successfully"
}

# Function to handle kernel source
setup_kernel_source() {
    local force_remove=$1
    
    if [[ "$force_remove" == "true" ]] && [[ -d "$KERNEL_DIR" ]]; then
        print_warning "Removing existing kernel directory: $KERNEL_DIR"
        rm -rf "$KERNEL_DIR"
    fi
    
    if [[ ! -d "$KERNEL_DIR" ]]; then
        print_header "📥 Cloning Linux kernel source..."
        git clone --depth 1 "$KERNEL_REPO" "$KERNEL_DIR"
        print_success "Kernel source cloned successfully"
    else
        print_status "Using existing kernel source at: $KERNEL_DIR"
        cd "$KERNEL_DIR"
        print_status "Updating kernel source..."
        git fetch origin
        git reset --hard origin/master
        print_success "Kernel source updated"
    fi
}

# Function to configure kernel
configure_kernel() {
    print_header "⚙️  Configuring kernel..."
    cd "$KERNEL_DIR"
    
    # Copy current kernel config if available
    if [[ -f "/boot/config-$(uname -r)" ]]; then
        print_status "Using current kernel configuration as base"
        cp "/boot/config-$(uname -r)" .config
        make olddefconfig
    else
        print_status "Creating default configuration"
        make defconfig
    fi
    
    print_success "Kernel configuration completed"
}

# Function to build kernel
build_kernel() {
    print_header "🔨 Building kernel (this may take a while)..."
    cd "$KERNEL_DIR"
    
    print_status "Building with $JOBS parallel jobs"
    
    # Build kernel image
    make -j"$JOBS" 
    
    # Build modules
    make -j"$JOBS" modules
    
    print_success "Kernel build completed successfully"
}

# Function to install kernel
install_kernel() {
    print_header "📦 Installing kernel..."
    cd "$KERNEL_DIR"
    
    # Install modules
    print_status "Installing kernel modules..."
    sudo make modules_install
    
    # Install kernel
    print_status "Installing kernel image..."
    sudo make install
    
    # Update initramfs
    print_status "Updating initramfs..."
    KERNEL_VERSION=$(make kernelrelease)
    sudo update-initramfs -c -k "$KERNEL_VERSION"
    
    # Update GRUB
    print_status "Updating GRUB configuration..."
    sudo update-grub
    
    print_success "Kernel installation completed"
    print_success "New kernel version: $KERNEL_VERSION"
}

# Function to update firmware
update_firmware() {
    print_header "🔄 Updating system firmware..."
    
    # Update linux-firmware package
    sudo apt update
    sudo apt install -y linux-firmware
    
    # Update microcode if available
    if lscpu | grep -q "Intel"; then
        print_status "Installing Intel microcode updates..."
        sudo apt install -y intel-microcode
    elif lscpu | grep -q "AMD"; then
        print_status "Installing AMD microcode updates..."
        sudo apt install -y amd64-microcode
    fi
    
    print_success "Firmware updates completed"
}

# Function to cleanup
cleanup() {
    print_header "🧹 Cleaning up build artifacts..."
    cd "$KERNEL_DIR"
    make clean
    print_success "Cleanup completed"
}

# Function to display completion message
display_completion() {
    print_header "🎉 Kernel Installation Complete!"
    echo
    print_success "Your new kernel has been successfully built and installed."
    print_warning "IMPORTANT: Please reboot your system to use the new kernel."
    echo
    print_status "After reboot, verify your kernel version with: uname -r"
    print_status "If you encounter issues, you can select the previous kernel from GRUB menu."
    echo
    
    read -p "Do you want to reboot now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Rebooting system..."
        sudo reboot
    else
        print_status "Remember to reboot when convenient to use the new kernel."
    fi
}

# Main function
main() {
    local force_remove=false
    
    # Parse command line arguments
    while getopts "f" opt; do
        case $opt in
            f)
                force_remove=true
                ;;
            \?)
                echo "Usage: $0 [-f]"
                echo "  -f : Force removal of existing kernel source directory"
                exit 1
                ;;
        esac
    done
    
    # Display welcome message
    clear
    echo "================================================================"
    print_header "🐧 Linux Kernel Compilation & Installation Script"
    print_header "📋 For Ubuntu 24.04 LTS"
    echo "================================================================"
    echo
    print_status "This script will:"
    echo "  • Install necessary build dependencies"
    echo "  • Clone/update Linux kernel source code"
    echo "  • Configure the kernel build"
    echo "  • Compile the kernel and modules"
    echo "  • Install the new kernel"
    echo "  • Update system firmware"
    echo "  • Update bootloader configuration"
    echo
    echo
    
    # Check if running as root
    check_root
    
    # Start the installation process
    print_header "🚀 Starting kernel installation process..."
    echo
    
    # Step 1: Install dependencies
    install_dependencies
    echo
    
    # Step 2: Setup kernel source
    setup_kernel_source "$force_remove"
    echo
    
    # Step 3: Configure kernel
    configure_kernel
    echo
    
    # Step 4: Build kernel
    build_kernel
    echo
    
    # Step 5: Install kernel
    install_kernel
    echo
    
    # Step 6: Update firmware
    update_firmware
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

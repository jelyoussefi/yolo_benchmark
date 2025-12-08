#!/bin/bash

# ================================================================
# Linux Kernel Installation Script for Ubuntu 24.04 (Enhanced)
# ================================================================
# This script automates the process of cloning, building, and 
# installing a Linux kernel from source, with automatic handling
# of Ubuntu-specific configuration issues.
# 
# Usage: ./install_kernel_improved.sh [-f] [-c]
#   -f : Force removal of existing kernel source directory
#   -c : Clean build (remove .config and build artifacts)
# ================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

print_step() {
    echo -e "${CYAN}➜${NC} $1"
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
        print_success "Skipping clone/update (source already present)"
    fi
}

# Function to fix Ubuntu-specific kernel configuration issues
fix_ubuntu_config() {
    print_header "🔧 Fixing Ubuntu-specific configuration issues..."
    cd "$KERNEL_DIR"
    
    if [[ ! -f .config ]]; then
        print_warning "No .config file found, skipping fixes"
        return
    fi
    
    print_step "Disabling Canonical certificate paths..."
    scripts/config --disable SYSTEM_TRUSTED_KEYS
    scripts/config --disable SYSTEM_REVOCATION_KEYS
    scripts/config --set-str SYSTEM_TRUSTED_KEYS ""
    scripts/config --set-str SYSTEM_REVOCATION_KEYS ""
    
    print_step "Disabling module signing to avoid certificate issues..."
    # Disable all module signing options
    scripts/config --disable MODULE_SIG
    scripts/config --disable MODULE_SIG_ALL
    scripts/config --disable MODULE_SIG_FORCE
    scripts/config --disable MODULE_SIG_KEY
    scripts/config --set-str MODULE_SIG_KEY ""
    scripts/config --set-str MODULE_SIG_HASH "sha256"
    
}

# Function to configure kernel
configure_kernel() {
    print_header "⚙️  Configuring kernel..."
    cd "$KERNEL_DIR"
    
    # Copy current kernel config if available
    if [[ -f "/boot/config-$(uname -r)" ]]; then
        print_status "Using current kernel configuration as base"
        cp "/boot/config-$(uname -r)" .config
        
        # Apply fixes for Ubuntu-specific issues
        fix_ubuntu_config
        
        # Update config for new kernel version
        print_step "Updating configuration for current kernel version..."
        make olddefconfig
    else
        print_status "Creating default configuration"
        make defconfig
    fi
    
    print_success "Kernel configuration completed"
}

# Function to validate configuration
validate_config() {
    print_header "🔍 Validating kernel configuration..."
    cd "$KERNEL_DIR"
    
    # Check for problematic settings
    local issues_found=false
    
    if grep -q 'CONFIG_SYSTEM_TRUSTED_KEYS=.*canonical-certs' .config 2>/dev/null; then
        print_warning "Found Canonical certificate path in config"
        issues_found=true
    fi
    
    if grep -q 'CONFIG_SYSTEM_REVOCATION_KEYS=.*canonical-revoked-certs' .config 2>/dev/null; then
        print_warning "Found Canonical revocation certificate path in config"
        issues_found=true
    fi
    
    if [[ "$issues_found" == "true" ]]; then
        print_warning "Configuration issues detected, re-applying fixes..."
        fix_ubuntu_config
        make olddefconfig
        print_success "Configuration issues resolved"
    else
        print_success "Configuration validated successfully"
    fi
}

# Function to build kernel
build_kernel() {
    print_header "🔨 Building kernel (this may take a while)..."
    cd "$KERNEL_DIR"
    
    print_status "Building with $JOBS parallel jobs"
    print_status "Build started at: $(date)"
    
    local start_time=$(date +%s)
    
    # Build kernel image and modules
    if make -j"$JOBS" 2>&1 | tee build.log; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))
        
        print_success "Kernel build completed successfully"
        print_status "Build time: ${minutes}m ${seconds}s"
    else
        print_error "Kernel build failed!"
        print_error "Check build.log for details"
        print_error "Common issues:"
        echo "  • Missing certificates: Run fix with scripts/config --disable SYSTEM_TRUSTED_KEYS"
        echo "  • Out of disk space: Check available space with df -h"
        echo "  • Missing dependencies: Re-run dependency installation"
        exit 1
    fi
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
    
    print_status "Updating GRUB configuration..."
    sudo update-grub
    
    print_success "Kernel installation completed"
    print_success "New kernel version: $KERNEL_VERSION"
}

# Function to cleanup
cleanup() {
    print_header "🧹 Cleaning up build artifacts..."
    cd "$KERNEL_DIR"
    make clean
    
    # Remove build log if it exists
    if [[ -f build.log ]]; then
        rm -f build.log
    fi
    
    print_success "Cleanup completed"
}

# Function to create backup information
create_backup_info() {
    print_header "📝 Creating kernel information file..."
    
    local info_file="$KERNEL_DIR/KERNEL_INFO.txt"
    cat > "$info_file" << EOF
Kernel Build Information
========================
Build Date: $(date)
Kernel Version: $(cd "$KERNEL_DIR" && make kernelrelease)
Built On: $(hostname)
Built By: $(whoami)
Source: $KERNEL_REPO
Configuration Base: /boot/config-$(uname -r)

Installation Locations:
- Kernel Image: /boot/vmlinuz-$(cd "$KERNEL_DIR" && make kernelrelease)
- System.map: /boot/System.map-$(cd "$KERNEL_DIR" && make kernelrelease)
- Config: /boot/config-$(cd "$KERNEL_DIR" && make kernelrelease)
- Modules: /lib/modules/$(cd "$KERNEL_DIR" && make kernelrelease)/

Recovery Information:
- Previous kernel can be selected from GRUB menu at boot
- Hold SHIFT during boot to show GRUB menu
- Select "Advanced options" and choose previous kernel

To remove this kernel:
sudo apt remove --purge linux-image-$(cd "$KERNEL_DIR" && make kernelrelease)
EOF
    
    print_success "Kernel information saved to: $info_file"
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
    print_status "Hold SHIFT during boot to show the GRUB menu."
    echo
    print_status "Kernel information file: $KERNEL_DIR/KERNEL_INFO.txt"
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

# Function to show system information
show_system_info() {
    print_header "💻 System Information"
    echo
    print_status "Current kernel: $(uname -r)"
    print_status "CPU cores: $JOBS"
    print_status "Architecture: $(uname -m)"
    print_status "Distribution: $(lsb_release -d | cut -f2)"
    echo
    
    # Check disk space
    local available_space=$(df -BG "$HOME" | tail -1 | awk '{print $4}' | sed 's/G//')
    if [[ $available_space -lt 50 ]]; then
        print_warning "Low disk space: ${available_space}GB available"
        print_warning "Kernel build requires at least 50GB of free space"
        read -p "Do you want to continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Installation cancelled"
            exit 1
        fi
    else
        print_success "Disk space: ${available_space}GB available"
    fi
    echo
}

# Main function
main() {
    local force_remove=false
    local clean_build=false
    
    # Parse command line arguments
    while getopts "fc" opt; do
        case $opt in
            f)
                force_remove=true
                ;;
            c)
                clean_build=true
                ;;
            \?)
                echo "Usage: $0 [-f] [-c]"
                echo "  -f : Force removal of existing kernel source directory"
                echo "  -c : Clean build (remove .config and build artifacts)"
                exit 1
                ;;
        esac
    done
    
    # Display welcome message
    clear
    echo "================================================================"
    print_header "🐧 Linux Kernel Compilation & Installation Script (Enhanced)"
    print_header "📋 For Ubuntu 24.04 LTS"
    echo "================================================================"
    echo
    print_status "This script will:"
    echo "  • Install necessary build dependencies"
    echo "  • Clone/update Linux kernel source code"
    echo "  • Fix Ubuntu-specific configuration issues"
    echo "  • Configure the kernel build"
    echo "  • Compile the kernel and modules"
    echo "  • Install the new kernel"
    echo "  • Update bootloader configuration"
    echo
    echo
    
    # Check if running as root
    check_root
    
    # Show system information
    show_system_info
    
    # Start the installation process
    print_header "🚀 Starting kernel installation process..."
    echo
    
    # Step 1: Install dependencies
    install_dependencies
    echo
    
    # Step 2: Setup kernel source
    setup_kernel_source "$force_remove"
    echo
    
    # Clean build if requested
    if [[ "$clean_build" == "true" ]]; then
        print_header "🧹 Performing clean build..."
        cd "$KERNEL_DIR"
        make mrproper
        print_success "Build environment cleaned"
        echo
    fi
    
    # Step 3: Configure kernel
    configure_kernel
    echo
    
    # Step 4: Validate configuration
    validate_config
    echo
    
    # Step 5: Build kernel
    build_kernel
    echo
    
    # Step 6: Install kernel
    install_kernel
    echo
    
    # Step 7: Create backup information
    create_backup_info
    echo
    
    # Step 8: Cleanup
    cleanup
    echo
    
    # Step 9: Display completion message
    display_completion
}

# Trap to handle script interruption
trap 'print_error "Script interrupted by user"; exit 1' INT

# Run main function
main "$@"

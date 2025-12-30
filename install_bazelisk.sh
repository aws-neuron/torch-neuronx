#!/bin/bash

# Script to install bazelisk on Ubuntu.
# Bazelisk is the recommended launcher for Bazel that automatically downloads and runs the appropriate version.
# https://bazel.build/install/bazelisk

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

INSTALL_DIR="/usr/local/bin"

# Detect architecture
ARCH=$(uname -m)
case $ARCH in
    x86_64)
        BAZELISK_ARCH="linux-amd64"
        ;;
    aarch64|arm64)
        BAZELISK_ARCH="linux-arm64"
        ;;
    *)
        print_error "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

print_info "Detected architecture: $ARCH (using $BAZELISK_ARCH)"

# Check if bazelisk is already installed
if command -v bazelisk &> /dev/null; then
    CURRENT_VERSION=$(bazelisk version 2>/dev/null | head -n1 || echo "unknown")
    print_warning "bazelisk is already installed: $CURRENT_VERSION"
    read -p "Do you want to reinstall? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Installation cancelled."
        exit 0
    fi
fi

# Get the latest release version from GitHub API
print_info "Fetching latest bazelisk release information..."
LATEST_VERSION=$(curl -s https://api.github.com/repos/bazelbuild/bazelisk/releases/latest | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')

if [[ -z "$LATEST_VERSION" ]]; then
    print_error "Failed to fetch latest version information"
    exit 1
fi

print_info "Latest version: $LATEST_VERSION"

# Download URL
DOWNLOAD_URL="https://github.com/bazelbuild/bazelisk/releases/download/${LATEST_VERSION}/bazelisk-${BAZELISK_ARCH}"

# Retry configuration
MAX_RETRIES=3
RETRY_DELAY=2  # seconds

# Function to perform installation and verification
install_and_verify() {
    print_info "Downloading bazelisk from: $DOWNLOAD_URL"

    # Download bazelisk
    TEMP_FILE=$(mktemp)
    if ! curl -L -o "$TEMP_FILE" "$DOWNLOAD_URL"; then
        print_error "Failed to download bazelisk"
        rm -f "$TEMP_FILE"
        return 1
    fi

    # Verify the download
    if [[ ! -s "$TEMP_FILE" ]]; then
        print_error "Downloaded file is empty"
        rm -f "$TEMP_FILE"
        return 1
    fi

    # Make it executable
    chmod +x "$TEMP_FILE"

    # Move to installation directory
    if ! mv "$TEMP_FILE" "$INSTALL_DIR/bazelisk"; then
        print_error "Failed to install bazelisk to $INSTALL_DIR. Must run as root user (sudo)."
        rm -f "$TEMP_FILE"
        return 1
    fi

    print_info "bazelisk installed successfully to $INSTALL_DIR/bazelisk"

    # Create bazel symlink if it doesn't exist
    if [[ ! -f "$INSTALL_DIR/bazel" ]]; then
        ln -s "$INSTALL_DIR/bazelisk" "$INSTALL_DIR/bazel"
        print_info "Created bazel symlink"
    fi

    # Verify installation
    print_info "Verifying installation..."
    if ! command -v bazelisk &> /dev/null; then
        print_error "bazelisk command not found in PATH"
        return 1
    fi

    # Check if bazelisk can run and get version (this may download Bazel)
    # Use GitHub releases as fallback if releases.bazel.build has certificate issues
    export BAZELISK_BASE_URL="${BAZELISK_BASE_URL:-https://github.com/bazelbuild/bazel/releases/download}"
    if VERSION_OUTPUT=$(bazelisk version 2>&1); then
        print_info "Installation successful!"
        echo "bazelisk version output:"
        echo "$VERSION_OUTPUT" | head -5
        return 0
    else
        print_error "Failed to get bazelisk version (exit code: $?)"
        echo "Output: $VERSION_OUTPUT" | head -10
        return 1
    fi
}

# Perform installation with retries
for ((attempt=1; attempt<=MAX_RETRIES; attempt++)); do
    print_info "Installation attempt $attempt/$MAX_RETRIES..."
    if install_and_verify; then
        break
    fi

    if [[ $attempt -lt $MAX_RETRIES ]]; then
        print_warning "Installation attempt $attempt/$MAX_RETRIES failed. Retrying in ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
    else
        print_error "Installation failed after $MAX_RETRIES attempts"
        print_info "You may need to restart your shell or run: export PATH=\"$INSTALL_DIR:\$PATH\""
        exit 1
    fi
done

# Check if .bazelversion file exists in current directory
if [[ -f ".bazelversion" ]]; then
    BAZEL_VERSION=$(cat .bazelversion)
    print_info "Found .bazelversion file specifying Bazel version: $BAZEL_VERSION"
    print_info "bazelisk will automatically use this version when run in this directory"
fi

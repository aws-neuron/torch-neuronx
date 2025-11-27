#!/bin/bash

# Function to install Neuron components
install_neuron() {
    echo "Installing Neuron driver, runtime and tools..."

    # Configure Linux for Neuron repository updates
    . /etc/os-release
    sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
    wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

    # Update OS packages
    sudo apt-get update -y

    # Install OS headers
    sudo apt-get install linux-headers-$(uname -r) -y

    # Install git
    sudo apt-get install git -y

    # install Neuron Driver
    sudo apt-get install aws-neuronx-dkms=2.* -y

    # Install Neuron Runtime
    sudo apt-get install aws-neuronx-collectives=2.* -y
    sudo apt-get install aws-neuronx-runtime-lib=2.* -y

    # Install Neuron Tools
    sudo apt-get install aws-neuronx-tools=2.* -y

    # Add PATH
    echo 'export PATH=/opt/aws/neuron/bin:$PATH' >> ~/.bashrc
    export PATH=/opt/aws/neuron/bin:$PATH
    export NRT_LOCAL_PATH=/opt/aws/neuron

    echo "Neuron installation completed."
}

# Function to install EFA
install_efa() {
    echo "Installing EFA driver..."

    # Install EFA Driver
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
    wget https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key
    cat aws-efa-installer.key | gpg --fingerprint
    wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && gpg --verify ./aws-efa-installer-latest.tar.gz.sig
    tar -xvf aws-efa-installer-latest.tar.gz
    cd aws-efa-installer && sudo bash efa_installer.sh --yes
    cd
    sudo rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer

    echo "EFA installation completed."
}

# Main script
echo "Torch Neuron Eager Setup Script (Neuron Dependencies and Optional EFA)"
echo "-------------------------------------"

install_neuron

echo "=== C++ Test Dependencies Installation ==="
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")/tools"
"$SCRIPT_DIR/install-cpp-test-dependencies"


echo "=== Bazel Build Installation ==="
# Check if Bazel is installed
if ! command -v bazel &> /dev/null; then
    echo "Bazel is not installed. Installing..."
    sudo ./install_bazelisk.sh
    if ! command -v bazel &> /dev/null; then
        echo "Error: Failed to install Bazel. Please install manually."
        exit 1
    fi
    echo "Bazel installed successfully at $(which bazel)"
fi

echo "Using: $(bazel version)"

# Install EFA (optional)
read -p "Do you want to install EFA driver? (y/n): " install_efa_choice
if [[ $install_efa_choice =~ ^[Yy]$ ]]; then
    install_efa
fi

echo "Installation process completed."

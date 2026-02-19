# Base image
FROM public.ecr.aws/ubuntu/ubuntu:22.04_stable

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies and add deadsnakes PPA for Python 3.11
RUN apt-get update -y && \
    apt-get install -y \
    wget \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create python symlink to Python 3.11 and install pip for Python 3.11
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Configure Linux for Neuron repository updates
RUN . /etc/os-release && \
    echo "deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" | tee /etc/apt/sources.list.d/neuron.list > /dev/null && \
    wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -

# Update OS packages
RUN apt-get update -y

# Install git and basic build dependencies
RUN apt-get install -y \
    git \
    build-essential \
    python3.11-venv \
    python3-numpy \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Install python3-apt for apt_pkg module and add LLVM repository manually
RUN apt-get update -y && \
    apt-get install -y python3-apt && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" >> /etc/apt/sources.list && \
    apt-get update -y && \
    apt-get install -y clang-18 && \
    rm -rf /var/lib/apt/lists/*

# Install Neuron Runtime (need to update apt cache first)
RUN apt-get update -y && \
    apt-get install -y \
    aws-neuronx-collectives=2.* \
    aws-neuronx-runtime-lib=2.*

# Print Neuron Runtime and Collectives versions
RUN echo "=== Neuron Runtime & Collectives Versions ===" && \
    dpkg -l | grep aws-neuronx

# Install Neuron Tools
RUN apt-get install -y aws-neuronx-tools=2.*

# Print Neuron Tools version
RUN echo "=== Neuron Tools Version ===" && \
    dpkg -l | grep aws-neuronx-tools

# Install EFA Driver (only required for multi-instance training)
# Using --skip-kmod to skip kernel module installation in container
RUN curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz && \
    wget https://efa-installer.amazonaws.com/aws-efa-installer.key && \
    gpg --import aws-efa-installer.key && \
    wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && \
    gpg --verify ./aws-efa-installer-latest.tar.gz.sig && \
    tar -xvf aws-efa-installer-latest.tar.gz && \
    cd aws-efa-installer && \
    bash efa_installer.sh --yes --skip-kmod && \
    cd / && \
    rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer

# Print EFA version
RUN echo "=== EFA Driver Version ===" && \
    /opt/amazon/efa/bin/fi_info --version 2>/dev/null || echo "EFA version check failed - may not be available in container"

# Add PATH
ENV PATH=/opt/aws/neuron/bin:$PATH

# Configure Bazelisk to use GitHub releases instead of releases.bazel.build
# This works around SSL certificate issues with releases.bazel.build
ENV BAZELISK_BASE_URL=https://github.com/bazelbuild/bazel/releases/download

# Set Neuron repository as additional index (PyPI remains primary)
RUN python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# Install compiler (will use PyPI for dependencies and Neuron repo for neuronx-cc)
RUN python -m pip install neuronx-cc==2.23.*

# Install additional dependencies for PyTorch CPU build
RUN apt-get update -y && \
    apt-get install -y \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for PyTorch build
RUN python -m pip install cmake ninja

# Clone PyTorch repository
RUN git clone https://github.com/pytorch/pytorch /code/pytorch
WORKDIR /code/pytorch

# Initialize and update submodules
RUN git submodule sync && \
    git submodule update --init --recursive

# Install PyTorch dependencies
RUN python -m pip install -r requirements.txt

# Install additional dependencies for Linux CPU build
RUN python -m pip install mkl-static mkl-include

# Set environment variables for CPU-only build
ENV USE_CUDA=0
ENV CMAKE_PREFIX_PATH=/usr/local

# Build and install PyTorch from source (CPU-only)
RUN python -m pip install --no-build-isolation -v -e .

# Clone pytorch/xla repository for torchax
RUN git clone https://github.com/pytorch/xla.git /code/torch_xla

# Install torchax from local repository
RUN python -m pip install /code/torch_xla/torchax

# Create and set working directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]

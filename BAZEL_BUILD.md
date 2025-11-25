# Bazel Build  for TorchNeuronEager
## Overview

[Bazel](https://bazel.build/) is a free software tool used for the
automation of building and testing software.
[Pytorch](https://github.com/pytorch/pytorch),
[Pytorch/XLA](https://github.com/pytorch/xla/),
[OpenXLA](https://github.com/openxla/xla), and [StableHLO](https://github.com/openxla/stablehlo) all use it.


## Prerequisites

1. **Bazel Installation**: Install Bazel following the instructions below.
2. **Torch Dependency**: The Bazel configuration automatically detects local PyTorch installation, or installs from source.
3. **Neuron Runtime Dependency**: The Bazel configuration relies on local Neuron Runtime (NRT) installation.


### Installation

Install Bazelisk, the recommended way to install and manage bazel versions
(https://bazel.build/install/bazelisk). This can be done using the script:

```
./install_bazelisk.sh
```

This script is also added to ```prerequisites.sh```.

This installs the bazel version as specified in ```.bazelversion```.

### Build Architecture

1. **`MODULE.bazel`** - Bazel dependency management
2. **`BUILD.bazel`**: Main build targets
3. **`.bazelrc`** - Build configuration, compiler flags, env variables
4. **`torch_neuronx/bazel/`** - bazel build external dependencies setup.
5. **`torch_neuronx/tests/csrc/BUILD.bazel`** - C++ unit tests build targets


### Dependencies
The build dependency and setup configurations are found in the `torch_neuronx/bazel/` directory. This builds in the external pytorch and nrt libraries.
Enviroment variables that dictate versions and locations for these libraries are set in the build flags in `.bazelrc`.

**PyTorch**:
The pytorch build is handled by `bazel/torch_extension.bzl`.
To change pytorch version, local directory, or pull from external source, override the env vars in `bazel.rc` or run bazel build with explicit env vars.

To override this repository with a different local path, can specify with

``` bash
bazel build --override_repository=torch=/path/to/built/torch_repo //...
```

**Neuron Runtime (NRT)**:
The nrt build is handled by `bazel/nrt_extension.bzl`.
To change nrt local directory, override the env vars in `bazel.rc` or run bazel build with explicit env vars.

To override this library with a different local path, can specify with

``` bash
bazel build --override_repository=nrt=/path/to/nrt/library //...
```

## Usage

### Using Bazel Build
Run `./tools/build` to build with bazel and copy extension to `torch_neuronx` directory.
To build with Bazel directly, run bazel build with the desired target:

TorchNeuronx C++ Extension:
``` bash
bazel build //:_C.so
```

Or build a component ```cc_library```:
``` bash
bazel build //:torch_neuronx_core
```

### Running Tests
Use the `./tools/run-cpp-test` script to run bazel tests with proper prefix (LD_LIBRARY_PATH).
No target runs all tests - equivalent to:
```bash
# Run all tests
LD_LIBRARY_PATH='/paths/to/torch/cuda/libs/:$LD_LIBRARY_PATH' bazel test //:cpp_unit_tests
```
To run specific test, run `./tools/run-cpp-test //tests/csrc:{TEST_NAME}`.
Equivalent to:
```bash
# Run NeuronDeviceTest
LD_LIBRARY_PATH='/paths/to/torch/cuda/libs/:$LD_LIBRARY_PATH' bazel test //tests/csrc:NeuronDeviceTest
```

# Contributing guidelines

## Setup Development Environment

```sh
# Clone the repository
git clone <repository-url>
cd TorchNeuronEager

# Install prerequisites
# Installs necessary drivers and bazel (See BAZEL_BUILD.md)
./prerequisites.sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install in editable mode for development
uv pip install -e .

#If permission are needed, execute below command
chmod +x ./tools/*

#Perform the build
./tools/build

## Install neuron compiler from neuron repo
uv pip install neuronx-cc==2.* --extra-index-url https://pip.repos.neuron.amazonaws.com --prerelease=allow --index-strategy unsafe-best-match

# Install development dependencies including Ruff
uv pip install ruff

# Install pre-commit hooks
uv pip install pre-commit
uv run pre-commit install
```

### Building and Testing

**Build only:**
```sh
./tools/build
```

**Build and generate wheel:**
```sh
./tools/build --wheel
```

**Generate wheel only (after build):**
```sh
USE_BAZEL=1 python setup.py bdist_wheel release
```

**Python tests:**
```sh
# Run all tests in parallel
./tools/build-and-test-parallel

# Run all tests sequentially
./tools/build-and-test

# Run specific test
./tools/run-test device/test_device.py

# Pass pytest options
./tools/run-test device/test_device.py -v
```

**C++ tests:**
```sh
# Run all C++ unit tests
./tools/run-cpp-test

# Run specific test
./tools/run-cpp-test KernelExecutionTest

# Verbose output
./tools/run-cpp-test -v

# Rebuild and test
./tools/run-cpp-test -r
```

### Test Reports and Coverage

> [!NOTE]
> Currently all tests are running with the legacy synchronous path which is controlled with TORCH_NEURONX_SYNC_MODE=1

```sh
./tools/run-test-parallel                              # HTML reports (default)
./tools/run-test-parallel --coverage                   # With coverage analysis
./tools/run-test-parallel --skip-distributed           # Skip distributed tests
./tools/run-test-parallel --coverage --skip-distributed
```

**Reports:** `test-reports/report.html`, `test-reports/coverage/index.html`

### Huggingface Tests

Tests top 50 Causal-LM models. Excluded by default (must specify path explicitly):

```sh
./tools/run-test tests/huggingface                    # All HF tests
./tools/run-test tests/huggingface/test_hf_models.py  # Specific test
./tools/run-test-parallel tests/huggingface           # Parallel
```

### Linting and Formatting

```sh
uv run ruff check .                            # Check linting
uv run ruff check . --fix                       # Auto-fix
uv run ruff format .                           # Format Python
clang-format -i torch_neuronx/csrc/**/*.cpp     # Format C++
uv run pre-commit run --all-files               # Run all checks
```

## Expand op coverage

There are two ways to find what ops to implement:

1. **Generate a HTML table of categorized aten ops**: This shows all aten ops with core ops highlighted and already implemented ops struck out.
   ```sh
   uv run python ./tools/aten-functions.py \
       --yaml-path ~/pytorch/aten/src/ATen/native/native_functions.yaml \
       --html-table \
       > ops.html
   ```

2. **Run arbitrary PyTorch code**: Write the PyTorch code you want to be supported, run it with torch_neuronx, and get the list of unimplemented ops required to run that code without CPU fallback from `torch_neuronx_dump/MMDDHHMMSS_PID/offloaded_ops.txt` (or `$TORCH_NEURONX_DUMP/MMDDHHMMSS_PID/offloaded_ops.txt` if the environment variable is set)

Once you pick the op to implement, follow [docs/new_op.md](docs/new_op.md) to implement it.

# Contributing guidelines

## Setup Development Environment

```sh
# Clone the repository
git clone <repository-url>

# Install prerequisites
# Installs necessary drivers and bazel (See BAZEL_BUILD.md)
./prerequisites.sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate

#If permission are needed, execute below command
chmod +x ./tools/*

#Perform the build
./tools/build

## Install neuron compiler from neuron repo
uv pip install 'neuronx-cc==2.*' --extra-index-url https://pip.repos.neuron.amazonaws.com --prerelease=allow --index-strategy unsafe-best-match

## Install Mlir

# Download the wheel matching your python version
# from https://prod.artifactbrowser.brazil.aws.dev/packages/Torch-mlir/versions/

#install the wheel.
uv pip install <torch_mlir-latest-version>-linux_x86_64.whl --no-deps

# Install development dependencies including Ruff
uv pip install ruff

# Install pre-commit hooks
uv pip install pre-commit
pre-commit install
```

> [!NOTE]
> **Why `uv pip` instead of `pip`?** Virtual environments created by `uv` don't include `pip` by default. When your venv is activated, running `pip` will use the system pip (`/usr/bin/pip`) instead of the venv's. Always use `uv pip` for installing packages, or install pip into the venv with `uv pip install pip`.

> [!NOTE]
> **When to use `uv run`?** Use `uv run <command>` when your venv is **not activated** — it automatically finds the venv in current dir and runs the command inside it. If your venv is already activated (check with `echo $VIRTUAL_ENV`), you can run commands directly without `uv run`. Keep in mind that `uv run` looks for `.venv` in the current or parent directories, so it may fail if your venv is located elsewhere.

### Building and Testing

**Build only:**
```sh
./tools/build
```

**Build and generate wheel:**
```sh
./tools/build --wheel
```

**Build documentation:**
```sh
uv sync --group docs
./tools/build --docs
```
Documentation will be generated in `build/docs/html/`.

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
./tools/run-cpp-test  //tests/csrc:KernelExecutionTest

# Verbose output
./tools/run-cpp-test -v

# Rebuild and test
./tools/run-cpp-test -r

# To run the profiler test suite.
./tools/run-cpp-test //:ProfilerTests
```

### Test Reports and Coverage

> [!NOTE]
> Currently all sync tests are running with the flag `NEURON_LAUNCH_BLOCKING=1`

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

Once you pick the op to implement, follow [docs/new_op.md](docs/new_op.md) to implement it.

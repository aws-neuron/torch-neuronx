# Contributing to torch-neuronx

Thank you for your interest in contributing to torch-neuronx! This document covers the guidelines and technical details for contributing to this project.

## Table of Contents

- [Contributing](#contributing)
- [Setup Development Environment](#setup-development-environment)
- [Building](#building)
- [Testing](#testing)
- [Linting and Formatting](#linting-and-formatting)
- [Codebase Structure](#codebase-structure)
- [CI and the PR Lifecycle](#ci-and-the-pr-lifecycle)

## Contributing

### Contribution Policy

- For new features or significant changes, please open an issue first to discuss your proposal. Wait for maintainer approval before submitting a PR.
- AI-generated code is welcome, but the contributor is fully responsible for its correctness, testing, and compliance with project guidelines.

### Contribution Paths

There are several ways to contribute to torch-neuronx:

| Path | Description | Guide |
|------|-------------|-------|
| **Expand Op Coverage** | Implement new ATen operations for the Neuron backend | [docs/decompositions.md](docs/decompositions.md) |
| **Add Decompositions** | Decompose complex ops into simpler primitives | [docs/decompositions.md](docs/decompositions.md) |
| **Bug Fixes & Improvements** | Fix bugs, improve performance, or enhance existing features | — |
| **Statistics & Metrics** | Add instrumentation for debugging and monitoring | [docs/statistics/](docs/statistics/torch_neuronx_stats_readme.md) |

To find ops that need implementation, generate the op coverage table:

```sh
uv run python ./tools/aten-functions.py \
    --yaml-path ~/pytorch/aten/src/ATen/native/native_functions.yaml \
    --html-table \
    > ops.html
```

## Setup Development Environment

### Prerequisites

torch-neuronx requires:
- **Linux** (only supported platform)
- **Python 3.10–3.13**
- **AWS Neuron hardware** (trn1/trn2) for running tests — linting and building work without hardware

### Installation

```sh
# Clone the repository
git clone https://github.com/aws-neuron/torch-neuronx.git
cd torch-neuronx

# Install system prerequisites (drivers, Bazel)
./prerequisites.sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Make tool scripts executable
chmod +x ./tools/*

# Build the project
./tools/build

# Install Neuron compiler
uv pip install 'neuronx-cc==2.*' --extra-index-url https://pip.repos.neuron.amazonaws.com --prerelease=allow --index-strategy unsafe-best-match

# Install development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
uv pip install pre-commit
pre-commit install
```

> [!NOTE]
> **Why `uv pip` instead of `pip`?** Virtual environments created by `uv` don't include `pip` by default. When your venv is activated, running `pip` will use the system pip (`/usr/bin/pip`) instead of the venv's. Always use `uv pip` for installing packages.

> [!NOTE]
> **When to use `uv run`?** Use `uv run <command>` when your venv is **not activated** — it automatically finds the `.venv` in the current directory and runs the command inside it. If your venv is already activated, you can run commands directly.

## Building

torch-neuronx uses Bazel to build C++ extensions. See [BAZEL_BUILD.md](BAZEL_BUILD.md) for detailed build system documentation.

### Common Build Commands

```sh
# Build the C++ extension
./tools/build

# Build and generate a wheel
./tools/build --wheel

# Build for a specific Python version
./tools/build --python 3.12

# Build documentation (Sphinx)
uv sync --group docs
./tools/build --docs
# Output: build/docs/html/index.html
```

### Building with Bazel Directly

```sh
# Build the C++ extension
bazel build //:_C.so

# Build a specific library
bazel build //:torch_neuronx_core
```

## Testing

### Python Tests

torch-neuronx uses pytest with parallel execution support.

```sh
# Run all tests in parallel (requires Neuron hardware)
./tools/run-test-parallel

# Run all tests sequentially
./tools/run-test

# Run a specific test file
./tools/run-test device/test_device.py

# Run with pytest options
./tools/run-test device/test_device.py -v -k "test_add"
```

### C++ Tests

C++ tests use Google Test via Bazel.

```sh
# Run all C++ unit tests
./tools/run-cpp-test

# Run a specific test target
./tools/run-cpp-test //tests/csrc:KernelExecutionTest

# Verbose output
./tools/run-cpp-test -v

# Rebuild and test
./tools/run-cpp-test -r
```

### Test Reports and Coverage

```sh
./tools/run-test-parallel --coverage                    # With coverage analysis
./tools/run-test-parallel --skip-distributed            # Skip distributed tests
./tools/run-test-parallel --coverage --json-report      # Coverage + JSON report
```

Reports are generated in `test-reports/`:
- `test-reports/report.html` — test results
- `test-reports/coverage/index.html` — coverage report

### Testing Without Neuron Hardware

If you don't have access to Neuron hardware, you can still:
- Run linting and formatting checks locally
- Build the project (compilation does not require hardware)
- Write tests and validate they are syntactically correct

The CI system will run your tests on Neuron hardware when an internal developer triggers the test suite on your PR.

## Linting and Formatting

torch-neuronx uses [Ruff](https://docs.astral.sh/ruff/) for Python and [clang-format](https://clang.llvm.org/docs/ClangFormat.html) for C++. All checks are enforced via pre-commit hooks and CI.

```sh
# Run all pre-commit checks
uv run pre-commit run --all-files

# Python linting
uv run ruff check .                # Check for issues
uv run ruff check . --fix          # Auto-fix issues
uv run ruff format .               # Format Python code

# C++ formatting
clang-format -i torch_neuronx/csrc/**/*.cpp
```

### Configuration

- **Ruff**: Configured in `pyproject.toml` — line length 100, Python 3.10 target
- **clang-format**: Configured in `.clang-format`
- **pre-commit**: Configured in `.pre-commit-config.yaml`

## Codebase Structure

```
torch-neuronx/
├── torch_neuronx/              # Main Python package
│   ├── csrc/                   # C++ source (ATen ops, allocator, runtime, distributed)
│   ├── neuron_dynamo_backend/  # torch.compile backend (decompositions, lowering)
│   ├── distributed/            # Distributed training support
│   ├── python_ops/             # Python ATen op implementations
│   ├── kernels/                # NKI kernel implementations
│   └── protos/                 # gRPC protocol definitions
├── tests/                      # Test suite
│   ├── allocator/              # Memory allocator tests
│   ├── device/                 # Device operation tests
│   ├── distributed/            # Distributed tests (multi-device)
│   ├── csrc/                   # C++ unit tests
│   ├── python_ops/             # Python ATen op tests
│   ├── neuron_dynamo_backend/  # torch.compile backend tests
│   ├── nki/                    # NKI kernel tests
│   ├── streams/                # Stream/async execution tests
│   └── pytorch_tests/          # PyTorch upstream compatibility tests
├── bazel/                      # Bazel build configuration
├── docs/                       # Documentation (Sphinx + guides)
├── examples/                   # Usage examples
└── tools/                      # Build, test, and utility scripts
```

### Key Components

| Directory | Purpose |
|-----------|---------|
| `torch_neuronx/csrc/` | C++ core: device management, kernel execution, memory allocation, NRT integration |
| `torch_neuronx/neuron_dynamo_backend/` | `torch.compile` backend: graph capture, StableHLO lowering, compilation |
| `torch_neuronx/distributed/` | Collective operations and distributed training |
| `tests/python_ops/` | Tests for individual ATen op implementations |
| `tests/distributed/` | Multi-device and collective operation tests |
| `tools/` | Developer scripts (`build`, `run-test`, `run-test-parallel`, `run-cpp-test`) |

## CI and the PR Lifecycle

### How CI Works for External Contributors

torch-neuronx tests require Neuron hardware, so CI does **not** run automatically on external PRs. Here's the workflow:

1. **You submit a PR** — lint checks run automatically.
2. **An internal developer reviews your PR** — once reviewed, they will trigger the hardware test suite.
3. **Tests run on Neuron hardware** — results are reported back as commit status checks.
4. **Merge** — once tests pass and the PR is approved, an internal developer merges it.

### What Gets Tested in CI

| Test Suite | Description |
|------------|-------------|
| Lint (pre-commit) | Ruff, clang-format, trailing whitespace, YAML validation |
| ATen Unit Tests | Core op correctness on Neuron hardware |
| C++ Unit Tests | C++ component tests via Bazel |
| Distributed Tests | Multi-device collective operations |

### If Your PR is Stuck

If your PR hasn't received a response within 2 weeks, leave a comment on the PR or mention the assigned reviewer to nudge it back onto the team's radar.

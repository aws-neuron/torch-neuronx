# PyTorch Native Enablement

Run upstream PyTorch tests directly on Neuron — no wrappers, no JSON specs, no dynamic class generation. Tests are cloned from PyTorch, patched minimally for backend registration, and executed with pytest.

## Architecture

```
tests/pytorch_tests_native/
├── conftest.py                 # Overlay setup + skip/xfail hook + test duration
├── neuron_test_registry.py     # Single source of truth for skip/xfail
├── clone_tests.py              # Clone upstream tests (TODO: migrate)
├── torch/testing/_internal/    # Testing library overlay (cloned)
└── test/                       # Upstream PyTorch tests (cloned)
    └── distributed/
        └── _composable/fsdp/   # FSDP v2 tests
```

## Setup

Clone PyTorch tests from a local fork:

```sh
python tests/pytorch_tests_native/clone_device_agnostic_tests.py --source-repo ../pytorch
```

Install the nightly and reinstall TorchNeuronEager:

```sh
# If you want to install another version replace the /cpu with what you want
# Ex: https://download.pytorch.org/whl/nightly/cu126
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu --force-reinstall
cd TorchNeuronEager
uv pip install -e .[dev] --no-deps

# You might need to remove "torch~=2.10" from pyproject.toml in dependencies and then build
uv run ./tools/build
```

## Running Tests

```sh
# Run a single test file
pytest tests/pytorch_tests_native/test/distributed/_composable/fsdp/test_fully_shard_comm.py -v -rs

# Run a single test
pytest tests/pytorch_tests_native/test/distributed/_composable/fsdp/test_fully_shard_comm.py::TestFullyShardForceSumReduction::test_fully_shard_force_sum_both_reductions -v -rs

# Dry run — see which tests are collected
pytest tests/pytorch_tests_native/test/distributed/_composable/fsdp/test_fully_shard_comm.py --collect-only -v
```

Use `-rs` to show skip reasons in the summary. Use `-rx` to show xfail reasons.

## Skip/Xfail Registry

`neuron_test_registry.py` is the single source of truth for all tests that need to be skipped or xfailed on Neuron. Applied automatically via `conftest.py` — no upstream patches needed.

### How it works

Each entry is a `SkipGroup` tied to one root cause. When the root cause is fixed, delete the entire group and all its tests are re-enabled.

```python
SkipGroup(
    reason="NCCL log comparison does not apply to Neuron",
    tests=["test_fully_shard_force_sum_both_reductions"],  # exact match in nodeid
    patterns=["*nccl*"],                                     # glob against test name
    files=["test_sparse.py"],                                # skip entire file
)
```

Three granularity levels:
- **files** — skip all tests in a file
- **tests** — exact test name (matched anywhere in the nodeid)
- **patterns** — glob patterns against the test function name

### Adding entries

Add a `SkipGroup` to `SKIP_GROUPS` (test can't run at all) or `XFAIL_GROUPS` (test runs but produces wrong results):

```python
SkipGroup(
    reason="Brief description of root cause",
    tests=["test_something", "test_another"],
)
```

## Design Principles

1. **No upstream modifications** — all customization via overlay, conftest hooks, and registry
2. **Root-cause grouping** — every skip/xfail is tied to a reason; fix the cause, delete the group
3. **Generic PrivateUse1** — solutions work for any PrivateUse1 backend, not Neuron-hardcoded
4. **100% categorized** — every test is passing, skipped-with-reason, or xfailed-with-root-cause

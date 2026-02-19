# Vertical Tests

Automated testing framework for Neuron ops using PyTorch's `op_db` for comprehensive test input generation.

## Overview

This framework leverages PyTorch's `op_db` and `sample_inputs_func` to automatically generate test inputs, similar to PyTorch's `TestCommon` in `test_ops.py`.

### Key Benefits
- **Automatic op discovery**: Uses `torch._C._dispatch_get_registrations_for_dispatch_key` to find all Neuron-registered ops
- **Automatic input generation**: Reuses PyTorch's battle-tested `sample_inputs_func` for each op
- **Extra ops support**: Custom `OpInfo` definitions for ops not in PyTorch's `op_db` (attention, grouped_mm, etc.)
- **Per-op test methods**: Each op × dtype becomes a standalone pytest (e.g., `test_correctness[add_float32]`)

## Directory Structure
```
tests/vertical/
├── README.md
├── __init__.py
├── conftest.py              # Aggregates perf metrics, provides fixtures
├── neuron_ops.py            # @NeuronOps decorator
├── neuron_op_db.py          # Filtered op_db for Neuron (auto + extra)
├── neuron_extra_ops.py      # Custom OpInfo for ops not in PyTorch's op_db
├── skip_xfail_ops.py        # Skip/xfail lists per test class
├── test_neuron_ops_correctness.py  # Correctness tests (Neuron vs CPU)
├── test_neuron_ops_backward.py  # Backward/gradient tests
└── test_neuron_ops_perf.py  # Performance tests
```

## Running Tests

Vertical tests are excluded from default test runs. Run them explicitly:

### All Vertical Tests

```sh
./tools/run-test-parallel tests/vertical
```

### Correctness Tests

```sh
# Run all correctness tests
./tools/run-test-parallel tests/vertical/test_neuron_ops_correctness.py

# Run specific op
python -m pytest tests/vertical/test_neuron_ops_correctness.py -v -k "add"

# Run specific dtype
python -m pytest tests/vertical/test_neuron_ops_correctness.py -v -k "float32"

# Run specific op+dtype
python -m pytest tests/vertical/test_neuron_ops_correctness.py -v -k "test_correctness[add_float32]"
```

### Backward Tests

Tests gradient computation matches between CPU and Neuron for ops with `supports_autograd=True`.

```sh
# Run all backward tests
./tools/run-test-parallel tests/vertical/test_neuron_ops_backward.py

# Run backward test for specific ops
python -m pytest tests/vertical/test_neuron_ops_backward.py -v -k "scaled_dot_product"
python -m pytest tests/vertical/test_neuron_ops_backward.py -v -k "relu or gelu or silu"
python -m pytest tests/vertical/test_neuron_ops_backward.py -v -k "linear or embedding"

# List all available backward test names
python -m pytest tests/vertical/test_neuron_ops_backward.py --collect-only -q
```

### Performance Tests

```sh
# Run performance tests for specific ops (single worker)
python -m pytest tests/vertical/test_neuron_ops_perf.py -v -k "log_softmax or amax or add" -s

# Run all performance tests in parallel
./tools/run-test-parallel tests/vertical/test_neuron_ops_perf.py
```

Results are aggregated into `neuron_ops_perf.csv`.

#### Output CSV Schema

| Column | Description |
|--------|-------------|
| `timestamp` | ISO timestamp of test run |
| `op_name` | Op name with dtype (e.g., `where[float32]`, `_grouped_mm/nki_2dx2d[bfloat16]`) |
| `ttf_ms` | Max time-to-first-result across samples (ms) |
| `exec_min_ms` | Min execution time across all samples/iterations (ms) |
| `exec_max_ms` | Max execution time across all samples/iterations (ms) |
| `exec_avg_ms` | Average execution time across samples (ms) |

Samples are aggregated per op/dtype unless they have meaningful variant names (e.g., NKI vs MLIR).

Note: NEFF cache is automatically cleared before each perf test to ensure accurate TTF measurements.

## Skip and Expected Failure Lists

Edit `skip_xfail_ops.py` to skip or mark ops as expected failures. Use the test class name to target specific tests:

```python
# skip_xfail_ops.py

# Ops to skip entirely (won't run)
NEURON_SKIP_OPS = {
    # Skip for all test classes
    "broken_op": {"tests": None, "reason": "Crashes on Neuron"},
    # Skip only for perf tests
    "slow_op": {"tests": ["TestNeuronOpsPerformance"], "reason": "Too slow"},
}

# Ops expected to fail (will run but marked as xfail)
NEURON_XFAIL_OPS = {
    # xfail for correctness and backward tests
    "flaky_op": {"tests": ["TestNeuronOps", "TestNeuronOpsBackward"], "reason": "Numerical instability #123"},
}
```

Test class names:
- `TestNeuronOps` - correctness tests
- `TestNeuronOpsBackward` - backward/gradient tests
- `TestNeuronOpsPerformance` - performance tests

## Extra Ops

Ops not in PyTorch's `op_db` are defined in `neuron_extra_ops.py` with custom `sample_inputs_func`:

- `_scaled_dot_product_fused_attention_overrideable` (NKI + MLIR variants)
- `_native_multi_head_attention`
- `_grouped_mm` (NKI 2Dx2D + MLIR 2Dx3D variants)
- `relu`, `gelu`, `silu`, `softplus`, `threshold`
- `linear`, `embedding`, `convolution`
- `_softmax`, `_log_softmax`, `native_dropout`
- `nll_loss_forward`, `one_hot`, `linalg_vector_norm`

## Adding New Extra Ops

Add to `NEURON_EXTRA_OPS` in `neuron_extra_ops.py`:

```python
NEURON_EXTRA_OPS = {
    "my_op": {
        "op": torch.ops.aten.my_op.default,
        "sample_inputs_func": _sample_inputs_my_op,
        "dtypes": (torch.float32, torch.bfloat16),
        "supports_autograd": True,  # Set True if backward is implemented
    },
}
```

## Writing Custom Tests

```python
from tests.vertical.neuron_op_db import get_neuron_op_db, NEURON_DEFAULT_DTYPES
from tests.vertical.neuron_ops import NeuronOps, allocate_to_device

neuron_op_db = get_neuron_op_db(dtypes=NEURON_DEFAULT_DTYPES)

class TestMyFeature:
    @NeuronOps(neuron_op_db, dtypes=NEURON_DEFAULT_DTYPES)
    def test_my_feature(self, op, dtype):
        for sample in op.sample_inputs("cpu", dtype):
            cpu_out = op(sample.input, *sample.args, **sample.kwargs)

            neuron_input = allocate_to_device(sample.input, "neuron")
            neuron_args = tuple(allocate_to_device(a, "neuron") for a in sample.args)
            neuron_out = op(neuron_input, *neuron_args, **sample.kwargs)

            torch.testing.assert_close(cpu_out, neuron_out.cpu())
```

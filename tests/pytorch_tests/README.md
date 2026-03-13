# Guidelines for running upstream PyTorch tests

Framework for executing PyTorch upstream tests against torch_neuronx. The framework:
1. Clones PyTorch test directory from GitHub
2. Auto-imports test modules and classes from the JSON spec
3. Dynamically wraps PyTorch test classes for Neuron device compatibility (appends suffix like `PRIVATEUSE1` or `NEURON`)
4. Executes each test method as a separate pytest

Test configurations are defined in JSON spec files (`specs/`).

## Setup

Clone PyTorch tests before running any tests:

```sh
python tests/pytorch_tests/clone_tests.py --torch-version v2.8.0
```

## Running Tests

### Non-Distributed Tests

```sh
# Run all non-distributed tests from aten_spec.json
tools/run-test pytorch_tests/test_aten.py

# Run with command-line overrides
python tests/pytorch_tests/test_aten.py \
    --test-file-name test_tensor_creation_ops.py \
    --test-class-name TestLikeTensorCreation \
    --class-name-suffix-options PRIVATEUSE1,NEURON \
    --test-method-name test_ones_like_neuron,test_zeros_like_neuron \
    --debug
```

### Distributed Tests

```sh
# Run specific distributed test suite
pytest -n 8 tests/pytorch_tests/distributed/test_c10d_common.py

# Run all distributed tests in parallel
export DISTRIBUTED_TESTS_DEFAULT_TIMEOUT=1200 && pytest -n 8 tests/pytorch_tests/distributed/

# Run with command-line overrides
python tests/pytorch_tests/distributed/test_c10d_common.py \
    --test-file-name distributed/test_c10d_object_collectives.py \
    --test-class-name TestObjectCollectives \
    --class-name-suffix-options NEURON \
    --test-method-name test_send_recv_object_list \
    --debug
```

### Available Command-Line Flags

- `--test-file-name`: Override spec file, specify PyTorch test file path
- `--test-class-name`: Specify test class to run
- `--class-name-suffix-options`: Comma-separated suffixes to try appending to class name (e.g., `PRIVATEUSE1,NEURON`)
- `--test-method-name`: Comma-separated list of test methods to run
- `--torch-version`: PyTorch version to clone (if tests not already cloned)
- `--debug`: Enable debug logging

## JSON Spec File Configuration

Spec files in `specs/` define which tests to run. Example (`c10d_common_spec.json`):

```json
{
    "test_configurations": [
        {
            "file": "distributed/test_c10d_object_collectives.py",
            "class": "TestObjectCollectives",
            "xfail": [
                {
                    "test_send_recv_object_list": {
                        "reason": "Pending investigation"
                    }
                }
            ]
        },
        {
            "folder": "distributed/algorithms/ddp_comm_hooks"
        }
    ]
}
```

### Configuration Options

- `file`: Single test file path (alternative to `folder`)
- `folder`: Directory path to run all `test_*.py` files recursively
- `exclude_subfolders`: List of subdirectory names to exclude from folder scan
- `class`: (Optional) Specific test class. If omitted, auto-discovers all classes
- `class_name_suffix_options`: (Optional) List of suffixes to try appending to class name (e.g., `["PRIVATEUSE1", "NEURON"]`)
- `methods`: (Optional) List of specific methods. If omitted/null, runs all methods
- `xfail`: List of expected failures with reasons
- `skip_tests`: List of tests to skip entirely (similar format to xfail)

### Xfail Formats

```json
// Grouped format (recommended)
"xfail": [{"tests": ["test1", "test2"], "reason": "description"}]

// Individual format
"xfail": [{"test1": {"reason": "desc1"}}, {"test2": {"reason": "desc2"}}]

// External file reference
"xfail": [{"test_list_filename": "failing_test_methods/failures.json", "reason": "tracked separately"}]
```

**Note**: Include empty `xfail: []` array with folder configs to prevent auto-xfail of all tests.

## Adding New Test Suites

1. Create spec file in `specs/` (e.g., `my_test_spec.json`)
2. Create test file in `distributed/` (e.g., `test_my_feature.py`):

```python
import pytest
from common_distributed import NeuronCommonTest, create_test_classes
import torch_neuronx

create_test_classes(NeuronCommonTest, globals(), "my_test_spec.json")

if __name__ == "__main__":
    pytest.main(["-vs", __file__])
```

## Dynamo Tests

Dynamo tests validate `torch.compile` functionality with the neuron backend.

### Running Dynamo Tests

```sh
# Run all dynamo tests
pytest tests/pytorch_tests/test_dynamo.py -v

# Run with parallelization (recommended)
pytest tests/pytorch_tests/test_dynamo.py -v -n 8
```

### Dynamo Test Configuration

Dynamo tests use `specs/dynamo_spec.json` to control which PyTorch dynamo test files run. The framework:
1. Applies neuron-specific patches (`dynamo/neuron_dynamo_patch.py`)
2. Redirects backends (eager, inductor) to neuron
3. Dynamically generates pytest methods for each PyTorch test

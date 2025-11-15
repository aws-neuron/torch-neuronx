# Vertical Tests Guidelines

This framework provides a structured way to test PyTorch ops across different configurations and input parameters.

# Directory Structure
```
tests/vertical/
├── README.md
├── __init__.py
├── op_spec_base.py
├── op_test_result.py
├── test_cli.py
├── test_registry.py
└── vertical_test_base.py
├── shapes
    ├── __init__.py
    ├── shape_test.py
    └── specs.py
```

# Components

1. **TestVerticalWrapper**: Main test class with each method running a single op spec pytest.
2. **VerticalTest**: Vertical test class that takes an op spec, execute the op for different input combinations and asserts expected behavior.
3. **OpSpec**: User defined args and kwargs spec to check for a specific op in a specific vertical test.
4. **VerticalTestsRegistry**: Registry class for auto-discovering and managing test-spec relationships.

## Run tests

- Run specific vertical test(s)
```sh
python tests/vertical/test_cli.py --test-name MyTest
```

- Run specific op spec(s)
```sh
python tests/vertical/test_cli.py --op-spec OpSpec1,OpSpec2
```

- Run all vertical tests against all specs
```sh
./tools/run-test vertical/test_cli.py -s --log-cli-level=INFO
```

- Alternative command to run run all vertical tests:
```sh
./tools/run-test vertical/test_cli.py -s
```

## Debugging

Enable debug logging for more detailed output

```sh
python tests/vertical/test_cli.py --test-name MyTest --debug
```


## Adding a Vertical Test

### Step 1: Create Test Directory Structure
Create a new subdirectory under `tests/vertical/` for your test category:
```
tests/vertical/my_test_category/
├── __init__.py
├── my_test.py
└── specs.py
```

### Step 2: Implement Your Vertical Test
Create a test class that inherits from `VerticalTestBase`:

Define `execute` and `validate_result` methods for the new test class.

### Step 3: Create Op Specs
Define op specs to run with this test:

Each new spec class inherits from OpSpecBase and must include
* op_name
* args_specs
* kwargs_specs

Optionally you can overwrite:
* output_indices to specify which indices which outputs to validate
* generate_combinations method to change default cartesian product behavior
* expected_neuron_op method to specify name of the op expected to run on neuron device


### Step 4: Register Your Components
Ensure both your test class and specifications use the appropriate decorators:
- `@register_test()` for test classes
- `@register_spec("TestName")` for op spec class

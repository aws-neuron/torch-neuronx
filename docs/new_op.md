# Writing new ops

There are three ways to write a new op in torch-neuronx.
- Use C++ when you can implement an ops with a simple neuron runtime call (like memory allocation) or ops that can be implemented by calling existing CPU ops (like creating a view).
- Use JAX/XLA when the op can be expressed as a JIT compiled Jax function.
- Use NKI when the op is hard to express in JAX - like arbitary DMA operations or custom fused kernels.

Since most ops are JAX/XLA ops, this document will discuss that.

## JAX Op

Here's a step-by-step guide to implement a new JAX op. We'll use the `sqrt` (square root) operation as an example to show how existing ops are structured.

### Step 1: Create the XLA Implementation

Create a new file `torch_neuronx/python_ops/xla_ops/sqrt_xla.py`:

```python
import torch
import jax.numpy as jnp
from ...kernels.xla_kernel import TorchNeuronXLAKernel
from ..base import ExecutionResult, OperationImplementation

class SqrtXLAImpl(OperationImplementation):
    """XLA implementation for element-wise square root using JAX"""

    def __init__(self):
        # Define the JAX computation
        def sqrt_computation(x):
            return jnp.sqrt(x)

        # Create the kernel
        self.kernel = TorchNeuronXLAKernel(sqrt_computation, "sqrt")

    def can_handle(self, *args, **kwargs) -> bool:
        """Check if this implementation can handle the given inputs"""
        if len(args) != 1:
            return False

        input_tensor = args[0]

        # Must be on Neuron device
        if not input_tensor.device.type == "neuron":
            return False

        return True

    def execute(self, input: torch.Tensor, *, out=None) -> ExecutionResult:
        """Execute the square root operation"""
        try:
            # Create output tensor if not provided
            if out is None:
                output = torch.empty_like(input)
            else:
                output = out

            # Execute the kernel
            self.kernel(input, output=output)

            return ExecutionResult(success=True, output=output)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))
```

### Step 2: Create the Operation Wrapper

Create a new file `torch_neuronx/python_ops/sqrt.py`:

```python
from .base import Operation
from .xla_ops.sqrt_xla import SqrtXLAImpl

class SqrtOp(Operation):
    """Square root operation for element-wise computation"""

    def _setup_implementations(self):
        """Setup available implementations"""
        self._implementations.append(SqrtXLAImpl())

    @property
    def op_name(self) -> str:
        """Return the operation name"""
        return "sqrt"
```

### Step 3: Register the Operation

Update `torch_neuronx/python_ops/__init__.py`:

1. Add import:
```python
from .sqrt import SqrtOp
```

2. Create operation instance in `register_python_operations()`:
```python
sqrt_op = SqrtOp()
```

3. Define the dispatcher functions:
```python
def sqrt_neuron(self):
    return sqrt_op(self)

def sqrt_out_neuron(self, *, out):
    return sqrt_op(self, out=out)
```

4. Register with PyTorch:
```python
aten_lib.impl("sqrt", sqrt_neuron, "PrivateUse1")
aten_lib.impl("sqrt_", sqrt_neuron, "PrivateUse1")  # In-place variant
aten_lib.impl("sqrt.out", sqrt_out_neuron, "PrivateUse1")
```

### Step 4: Write Tests

Create `torch_neuronx/tests/python_ops/test_sqrt.py`:

```python
import pytest
import torch

class TestSqrt:
    def test_sqrt_basic(self):
        """Test basic square root operation"""
        input_tensor = torch.tensor([4.0, 9.0, 16.0, 25.0], device='neuron')

        result = torch.sqrt(input_tensor)
        expected = torch.tensor([2.0, 3.0, 4.0, 5.0], device='neuron')

        torch.testing.assert_close(result, expected)

    def test_sqrt_with_output(self):
        """Test sqrt with pre-allocated output tensor"""
        input_tensor = torch.tensor([1.0, 4.0, 9.0], device='neuron')
        output = torch.empty_like(input_tensor)

        torch.sqrt(input_tensor, out=output)
        expected = torch.tensor([1.0, 2.0, 3.0], device='neuron')

        torch.testing.assert_close(output, expected)

    def test_sqrt_inplace(self):
        """Test in-place square root operation"""
        tensor = torch.tensor([4.0, 16.0, 64.0], device='neuron')
        expected = torch.tensor([2.0, 4.0, 8.0], device='neuron')

        tensor.sqrt_()

        torch.testing.assert_close(tensor, expected)
```

### Key Points:

1. **JAX Function**: Keep the JAX computation simple and pure - it should only contain mathematical operations. It has to be JIT compilable. Not every op is JIT compilable.
2. **Type Promotion**: For mixed-type operations, use `promote_binary_op` from `torch_neuronx.kernels.type_promotion`. See [Type Promotion Guide](type_promotion.md) for detailed information.
3. **Broadcasting**: Always check and handle broadcasting compatibility. See [Broadcasting Guide](broadcast.md) for detailed information.
4. **Error Handling**: Return `ExecutionResult` with success/failure status
5. **Output Tensor**: Support both creating new tensors and writing to pre-allocated `out` tensors
6. **Device Check**: Ensure all inputs are on the Neuron device before processing

### Understanding Key Components

#### The `can_handle` Method
The `can_handle` method in `OperationImplementation` determines if a specific implementation can process the given inputs:

```python
def can_handle(self, *args, **kwargs) -> bool:
    """Check if this implementation can handle the given inputs"""
    # Common checks include:
    # - Correct number of arguments
    # - Tensors are on Neuron device
    # - Shape compatibility (e.g., broadcasting)
    # - Supported data types
```

While the design supports multiple implementations per operation, currently all operations have only a single implementation. This design allows future extensibility for:
- Optimized implementations for specific cases. Like, different implementations for different tensor layouts
- Fallback handling. For example, try NKI and fall back to XLA is NKI kernel fails. Fall back to op-by-op if even XLA op fails.

#### The `op_name` Parameter in TorchNeuronXLAKernel
The second parameter in `TorchNeuronXLAKernel(computation, "op_name")` is a perf optimization to avoid tracing everytime:

```python
self.kernel = TorchNeuronXLAKernel(sqrt_computation, "sqrt")
```

This string is used for **NEFF caching**:
- Cache key combines: `op_name + tensor_shapes + tensor_dtypes`
- Example: `"sqrt_t0_(10, 20)_torch.float32"`
- Avoids recompilation when calling the same op with same shapes/dtypes
- Without it, the kernel would need to generate HLO first to create a hash

#### Avoiding Name Duplication
Instead of hardcoding the operation name in multiple places, pass it from the Operation to the Implementation:

```python
class SqrtXLAImpl(OperationImplementation):
    def __init__(self, op_name: str):
        self.kernel = TorchNeuronXLAKernel(sqrt_computation, op_name)

class SqrtOp(Operation):
    @property
    def op_name(self) -> str:
        return "sqrt"

    def _setup_implementations(self):
        self._implementations.append(SqrtXLAImpl(self.op_name))
```

#### Kernel Variants for Optimization
A single operation can have multiple kernel variants for different cases:

```python
class AddXLAImpl(OperationImplementation):
    def __init__(self, op_name: str):
        # Optimized kernel for alpha=1
        self.kernel_default = TorchNeuronXLAKernel(
            lambda x, y: x + y,
            f"{op_name}_default"  # "add_default"
        )

        # General kernel for custom alpha
        self.kernel_custom = TorchNeuronXLAKernel(
            lambda x, y, alpha: x + alpha * y,
            f"{op_name}_alpha"    # "add_alpha"
        )
```

This allows:
- Different optimized compilations for different parameter values
- Each variant is cached separately
- Better performance for common cases

#### Error Handling with ExecutionResult
The `ExecutionResult` pattern provides structured error handling without throwing exceptions immediately:

```python
@dataclass
class ExecutionResult:
    success: bool
    output: Any | None = None
    error_msg: str | None = None
```

Every operation implementation must return an `ExecutionResult` from its `execute()` method:

```python
def execute(self, input: torch.Tensor, *, out=None) -> ExecutionResult:
    try:
        # Your operation logic here
        output = torch.empty_like(input) if out is None else out
        self.kernel(input, output=output)
        return ExecutionResult(success=True, output=output)
    except Exception as e:
        return ExecutionResult(success=False, error_msg=str(e))
```

**Why This Pattern?**
1. **Graceful Failure Handling**: Operations can fail without throwing exceptions immediately, allowing the framework to try alternative implementations
2. **Multi-Implementation Support**: If one implementation returns `success=False`, the framework logs the error and tries the next implementation in the priority list
3. **Detailed Error Information**: The `error_msg` field provides context about why an operation failed for debugging
4. **Consistent Interface**: All operations follow the same pattern, making the codebase predictable and maintainable

The `Operation` base class's `__call__` method processes these results:
- Tries each implementation in priority order
- Returns the output if `success=True`
- Logs the error and continues to next implementation if `success=False`
- Raises `RuntimeError` only if all implementations fail

This design enables robust error handling while maintaining clean, predictable code structure across all operations.

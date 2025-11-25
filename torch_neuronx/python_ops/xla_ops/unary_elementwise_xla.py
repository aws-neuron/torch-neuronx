"""XLA implementation of elementwise unary operators."""

import jax.nn as jnn
import jax.numpy as jnp
import torch

from torch_neuronx.kernels import TorchNeuronXLAKernel
from torch_neuronx.python_ops.auto_registration import neuron_unary_op
from torch_neuronx.python_ops.base import ExecutionResult, UnaryOpImplementation

# For ops that don't have a 1:1 translation, please write a custom function and add it
# here.
UNARY_OP_LIST = {
    "log": jnp.log,
    "sin": jnp.sin,
    "cos": jnp.cos,
    "silu": jnn.silu,
    "exp": jnp.exp,
    "abs": jnp.abs,
}

# Mapping of operation names to their corresponding ATen operation names
# To add a new unary operation:
# 1. Add the operation to UNARY_OP_LIST above with its JAX implementation
# 2. Add the operation to this mapping with its ATen operation name(s)
ATEN_OP_MAPPING = {
    "log": ["aten::log.out"],
    "sin": ["aten::sin.out"],
    "cos": ["aten::cos.out"],
    "silu": ["aten::silu.out"],
    "exp": ["aten::exp.out"],
    "abs": ["aten::abs"],
    # Add new operations here with their corresponding ATen operation names
}


class UnaryElementwiseXLAImpl(UnaryOpImplementation):
    """Base XLA implementation of unary operators."""

    def __init__(self, op_name):
        # Define JAX computation
        jax_fn = UNARY_OP_LIST.get(op_name)
        if jax_fn is None:
            raise ValueError(f"{op_name} does not have an equivalent XLA lowering in UNARY_OP_LIST")

        def computation(*inputs):
            return jax_fn(*inputs)

        self.kernel = TorchNeuronXLAKernel(computation, op_name)

    def _execute_impl(self, input: torch.Tensor, out=None) -> ExecutionResult:
        """Execute unary operators using XLA."""
        try:
            # Use provided output tensor or create a new one
            output = (
                torch.empty(input.shape, dtype=input.dtype, device=input.device)
                if out is None
                else out
            )

            # Execute kernel
            self.kernel(input, output=output)

            return ExecutionResult(success=True, output=output)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


# Dynamically create and decorate specialized UnaryElementwiseXLAImpl classes for each operation
for op_name, _jax_fn in UNARY_OP_LIST.items():
    # Get the corresponding ATen operation name(s)
    aten_op_names = ATEN_OP_MAPPING.get(op_name)

    # Raise an error if no ATen operation name is defined for this operation
    if aten_op_names is None:
        raise ValueError(
            f"No ATen operation name defined for '{op_name}'. "
            f"Please add it to ATEN_OP_MAPPING in unary_elementwise_xla.py"
        )

    # Create the specialized implementation class
    class_name = f"{op_name.capitalize()}XLAImpl"

    # Create the class dynamically
    op_class = type(
        class_name,
        (UnaryElementwiseXLAImpl,),
        {"__init__": lambda self, op=op_name: UnaryElementwiseXLAImpl.__init__(self, op)},
    )

    # Apply the neuron_unary_op decorator for each ATen operation name
    for aten_op_name in aten_op_names:
        op_class = neuron_unary_op(aten_op_name)(op_class)

    # Add the class to the module's globals
    globals()[class_name] = op_class

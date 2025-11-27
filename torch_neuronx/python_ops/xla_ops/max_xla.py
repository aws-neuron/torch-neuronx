"""XLA implementation of max/reduction operation."""

import jax.numpy as jnp

from torch_neuronx.python_ops.auto_registration import neuron_reduction_op
from torch_neuronx.python_ops.xla_ops.reduction_op_xla import ReductionXLAImpl


@neuron_reduction_op("aten::max")
@neuron_reduction_op("aten::max.unary_out")
# TODO(apoorvgu): Dim based max requires argmax and argmin
# which will be added at a later date
class MaxXLAImpl(ReductionXLAImpl):
    """XLA implementation of max/reduction operation."""

    def __init__(self):
        def max_computation(input, axis, dtype, keepdims):
            # dtype is intentionally unused
            return jnp.max(input, axis=axis, keepdims=keepdims)

        super().__init__("max", max_computation)

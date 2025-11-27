"""XLA implementation of min/reduction operation."""

import jax.numpy as jnp

from torch_neuronx.python_ops.auto_registration import neuron_reduction_op
from torch_neuronx.python_ops.xla_ops.reduction_op_xla import ReductionXLAImpl


@neuron_reduction_op("aten::min")
@neuron_reduction_op("aten::min.unary_out")
# TODO(apoorvgu): Dim based min requires argmax and argmin
# which will be added at a later date
class MinXLAImpl(ReductionXLAImpl):
    """XLA implementation of min/reduction operation."""

    def __init__(self):
        def min_computation(input, axis, dtype, keepdims):
            # dtype is intentionally unused
            return jnp.min(input, axis=axis, keepdims=keepdims)

        super().__init__("min", min_computation)

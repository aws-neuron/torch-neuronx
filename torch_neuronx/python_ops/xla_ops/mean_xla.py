"""XLA implementation of mean operation."""

import jax.numpy as jnp

from torch_neuronx.python_ops.auto_registration import neuron_reduction_op
from torch_neuronx.python_ops.xla_ops.reduction_op_xla import ReductionXLAImpl


@neuron_reduction_op("aten::mean")
@neuron_reduction_op("aten::mean.dim")
@neuron_reduction_op("aten::mean.out")
@neuron_reduction_op("aten::mean.dtype_out")
class MeanXLAImpl(ReductionXLAImpl):
    """XLA implementation of mean operation."""

    def __init__(self):
        # Mean has no mathematical identity value, but for empty tensors we return NaN
        super().__init__(op_name="mean", reduction_fn=jnp.mean, identity_value=float("nan"))

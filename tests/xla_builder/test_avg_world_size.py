"""Test that AVG reduce derives world_size from replica_groups (no device-to-host sync)."""

from unittest.mock import MagicMock, patch

import torch
from torch._C._distributed_c10d import ReduceOp

from torch_neuronx.python_ops.xla_builder.kernel import XLABuilderKernel


def _make_kernel():
    """Create an XLABuilderKernel with mocked dependencies."""
    with (
        patch("torch_neuronx.python_ops.xla_builder.kernel.XLABuilderCompiler"),
        patch("torch_neuronx.python_ops.xla_builder.kernel.HloCompiler"),
        patch("torch_neuronx.python_ops.xla_builder.kernel.CompilationCache"),
        patch("torch_neuronx.python_ops.xla_builder.kernel.ArgumentProcessor"),
    ):
        kernel = XLABuilderKernel(hlo_fn=lambda: None, op_name="all_reduce")
    return kernel


@patch("torch_neuronx.python_ops.xla_builder.kernel._C._submit_collective_task_to_pipeline")
def test_avg_world_size_from_replica_groups(mock_submit):
    """world_size must come from replica_groups, not from divisor tensor.item()."""
    kernel = _make_kernel()

    replica_groups = [[0, 1, 2, 3]]  # world_size = 4
    data = torch.full((4,), 8.0)
    divisor = MagicMock(spec=torch.Tensor)  # .item() should never be called
    output = torch.full((4,), 8.0)

    kernel._execute_collective_direct(
        non_static_inputs=(data, divisor),
        kwargs={"reduce": "AVG", "replica_groups": replica_groups},
        output_tensors=(output,),
        cpu_fallback_context=None,
    )

    # divisor tensor must not have .item() called (no device-to-host sync)
    divisor.item.assert_not_called()
    # output should be divided by 4 (len of replica_groups[0])
    assert torch.allclose(output, torch.full((4,), 2.0))
    # submit should have received SUM (AVG converted to SUM + post-divide)
    # args: op_name(0), inputs(1), outputs(2), op_type(3), reduce_op(4), ...
    assert mock_submit.call_args[0][4] == ReduceOp.SUM

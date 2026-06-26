import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    track_neuron_ops,
)

# Shared (shape, dim, index) coverage -- mirrors the compile E2E suite
# (TestSelectBackwardDecompositionE2E) and spans both dim-0 (contiguous dest) and
# dim>0 (non-contiguous dest) across 2D/3D/4D.
SELECT_CASES = [
    ((4, 5), 0, 0),  # 2D, select first row (dim 0 -> contiguous dest)
    ((4, 5), 0, 2),  # 2D, select middle row
    ((4, 5), 1, 3),  # 2D, select column (dim 1 -> non-contiguous dest)
    ((3, 4, 5), 0, 1),  # 3D, dim 0
    ((3, 4, 5), 1, 2),  # 3D, dim 1
    ((3, 4, 5), 2, 0),  # 3D, dim 2
    ((2, 3, 4, 5), 0, 0),  # 4D, dim 0
    ((2, 3, 4, 5), 2, 1),  # 4D, dim 2
    ((4, 5), 0, 3),  # 2D, last row -> high pad = 0 (symmetric to index 0)
    ((3, 4, 5), 2, 4),  # 3D, dim 2, last index -> high pad = 0
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# device<->host transfer ops emitted when a kernel round-trips through CPU
_COPY_OPS = ("neuron::copy::neuron_to_cpu", "neuron::copy::cpu_to_neuron")


def _assert_select_grad_matches_cpu(shape, dim, index, dtype):
    """Run select+backward on CPU and neuron; assert the input gradients match."""
    x_cpu = torch.randn(shape, dtype=dtype, requires_grad=True)
    x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

    selected_cpu = torch.select(x_cpu, dim, index)
    selected_neuron = torch.select(x_neuron, dim, index)

    grad_output = torch.randn_like(selected_cpu)
    selected_cpu.backward(grad_output)
    selected_neuron.backward(grad_output.to("neuron"))

    torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad)


def _backward_copy_ops(shape, dim, index, dtype=torch.float32):
    """Return the device<->host copy ops emitted by JUST select_backward.

    randn runs on CPU then moves to device (device='neuron' randn would emit copy
    ops via its normal_ CPU fallback); op tracking is cleared AFTER setup so only
    the backward is measured.
    """
    x_neuron = torch.randn(shape, dtype=dtype).to("neuron").requires_grad_(True)
    out = torch.select(x_neuron, dim, index).sum()
    torch.neuron.synchronize()

    torch_neuronx.clear_op_tracking()  # measure ONLY the backward
    out.backward()
    torch.neuron.synchronize()
    ops = torch_neuronx.get_executed_ops()
    return [o for o in ops if any(c in o for c in _COPY_OPS)]


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestSelectBackward:
    """Eager-mode tests for select_backward on Neuron.

    select_backward is handled by the @register_aten torch_select_backward op
    (torch_mlir/ops/indexing.py), which decomposes to on-device ops -- no C++ eager
    kernel and no CPU round-trip. Covers correctness, op tracking, and no-CPU-bounce
    across dims/shapes/dtypes.
    """

    @pytest.mark.parametrize("shape,dim,index", SELECT_CASES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_select_backward(self, shape, dim, index, dtype):
        """select_backward across dims/shapes/dtypes must:
        1. produce gradients matching CPU (correctness),
        2. run on neuron and be tracked by name (registered via @register_aten),
        3. stay fully on-device with no CPU round-trip.
        """
        with track_neuron_ops():
            _assert_select_grad_matches_cpu(shape, dim, index, dtype)
            assert_op_runs_on_neuron("aten::select_backward")

        # no device<->host round-trip (measured on the backward in isolation)
        copies = _backward_copy_ops(shape, dim, index, dtype)
        assert not copies, f"dim={dim} dtype={dtype} round-tripped through CPU: {copies}"

    @pytest.mark.parametrize("dim", [0, 1, -1, -2])
    def test_select_backward_negative_dim(self, dim):
        """Negative dims must be normalized and match CPU."""
        with track_neuron_ops():
            _assert_select_grad_matches_cpu((3, 4, 5), dim, 1, torch.float32)

    # dim 1 has size 4, so valid negative indices are -1..-4 (-4 normalizes to 0).
    @pytest.mark.parametrize("index", [-1, -2, -4])
    def test_select_backward_negative_index(self, index):
        """Negative indices must be normalized (index % size) and match CPU."""
        with track_neuron_ops():
            _assert_select_grad_matches_cpu((3, 4, 5), 1, index, torch.float32)

    def test_select_backward_1d(self):
        """1D input: select(0, i) -> scalar grad scattered back."""
        with track_neuron_ops():
            _assert_select_grad_matches_cpu((8,), 0, 3, torch.float32)

    def test_select_backward_with_computation(self):
        """select followed by elementwise computation."""
        with track_neuron_ops():
            x_cpu = torch.randn(4, 5, requires_grad=True)
            x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

            out_cpu = (torch.select(x_cpu, 0, 1) * 2 + 1).sum()
            out_neuron = (torch.select(x_neuron, 0, 1) * 2 + 1).sum()

            out_cpu.backward()
            out_neuron.backward()

            torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad)

    def test_select_backward_multiple_selects(self):
        """Multiple selects on the same tensor -> grads accumulate into one buffer.

        Mirrors the attention qkv[0]/qkv[1]/qkv[2] pattern: three dim-0
        select_backwards scattering into the same input gradient.
        """
        with track_neuron_ops():
            x_cpu = torch.randn(3, 4, 5, requires_grad=True)
            x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

            out_cpu = (x_cpu[0] + x_cpu[1] + x_cpu[2]).sum()
            out_neuron = (x_neuron[0] + x_neuron[1] + x_neuron[2]).sum()

            out_cpu.backward()
            out_neuron.backward()

            torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad)

    def test_select_backward_non_contiguous_grad(self):
        """Gradient flowing into select_backward may itself be non-contiguous.

        The transpose makes the incoming grad_output strided, exercising a
        non-contiguous SOURCE into a (dim-0, contiguous) destination.
        """
        with track_neuron_ops():
            x_cpu = torch.randn(3, 4, 5, requires_grad=True)
            x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

            out_cpu = torch.select(x_cpu, 0, 1).transpose(0, 1).sum()
            out_neuron = torch.select(x_neuron, 0, 1).transpose(0, 1).sum()

            out_cpu.backward()
            out_neuron.backward()

            torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad)

    def test_select_backward_attention_shape(self):
        """qkv[0]/qkv[1]/qkv[2] selects at attention shape, backprop through the
        selects only (explicit grad_outputs) to isolate select_backward."""
        with track_neuron_ops():
            shape = (3, 8, 6, 256, 64)  # (3, B, heads, N, head_dim)
            qkv_cpu = torch.randn(shape, requires_grad=True)
            qkv_neuron = qkv_cpu.detach().clone().to("neuron").requires_grad_(True)

            grad_outs = [torch.randn(shape[1:]) for _ in range(3)]

            torch.autograd.backward([qkv_cpu[0], qkv_cpu[1], qkv_cpu[2]], grad_outs)
            torch.autograd.backward(
                [qkv_neuron[0], qkv_neuron[1], qkv_neuron[2]],
                [g.to("neuron") for g in grad_outs],
            )

            torch.testing.assert_close(qkv_neuron.grad.cpu(), qkv_cpu.grad)

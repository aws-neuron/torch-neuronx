import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

# (shape, dim). The split dimension must be even. glu_backward splits ``self``
# along ``dim`` into two equal halves, so ``grad_output`` has half the size there.
SHAPE_DIM_CONFIGS = [
    pytest.param((8,), 0, id="1d_dim0"),
    pytest.param((4, 8), 1, id="2d_dim1"),
    pytest.param((4, 8), -1, id="2d_negdim"),
    pytest.param((8, 4), 0, id="2d_dim0"),
    pytest.param((2, 3, 8), -1, id="3d_lastdim"),
    pytest.param((2, 8, 4), 1, id="3d_middim"),
    pytest.param((4, 6, 8, 10), 2, id="4d_dim2"),
    pytest.param((2,), 0, id="1d_min_even"),
    pytest.param((32, 128), 1, id="2d_medium"),
    pytest.param((10000,), 0, id="1d_large"),
    pytest.param((8, 64, 32, 32), 3, id="4d_large"),
]

DTYPE_TOLERANCE_CONFIGS = [
    pytest.param(torch.float32, 1e-5, 1.3e-6, id="float32"),
    pytest.param(torch.float16, 1e-2, 1e-2, id="float16"),
    pytest.param(torch.bfloat16, 1e-1, 1e-2, id="bfloat16"),
]


def _grad_output_shape(shape, dim):
    out = list(shape)
    out[dim] = out[dim] // 2
    return tuple(out)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestGluBackward:
    """Test aten::glu_backward on Neuron against the CPU reference."""

    @pytest.mark.parametrize("shape, dim", SHAPE_DIM_CONFIGS)
    @pytest.mark.parametrize("dtype, atol, rtol", DTYPE_TOLERANCE_CONFIGS)
    def test_glu_backward_matches_cpu(self, shape, dim, dtype, atol, rtol):
        """glu_backward runs on Neuron and matches the CPU reference."""
        torch.manual_seed(0)
        self_cpu = torch.randn(shape, dtype=dtype)
        grad_output_cpu = torch.randn(_grad_output_shape(shape, dim), dtype=dtype)

        expected = torch.ops.aten.glu_backward(grad_output_cpu, self_cpu, dim)

        with track_neuron_ops():
            got = torch.ops.aten.glu_backward(
                grad_output_cpu.to("neuron"), self_cpu.to("neuron"), dim
            )
            torch.neuron.synchronize()
        assert_op_runs_on_neuron("aten::glu_backward")

        assert got.dtype == dtype
        torch.testing.assert_close(got.cpu(), expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("shape, dim", [((4, 8), 1), ((2, 3, 8), -1), ((8, 4), 0)])
    def test_glu_autograd_end_to_end(self, shape, dim):
        """Autograd through glu forward drives glu_backward on Neuron."""
        torch.manual_seed(1)
        x_cpu = torch.randn(shape, dtype=torch.float32, requires_grad=True)
        x_neuron = x_cpu.detach().to("neuron").requires_grad_(True)

        out_cpu = torch.nn.functional.glu(x_cpu, dim=dim)
        out_cpu.sum().backward()

        with track_neuron_ops():
            out_neuron = torch.nn.functional.glu(x_neuron, dim=dim)
            out_neuron.sum().backward()
            torch.neuron.synchronize()
        # The backward op must run on Neuron, not fall back to CPU.
        assert_op_runs_on_neuron("aten::glu_backward")

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("shape, dim", [((4, 8), 1), ((2, 8, 4), 1)])
    def test_glu_backward_grad_input_variant(self, shape, dim):
        """The .grad_input overload writes the buffer and leaves self untouched."""
        torch.manual_seed(2)
        self_neuron = torch.randn(shape, dtype=torch.float32, device="neuron")
        grad_output_neuron = torch.randn(
            _grad_output_shape(shape, dim), dtype=torch.float32, device="neuron"
        )
        grad_input = torch.empty(shape, dtype=torch.float32, device="neuron")
        self_before = self_neuron.clone()

        with track_neuron_ops():
            result = torch.ops.aten.glu_backward.grad_input(
                grad_output_neuron, self_neuron, dim, grad_input=grad_input
            )
            torch.neuron.synchronize()
        assert_op_runs_on_neuron("aten::glu_backward")

        expected = torch.ops.aten.glu_backward(grad_output_neuron, self_neuron, dim)
        torch.testing.assert_close(result.cpu(), expected.cpu())
        torch.testing.assert_close(grad_input.cpu(), expected.cpu())
        # out= must not corrupt self.
        torch.testing.assert_close(self_neuron.cpu(), self_before.cpu())

    def test_glu_backward_noncontiguous_grad_input(self):
        """A non-contiguous grad_input buffer receives the correct value."""
        torch.manual_seed(3)
        shape, dim = (4, 8), 1
        self_neuron = torch.randn(shape, dtype=torch.float32, device="neuron")
        grad_output_neuron = torch.randn(4, 4, dtype=torch.float32, device="neuron")
        # A transposed view is non-contiguous.
        grad_input = torch.empty(8, 4, dtype=torch.float32, device="neuron").t()

        with track_neuron_ops():
            result = torch.ops.aten.glu_backward.grad_input(
                grad_output_neuron, self_neuron, dim, grad_input=grad_input
            )
            torch.neuron.synchronize()
        assert_op_runs_on_neuron("aten::glu_backward")

        expected = torch.ops.aten.glu_backward(grad_output_neuron, self_neuron, dim)
        torch.testing.assert_close(result.cpu(), expected.cpu())
        torch.testing.assert_close(grad_input.cpu(), expected.cpu())

    def test_glu_backward_odd_dimension_raises(self):
        """glu_backward rejects an odd split dimension, like aten."""
        self_neuron = torch.ones(4, 7, dtype=torch.float32, device="neuron")
        grad_output_neuron = torch.ones(4, 3, dtype=torch.float32, device="neuron")
        with pytest.raises(RuntimeError, match="[Hh]alving dimension must be even"):
            torch.ops.aten.glu_backward(grad_output_neuron, self_neuron, 1)

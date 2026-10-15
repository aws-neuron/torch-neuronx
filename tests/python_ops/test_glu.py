"""Tests for aten::glu forward (gated linear unit)."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

# (shape, dim). glu halves ``self`` along ``dim``, so that dimension must be even.
SHAPE_DIM_CONFIGS = [
    pytest.param((8,), 0, id="1d_dim0"),
    pytest.param((2,), 0, id="1d_min_even"),
    pytest.param((4, 8), 1, id="2d_dim1"),
    pytest.param((4, 8), -1, id="2d_negdim"),
    pytest.param((8, 5), 0, id="2d_dim0_odd_freedim"),
    pytest.param((2, 3, 8), -1, id="3d_lastdim"),
    pytest.param((2, 8, 4), 1, id="3d_middim"),
    pytest.param((4, 6, 8, 10), 2, id="4d_dim2"),
    pytest.param((32, 128), 1, id="2d_medium"),
    pytest.param((1048576, 2), -1, id="large_1m"),
    pytest.param((8, 64, 32, 32), 3, id="4d_large"),
]

DTYPE_TOLERANCE_CONFIGS = [
    pytest.param(torch.float32, 1e-5, 1.3e-6, id="float32"),
    pytest.param(torch.float16, 1e-2, 1e-2, id="float16"),
    pytest.param(torch.bfloat16, 1e-1, 1e-2, id="bfloat16"),
]


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestGlu:
    """Test aten::glu and aten::glu.out on Neuron against the CPU reference."""

    @pytest.fixture
    def device(self):
        return torch.device("neuron")

    @pytest.mark.parametrize("shape, dim", SHAPE_DIM_CONFIGS)
    @pytest.mark.parametrize("dtype, atol, rtol", DTYPE_TOLERANCE_CONFIGS)
    def test_glu_matches_cpu(self, device, shape, dim, dtype, atol, rtol):
        """glu runs on Neuron and matches the CPU reference across shapes/dtypes."""
        torch.manual_seed(0)
        x_cpu = torch.randn(shape, dtype=dtype)
        expected = torch.nn.functional.glu(x_cpu, dim=dim)

        with track_neuron_ops():
            got = torch.nn.functional.glu(x_cpu.to(device), dim=dim)
            torch.neuron.synchronize()
        assert_op_runs_on_neuron("aten::glu")

        assert got.dtype == dtype
        expected_shape = list(shape)
        expected_shape[dim] = expected_shape[dim] // 2
        assert tuple(got.shape) == tuple(expected_shape)
        torch.testing.assert_close(got.cpu(), expected, atol=atol, rtol=rtol)

    def test_glu_default_dim(self, device):
        """The default dim is -1 (the last dimension)."""
        with track_neuron_ops():
            x = torch.randn(3, 8, device=device)
            got = torch.nn.functional.glu(x)
            torch.neuron.synchronize()
        assert_op_runs_on_neuron("aten::glu")
        torch.testing.assert_close(got.cpu(), torch.nn.functional.glu(x.cpu()))

    def test_glu_float64_downcast(self, device):
        """A float64 input keeps its dtype (computed at float32 precision).

        Neuron has no native float64, so compare at a relaxed tolerance.
        """
        with track_neuron_ops():
            x = torch.randn(4, 8, dtype=torch.float64, device=device)
            got = torch.nn.functional.glu(x, dim=1)
            torch.neuron.synchronize()
        assert_op_runs_on_neuron("aten::glu")

        assert got.dtype == torch.float64
        expected = torch.nn.functional.glu(x.cpu(), dim=1)
        torch.testing.assert_close(got.cpu().float(), expected.float(), rtol=1e-4, atol=1e-4)

    def test_glu_special_values(self, device):
        """Test inf, -inf and nan inputs. glu = a * sigmoid(b)."""
        with track_neuron_ops():
            x = torch.tensor(
                [[float("inf"), -1.0, float("-inf"), 2.0, float("nan"), 0.5]],
                dtype=torch.float32,
                device=device,
            )
            got = torch.nn.functional.glu(x, dim=1)
            torch.neuron.synchronize()
        assert_op_runs_on_neuron("aten::glu")
        expected = torch.nn.functional.glu(x.cpu(), dim=1)
        torch.testing.assert_close(got.cpu(), expected, equal_nan=True)

    def test_glu_empty_free_dim(self, device):
        """An empty tensor (numel=0) with an even split dimension is valid."""
        with track_neuron_ops():
            x = torch.empty(0, 8, device=device)
            got = torch.nn.functional.glu(x, dim=1)
            torch.neuron.synchronize()
        assert_op_runs_on_neuron("aten::glu")
        assert tuple(got.shape) == (0, 4)

    def test_glu_channels_last(self, device):
        """glu keeps a channels-last input layout, like aten."""
        x = torch.randn(2, 8, 4, 4, device=device).to(memory_format=torch.channels_last)
        with track_neuron_ops():
            got = torch.nn.functional.glu(x, dim=1)
            torch.neuron.synchronize()
        assert_op_runs_on_neuron("aten::glu")
        assert got.is_contiguous(memory_format=torch.channels_last)
        torch.testing.assert_close(got.cpu(), torch.nn.functional.glu(x.cpu(), dim=1))

    def test_glu_out_variant(self, device):
        """The .out variant writes the value and aliases the out tensor."""
        with track_neuron_ops():
            x = torch.randn(4, 8, device=device)
            out = torch.empty(4, 4, device=device)
            result = torch.ops.aten.glu(x, 1, out=out)
            torch.neuron.synchronize()
        assert_op_runs_on_neuron("aten::glu")

        assert out.data_ptr() == result.data_ptr()
        torch.testing.assert_close(out.cpu(), torch.nn.functional.glu(x.cpu(), dim=1))

    def test_glu_out_leaves_self_untouched(self, device):
        """The .out variant does not corrupt the input tensor."""
        with track_neuron_ops():
            x = torch.randn(4, 8, device=device)
            self_before = x.clone()
            out = torch.empty(4, 4, device=device)
            torch.ops.aten.glu(x, 1, out=out)
            torch.neuron.synchronize()
        assert_op_runs_on_neuron("aten::glu")
        torch.testing.assert_close(x.cpu(), self_before.cpu())

    def test_glu_out_noncontiguous(self, device):
        """The .out variant writes a non-contiguous view correctly."""
        with track_neuron_ops():
            x = torch.randn(4, 8, device=device)
            base = torch.empty(4, 4, 2, device=device)
            out = base[:, :, 0]
            assert not out.is_contiguous()
            torch.ops.aten.glu(x, 1, out=out)
            torch.neuron.synchronize()
        assert_op_runs_on_neuron("aten::glu")
        torch.testing.assert_close(out.cpu(), torch.nn.functional.glu(x.cpu(), dim=1))

    def test_glu_out_dtype_mismatch_raises(self, device):
        """A non-float out= dtype for the float glu must raise RuntimeError.

        CPU allows a float-to-float out= downcast but rejects a float-to-integer
        out=. The Neuron dispatch path skips PyTorch's structured-kernel guard,
        so the lowering validates this itself.
        """
        x = torch.randn(4, 8, device=device)
        out = torch.empty(4, 4, dtype=torch.int32, device=device)
        with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
            torch.ops.aten.glu(x, 1, out=out)

    def test_glu_odd_dimension_raises(self, device):
        """glu rejects an odd split dimension, like aten."""
        x = torch.ones(4, 7, device=device)
        with pytest.raises(RuntimeError, match="[Hh]alving dimension must be even"):
            torch.nn.functional.glu(x, dim=1)

    def test_glu_zero_dim_raises(self, device):
        """glu rejects a 0-dimensional input, like aten."""
        x = torch.tensor(1.0, device=device)
        with pytest.raises(RuntimeError, match="0-dimensional"):
            torch.ops.aten.glu(x, 0)

    def test_glu_integer_raises(self, device):
        """glu is float-only; an integer input must raise, like aten."""
        x = torch.ones(4, 8, dtype=torch.int32, device=device)
        with pytest.raises(RuntimeError, match="not implemented for"):
            torch.nn.functional.glu(x, dim=1)

    def test_glu_no_recompilation(self, device):
        """Calling glu twice with the same shape adds no compilation cache entry."""
        from torch_neuronx.python_ops.kernel_cache import kernel_cache_size

        with track_neuron_ops():
            x = torch.randn(64, 64, device=device)
            torch.nn.functional.glu(x, dim=1)
            torch.neuron.synchronize()
            cache_after_first = kernel_cache_size()

            torch.nn.functional.glu(x, dim=1)
            torch.neuron.synchronize()
            cache_after_second = kernel_cache_size()

        assert cache_after_second == cache_after_first
        assert_op_runs_on_neuron("aten::glu")

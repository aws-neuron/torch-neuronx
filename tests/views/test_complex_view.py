"""Tests for view_as_real and view_as_complex operations"""

import pytest
import torch

import torch_neuronx

from .view_test_utils import (
    assert_storage_shared,
    is_view_of,
)


class TestViewAsReal:
    """Test view_as_real operation - converts complex tensor to real tensor"""

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    @pytest.mark.parametrize("shape", [(4,), (3, 4), (2, 3, 4)])
    def test_view_as_real_matches_cpu(self, dtype, shape):
        """Test shape, dtype, stride, values, and view properties match CPU."""
        x_cpu = torch.randn(*shape, dtype=dtype)
        x_neuron = x_cpu.to("neuron")

        y_cpu = torch.view_as_real(x_cpu)
        y_neuron = torch.view_as_real(x_neuron)

        assert y_neuron.shape == y_cpu.shape
        assert y_neuron.dtype == y_cpu.dtype
        assert y_neuron.stride() == y_cpu.stride()
        assert is_view_of(x_neuron, y_neuron)
        assert_storage_shared(x_neuron, y_neuron)
        torch.testing.assert_close(y_neuron.cpu(), y_cpu)

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_view_as_real_data_consistency(self, dtype):
        """Test that real/imag parts map correctly and match CPU."""
        float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        real_cpu = torch.tensor([1.0, 2.0, 3.0], dtype=float_dtype)
        imag_cpu = torch.tensor([4.0, 5.0, 6.0], dtype=float_dtype)

        x_cpu = torch.complex(real_cpu, imag_cpu)
        x_neuron = x_cpu.to("neuron")

        y_cpu = torch.view_as_real(x_cpu)
        y_neuron = torch.view_as_real(x_neuron)

        torch.testing.assert_close(y_neuron.cpu(), y_cpu)

    def test_view_as_real_modification_propagates(self):
        """Test that modifications to real view affect the complex tensor."""
        x = torch.randn(4, dtype=torch.complex64, device="neuron")
        y = torch.view_as_real(x)

        modification = torch.tensor(100.0, dtype=y.dtype, device=y.device)
        y[0, 0].copy_(modification)
        assert x[0].real.item() == 100.0

        modification = torch.tensor(200.0, dtype=y.dtype, device=y.device)
        y[1, 1].copy_(modification)
        assert x[1].imag.item() == 200.0

    def test_view_as_real_empty_tensor(self):
        x_cpu = torch.empty(0, dtype=torch.complex64)
        x_neuron = x_cpu.to("neuron")

        y_cpu = torch.view_as_real(x_cpu)
        y_neuron = torch.view_as_real(x_neuron)

        assert y_neuron.shape == y_cpu.shape
        assert is_view_of(x_neuron, y_neuron)

    def test_view_as_real_scalar_tensor(self):
        x_cpu = torch.tensor(3.0 + 4.0j, dtype=torch.complex64)
        x_neuron = x_cpu.to("neuron")

        y_cpu = torch.view_as_real(x_cpu)
        y_neuron = torch.view_as_real(x_neuron)

        assert y_neuron.shape == y_cpu.shape
        torch.testing.assert_close(y_neuron.cpu(), y_cpu)


class TestViewAsComplex:
    """Test view_as_complex operation - converts real tensor to complex tensor"""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("shape", [(4, 2), (3, 4, 2), (2, 3, 4, 2)])
    def test_view_as_complex_matches_cpu(self, dtype, shape):
        """Test shape, dtype, stride, values, and view properties match CPU."""
        x_cpu = torch.randn(*shape, dtype=dtype)
        x_neuron = x_cpu.to("neuron")

        y_cpu = torch.view_as_complex(x_cpu)
        y_neuron = torch.view_as_complex(x_neuron)

        assert y_neuron.shape == y_cpu.shape
        assert y_neuron.dtype == y_cpu.dtype
        assert y_neuron.stride() == y_cpu.stride()
        assert is_view_of(x_neuron, y_neuron)
        assert_storage_shared(x_neuron, y_neuron)
        torch.testing.assert_close(y_neuron.cpu(), y_cpu)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_view_as_complex_data_consistency(self, dtype):
        """Test that real/imag parts are correctly interpreted and match CPU."""
        x_cpu = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=dtype)
        x_neuron = x_cpu.to("neuron")

        y_cpu = torch.view_as_complex(x_cpu)
        y_neuron = torch.view_as_complex(x_neuron)

        torch.testing.assert_close(y_neuron.cpu(), y_cpu)

    def test_view_as_complex_modification_propagates(self):
        """Test that modifications to complex view affect the real tensor."""
        x = torch.zeros(4, 2, dtype=torch.float32, device="neuron")
        y = torch.view_as_complex(x)

        new_value = torch.tensor(10.0 + 20.0j, dtype=torch.complex64, device="neuron")
        y[0].copy_(new_value)

        assert abs(x[0, 0].item() - 10.0) < 1e-5
        assert abs(x[0, 1].item() - 20.0) < 1e-5

    def test_view_as_complex_requires_last_dim_size_2(self):
        x = torch.randn(4, 3, dtype=torch.float32, device="neuron")
        with pytest.raises(RuntimeError, match="last dimension.*size 2"):
            torch.view_as_complex(x)

    def test_view_as_complex_requires_contiguous_last_dim(self):
        x = torch.randn(2, 4, dtype=torch.float32, device="neuron").transpose(0, 1)
        with pytest.raises(RuntimeError, match="last dimension with stride 1"):
            torch.view_as_complex(x)

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128, torch.int32])
    def test_view_as_complex_on_non_real_fails(self, dtype):
        if dtype in [torch.complex64, torch.complex128]:
            x = torch.randn(4, 2, dtype=dtype, device="neuron")
        else:
            x = torch.randint(0, 10, (4, 2), dtype=dtype, device="neuron")
        with pytest.raises(
            RuntimeError, match="view_as_complex is only supported for half, float and double"
        ):
            torch.view_as_complex(x)

    def test_view_as_complex_empty_tensor(self):
        x_cpu = torch.empty(0, 2, dtype=torch.float32)
        x_neuron = x_cpu.to("neuron")

        y_cpu = torch.view_as_complex(x_cpu)
        y_neuron = torch.view_as_complex(x_neuron)

        assert y_neuron.shape == y_cpu.shape
        assert is_view_of(x_neuron, y_neuron)


class TestViewAsRealComplexRoundTrip:
    """Test round-trip conversions between complex and real views"""

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    @pytest.mark.parametrize("shape", [(4,), (3, 4), (2, 3, 4)])
    def test_complex_to_real_to_complex(self, dtype, shape):
        x_cpu = torch.randn(*shape, dtype=dtype)
        x_neuron = x_cpu.to("neuron")

        y_neuron = torch.view_as_complex(torch.view_as_real(x_neuron))
        y_cpu = torch.view_as_complex(torch.view_as_real(x_cpu))

        torch.testing.assert_close(y_neuron.cpu(), y_cpu)
        assert_storage_shared(x_neuron, y_neuron)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_real_to_complex_to_real(self, dtype):
        x_cpu = torch.randn(3, 4, 2, dtype=dtype)
        x_neuron = x_cpu.to("neuron")

        y_neuron = torch.view_as_real(torch.view_as_complex(x_neuron))
        y_cpu = torch.view_as_real(torch.view_as_complex(x_cpu))

        torch.testing.assert_close(y_neuron.cpu(), y_cpu)
        assert_storage_shared(x_neuron, y_neuron)

    def test_modification_through_round_trip(self):
        x = torch.randn(4, dtype=torch.complex64, device="neuron")
        x_rt = torch.view_as_complex(torch.view_as_real(x))

        new_value = torch.tensor(99.0 + 88.0j, dtype=torch.complex64, device="neuron")
        x_rt[0].copy_(new_value)

        result = x[0].item()
        assert abs(result.real - 99.0) < 1e-5 and abs(result.imag - 88.0) < 1e-5


class TestViewAsRealComplexEdgeCases:
    """Test edge cases"""

    def test_requires_grad_preserved(self):
        x = torch.randn(4, dtype=torch.complex64, device="neuron", requires_grad=True)
        assert torch.view_as_real(x).requires_grad

        x = torch.randn(4, 2, dtype=torch.float32, device="neuron", requires_grad=True)
        assert torch.view_as_complex(x).requires_grad

    def test_multiple_views_share_storage(self):
        x = torch.randn(3, 4, dtype=torch.complex64, device="neuron")
        y1 = torch.view_as_real(x)
        y2 = torch.view_as_real(x)

        assert_storage_shared(y1, y2)

        new_value = torch.tensor(123.0, dtype=torch.float32, device="neuron")
        y1[0, 0, 0].copy_(new_value)
        assert abs(y2[0, 0, 0].item() - 123.0) < 1e-5

    def test_large_tensor(self):
        x_cpu = torch.randn(1000, dtype=torch.complex64)
        x_neuron = x_cpu.to("neuron")
        torch.testing.assert_close(torch.view_as_real(x_neuron).cpu(), torch.view_as_real(x_cpu))

        x_cpu = torch.randn(10, 20, 30, 2, dtype=torch.float32)
        x_neuron = x_cpu.to("neuron")
        torch.testing.assert_close(
            torch.view_as_complex(x_neuron).cpu(), torch.view_as_complex(x_cpu)
        )

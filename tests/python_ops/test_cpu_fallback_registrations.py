"""Tests for CPU fallback registrations (linalg ops with non-contiguous outputs)."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_did_not_run_on_neuron, track_neuron_ops


class TestCpuFallbackRegistrations:
    """Test CPU fallback ops return correct results with proper strides on Neuron."""

    def skip_if_no_device(self):
        """Skip test if neuron device is not available."""
        if not torch.neuron.is_available():
            pytest.skip("Neuron device not available")

    @pytest.mark.parametrize(
        "shape,mode",
        [
            ((4, 4), "reduced"),
            ((6, 4), "reduced"),
            ((4, 6), "reduced"),
            ((4, 4), "complete"),
            ((3, 5), "r"),
        ],
    )
    def test_linalg_qr(self, shape, mode):
        """Test torch.linalg.qr returns correct values and strides."""
        self.skip_if_no_device()

        cpu_input = torch.randn(shape, dtype=torch.float32)
        neuron_input = cpu_input.to("neuron")

        cpu_q, cpu_r = torch.linalg.qr(cpu_input, mode=mode)

        with track_neuron_ops():
            neuron_q, neuron_r = torch.linalg.qr(neuron_input, mode=mode)
            assert_op_did_not_run_on_neuron("aten::linalg_qr")

        # Check values match
        torch.testing.assert_close(neuron_q.cpu(), cpu_q, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(neuron_r.cpu(), cpu_r, rtol=1e-4, atol=1e-4)

        # Check strides preserved
        assert (
            neuron_q.stride() == cpu_q.stride()
        ), f"Q stride mismatch: {neuron_q.stride()} vs {cpu_q.stride()}"
        assert (
            neuron_r.stride() == cpu_r.stride()
        ), f"R stride mismatch: {neuron_r.stride()} vs {cpu_r.stride()}"

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 4),
            (6, 4),
            (4, 6),
        ],
    )
    def test_linalg_svd(self, shape):
        """Test torch.linalg.svd returns correct values and strides."""
        self.skip_if_no_device()

        cpu_input = torch.randn(shape, dtype=torch.float32)
        neuron_input = cpu_input.to("neuron")

        cpu_u, cpu_s, cpu_vh = torch.linalg.svd(cpu_input)

        with track_neuron_ops():
            neuron_u, neuron_s, neuron_vh = torch.linalg.svd(neuron_input)
            assert_op_did_not_run_on_neuron("aten::_linalg_svd")

        torch.testing.assert_close(neuron_u.cpu(), cpu_u, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(neuron_s.cpu(), cpu_s, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(neuron_vh.cpu(), cpu_vh, rtol=1e-4, atol=1e-4)

        assert neuron_u.stride() == cpu_u.stride()
        assert neuron_vh.stride() == cpu_vh.stride()

    @pytest.mark.parametrize("shape", [(4, 4), (6, 6)])
    def test_linalg_eigh(self, shape):
        """Test torch.linalg.eigh returns correct values and strides."""
        self.skip_if_no_device()

        # Create symmetric positive definite matrix
        cpu_input = torch.randn(shape, dtype=torch.float32)
        cpu_input = cpu_input @ cpu_input.T + torch.eye(shape[0])
        neuron_input = cpu_input.to("neuron")

        cpu_vals, cpu_vecs = torch.linalg.eigh(cpu_input)

        with track_neuron_ops():
            neuron_vals, neuron_vecs = torch.linalg.eigh(neuron_input)
            assert_op_did_not_run_on_neuron("aten::_linalg_eigh")

        torch.testing.assert_close(neuron_vals.cpu(), cpu_vals, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(neuron_vecs.cpu().abs(), cpu_vecs.abs(), rtol=1e-4, atol=1e-4)

        assert neuron_vecs.stride() == cpu_vecs.stride()

    @pytest.mark.parametrize("shape", [(4, 4), (6, 6)])
    def test_linalg_cholesky(self, shape):
        """Test torch.linalg.cholesky returns correct values and strides."""
        self.skip_if_no_device()

        # Create symmetric positive definite matrix
        cpu_input = torch.randn(shape, dtype=torch.float32)
        cpu_input = cpu_input @ cpu_input.T + torch.eye(shape[0])
        neuron_input = cpu_input.to("neuron")

        cpu_l = torch.linalg.cholesky(cpu_input)

        with track_neuron_ops():
            neuron_l = torch.linalg.cholesky(neuron_input)
            assert_op_did_not_run_on_neuron("aten::linalg_cholesky_ex")

        torch.testing.assert_close(neuron_l.cpu(), cpu_l, rtol=1e-4, atol=1e-4)
        assert neuron_l.stride() == cpu_l.stride()

    @pytest.mark.parametrize("shape", [(4, 4), (6, 6)])
    def test_linalg_inv(self, shape):
        """Test torch.linalg.inv returns correct values and strides."""
        self.skip_if_no_device()

        # Create invertible matrix
        cpu_input = torch.randn(shape, dtype=torch.float32)
        cpu_input = cpu_input @ cpu_input.T + torch.eye(shape[0])
        neuron_input = cpu_input.to("neuron")

        cpu_inv = torch.linalg.inv(cpu_input)

        with track_neuron_ops():
            neuron_inv = torch.linalg.inv(neuron_input)
            assert_op_did_not_run_on_neuron("aten::linalg_inv_ex")

        torch.testing.assert_close(neuron_inv.cpu(), cpu_inv, rtol=1e-4, atol=1e-4)
        assert neuron_inv.stride() == cpu_inv.stride()

    @pytest.mark.parametrize("shape", [(4, 4), (6, 6)])
    def test_linalg_lu_factor(self, shape):
        """Test torch.linalg.lu_factor returns correct values and strides."""
        self.skip_if_no_device()

        cpu_input = torch.randn(shape, dtype=torch.float32)
        neuron_input = cpu_input.to("neuron")

        cpu_lu, cpu_pivots = torch.linalg.lu_factor(cpu_input)

        with track_neuron_ops():
            neuron_lu, neuron_pivots = torch.linalg.lu_factor(neuron_input)
            assert_op_did_not_run_on_neuron("aten::linalg_lu_factor_ex")

        torch.testing.assert_close(neuron_lu.cpu(), cpu_lu, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(neuron_pivots.cpu(), cpu_pivots)

        assert neuron_lu.stride() == cpu_lu.stride()

    @pytest.mark.parametrize("shape", [(4, 4), (6, 6)])
    def test_linalg_det(self, shape):
        """Test torch.linalg.det returns correct values."""
        self.skip_if_no_device()

        cpu_input = torch.randn(shape, dtype=torch.float32)
        neuron_input = cpu_input.to("neuron")

        cpu_det = torch.linalg.det(cpu_input)

        with track_neuron_ops():
            neuron_det = torch.linalg.det(neuron_input)
            assert_op_did_not_run_on_neuron("aten::_linalg_det")

        torch.testing.assert_close(neuron_det.cpu(), cpu_det, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("shape", [(4, 4), (6, 6)])
    def test_linalg_slogdet(self, shape):
        """Test torch.linalg.slogdet returns correct values."""
        self.skip_if_no_device()

        cpu_input = torch.randn(shape, dtype=torch.float32)
        neuron_input = cpu_input.to("neuron")

        cpu_sign, cpu_logabsdet = torch.linalg.slogdet(cpu_input)

        with track_neuron_ops():
            neuron_sign, neuron_logabsdet = torch.linalg.slogdet(neuron_input)
            assert_op_did_not_run_on_neuron("aten::_linalg_slogdet")

        torch.testing.assert_close(neuron_sign.cpu(), cpu_sign, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(neuron_logabsdet.cpu(), cpu_logabsdet, rtol=1e-4, atol=1e-4)

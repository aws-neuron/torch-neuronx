import pytest
import torch

from tests.utils.neuron_test_utils import assert_raises


class TestAddmv:
    """Test cases for addmv operation"""

    def test_addmv_basic(self):
        """Check addmv with default alpha and beta"""
        vec_input = torch.randn(3)
        mat_input = torch.randn(3, 4)
        vec = torch.randn(4)

        expected = torch.addmv(vec_input, mat_input, vec)

        result = torch.addmv(
            vec_input.to("neuron"),
            mat_input.to("neuron"),
            vec.to("neuron"),
        )

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_addmv_alpha_beta(self):
        """Validate alpha and beta scaling"""
        input_vec = torch.randn(5)
        mat = torch.randn(5, 6)
        vec = torch.randn(6)

        expected = torch.addmv(input_vec, mat, vec, beta=0.5, alpha=2.0)

        result = torch.addmv(
            input_vec.to("neuron"),
            mat.to("neuron"),
            vec.to("neuron"),
            beta=0.5,
            alpha=2.0,
        )

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_addmv_scalar_self(self):
        """Ensure scalar self argument broadcasts correctly"""
        scalar_self = torch.tensor(1.5)
        mat = torch.randn(4, 3)
        vec = torch.randn(3)

        expected = torch.addmv(scalar_self, mat, vec)

        result = torch.addmv(
            scalar_self.to("neuron"),
            mat.to("neuron"),
            vec.to("neuron"),
        )

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_addmv_out(self):
        """Verify out tensor is populated"""
        input_vec = torch.randn(2)
        mat = torch.randn(2, 3)
        vec = torch.randn(3)

        expected = torch.addmv(input_vec, mat, vec, beta=1.25, alpha=0.75)

        out = torch.empty(2).to("neuron")
        torch.addmv(
            input_vec.to("neuron"),
            mat.to("neuron"),
            vec.to("neuron"),
            beta=1.25,
            alpha=0.75,
            out=out,
        )

        torch.testing.assert_close(out.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_addmv_inplace(self):
        """Check addmv in-place mutation"""
        input_vec = torch.zeros(3)
        mat = torch.randn(3, 2)
        vec = torch.randn(2)

        expected = torch.addmv_(input_vec.clone(), mat, vec, beta=0.1, alpha=1.5)

        target = input_vec.to("neuron")
        result = torch.addmv_(
            target,
            mat.to("neuron"),
            vec.to("neuron"),
            beta=0.1,
            alpha=1.5,
        )

        torch.testing.assert_close(target.cpu(), expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @assert_raises(RuntimeError)
    def test_addmv_errors(self):
        """Expect dimension and shape constraints to raise"""
        mat = torch.randn(3, 2).to("neuron")
        vec = torch.randn(2).to("neuron")

        bad_dim = torch.randn(1, 3).to("neuron")
        torch.addmv(bad_dim, mat, vec)

        size_mismatch = torch.randn(4).to("neuron")
        torch.addmv(size_mismatch, mat, vec)

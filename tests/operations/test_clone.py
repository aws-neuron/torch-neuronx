"""Test that clone operation is properly registered with PyTorch dispatcher."""

import os

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestCloneRegistration:
    """Test clone operation registration and functionality."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    def test_clone_runs_on_neuron(self, device):
        """Test that clone runs on Neuron without CPU fallback"""
        with track_neuron_ops():
            x = torch.tensor([1.0, 2.0, 3.0], device=device)
            result = x.clone()
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::clone")

    def test_clone_basic(self, device):
        """Test basic clone functionality"""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            y = x.clone()

            assert y.device.type == "neuron"
            assert torch.equal(x, y)
            assert x.data_ptr() != y.data_ptr()
            assert_op_runs_on_neuron("aten::clone")

    def test_clone_empty_tensor(self, device):
        """Test clone on empty tensor"""
        with track_neuron_ops():
            x = torch.tensor([], device=device)
            y = x.clone()

            assert y.device.type == "neuron"
            assert y.shape == torch.Size([0])
            assert torch.equal(x, y)
            assert_op_runs_on_neuron("aten::clone")

    def test_clone_scalar_tensor(self, device):
        """Test clone on scalar tensor"""
        with track_neuron_ops():
            x = torch.tensor(5.0, device=device)
            y = x.clone()

            assert y.device.type == "neuron"
            assert y.shape == torch.Size([])
            assert torch.equal(x, y)
            assert x.data_ptr() != y.data_ptr()
            assert_op_runs_on_neuron("aten::clone")

    def test_clone_preserves_requires_grad(self, device):
        """Test that clone preserves requires_grad"""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device, requires_grad=True)
            y = x.clone()

            assert y.device.type == "neuron"
            assert y.requires_grad == x.requires_grad
            assert torch.equal(x, y)
            assert x.data_ptr() != y.data_ptr()
            assert_op_runs_on_neuron("aten::clone")

    @pytest.mark.parametrize("memory_format", [None, torch.preserve_format])
    def test_clone_memory_format(self, device, memory_format):
        """Test clone with different memory formats"""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            y = x.clone(memory_format=memory_format)

            assert y.device.type == "neuron"
            assert torch.equal(x, y)
            assert x.data_ptr() != y.data_ptr()
            assert_op_runs_on_neuron("aten::clone")

    def test_clone_independence(self, device):
        """Test that cloned tensor is independent from original"""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            y = x.clone()

            # Modify original tensor
            x.fill_(1.0)

            # Cloned tensor should remain unchanged
            assert not torch.equal(x, y)
            assert torch.all(x == 1.0)
            assert not torch.all(y == 1.0)
            assert_op_runs_on_neuron("aten::clone")

    @pytest.mark.parametrize("contiguous", [True, False])
    def test_clone_contiguous_tensor(self, device, contiguous):
        """Test clone on contiguous and non-contiguous tensors"""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            if not contiguous:
                x = x.t()  # transpose to make non-contiguous

            assert x.is_contiguous() == contiguous
            y = x.clone()

            assert y.device.type == "neuron"
            assert torch.equal(x, y)
            assert x.data_ptr() != y.data_ptr()
            assert_op_runs_on_neuron("aten::clone")

    def test_clone_special_values(self, device):
        """Test clone with special float values"""
        with track_neuron_ops():
            x = torch.tensor([float("inf"), float("-inf"), float("nan")], device=device)
            y = x.clone()

            assert y.device.type == "neuron"
            assert torch.isinf(y[0]) and y[0] > 0
            assert torch.isinf(y[1]) and y[1] < 0
            assert torch.isnan(y[2])
            assert_op_runs_on_neuron("aten::clone")

    def test_clone_zero_tensor(self, device):
        """Test clone on tensor filled with zeros"""
        with track_neuron_ops():
            x = torch.zeros(3, 4, device=device)
            y = x.clone()

            assert y.device.type == "neuron"
            assert torch.equal(x, y)
            assert torch.all(y == 0.0)
            assert x.data_ptr() != y.data_ptr()
            assert_op_runs_on_neuron("aten::clone")

    def test_clone_ones_tensor(self, device):
        """Test clone on tensor filled with ones"""
        with track_neuron_ops():
            x = torch.ones(3, 4, device=device)
            y = x.clone()

            assert y.device.type == "neuron"
            assert torch.equal(x, y)
            assert torch.all(y == 1.0)
            assert x.data_ptr() != y.data_ptr()
            assert_op_runs_on_neuron("aten::clone")

    def test_clone_computation_graph_connection(self, device):
        """Test that clone maintains computation graph connection"""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device, requires_grad=True)
            y = x.clone()

            # Perform operation on cloned tensor
            z = y.sum()
            z.backward()

            # Original tensor should have gradients
            assert x.grad is not None
            assert x.grad.device.type == "neuron"
            assert_op_runs_on_neuron("aten::clone")

    def test_clone_gradient_flow(self, device):
        """Test gradient flow through clone operation"""
        with track_neuron_ops():
            x = torch.randn(2, 3, device=device, requires_grad=True)
            y = x.clone()
            z = y * 2
            loss = z.sum()
            loss.backward()

            # Check gradients flow back to original tensor
            assert x.grad is not None
            expected_grad = torch.full_like(x, 2.0)
            assert torch.allclose(x.grad, expected_grad)
            assert_op_runs_on_neuron("aten::clone")

    def test_clone_detach_vs_clone(self, device):
        """Test difference between clone and detach for computation graph"""
        with track_neuron_ops():
            x = torch.randn(2, 2, device=device, requires_grad=True)

            # Clone maintains graph connection
            y_clone = x.clone()
            z_clone = y_clone.sum()
            z_clone.backward()

            assert x.grad is not None

            # Reset gradients
            x.grad = None

            # Detach breaks graph connection
            y_detach = x.detach().clone()
            z_detach = y_detach.sum()
            # This should not affect x.grad since detach breaks the connection

            assert z_detach.grad_fn is None
            assert x.grad is None
            assert_op_runs_on_neuron("aten::clone")

    def test_clone_multiple_operations(self, device):
        """Test clone in chain of operations maintains gradients"""
        with track_neuron_ops():
            x = torch.randn(2, 2, device=device, requires_grad=True)
            y = x.clone()
            z = y.clone()
            w = z * 3
            loss = w.sum()
            loss.backward()

            # Gradients should flow through multiple clones
            assert x.grad is not None
            expected_grad = torch.full_like(x, 3.0)
            assert torch.allclose(x.grad, expected_grad)
            assert_op_runs_on_neuron("aten::clone")

    def test_clone_with_leaf_variable(self, device):
        """Test clone preserves leaf variable status for autograd"""
        with track_neuron_ops():
            x = torch.randn(2, 2, device=device, requires_grad=True)
            y = x.clone()

            assert x.is_leaf  # User created and marked as requires_grad leaf tensor
            assert not y.is_leaf  # Clone creates non-leaf tensor

            # But gradients should still flow
            z = y.sum()
            z.backward()
            assert x.grad is not None
            assert_op_runs_on_neuron("aten::clone")

    @pytest.mark.xfail(
        condition=os.environ.get("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS") == "1",
        reason="Intentionally fail the op to trigger fallback.",
    )
    def test_clone_recursion(self, device):
        with track_neuron_ops():

            def non_op(tensor):
                raise Exception("Op Not Implemented")

            # Unregister the op so that conj falls back to CPU and
            # trigger infinite recursion if is_pinned is called
            from torch.library import Library

            lib = Library("aten", "IMPL")
            lib.impl("conj", non_op, "PrivateUse1")

            x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j], device=device)
            y = x.conj()
            z = y.clone()
            assert z.device.type == "neuron"
            assert_op_runs_on_neuron("aten::clone")

    @pytest.mark.parametrize("memory_format", [torch.channels_last, torch.contiguous_format])
    def test_clone_channels_last_memory_format(self, device, memory_format):
        """Test clone with ChannelsLast memory format (4D NHWC)"""
        with track_neuron_ops():
            x = torch.randn(2, 3, 4, 5, device=device)
            y = x.clone(memory_format=memory_format)

            assert y.device.type == "neuron"
            assert torch.equal(x, y)
            assert x.data_ptr() != y.data_ptr()
            if memory_format == torch.channels_last:
                assert y.is_contiguous(memory_format=torch.channels_last)
            assert_op_runs_on_neuron("aten::clone")

    def test_clone_channels_last_3d_memory_format(self, device):
        """Test clone with ChannelsLast3d memory format (5D NDHWC)"""
        with track_neuron_ops():
            x = torch.randn(2, 3, 4, 5, 6, device=device)
            y = x.clone(memory_format=torch.channels_last_3d)

            assert y.device.type == "neuron"
            assert torch.equal(x, y)
            assert y.is_contiguous(memory_format=torch.channels_last_3d)
            assert_op_runs_on_neuron("aten::clone")

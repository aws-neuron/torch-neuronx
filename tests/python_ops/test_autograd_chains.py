"""
Test to check if registered operations may break autograd chains.

This test validates that neuron-registered operations don't intercept
CompositeImplicitAutograd operations that should be decomposed by PyTorch.
"""

import pytest
import torch
import torch._C

import torch_neuronx

# Whitelist of CompositeImplicitAutograd operations that are safe
SAFE_COMPOSITE_OPS = {
    "set_",  # PyTorch prevents set_ on tensors with requires_grad=True
    "set_.source_Tensor_storage_offset",
    "argsort.stable",  # Non-differentiable operation
    "_has_compatible_shallow_copy_type",  # Internal utility function
    "argsort",  # Non-differentiable operation (returns indices)
    "contiguous",  # View operation, doesn't affect gradients
    "isfinite",  # Non-differentiable boolean operation
    "one_hot",  # Typically used with integer tensors (no gradients)
    "rms_norm",  # Dispatched to _fused_rms_norm for perf
}


class TestAutogradChains:
    """Test that registered operations don't break autograd chains"""

    def get_registered_ops(self):
        """Get list of registered neuron operations."""
        from torch_neuronx.utils import get_neuron_registered_ops

        return get_neuron_registered_ops(as_sorted=True, keep_variant=True)

    def get_composite_implicit_autograd_ops(self):
        """Get operations that use CompositeImplicitAutograd dispatch."""
        # Get all CompositeImplicitAutograd registrations from PyTorch
        registrations = torch._C._dispatch_get_registrations_for_dispatch_key(
            "CompositeImplicitAutograd"
        )

        # Extract unique aten operations with full variant names
        composite_ops = set()
        for reg in registrations:
            if reg.startswith("aten::"):
                op_name = reg[6:]  # Remove "aten::" prefix
                composite_ops.add(op_name)

        return composite_ops

    def get_ops_with_explicit_derivatives(self):
        """Get operations that have explicit autograd derivatives."""
        # Operations registered with the Autograd dispatch key have explicit derivatives
        autograd_registrations = torch._C._dispatch_get_registrations_for_dispatch_key("Autograd")

        derivative_ops = set()
        for reg in autograd_registrations:
            if reg.startswith("aten::"):
                op_name = reg[6:]
                derivative_ops.add(op_name)

        return derivative_ops

    def test_no_problematic_composite_operations(self):
        """Test that no problematic CompositeImplicitAutograd operations are registered."""
        # Get data from PyTorch APIs
        registered_ops = self.get_registered_ops()
        composite_ops = self.get_composite_implicit_autograd_ops()
        derivative_ops = self.get_ops_with_explicit_derivatives()

        # Problematic operations:
        # Registered by neuron AND CompositeImplicitAutograd AND no explicit derivatives
        problematic_ops = []

        for op in registered_ops:
            if (
                op in composite_ops
                and op not in derivative_ops
                and op not in SAFE_COMPOSITE_OPS
                and not op.endswith("_backward")
            ):
                problematic_ops.append(op)

        # Assert no problematic operations
        error_msg = (
            "Found CompositeImplicitAutograd operations that may break autograd chains:\n"
            + "\n".join([f"  - {op}" for op in problematic_ops])
            + "\n\nThese ops should decompose naturally in PyTorch but are intercepted by neuron.\n"
            "Consider removing these registrations to allow PyTorch's natural decomposition."
        )
        assert len(problematic_ops) == 0, error_msg

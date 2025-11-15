"""Common utilities for view operation tests"""

import pytest
import torch


def is_view_of(base, view):
    """Check if 'view' is a view of 'base' tensor"""
    return view._is_view() and view._base is base and view.data_ptr() == base.data_ptr()


def assert_view_properties(base, view, expected_shape=None, expected_stride=None):
    """Assert basic view properties"""
    assert is_view_of(base, view), "Tensor is not a view"
    assert view.device == base.device, "Device mismatch"
    assert view.dtype == base.dtype, "Dtype mismatch"

    if expected_shape is not None:
        assert view.shape == expected_shape, f"Shape mismatch: {view.shape} != {expected_shape}"

    if expected_stride is not None:
        assert (
            view.stride() == expected_stride
        ), f"Stride mismatch: {view.stride()} != {expected_stride}"


def assert_storage_shared(tensor1, tensor2):
    """Assert two tensors share the same storage"""
    assert tensor1.data_ptr() == tensor2.data_ptr(), "Tensors do not share storage"


def generate_test_shapes():
    """Generate common test shapes for parametrized tests"""
    return [
        (12,),  # 1D
        (3, 4),  # 2D
        (2, 3, 4),  # 3D
        (2, 2, 3, 4),  # 4D
        (1, 1, 1),  # All ones
        (1,),  # Single element
        # Note: Scalar shape () may need special handling
    ]


def make_test_tensor(shape, device="neuron", dtype=torch.float32, contiguous=True):
    """Create a test tensor with predictable values"""
    if shape == ():  # Scalar
        return torch.tensor(42.0, dtype=dtype, device=device)

    numel = 1
    for s in shape:
        numel *= s

    tensor = torch.arange(numel, dtype=dtype).reshape(shape).to(device)

    if not contiguous and len(shape) >= 2:
        # Make non-contiguous by transposing
        tensor = tensor.transpose(0, 1)

    return tensor


def check_view_semantics(base, view):
    """Check that modifications to view affect base"""
    # Save original value
    original_val = base.flatten()[0].item()

    # Modify view
    view.flatten()[0] = -999

    # Check base is modified
    assert base.flatten()[0].item() == -999, "Modification to view did not affect base"

    # Restore
    base.flatten()[0] = original_val

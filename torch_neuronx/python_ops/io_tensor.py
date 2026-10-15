"""
Centralized I/O tensor creation with handling for Dynamo.
Needed for dynamo compatibility under eager / compile interop
to create physical tensors instead of incompatible traced tensors.
"""

import torch


@torch._dynamo.disable
def empty(*args, **kwargs):
    """Create an uninitialized physical tensor outside Dynamo tracing.

    Args:
        *args (Any): Positional arguments forwarded to ``torch.empty``.
        **kwargs (Any): Keyword arguments forwarded to ``torch.empty``.

    Returns:
        torch.Tensor: Physical tensor created by ``torch.empty``.
    """
    return torch.empty(*args, **kwargs)


@torch._dynamo.disable
def empty_like(*args, **kwargs):
    """Create an uninitialized physical tensor like another tensor.

    Args:
        *args (Any): Positional arguments forwarded to ``torch.empty_like``.
        **kwargs (Any): Keyword arguments forwarded to ``torch.empty_like``.

    Returns:
        torch.Tensor: Physical tensor created by ``torch.empty_like``.
    """
    return torch.empty_like(*args, **kwargs)


@torch._dynamo.disable
def tensor(*args, **kwargs):
    """Create a physical tensor outside Dynamo tracing.

    Args:
        *args (Any): Positional arguments forwarded to ``torch.tensor``.
        **kwargs (Any): Keyword arguments forwarded to ``torch.tensor``.

    Returns:
        torch.Tensor: Physical tensor created by ``torch.tensor``.
    """
    return torch.tensor(*args, **kwargs)


@torch._dynamo.disable
def zeros(*args, **kwargs):
    """Create a zero-filled physical tensor outside Dynamo tracing.

    Args:
        *args (Any): Positional arguments forwarded to ``torch.zeros``.
        **kwargs (Any): Keyword arguments forwarded to ``torch.zeros``.

    Returns:
        torch.Tensor: Physical tensor created by ``torch.zeros``.
    """
    return torch.zeros(*args, **kwargs)

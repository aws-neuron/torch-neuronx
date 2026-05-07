"""
Centralized I/O tensor creation with handling for Dynamo.
Needed for dynamo compatibility under eager / compile interop
to create physical tensors instead of incompatible traced tensors.
"""

import torch


@torch._dynamo.disable
def empty(*args, **kwargs):
    """
    Create empty tensor for I/O operations. Accepts same arguments as torch.empty.
    """
    return torch.empty(*args, **kwargs)


@torch._dynamo.disable
def empty_like(*args, **kwargs):
    """
    Create empty tensor like another tensor for I/O operations.
    Accepts same arguments as torch.empty_like.
    """
    return torch.empty_like(*args, **kwargs)


@torch._dynamo.disable
def tensor(*args, **kwargs):
    """
    Create tensor for I/O operations. Accepts same arguments as torch.tensor.
    """
    return torch.tensor(*args, **kwargs)


@torch._dynamo.disable
def zeros(*args, **kwargs):
    """
    Create zero tensor for I/O operations. Accepts same arguments as torch.zeros.
    """
    return torch.zeros(*args, **kwargs)

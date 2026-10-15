import torch

from .feedforward import FeedForward
from .llama import LlamaTransformerBlock


def force_local_contiguous(t: torch.Tensor) -> torch.Tensor:
    """Force DTensor local shards to be contiguous.

    Workaround for PyTorch issue #173041: DTensor.contiguous() doesn't always
    make local shards contiguous under torch.compile, causing view() to fail.
    In eager mode, DTensor.contiguous() works correctly.

    Ported from NeuronTorchTitan torchtitan/models/common/attention.py.
    """
    from torch.distributed._tensor import DTensor

    if torch.compiler.is_compiling() and isinstance(t, DTensor):
        local = t._local_tensor.contiguous()
        return DTensor.from_local(local, t.device_mesh, t.placements, run_check=False)
    return t.contiguous()


__all__ = ["FeedForward", "LlamaTransformerBlock", "force_local_contiguous"]

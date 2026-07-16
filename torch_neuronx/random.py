"""Device-agnostic ``torch.random.fork_rng`` for the neuron backend.

On torch <=2.12 ``fork_rng`` hard-defaults ``device_type="cuda"``, so forking
RNG state fails on the neuron (PrivateUse1) backend. torch 2.13 infers
``device_type`` from ``devices``; :func:`_apply_fork_rng_patch` reproduces that
behavior and auto-disables once ``fork_rng`` already defaults ``device_type``
to ``None``.

The patch is applied at ``torch_neuronx`` import so it is present in spawned
test workers, which re-import torch (and thus this module via backend autoload)
in a fresh interpreter and would otherwise bypass conftest-level patches.
"""

import functools
import inspect
import logging

import torch

__all__ = ["_apply_fork_rng_patch"]

logger = logging.getLogger(__name__)

_patch_applied = False


def _apply_fork_rng_patch() -> None:
    """Apply the fork_rng device_type inference patch. Safe to call multiple times."""
    global _patch_applied
    if _patch_applied:
        return

    fork_rng = torch.random.fork_rng
    if inspect.signature(fork_rng).parameters["device_type"].default is None:
        # torch >=2.13 already infers device_type from devices.
        _patch_applied = True
        return

    @functools.wraps(fork_rng)
    def _patched_fork_rng(devices=None, enabled=True, device_type=None, **kwargs):
        if device_type is None:
            device_type = "cuda"
            if devices:
                types = {torch.device(d).type for d in devices}
                if len(types) == 1:
                    device_type = types.pop()
        return fork_rng(devices=devices, enabled=enabled, device_type=device_type, **kwargs)

    torch.random.fork_rng = _patched_fork_rng
    _patch_applied = True
    logger.debug("Applied fork_rng device_type inference patch")

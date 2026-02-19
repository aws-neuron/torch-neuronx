"""
MegaCache integration for Neuron compilation caches.

Enables torch.compiler.save_cache_artifacts() / load_cache_artifacts() to include
compiled NEFFs for cross-machine cache sharing.
"""

import logging
from enum import Enum

import torch
from torch.compiler._cache import CacheArtifact, CacheArtifactFactory, CacheArtifactManager

import torch_neuronx

logger = logging.getLogger(__name__)


class ArtifactType(Enum):
    """MegaCache artifact types for Neuron compilation caches."""

    NEURON_NEFF = "neuron_neff"  # StableHLO → NEFF


@CacheArtifactFactory.register
class NeuronNeffCacheArtifact(CacheArtifact):
    """Cache artifact for Neuron NEFF binaries."""

    @staticmethod
    def type() -> str:
        return ArtifactType.NEURON_NEFF.value

    def populate_cache(self) -> None:
        """Write NEFF to persistent cache during load_cache_artifacts().

        Logs warning on failure (consistent with PyTorch's silent-success API contract).
        """
        if not torch_neuronx._C.put_neff_cache(self.key, self.content):
            logger.warning(f"Failed to write NEFF to persistent cache: {self.key}")


def _collect_neuron_artifacts() -> None:
    """Record compiled NEFFs as MegaCache artifacts.

    Iterates tracked cache keys and records each NEFF with its persistent key.
    Keys are cleared after collection to prevent unbounded memory growth.
    """
    # Import here to avoid circular dependency: compile.py imports ArtifactType from this module
    from torch_neuronx.neuron_dynamo_backend.compile import (
        clear_compiled_cache_keys,
        get_compiled_cache_keys,
    )

    for cache_key in get_compiled_cache_keys(ArtifactType.NEURON_NEFF):
        result = torch_neuronx._C.get_neff_info(cache_key)
        if result:
            persistent_key, neff_bytes = result
            CacheArtifactManager.record_artifact(
                ArtifactType.NEURON_NEFF.value, persistent_key, neff_bytes
            )
    clear_compiled_cache_keys(ArtifactType.NEURON_NEFF)


# Wrap serialize() to collect Neuron artifacts just-in-time when saving
_original_serialize = CacheArtifactManager.serialize.__func__


def _wrapped_serialize(cls):
    """Sync device and collect Neuron artifacts before serializing."""
    torch.neuron.synchronize()
    _collect_neuron_artifacts()
    return _original_serialize(cls)


CacheArtifactManager.serialize = classmethod(_wrapped_serialize)

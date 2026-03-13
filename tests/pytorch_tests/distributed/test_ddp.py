import sys
from typing import Any

import pytest

import torch_neuronx
from tests.pytorch_tests.distributed.common_distributed import (
    NeuronMultiProcessTestDDP,
    create_test_classes,
)


def _neuron_gpus_for_rank(world_size):
    """
    Neuron fix: Each rank uses device index = rank.

    In Neuron's distributed setup:
    - set_local_device_start_index(local_rank) is called during process group init
    - Each rank's valid device index equals its local_rank

    The original gpus_for_rank divides torch.cuda.device_count() (which returns 64
    cores) among ranks, giving rank 1 devices [32-63] which fails validation.
    This returns [[0], [1], ...] matching Neuron's local_device_start_index.
    """
    return [[rank] for rank in range(world_size)]


class _GpusForRankImportHook:
    """Import hook to patch gpus_for_rank when test_c10d_common is imported."""

    def find_module(self, fullname, path=None):
        if fullname == "test_c10d_common":
            return self
        return None

    def load_module(self, fullname):
        # If already loaded, return cached module
        if fullname in sys.modules:
            return sys.modules[fullname]

        # Remove ourselves temporarily to avoid recursion
        sys.meta_path = [h for h in sys.meta_path if not isinstance(h, _GpusForRankImportHook)]

        try:
            # Let Python do the normal import
            import test_c10d_common as module

            # Patch gpus_for_rank
            module.gpus_for_rank = _neuron_gpus_for_rank
            return module
        finally:
            # Re-add ourselves
            sys.meta_path.insert(0, _GpusForRankImportHook())


# DDP-specific patches that need to be applied before test class creation
def _apply_ddp_patches():
    """Apply DDP-specific patches for Neuron compatibility."""

    # Install the import hook if not already installed
    for hook in sys.meta_path:
        if isinstance(hook, _GpusForRankImportHook):
            return  # Already installed

    sys.meta_path.insert(0, _GpusForRankImportHook())

    # Also patch if already loaded
    module_name = "test_c10d_common"
    if module_name in sys.modules:
        sys.modules[module_name].gpus_for_rank = _neuron_gpus_for_rank


# Apply DDP patches before test class creation
_apply_ddp_patches()

create_test_classes(NeuronMultiProcessTestDDP, globals(), "ddp_spec.json")


if __name__ == "__main__":
    import pytest

    pytest.main(["-vs", __file__])

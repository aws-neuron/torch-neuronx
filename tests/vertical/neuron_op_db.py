"""Neuron op database - filtered subset of PyTorch's op_db for Neuron testing."""

import torch
from torch.testing._internal.common_methods_invocations import op_db

from .neuron_extra_ops import get_neuron_extra_op_db


def get_neuron_registered_op_names() -> set[str]:
    """Get set of op names registered for Neuron device (PrivateUse1).

    Uses torch dispatch API to discover registered ops.

    Returns:
        Set of op names (e.g., {"add", "mul", "relu"})
    """
    from torch_neuronx.utils import get_neuron_registered_ops

    return get_neuron_registered_ops()


# Default dtypes to test on Neuron
NEURON_DEFAULT_DTYPES = (
    torch.float32,
    torch.bfloat16,
)


def get_neuron_op_db(
    dtypes: tuple[torch.dtype, ...] | None = None,
) -> list:
    """Get filtered op_db for Neuron testing.

    Combines:
    1. Auto-discovered ops from PyTorch's op_db (filtered to Neuron-registered)
    2. Extra ops defined in neuron_extra_ops.py (for ops not in PyTorch's op_db)

    Args:
        dtypes: Dtypes to filter for. If None, includes ops supporting any dtype.

    Returns:
        List of OpInfo objects for Neuron-supported ops.
    """
    neuron_op_names = get_neuron_registered_op_names()

    # 1. Filter PyTorch's op_db to Neuron-registered ops
    filtered = []
    covered_names = set()
    for op in op_db:
        if op.name not in neuron_op_names:
            continue

        # Check if op supports at least one of the requested dtypes
        if dtypes and op.dtypes and not any(d in op.dtypes for d in dtypes):
            continue

        filtered.append(op)
        covered_names.add(op.name)

    # 2. Add extra ops (only if not already covered by op_db)
    for op in get_neuron_extra_op_db():
        if op.name in covered_names:
            continue  # op_db version takes precedence
        if op.name not in neuron_op_names:
            continue  # Only include if actually registered
        if dtypes and op.dtypes and not any(d in op.dtypes for d in dtypes):
            continue
        filtered.append(op)

    return filtered


def _lazy_init_neuron_op_db():
    """Lazily initialize neuron_op_db after torch_neuronx is imported."""
    return get_neuron_op_db(dtypes=NEURON_DEFAULT_DTYPES)


# Will be populated when accessed after torch_neuronx import
neuron_op_db = []

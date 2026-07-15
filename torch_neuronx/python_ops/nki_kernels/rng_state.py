"""NKI kernels for RNG state management"""

import logging
from collections.abc import Sequence

import torch

from torch_neuronx.nki_hop import nki_op, wrap_nki
from torch_neuronx.utils import get_platform_target

logger = logging.getLogger(__name__)

try:
    from nkilib.experimental.rng.rng import (
        NUM_RNG_SEEDS,
        generate_random_fast,
        get_rng_state_gpsimd,
        set_rng_state_gpsimd,
    )

    # Fast multi-lane kernel (all 128 GPSIMD lanes, per-lane seeded). This is the
    # only RNG path: dropout (the sole consumer) only needs a correct independent
    # random distribution, for which the value->position mapping is irrelevant.
    wrapped_rand = wrap_nki(generate_random_fast)
    # Set has_side_effect as an attribute to avoid modifying wrap_nki,
    # which may be deprecated
    wrapped_rand.has_side_effect = True

    @nki_op("nki_kernels::get_rng_state", mutates_args={})
    def get_rng_state() -> torch.Tensor:
        """
        Get RNG state from GPSIMD engine.

        Returns:
            Tensor of shape (1, NUM_RNG_SEEDS) with dtype uint32 containing the RNG seeds
        """
        state = torch.empty(1, NUM_RNG_SEEDS, device="neuron", dtype=torch.uint32)
        result = get_rng_state_gpsimd(state)
        return result.squeeze(0)

    @nki_op("nki_kernels::set_rng_state", mutates_args={})
    def set_rng_state(seed: torch.Tensor) -> None:
        """
        Set RNG state for GPSIMD engine.

        Args:
            seed: Seed tensor of shape (1, NUM_RNG_SEEDS) with dtype uint32
        """
        set_rng_state_gpsimd(seed)

    def _generate_raw(ref: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        """Generate raw int32 random values in the given shape."""
        total = 1
        for s in shape:
            total *= s
        output_buffer = torch.empty((1, total), dtype=torch.int32, device=ref.device)
        flat = wrapped_rand(output_buffer, total)
        return flat.squeeze(0).reshape(shape)

    _has_nki_rng = get_platform_target() != "trn1"
except (ImportError, ModuleNotFoundError):
    _has_nki_rng = False

if not _has_nki_rng:
    logger.warning("On-device RNG not available - CPU RNG will be used")


def rand(
    ref: torch.Tensor, size: Sequence[int], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Generate uniform random floats in [0, 1), matching torch.rand convention.

    Args:
        size: Shape of the output tensor
        dtype: Output dtype (default: torch.float32)

    Returns:
        Tensor of uniform random values in [0, 1)
    """
    raw = _generate_raw(ref, size)
    # Take lower 23 bits (float32 mantissa), OR with 0x3f800000 to get [1.0, 2.0), subtract 1.0
    bits = (raw & 0x7FFFFF) | 0x3F800000
    result = bits.view(dtype=torch.float32) - 1.0
    return result.to(dtype=dtype)


@nki_op("nki_kernels::rand", mutates_args={})
def eager_rand(
    ref: torch.Tensor, size: Sequence[int], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    return rand(ref, size, dtype)


@nki_op("nki_kernels::randint", mutates_args={})
def randint(
    ref: torch.Tensor, low: int, high: int, size: Sequence[int], dtype: torch.dtype = torch.int64
) -> torch.Tensor:
    """
    Generate uniform random integers in [low, high), matching torch.randint convention.

    Args:
        ref: Reference tensor (used for device/tracing context)
        low: Inclusive lower bound
        high: Exclusive upper bound
        size: Shape of the output tensor
        dtype: Output dtype (default: torch.int64)

    Returns:
        Tensor of uniform random integers in [low, high)
    """
    raw = _generate_raw(ref, size).to(dtype=torch.int64)
    # Map to [low, high) using modulo
    result = low + (raw.abs() % (high - low))
    return result.to(dtype=dtype)

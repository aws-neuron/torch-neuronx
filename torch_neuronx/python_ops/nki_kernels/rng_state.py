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

    # ── Fake/meta kernels ──
    # These four ops call a NKI kernel directly, so the real body cannot run on a
    # meta tensor. Without an explicit fake, PyTorch registers the real body as
    # the Fake kernel and any meta-dispatch path (test_meta, FakeTensorMode,
    # torch.compile tracing) tries to compile the kernel and raises
    # "failed to run: nki_kernels.set_rng_state.default(... device='meta' ...)".
    # torch.manual_seed() reaches set_rng_state through the PrivateUse1 hook, so
    # every random op inherited the failure. A fake only reports output metadata.

    def _fake_get_rng_state() -> torch.Tensor:
        return torch.empty(NUM_RNG_SEEDS, device="neuron", dtype=torch.uint32)

    def _fake_set_rng_state(seed: torch.Tensor) -> None:
        return None

    @nki_op("nki_kernels::get_rng_state", mutates_args={}, fake_fn=_fake_get_rng_state)
    def get_rng_state() -> torch.Tensor:
        """Return the current GPSIMD random-number-generator state.

        Returns:
            torch.Tensor: One-dimensional tensor containing ``NUM_RNG_SEEDS`` seed words.
        """
        state = torch.empty(1, NUM_RNG_SEEDS, device="neuron", dtype=torch.uint32)
        result = get_rng_state_gpsimd(state)
        return result.squeeze(0)

    @nki_op("nki_kernels::set_rng_state", mutates_args={}, fake_fn=_fake_set_rng_state)
    def set_rng_state(seed: torch.Tensor) -> None:
        """Set the GPSIMD random-number-generator state.

        Args:
            seed (torch.Tensor): Seed words with shape ``(1, NUM_RNG_SEEDS)`` and a
                ``uint32`` bit pattern.
        """
        set_rng_state_gpsimd(seed)

    def _generate_raw(ref: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        """Generate raw int32 random values in the given shape."""
        total = 1
        for s in shape:
            total *= s
        # A zero-element output (any dim == 0, e.g. torch.randn(0, 5)) has no
        # values to draw. Skip the NKI kernel entirely: it requires every
        # dimension to be a positive integer and would assert on the (1, 0)
        # buffer otherwise.
        if total == 0:
            return torch.empty(tuple(shape), dtype=torch.int32, device=ref.device)
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
    """Generate uniformly distributed values in the interval ``[0, 1)``.

    Args:
        ref (torch.Tensor): Tensor providing the output device and tracing context.
        size (Sequence[int]): Output shape.
        dtype (torch.dtype, optional): Output data type. Defaults to ``torch.float32``.

    Returns:
        torch.Tensor: Random values with the requested shape and data type.

    Note:
        Zero-element shapes return without launching the NKI random kernel.
    """
    raw = _generate_raw(ref, size)
    # Take lower 23 bits (float32 mantissa), OR with 0x3f800000 to get [1.0, 2.0), subtract 1.0
    bits = (raw & 0x7FFFFF) | 0x3F800000
    result = bits.view(dtype=torch.float32) - 1.0
    return result.to(dtype=dtype)


def _fake_rand(
    ref: torch.Tensor, size: Sequence[int], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Fake kernel for ``nki_kernels::rand``: metadata only, no NKI kernel."""
    return torch.empty(tuple(size), dtype=dtype, device=ref.device)


def _fake_randint(
    ref: torch.Tensor,
    low: int,
    high: int,
    size: Sequence[int],
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Fake kernel for ``nki_kernels::randint``: metadata only, no NKI kernel."""
    return torch.empty(tuple(size), dtype=dtype, device=ref.device)


@nki_op("nki_kernels::rand", mutates_args={}, fake_fn=_fake_rand)
def eager_rand(
    ref: torch.Tensor, size: Sequence[int], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Generate uniform random values through the eager NKI op wrapper.

    Args:
        ref (torch.Tensor): Tensor providing the output device and tracing context.
        size (Sequence[int]): Output shape.
        dtype (torch.dtype, optional): Output data type. Defaults to ``torch.float32``.

    Returns:
        torch.Tensor: Random values in ``[0, 1)`` with the requested shape and type.
    """
    return rand(ref, size, dtype)


@nki_op("nki_kernels::randint", mutates_args={}, fake_fn=_fake_randint)
def randint(
    ref: torch.Tensor, low: int, high: int, size: Sequence[int], dtype: torch.dtype = torch.int64
) -> torch.Tensor:
    """Generate pseudorandom integers in the interval ``[low, high)``.

    Args:
        ref (torch.Tensor): Tensor providing the output device and tracing context.
        low (int): Inclusive lower bound.
        high (int): Exclusive upper bound, which must exceed ``low``.
        size (Sequence[int]): Output shape.
        dtype (torch.dtype, optional): Output data type. Defaults to ``torch.int64``.

    Returns:
        torch.Tensor: Random integers with the requested shape and data type.

    Note:
        Values are derived by applying modulo reduction to raw random words.
    """
    raw = _generate_raw(ref, size).to(dtype=torch.int64)
    # Map to [low, high) using modulo
    result = low + (raw.abs() % (high - low))
    return result.to(dtype=dtype)

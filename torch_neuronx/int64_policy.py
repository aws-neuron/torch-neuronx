"""Int64 execution mode policy for Neuron.

Controls how int64 tensors are handled during Neuron dispatch:

  - packed (default): Store int64 as two int32 lanes; execute using low 32 bits only.
    Avoids host roundtrip at the cost of losing upper 32 bits during compute.
  - cpu_bounce: Move to CPU, cast to int32, move back. Safe but slow.
  - native: Pass int64 through to compiler as S64 HLO. Requires compiler support.

Set via environment variable:
    export TORCH_NEURONX_INT64_MODE=packed|native|cpu_bounce

For backward compatibility, TORCH_NEURONX_NATIVE_INT64=1 is equivalent to
TORCH_NEURONX_INT64_MODE=native.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache


class Int64Mode(Enum):
    CPU_BOUNCE = "cpu_bounce"
    NATIVE = "native"
    PACKED = "packed"


@lru_cache(maxsize=1)
def get_int64_mode() -> Int64Mode:
    """Read the int64 execution mode from environment. Cached at first call.

    Resolution order:
      1. TORCH_NEURONX_INT64_MODE (explicit new API)
      2. TORCH_NEURONX_NATIVE_INT64=1 (legacy, maps to native)
      3. Default: cpu_bounce
    """
    mode_str = os.environ.get("TORCH_NEURONX_INT64_MODE")
    if mode_str is not None:
        mode_str = mode_str.strip().lower()
        try:
            return Int64Mode(mode_str)
        except ValueError:
            valid = ", ".join(m.value for m in Int64Mode)
            raise ValueError(
                f"Invalid TORCH_NEURONX_INT64_MODE={mode_str!r}. Valid: {valid}"
            ) from None

    # Legacy env var fallback
    legacy = os.environ.get("TORCH_NEURONX_NATIVE_INT64", "0")
    if legacy == "1":
        return Int64Mode.NATIVE

    return Int64Mode.PACKED


def is_packed_int64_mode() -> bool:
    """Return True if int64 should use packed physical storage."""
    return get_int64_mode() == Int64Mode.PACKED


def is_native_int64_mode() -> bool:
    """Return True if int64 passes through to compiler as S64."""
    return get_int64_mode() == Int64Mode.NATIVE


def is_cpu_bounce_int64_mode() -> bool:
    """Return True if int64 uses CPU roundtrip cast (default)."""
    return get_int64_mode() == Int64Mode.CPU_BOUNCE


def reset_int64_mode_cache() -> None:
    """Clear the cached mode. For testing only."""
    get_int64_mode.cache_clear()

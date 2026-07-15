"""Dynamic-shape NKI kernel registry.

Maps ATen op names to dynamic NKI kernel adapters. Each entry compiles once
at minimum tile size and is reused for all valid shapes at runtime.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DynamicNkiEntry:
    """Adapter for a dynamic-shape NKI kernel."""

    # compile_fn(compile_key) -> (hlo_bytes, cache_key). compile_key = dims baked
    # into NEFF (everything except runtime-dynamic M). Memoized per compile_key.
    compile_fn: Callable[..., tuple[bytes, str]]
    can_handle: Callable[..., bool]
    prepare_io: Callable[..., tuple[tuple, tuple, int]]  # -> (inputs, outputs, actual_M)

    # Optional: reshape kernel outputs to the op's expected shape.
    reshape_output: Callable[..., object] | None = None

    # Optional: derive compile_key from op args. When None, one NEFF (key=None).
    compile_key_fn: Callable[..., object] | None = None

    _compiled: dict = field(default_factory=dict, repr=False)

    def get_compiled(self, compile_key=None) -> tuple[bytes, str]:
        """Lazily compile the dynamic kernel for `compile_key`, memoized per key."""
        if compile_key not in self._compiled:
            logger.debug(
                "dynamic NKI get_compiled: memo MISS key=%s -> running compile_fn=%s",
                compile_key,
                getattr(self.compile_fn, "__name__", self.compile_fn),
            )
            self._compiled[compile_key] = self.compile_fn(compile_key)
            logger.info(
                "Compiled dynamic NKI NEFF (key=%s, hlo %d bytes)",
                compile_key,
                len(self._compiled[compile_key][0]),
            )
        else:
            logger.debug(
                "dynamic NKI get_compiled: memo HIT key=%s (cache_key=%s...)",
                compile_key,
                self._compiled[compile_key][1][:48],
            )
        return self._compiled[compile_key]


_registry: dict[str, DynamicNkiEntry] = {}


def register_dynamic_nki(op_name: str, entry: DynamicNkiEntry):
    _registry[op_name] = entry


def get_dynamic_nki_entry(op_name: str) -> DynamicNkiEntry | None:
    if os.environ.get("TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS") != "1":
        return None
    return _registry.get(op_name)

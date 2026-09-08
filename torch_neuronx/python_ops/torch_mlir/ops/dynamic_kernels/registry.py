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

    # Optional composite path: run the op as a few EXISTING dynamic kernels instead of a
    # NEFF of its own — for ops whose operand COUNT is data-dependent, which a NEFF cannot
    # express (see adapters/concat.py's dim-0 concat). ``run_fn(*args, dest=..., **kwargs)``
    # returning None means "not my shape": dispatch falls through to compile/execute below.
    run_fn: Callable[..., object] | None = None

    _compiled: dict = field(default_factory=dict, repr=False)

    # Instrumentation (for tests / debugging): every time this dynamic kernel
    # actually executes, _exec_count is bumped and the runtime dynamic dim
    # (actual_M from prepare_io) is recorded. This distinguishes "the dynamic
    # NEFF ran" from "the op silently fell back to the default lowering" — the
    # compiled-NEFF count alone can't tell those apart.
    _exec_count: int = field(default=0, repr=False)
    _exec_ms: set = field(default_factory=set, repr=False)

    def record_exec(self, actual_m: int):
        self._exec_count += 1
        self._exec_ms.add(int(actual_m))

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
    # Minimal list of dynamic-NKI kernels to (a) avoid recompilation on the pure-TP4
    # setup and (b) skip the slow serialized static index_put for grouped_mm-based MoE
    _allow_list = {
        "scatter_add",
        "scatter_add_",
        "_index_put_impl_",
        "index",
        "gather",
    }
    # TORCH_NEURONX_DYNAMIC_NKI_ALLOW_LIST gates the allowlist:
    #   "1"          -> restrict to _allow_list dynamic NKI kernels.
    #   "0" or unset -> skip the allowlist check; use ALL registered dynamic kernels.
    _use_allow_list = os.environ.get("TORCH_NEURONX_DYNAMIC_NKI_ALLOW_LIST", "0") == "1"
    if _use_allow_list and op_name not in _allow_list:
        return None
    return _registry.get(op_name)


def get_dynamic_nki_usage() -> dict[str, dict]:
    """Report the dynamic NKI kernels that ran in this process, per ATen op.

    An op appears only after it executes. ``kernels`` names the NEFFs it compiled; a
    composite op (dim-0 cat) has none of its own, because it runs other ops' kernels.
    Not flag-gated: the registry is already empty when the feature is off.
    """
    usage = {}
    for op_name, entry in _registry.items():
        if not entry._exec_count:
            continue
        ms = entry._exec_ms
        usage[op_name] = {
            "exec_count": entry._exec_count,
            "neffs": len(entry._compiled),
            "kernels": sorted({key.split("--")[0] for _hlo, key in entry._compiled.values()}),
            "m_min": min(ms) if ms else None,
            "m_max": max(ms) if ms else None,
            "m_distinct": len(ms),
        }
    return usage

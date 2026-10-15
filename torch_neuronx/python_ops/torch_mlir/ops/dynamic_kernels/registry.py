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
    """Describe compilation, eligibility, and I/O for a dynamic NKI kernel.

    Args:
        compile_fn (Callable): Function returning HLO bytes and a cache key for a compile key.
        can_handle (Callable): Predicate that accepts or declines an operation invocation.
        prepare_io (Callable): Function returning inputs, outputs, and the runtime dynamic size.
        reshape_output (Callable, optional): Function restoring the operation's output shape.
            Defaults to ``None``.
        compile_key_fn (Callable, optional): Function deriving dimensions baked into the NEFF.
            Defaults to ``None``.
        run_fn (Callable, optional): Composite execution path that may return ``None`` to
            fall through.
            Defaults to ``None``.
        _compiled (dict, optional): Per-key HLO-byte and cache-key tuples. Defaults to
            an empty dictionary.
        _exec_count (int, optional): Number of recorded dynamic executions. Defaults to ``0``.
        _exec_ms (set, optional): Distinct recorded runtime dynamic sizes. Defaults to an empty set.

    Note:
        Execution instrumentation distinguishes a dynamic NEFF run from silent CPU fallback
        or default lowering.
    """

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
        """Record one dynamic-kernel execution and its runtime row count.

        Args:
            actual_m (int): Runtime dynamic leading-dimension size.
        """
        self._exec_count += 1
        self._exec_ms.add(int(actual_m))

    def get_compiled(self, compile_key=None) -> tuple[bytes, str]:
        """Compile or retrieve the dynamic kernel for a compile key.

        Args:
            compile_key (object, optional): Dimensions and attributes baked into the NEFF.
                Defaults to ``None``.

        Returns:
            tuple: Compiled HLO bytes and cache key.

        Note:
            Compilation is memoized once per distinct key, including the single ``None`` key.
        """
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
    """Register a dynamic NKI entry under a dispatch key.

    Args:
        op_name (str): Short operation name or synthetic dispatch key.
        entry (DynamicNkiEntry): Entry to store in the process-wide registry.
    """
    _registry[op_name] = entry


def get_dynamic_nki_entry(op_name: str) -> DynamicNkiEntry | None:
    """Return a registered dynamic NKI entry when dynamic dispatch permits it.

    Args:
        op_name (str): Short operation name or synthetic dispatch key to look up.

    Returns:
        DynamicNkiEntry or None: Registered entry, or ``None`` when dynamic NKI
            kernels are disabled, the optional allowlist excludes the key, or the key
            is unregistered.
    """
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
    """Report dynamic NKI kernels executed by the current process.

    Returns:
        dict: Per-operation dictionaries with ``exec_count``, ``neffs``, ``kernels``,
            ``m_min``, ``m_max``, and ``m_distinct`` keys.

    Note:
        Operations appear only after execution; composite operations list no kernels of their
        own. This query is not feature-flag gated because feature-off dispatch does not record
        executions.
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

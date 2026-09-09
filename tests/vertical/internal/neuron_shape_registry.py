"""Op-specific shape registry backed by a consolidated CSV.

CSV format:
    op_name,arg_shapes,source_model,count,capture_paths
    aten::linear,"{""input"":[1,2048,4096],""weight"":[11008,4096],""bias"":null}",llama-7b,47,

Deduplicates on (op_name, arg_shapes). Merges counts across models.
"""

from __future__ import annotations

import csv
import functools
import json
import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from math import prod
from pathlib import Path

# Source tag every entry in ``neuron_shape_defaults_generated`` carries. Anything else is
# a model name, so ``source != BUILTIN_SOURCE`` is the "came from a real trace" test and
# ``--traced-model <name>`` selects one such model.
BUILTIN_SOURCE = "builtin"


@dataclass
class OpShapeEntry:
    """A unique (op_name, arg_shapes) pair with metadata."""

    op_name: str
    arg_shapes: dict[str, tuple[int, ...] | None]
    sources: dict[str, int] = field(default_factory=dict)  # model → count

    # Capture paths this shape was actually seen on, e.g. {"eager"}, {"compiled"} or both.
    # Property for a model-traced entry only; Empty for builtin/manual entries.
    paths: set[str] = field(default_factory=set)

    @property
    def total_count(self) -> int:
        return sum(self.sources.values())

    @property
    def shapes_key(self) -> str:
        """Stable JSON key for dedup."""
        return json.dumps(self.arg_shapes, sort_keys=True)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.arg_shapes.items())
        return f"OpShapeEntry({self.op_name}, {args}, count={self.total_count})"


class NeuronShapeRegistry:
    """Deduplicated registry of op shape combinations.

    Usage:
        # Register from model trace
        SHAPE_REGISTRY.register("aten::linear",
            arg_shapes={"input": (1, 2048, 4096), "weight": (11008, 4096), "bias": None},
            source="llama-7b", count=47)

        # Query
        for entry in SHAPE_REGISTRY.get("aten::linear"):
            sample = op.make_sample(entry.arg_shapes, device, dtype)

    """

    def __init__(self) -> None:
        # {op_name: {shapes_key: OpShapeEntry}}
        self._entries: dict[str, dict[str, OpShapeEntry]] = defaultdict(dict)

    def register(
        self,
        op_name: str,
        arg_shapes: dict[str, tuple[int, ...] | list[int] | None],
        source: str = "manual",
        count: int = 1,
        paths: Iterable[str] = (),
    ) -> None:
        """Register a shape combination. Deduplicates and merges counts.

        ``paths``: the paths that an op shape was captured from.
        """
        normalized = {
            k: tuple(v) if isinstance(v, list | tuple) else v for k, v in arg_shapes.items()
        }
        entry = OpShapeEntry(op_name=op_name, arg_shapes=normalized)
        key = entry.shapes_key

        if key in self._entries[op_name]:
            existing = self._entries[op_name][key]
            existing.sources[source] = existing.sources.get(source, 0) + count
            existing.paths.update(paths)
        else:
            entry.sources[source] = count
            entry.paths.update(paths)
            self._entries[op_name][key] = entry

    def get(self, op_name: str, model: str | None = None) -> list[OpShapeEntry]:
        """Get all shape entries for an op, sorted by total count (most common first).

        ``model`` restricts the result to entries that carry that source tag.
        """
        entries = self._entries.get(op_name, {}).values()
        if model is not None:
            entries = [e for e in entries if model in e.sources]
        return sorted(entries, key=lambda e: -e.total_count)

    def has(self, op_name: str) -> bool:
        return bool(self._entries.get(op_name))

    def all_ops(self, model: str | None = None) -> list[str]:
        """Registered op names, optionally only those carrying ``model``'s shapes."""
        if model is None:
            return sorted(self._entries.keys())
        return sorted(op for op in self._entries if self.get(op, model=model))

    def models(self) -> list[str]:
        """Source tags other than ``BUILTIN_SOURCE`` -- i.e. the model traces loaded."""
        return sorted(
            {s for by_key in self._entries.values() for e in by_key.values() for s in e.sources}
            - {BUILTIN_SOURCE}
        )

    def __len__(self) -> int:
        return sum(len(v) for v in self._entries.values())

    def load_json(self, path: str | Path, source: str | None = None) -> int:
        """Load from model trace JSON. Source defaults to filename."""
        if source is None:
            source = Path(path).stem
        n = 0
        with open(path) as f:
            for entry in json.load(f):
                self.register(
                    op_name=entry["op_name"],
                    arg_shapes=entry["arg_shapes"],
                    source=source,
                    count=entry.get("count", 1),
                    paths=entry.get("capture_paths", ()),
                )
                n += 1
        return n

    def load_csv(self, path: str | Path) -> int:
        """Load from consolidated CSV. Returns number of entries loaded."""
        n = 0
        with open(path) as f:
            for row in csv.DictReader(f):
                self.register(
                    op_name=row["op_name"],
                    arg_shapes=json.loads(row["arg_shapes"]),
                    source=row.get("source_model", "csv"),
                    count=int(row.get("count", 1)),
                    paths=(row.get("capture_paths") or "").split(),
                )
                n += 1
        return n

    def save_csv(self, path: str | Path) -> int:
        """Save to consolidated CSV. Returns number of rows written."""
        n = 0
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["op_name", "arg_shapes", "source_model", "count", "capture_paths"])
            for op_name in sorted(self._entries):
                for entry in sorted(self._entries[op_name].values(), key=lambda e: -e.total_count):
                    capture_paths = " ".join(sorted(entry.paths))
                    for source, count in sorted(entry.sources.items()):
                        writer.writerow([op_name, entry.shapes_key, source, count, capture_paths])
                        n += 1
        return n

    def stats(self) -> dict:
        """Summary stats."""
        return {
            "total_ops": len(self._entries),
            "total_shapes": len(self),
            "top_ops": sorted(
                ((op, len(entries)) for op, entries in self._entries.items()),
                key=lambda x: -x[1],
            )[:10],
        }

    def propagate_to_compatible_overloads(self, target_ops) -> int:
        """Pre-resolve fallback lookups: for every registered op, copy its
        entries onto every schema-compatible write-out (or ``.default``)
        overload of the same base name found in ``target_ops`` that is not
        already registered.

        This replaces the runtime ``.out`` / ``.default`` fallbacks in
        NeuronOpInfo with a one-time, auditable transformation of the registry.

        Runs once over the registry; cost is O(|registry| x overloads_per_base)
        schema compatibility checks (~13 ms on ~800 ops / ~1360 targets).
        Returns the number of (src, tgt) pairs that were propagated.
        """
        from collections import defaultdict

        def _base(op_name: str) -> str:
            return op_name.replace("aten::", "").split(".", 1)[0]

        base_to_overloads: dict[str, list[str]] = defaultdict(list)
        for op in target_ops:
            base_to_overloads[_base(op)].append(op)

        schema_cache: dict[str, object] = {}

        def _schema(name: str):
            if name not in schema_cache:
                schema_cache[name] = _aten_schema(name)
            return schema_cache[name]

        propagated = 0
        for src_name in list(self.all_ops()):
            overloads = base_to_overloads.get(_base(src_name), ())
            if not overloads:
                continue
            src_entries = self.get(src_name)
            for tgt_name in overloads:
                if tgt_name == src_name or self.has(tgt_name):
                    continue  # skip already-registered targets
                tgt_schema = _schema(tgt_name)
                if tgt_schema is None:
                    continue
                # Propagate to write-out variants, and also to .default siblings
                # for parity with the pre-existing .default fallback.
                if not _is_write_out_schema(tgt_schema) and not tgt_name.endswith(".default"):
                    continue
                for entry in src_entries:
                    if not _entry_compatible_with_schema(entry.arg_shapes, tgt_schema):
                        continue
                    for source, count in entry.sources.items():
                        self.register(tgt_name, entry.arg_shapes, source, count, entry.paths)
                    propagated += 1
        return propagated


SHAPE_REGISTRY = NeuronShapeRegistry()


@functools.cache
def _tensor_arg_names(op_name: str) -> frozenset[str]:
    """Names of tensor-typed args for ``op_name`` per its aten schema.

    Used by ``max_tensor_elems`` to identify which keys in an ``arg_shapes``
    entry hold tensor shapes (as opposed to parameter lists like ``padding``
    or ``dim``). Returns an empty frozenset when the schema can't be
    resolved; callers degrade to registration-order sort in that case.
    Cached because ``_aten_schema`` is hit once per op across many entries.
    """
    schema = _aten_schema(op_name)
    if schema is None:
        return frozenset()
    return frozenset(a.name for a in schema.arguments if "Tensor" in str(a.type))


def _shape_numel(shape) -> int:
    """Element count of an int-tuple shape; 0 for malformed or empty."""
    if not shape or not all(isinstance(x, int) for x in shape):
        return 0
    return prod(shape)


def _value_numel(val) -> int:
    """Element count for a single arg value of any tensor-shape encoding:

    - ``None``                              → 0
    - ``{"shape": [...], "dtype": "..."}``  → numel of shape
    - ``(1, 2048, 4096)``                   → numel of shape
    - ``[{"shape": (4,)}, {"shape": (8,)}]`` → max over per-tensor specs
    - ``[(4, 4), (8, 8)]``                  → max over the shape tuples
    """
    if val is None:
        return 0
    if isinstance(val, dict):
        sh = val.get("shape")
        return _shape_numel(sh) if isinstance(sh, list | tuple) else 0
    if isinstance(val, list | tuple):
        if not val:
            return 0
        if all(isinstance(x, int) for x in val):
            return _shape_numel(val)
        return max((_value_numel(t) for t in val), default=0)
    return 0


def max_tensor_elems(arg_shapes: dict, op_name: str) -> int:
    """Max ``numel()`` across every tensor-typed arg value in an entry.

    Uses ``op_name``'s aten schema to pick which keys are tensors (so
    ``padding`` / ``stride`` / ``dim`` / ``split_sizes`` and the like
    are never mistaken for shape values) and supports every tensor-shape
    encoding accepted elsewhere in the registry: nested
    ``{"shape","dtype"}`` dicts, bare int tuples, lists of dicts, and
    lists of bare tuples.

    Returns 0 when the schema is unresolvable or the entry carries no
    tensor values; callers treat that as "unrankable" and keep the
    entry in its registered order.
    """
    if not arg_shapes:
        return 0
    tensor_names = _tensor_arg_names(op_name)
    if not tensor_names:
        return 0
    best = 0
    for key, val in arg_shapes.items():
        if key in tensor_names:
            best = max(best, _value_numel(val))
    return best


def _aten_schema(op_name: str):
    """Resolve ``aten::<base>.<overload>`` to its JIT schema, or ``None``."""
    import torch

    clean = op_name.replace("aten::", "")
    base, _, overload = clean.partition(".")
    pkt = getattr(torch.ops.aten, base, None)
    if pkt is None:
        return None
    op = getattr(pkt, overload or "default", None) or getattr(pkt, "default", None)
    return getattr(op, "_schema", None) if op is not None else None


def _is_write_out_schema(schema) -> bool:
    """True if schema has at least one kwarg-only write alias (e.g. ``out=``)."""
    return any(
        a.kwarg_only and getattr(getattr(a, "alias_info", None), "is_write", False)
        for a in schema.arguments
    )


def _value_compatible_with_type(value, type_str: str) -> bool:
    """Check that a registry entry value is plausibly compatible with a
    schema arg type. Catches cross-overload mismatches during propagation,
    e.g. propagating ``dim: 1`` (int) to a ``str`` / ``List[str]`` dim.
    Only rejects obvious mismatches; permissive otherwise to avoid false
    negatives on edge cases.
    """
    if value is None:
        return "?" in type_str or "Optional" in type_str
    # bool must be checked before int because bool subclasses int in Python
    if isinstance(value, bool):
        return "bool" in type_str
    if isinstance(value, int | float):
        # Numeric literal — accept numeric / Scalar / Tensor (shape-spec)
        # plain number can't go to a string-typed arg
        return not ("str" in type_str and "Tensor" not in type_str)
    if isinstance(value, str):
        return "str" in type_str
    if isinstance(value, list | tuple):
        if not value:
            return True  # empty literal; hard to classify, accept
        first = value[0]
        if isinstance(first, bool):
            return "bool" in type_str
        if isinstance(first, str):
            return "str" in type_str
        if isinstance(first, int | float):
            # int-list can be a tensor shape OR a List[int] — reject if the
            # target expects List[str] / List[Dimname].
            return not ("str" in type_str and "Tensor" not in type_str)
        return True
    if isinstance(value, dict):
        return "Tensor" in type_str  # dict spec describes a Tensor
    return True


def _entry_compatible_with_schema(arg_shapes: dict, tgt_schema) -> bool:
    """An entry (arg_shapes dict) is reusable for tgt_schema iff:
    - every key in the entry is a valid schema arg name,
    - every value's Python type is plausibly compatible with the schema
      arg's type (catches e.g. ``dim: 1`` (int) propagating to a
      ``dim: str`` dimname-variant), and
    - every arg in tgt that is not covered by the entry is either
        (a) defaulted,
        (b) Optional/nullable, or
        (c) a write-alias kwarg (populated later by ``_with_out_kwargs``).
    """
    schema_args = {a.name: a for a in tgt_schema.arguments}
    if any(k not in schema_args for k in arg_shapes):
        return False
    for k, v in arg_shapes.items():
        if not _value_compatible_with_type(v, str(schema_args[k].type)):
            return False
    for a in tgt_schema.arguments:
        if a.name in arg_shapes or a.has_default_value():
            continue
        type_str = str(a.type)
        if "?" in type_str or "Optional" in type_str:
            continue
        if a.kwarg_only and getattr(getattr(a, "alias_info", None), "is_write", False):
            continue  # write-alias kwargs are filled by _with_out_kwargs downstream
        return False
    return True


# Load all shape defaults (hand-crafted + auto-generated).
#
# The accuracy smoke test subsets live in the same registry as the builtin defaults.
# Registering them is opt-in: ``NEURON_TRACED_MODEL`` (set by ``--traced-model``) is an
# allowlist, and it is empty by default. A new subset does not impact the existing runs.
def _load_defaults():
    from .neuron_shape_defaults_generated import register_all_defaults

    traced = [m.strip() for m in os.environ.get("NEURON_TRACED_MODEL", "").split(",") if m.strip()]
    register_all_defaults(SHAPE_REGISTRY.register, smoke_subsets=traced)


_load_defaults()

# Load model-traced shapes from env var (set by --shape-registry pytest option).
# This ensures xdist workers also load the shapes.
_shape_registry_path = os.environ.get("NEURON_SHAPE_REGISTRY")
if _shape_registry_path and os.path.exists(_shape_registry_path):
    SHAPE_REGISTRY.load_json(_shape_registry_path, source="model_trace")

# Register the committed cache-key model shapes. These are on by default now, so a
# plain run tests the model shapes too. To turn them off, set
# NEURON_CACHE_KEY_SHAPES=0 (the --no-cache-key-shapes pytest option). The shapes
# live in neuron_shape_defaults_generated.py. Regenerate them with
# tests/vertical/tools/refresh_cache_key_shapes.py.
if os.environ.get("NEURON_CACHE_KEY_SHAPES", "1") != "0":
    from .neuron_shape_defaults_generated import register_cache_key_shapes

    register_cache_key_shapes(SHAPE_REGISTRY.register)

# Propagate base entries onto schema-compatible write-out / .default overloads
# that don't have explicit entries. Replaces runtime .out / .default fallbacks
# with a single deterministic registry transformation.
from .target_ops import TARGET_ATEN_OPS as _TARGET_ATEN_OPS  # noqa: E402

SHAPE_REGISTRY.propagate_to_compatible_overloads(_TARGET_ATEN_OPS)

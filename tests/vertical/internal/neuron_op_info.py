"""NeuronOpInfo — extends PyTorch OpInfo with an op-specific shape registry.

Design:
- The shape registry is the primary source of inputs.
- `_schema_make_sample` reads the op schema to build a SampleInput from a
  `{arg_name: shape_or_spec}` dict, so JSON-loaded shapes work for any op
  without per-op code.
- PyTorch's op_db is a secondary fallback for base/`.default`/out overloads.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable, Iterable

import torch
from torch.testing._internal.opinfo.core import OpInfo, SampleInput, _dispatch_dtypes

# Default length for ``Tensor[]`` / ``List[Tensor]`` args when the registry
# entry provides a single shape tuple rather than an explicit list-of-specs.
# Defined BEFORE the shape-registry import so neuron_shape_defaults_generated
# (imported transitively from neuron_shape_registry at module load) can
# reference it without a circular import.
TENSOR_LIST_DEFAULT_COUNT = 15

from .neuron_shape_registry import SHAPE_REGISTRY, max_tensor_elems  # noqa: E402

logger = logging.getLogger(__name__)


# Stub handed to OpInfo when the real op can't be resolved; never invoked
# because we override `sample_inputs_func`.
def _noop(*_args, **_kwargs):
    return None


_DTYPE_MAP = {
    "long": torch.long,
    "int64": torch.long,
    "int": torch.int32,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
    "double": torch.float64,
    "float64": torch.float64,
    "float": None,  # sentinel: use caller's default dtype
    "float32": None,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _resolve_op(op_name: str):
    clean = op_name.replace("aten::", "")
    parts = clean.split(".", 1)
    base, overload = parts[0], (parts[1] if len(parts) > 1 else "default")
    pkt = getattr(torch.ops.aten, base, None)
    if pkt is None:
        return None
    return getattr(pkt, overload, None) or getattr(pkt, "default", None)


def _is_out_variant(schema) -> bool:
    """Schema has at least one writable kwarg alias (e.g. `out=`)."""
    return any(
        a.kwarg_only and getattr(getattr(a, "alias_info", None), "is_write", False)
        for a in schema.arguments
    )


def _tensor_from_spec(spec, device, default_dtype):
    """Build a tensor from a shape tuple/list or a dict spec.

    Supported dict keys:
      shape (required), dtype, low, high, sigmoid, names, sorted, starts_at.

    When ``names`` is provided (tuple of strings matching ``len(shape)``),
    the tensor is tagged with those names via ``refine_names`` so it can be
    passed to ``.dimname`` / ``.names_out`` schema variants that require
    named-tensor inputs.

    When ``sorted`` is True, the tensor values are sorted in ascending order.
    When ``starts_at`` is set, the first element is forced to that value.
    These are useful for ops like embedding_bag that require monotonically
    increasing offsets starting at 0.
    """
    if isinstance(spec, dict):
        shape = tuple(spec["shape"])
        dt = _DTYPE_MAP.get(spec.get("dtype", "float"))
        dt = dt if dt is not None else default_dtype
        if dt in (torch.long, torch.int32):
            t = torch.randint(
                spec.get("low", 0), spec.get("high", 4), shape, device=device, dtype=dt
            )
        elif dt is torch.bool:
            t = torch.randint(0, 2, shape, device=device, dtype=torch.bool)
        else:
            t = torch.randn(shape, device=device, dtype=dt)
            if spec.get("sigmoid"):
                t = t.sigmoid()
        if spec.get("sorted"):
            t = t.sort().values
        if "starts_at" in spec:
            t[0] = spec["starts_at"]
        names = spec.get("names")
        if names:
            t = t.refine_names(*names)
        return t
    return torch.randn(tuple(spec), device=device, dtype=default_dtype)


# Default length for `Tensor[]` / `List[Tensor]` args when the registry entry
# provides a single shape tuple rather than an explicit list-of-specs. Foreach
# ops and any co-varying list-typed args (e.g. ``scalars``) in the same entry
# must have a matching element count — see neuron_shape_defaults_generated.py.


def _tensor_list_from_spec(raw, device, default_dtype, count_default=TENSOR_LIST_DEFAULT_COUNT):
    """Build a list of tensors from a shape, a list of shape-dicts, a list of
    shape-tuples, or a passthrough."""
    if isinstance(raw, tuple | list) and raw and isinstance(raw[0], int):
        # Single shape tuple → count_default identical tensors
        return [torch.randn(*raw, device=device, dtype=default_dtype) for _ in range(count_default)]
    if isinstance(raw, list | tuple) and raw and isinstance(raw[0], dict):
        # List of shape-dicts → one tensor per dict
        return [_tensor_from_spec(d, device, default_dtype) for d in raw]
    if isinstance(raw, list | tuple) and raw and isinstance(raw[0], tuple | list):
        # List of shape tuples → one tensor per shape. Each element must be
        # a tuple/list of ints (explicit shapes).
        return [torch.randn(tuple(s), device=device, dtype=default_dtype) for s in raw]
    return raw


def _build_value(type_str, raw, device, dtype):
    if (
        "Tensor[]" in type_str
        or "Tensor?[]" in type_str
        or "List[Tensor]" in type_str
        or "List[Optional[Tensor]]" in type_str
    ):
        return _tensor_list_from_spec(raw, device, dtype)
    if "Tensor" in type_str:
        if raw is None:
            return None
        if isinstance(raw, tuple | list) and (not raw or isinstance(raw[0], int)):
            return torch.randn(tuple(raw), device=device, dtype=dtype)
        if isinstance(raw, dict):
            return _tensor_from_spec(raw, device, dtype)
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    # Convert dtype strings (e.g., "bfloat16") to torch.dtype objects
    if isinstance(raw, str):
        attr = getattr(torch, raw, None)
        if isinstance(attr, torch.dtype):
            return attr
    return raw


def _schema_positional_index(schema, name) -> int:
    """Positional index of ``name`` among the schema's positional args, or a
    large sentinel if ``name`` is kwarg-only / absent."""
    idx = 0
    for a in schema.arguments:
        if a.kwarg_only:
            continue
        if a.name == name:
            return idx
        idx += 1
    return 1 << 30


def _schema_positional_bounds(schema) -> tuple[int, int]:
    """(min_required, max_accepted) positional-arg counts for schema."""
    positional = [a for a in schema.arguments if not a.kwarg_only]
    required = sum(1 for a in positional if not a.has_default_value())
    return required, len(positional)


def _arg_default(arg, type_str, dtype=None):
    """Return default value for a missing positional arg, or raise KeyError to bail."""
    if arg.has_default_value():
        return arg.default_value
    if "?" in type_str or "Optional" in type_str:
        return None
    # ``out_dtype`` is a required ScalarType positional on the *.dtype_out
    # matmul overloads (mm/bmm/addmm/baddbmm.dtype_out).
    if arg.name == "out_dtype" and dtype is not None:
        return dtype
    # Required ``dim`` on reduction ``.out``/dtype-out overloads (norm.out, ...)
    if arg.name == "dim" and "str" not in type_str:
        return [0] if ("[]" in type_str or "List" in type_str) else 0
    # Required ``keepdim`` bool flag with no schema default.
    if arg.name == "keepdim" and type_str == "bool":
        return False
    raise KeyError(arg.name)


def _schema_make_sample(op, arg_shapes, device, dtype):
    """Build SampleInput from arg_shapes dict + op schema for arg ordering.

    Returns None if a required positional arg is missing from the registry entry.
    """
    schema = getattr(op, "_schema", None)
    if schema is None:
        return None

    positional, kwargs = [], {}
    for arg in schema.arguments:
        type_str = str(arg.type)
        if arg.name in arg_shapes:
            val = _build_value(type_str, arg_shapes[arg.name], device, dtype)
        elif arg.kwarg_only:
            # Kwarg-only args not in the entry. If the schema has a default,
            # skip and PyTorch uses it. Otherwise, if the type is Optional or
            # has "?" (e.g. Generator?, str[]?) AND is not a write-alias, pass
            # None explicitly — some ops (aten::rand.generator_out,
            # aten::normal.out, etc.) require the kwarg even though it's
            # Optional. Write-alias kwargs are filled later by _with_out_kwargs.
            if arg.has_default_value():
                continue
            if ("?" in type_str or "Optional" in type_str) and not getattr(
                getattr(arg, "alias_info", None), "is_write", False
            ):
                kwargs[arg.name] = None
            continue
        else:
            try:
                val = _arg_default(arg, type_str, dtype)
            except KeyError:
                return None
        if arg.kwarg_only:
            kwargs[arg.name] = val
        else:
            positional.append(val)

    if not positional:
        return None

    return SampleInput(positional[0], args=tuple(positional[1:]), kwargs=kwargs)


def _is_out_schema(schema) -> bool:
    """True if any argument is a write-alias (i.e. an ``out=`` overload)."""
    return any(getattr(getattr(a, "alias_info", None), "is_write", False) for a in schema.arguments)


def _is_placeholder_out(val) -> bool:
    """True if ``val`` is a placeholder out buffer (a 0-element tensor, or a
    list of them) rather than a correctly-shaped one. Registry entries use a
    ``(0,)`` shape when the real out shape isn't known at authoring time."""
    if isinstance(val, torch.Tensor):
        return val.numel() == 0
    if isinstance(val, list | tuple):
        return bool(val) and all(isinstance(t, torch.Tensor) and t.numel() == 0 for t in val)
    return False


def _functional_probe_candidates(op, n_positional: int):
    """Functional (non-out) overloads of ``op``'s packet, best-match first."""
    schema = getattr(op, "_schema", None)
    if schema is None:
        return []
    base_name = schema.name.split("::")[-1]
    pkt = getattr(torch.ops.aten, base_name, None)
    if pkt is None:
        return []

    candidates = []
    for overload in pkt.overloads():
        cand = getattr(pkt, overload)
        cand_schema = getattr(cand, "_schema", None)
        if cand_schema is None or _is_out_schema(cand_schema):
            continue
        positional = [a for a in cand_schema.arguments if not a.kwarg_only]
        required = [a for a in positional if not a.has_default_value()]
        arity_ok = len(required) <= n_positional <= len(positional)
        candidates.append((arity_ok, cand))
    # Stable sort: arity-matching overloads first, original packet order within.
    candidates.sort(key=lambda c: not c[0])
    return [cand for _, cand in candidates]


def _probe_out_specs(op, sample) -> list[tuple[torch.Size, torch.dtype]] | None:
    """Infer expected out-variant output shapes+dtypes via FakeTensorMode.

    Runs the op's functional overload inside FakeTensorMode so no real kernel
    executes and no tensor storage is allocated. This prevents sample building
    from triggering Neuron compilation as a side effect (e.g., when a
    composite op decomposes onto device-sensitive primitives).

    Returns one ``(shape, dtype)`` per output tensor — a single-element list
    for ordinary ops and a multi-element list for multi-output ops.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    kw = {k: v for k, v in sample.kwargs.items() if k != "out"}
    schema = getattr(op, "_schema", None)

    # Non-out ops (or ops with no schema) probe directly.
    if schema is None or not _is_out_schema(schema):
        probe_ops = [op]
    else:
        # Out overload: probe via a functional sibling. Try arity-ordered
        # candidates and keep the first that dispatches under fake tensors.
        probe_ops = _functional_probe_candidates(op, 1 + len(sample.args)) or [op]

    for probe_op in probe_ops:
        try:
            with FakeTensorMode(allow_non_fake_inputs=True):
                out = probe_op(sample.input, *sample.args, **kw)
            if isinstance(out, torch.Tensor):
                return [(out.shape, out.dtype)]
            if (
                isinstance(out, tuple | list)
                and out
                and all(isinstance(o, torch.Tensor) for o in out)
            ):
                return [(o.shape, o.dtype) for o in out]
        except Exception as e:
            logger.debug("out-probe failed for %s via %s: %s", op, probe_op, e)
    return None


class NeuronOpInfo(OpInfo):
    """Extends OpInfo with the Neuron shape registry as primary input source.

    Input sources (in order):
      1. Shape registry entries (primary — extensible via CSV/JSON)
      2. PyTorch op_db sample_inputs_func (fallback for base/`.default`/out variants)
    """

    def __init__(
        self,
        name: str,
        *,
        op: Callable | None = None,
        op_name: str = "",
        op_group: str = "unknown",
        **kwargs,
    ):
        resolved = op if op is not None else _resolve_op(op_name or f"aten::{name}")
        self.op_resolved = resolved is not None
        self._resolved_op = resolved

        kwargs.pop("sample_inputs_func", None)
        self._pytorch_sample_inputs_func = kwargs.pop("_pytorch_sample_inputs_func", None)
        kwargs.setdefault("dtypes", _dispatch_dtypes({torch.float32, torch.bfloat16}))

        super().__init__(
            name,
            op=resolved or _noop,
            sample_inputs_func=self._sample_inputs,
            **kwargs,
        )

        self.op_name = op_name or f"aten::{name}"
        self.op_group = op_group

    # ── sample construction ──────────────────────────────────────────────────

    def make_sample(self, arg_shapes, device, dtype) -> SampleInput | None:
        """Build one SampleInput from an arg_shapes dict."""
        if self._resolved_op is None:
            return None
        return _schema_make_sample(self._resolved_op, arg_shapes, device, dtype)

    def _registry_entries(self) -> list:
        """Registry entries for this op. Propagation at registry-load time
        (see NeuronShapeRegistry.propagate_to_compatible_overloads) already
        covered schema-compatible ``.out`` / ``.default`` overloads, so no
        name-based runtime fallbacks are needed here.

        Caps the vended list to the top-N entries by max tensor element
        count, where N comes from the ``NEURON_MAX_SHAPES_PER_OP`` env var
        (default ``10``). Set by conftest from the ``--max-shapes-per-op``
        CLI option. Model-traced registries generated from compile-cache
        keys can carry 500-1000 entries per op with tensors up to multiple
        GB, which turns a single op's coverage run into a multi-hour
        cold-compile sweep and can OOM HBM on the largest shapes. Keeping
        the top-N by size preserves the strongest roofline signal per
        compile. A non-positive value disables the cap.
        """
        entries = SHAPE_REGISTRY.get(self.op_name)
        if entries and os.environ.get("NEURON_MODEL_SHAPES_ONLY"):
            entries = [e for e in entries if any(s != "builtin" for s in e.sources)]

        max_per_op = int(os.environ.get("NEURON_MAX_SHAPES_PER_OP", "10"))
        if max_per_op > 0 and len(entries) > max_per_op:
            entries = sorted(
                entries,
                key=lambda e: max_tensor_elems(e.arg_shapes, self.op_name),
                reverse=True,
            )[:max_per_op]
        return entries

    def _registry_samples(self, device, dtype) -> Iterable[SampleInput]:
        for config in self._registry_entries():
            try:
                sample = self.make_sample(config.arg_shapes, device, dtype)
            except Exception as e:
                logger.debug("%s: sample build failed: %s", self.op_name, e)
                continue
            if sample is not None:
                yield self._with_out_kwargs(sample)

    def _opdb_samples(self, op_info, device, dtype, requires_grad, **kwargs):
        if self._pytorch_sample_inputs_func is None or not self.op_resolved:
            return
        schema = getattr(self._resolved_op, "_schema", None)
        overload = self.op_name.split(".", 1)[1] if "." in self.op_name else ""
        is_base = not overload or overload == "default"
        if not (is_base or (schema is not None and _is_out_variant(schema))):
            return
        # Schema arg names the target op will accept — used to filter kwargs
        # coming from the functional variant's opdb sample_inputs_func. Factory
        # `.out` overloads (aten::arange.out, aten::full.out, aten::randn.out,
        # ...) don't accept dtype/layout/device/pin_memory/memory_format/names
        # kwargs that opdb includes for the functional variant. Dropping those
        # prevents spurious "expected at most N args but received M" errors
        # that were surfacing as false-positive `status=supported` rows.
        schema_arg_names = {a.name for a in schema.arguments} if schema is not None else None
        # Required args the target overload declares but the functional opdb
        # sample omits (no schema default, not Optional, not the write-alias
        # `out`). Synthesized by name via `_arg_default` so the opdb sample
        # matches the overload — e.g. ``out_dtype`` on *.dtype_out, ``dim``/
        # ``keepdim`` on reduction .out/dtype-out. A positional arg passed as a
        # kwarg is accepted by the ATen dispatcher.
        synth_required = []
        if schema is not None:
            for a in schema.arguments:
                ts = str(a.type)
                if a.has_default_value() or "?" in ts or "Optional" in ts:
                    continue
                if getattr(getattr(a, "alias_info", None), "is_write", False):
                    continue  # `out` filled by _with_out_kwargs
                # KeyError = tensor/self args etc. — nothing to synthesize.
                with contextlib.suppress(KeyError):
                    synth_required.append((a.name, _arg_default(a, ts, dtype)))
        # Positional-arity window of the overload bound to this op row. The
        # functional opdb sample_inputs_func emits samples for every functional
        # shape of the base op (e.g. eye(n) AND eye(n, m));
        pos_max = _schema_positional_bounds(schema)[1] if schema is not None else None
        try:
            for sample in self._pytorch_sample_inputs_func(
                op_info, device, dtype, requires_grad, **kwargs
            ):
                if pos_max is not None and (1 + len(sample.args)) > pos_max:
                    continue
                if schema_arg_names is not None and sample.kwargs:
                    filtered = {k: v for k, v in sample.kwargs.items() if k in schema_arg_names}
                    if len(filtered) != len(sample.kwargs):
                        sample = SampleInput(sample.input, args=sample.args, kwargs=filtered)
                if synth_required and schema is not None:
                    n_pos = 1 + len(sample.args)
                    missing = {
                        nm: v
                        for nm, v in synth_required
                        if nm not in sample.kwargs and _schema_positional_index(schema, nm) >= n_pos
                    }
                    if missing:
                        sample = SampleInput(
                            sample.input, args=sample.args, kwargs={**sample.kwargs, **missing}
                        )
                yield self._with_out_kwargs(sample)
        except Exception as e:
            logger.debug("%s: opdb fallback failed: %s", self.op_name, e)

    def _sample_inputs(self, op_info, device, dtype, requires_grad, **kwargs):
        """Yield samples in priority order:

        1. Shape registry (exact op_name match, or base name for .default)
        2. PyTorch op_db (only for base, .default, and .out overloads)
        3. If neither yields anything, the test marks the op as pending_registry.
        """
        yielded = False
        for s in self._registry_samples(device, dtype):
            yielded = True
            yield s
        if not yielded:
            yield from self._opdb_samples(op_info, device, dtype, requires_grad, **kwargs)

    def _with_out_kwargs(self, sample: SampleInput) -> SampleInput:
        """Return a new SampleInput with writable alias kwargs populated when missing.

        Uses a functional probe to learn expected shape/dtype instead of string heuristics.

        Registry entries sometimes carry a placeholder out value (a 0-element
        ``(0,)`` tensor) when the true out shape isn't known at authoring time.
        Treat such placeholders the same as a missing kwarg so the FakeTensor
        probe re-derives the correct shape. For data-dependent ops
        (``masked_select.out``, ``nonzero.out``, ...) the probe raises and returns
        None, so the placeholder is left untouched — exactly as before.
        """
        if self._resolved_op is None:
            return sample
        schema = getattr(self._resolved_op, "_schema", None)
        if schema is None:
            return sample

        write_args = [
            a
            for a in schema.arguments
            if a.kwarg_only
            and (a.name not in sample.kwargs or _is_placeholder_out(sample.kwargs[a.name]))
            and getattr(getattr(a, "alias_info", None), "is_write", False)
        ]
        if not write_args:
            return sample

        # Probe against a copy with placeholder out values dropped, so the
        # functional probe sees a clean call and no stale (0,) buffer leaks in.
        # Keep the original sample for the fallback paths below — if the probe
        # can't derive a shape (e.g. data-dependent ops), the original (with its
        # placeholder) is returned unchanged, exactly as before this change.
        write_names = {a.name for a in write_args}
        probe_sample = SampleInput(
            sample.input,
            args=sample.args,
            kwargs={k: v for k, v in sample.kwargs.items() if k not in write_names},
        )

        specs = _probe_out_specs(self._resolved_op, probe_sample)
        if specs is None:
            return sample

        single_tensor_list = len(write_args) == 1 and (
            "Tensor[]" in str(write_args[0].type) or "List[Tensor]" in str(write_args[0].type)
        )
        if not single_tensor_list and len(specs) < len(write_args):
            return sample

        extra = {}
        for i, arg in enumerate(write_args):
            type_str = str(arg.type)
            # Match both C++ schema style ("Tensor[]") and Python-style
            # ("List[Tensor]") — torch returns the latter for _foreach_*_out.
            if "Tensor[]" in type_str or "List[Tensor]" in type_str:
                # list length inferred from first tensor-list input, else 2
                n_list = 2
                for a in (sample.input, *sample.args):
                    if isinstance(a, list) and a and isinstance(a[0], torch.Tensor):
                        n_list = len(a)
                        break
                shape, out_dtype = specs[0]
                extra[arg.name] = [torch.empty(shape, dtype=out_dtype) for _ in range(n_list)]
            elif "Tensor" in type_str:
                shape, out_dtype = specs[i]
                extra[arg.name] = torch.empty(shape, dtype=out_dtype)
            else:
                return sample

        return SampleInput(sample.input, args=sample.args, kwargs={**sample.kwargs, **extra})

    # ── public helpers ───────────────────────────────────────────────────────

    def shape_sweep_inputs(self, device, dtype, **_kwargs) -> Iterable[SampleInput]:
        """All registered shapes for this op — for perf sweeps."""
        yield from self._registry_samples(device, dtype)

    @classmethod
    def from_schema(cls, op_name: str, op_group: str = "unknown") -> NeuronOpInfo:
        """Auto-generate from op schema."""
        base = op_name.replace("aten::", "").split(".")[0]
        return cls(name=base, op_name=op_name, op_group=op_group)

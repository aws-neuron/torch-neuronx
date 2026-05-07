"""NeuronOpInfo — extends PyTorch OpInfo with an op-specific shape registry.

Design:
- The shape registry is the primary source of inputs.
- `_schema_make_sample` reads the op schema to build a SampleInput from a
  `{arg_name: shape_or_spec}` dict, so JSON-loaded shapes work for any op
  without per-op code.
- PyTorch's op_db is a secondary fallback for base/`.default`/out overloads.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable

import torch
from torch.testing._internal.opinfo.core import OpInfo, SampleInput, _dispatch_dtypes

from .neuron_shape_registry import SHAPE_REGISTRY

logger = logging.getLogger(__name__)


# Stub handed to OpInfo when the real op can't be resolved; never invoked
# because we override `sample_inputs_func`.
def _noop(*_args, **_kwargs):
    return None


_DTYPE_MAP = {
    "long": torch.long,
    "int": torch.int32,
    "bool": torch.bool,
    "double": torch.float64,
    "float": None,  # sentinel: use caller's default dtype
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
    """Build a tensor from a shape tuple/list or a dict {shape, dtype, low, high, sigmoid}."""
    if isinstance(spec, dict):
        shape = tuple(spec["shape"])
        dt = _DTYPE_MAP.get(spec.get("dtype", "float"))
        dt = dt if dt is not None else default_dtype
        if dt in (torch.long, torch.int32):
            return torch.randint(
                spec.get("low", 0), spec.get("high", 4), shape, device=device, dtype=dt
            )
        if dt is torch.bool:
            return torch.randint(0, 2, shape, device=device, dtype=torch.bool)
        t = torch.randn(shape, device=device, dtype=dt)
        return t.sigmoid() if spec.get("sigmoid") else t
    return torch.randn(tuple(spec), device=device, dtype=default_dtype)


def _tensor_list_from_spec(raw, device, default_dtype, count_default=2):
    """Build a list of tensors from a shape, a list of shape-dicts, or a passthrough."""
    if isinstance(raw, tuple | list) and raw and isinstance(raw[0], int):
        return [torch.randn(*raw, device=device, dtype=default_dtype) for _ in range(count_default)]
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return [_tensor_from_spec(d, device, default_dtype) for d in raw]
    return raw


def _build_value(type_str, raw, device, dtype):
    if "Tensor[]" in type_str or "List[Tensor]" in type_str:
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


def _arg_default(arg, type_str):
    """Return default value for a missing positional arg, or raise KeyError to bail."""
    if arg.has_default_value():
        return arg.default_value
    if "?" in type_str or "Optional" in type_str:
        return None
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
            continue
        else:
            try:
                val = _arg_default(arg, type_str)
            except KeyError:
                return None
        if arg.kwarg_only:
            kwargs[arg.name] = val
        else:
            positional.append(val)

    if not positional:
        return None
    return SampleInput(positional[0], args=tuple(positional[1:]), kwargs=kwargs)


def _probe_out_shape_dtype(op, sample) -> tuple[torch.Size, torch.dtype] | None:
    """Call functional overload once to learn expected out shape+dtype. None on failure."""
    try:
        # Drop any out= kwargs that may have been pre-populated
        kw = {k: v for k, v in sample.kwargs.items() if k != "out"}
        out = op(sample.input, *sample.args, **kw)
        if isinstance(out, torch.Tensor):
            return out.shape, out.dtype
        if isinstance(out, tuple | list) and out and isinstance(out[0], torch.Tensor):
            return out[0].shape, out[0].dtype
    except Exception as e:
        logger.debug("out-probe failed for %s: %s", op, e)
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
        """Registry entries for this op; falls back to base name for `.default`."""
        entries = SHAPE_REGISTRY.get(self.op_name)
        if not entries and self.op_name.endswith(".default"):
            entries = SHAPE_REGISTRY.get(self.op_name.rsplit(".", 1)[0])
        if entries and os.environ.get("NEURON_MODEL_SHAPES_ONLY"):
            entries = [e for e in entries if any(s != "builtin" for s in e.sources)]
        return entries

    def _registry_samples(self, device, dtype) -> Iterable[SampleInput]:
        for config in self._registry_entries():
            try:
                sample = self.make_sample(config.arg_shapes, device, dtype)
            except Exception as e:
                logger.debug("%s: sample build failed: %s", self.op_name, e)
                continue
            if sample is not None:
                yield sample

    def _opdb_samples(self, op_info, device, dtype, requires_grad, **kwargs):
        if self._pytorch_sample_inputs_func is None or not self.op_resolved:
            return
        schema = getattr(self._resolved_op, "_schema", None)
        overload = self.op_name.split(".", 1)[1] if "." in self.op_name else ""
        is_base = not overload or overload == "default"
        if not (is_base or (schema is not None and _is_out_variant(schema))):
            return
        try:
            for sample in self._pytorch_sample_inputs_func(
                op_info, device, dtype, requires_grad, **kwargs
            ):
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
            and a.name not in sample.kwargs
            and getattr(getattr(a, "alias_info", None), "is_write", False)
        ]
        if len(write_args) != 1:
            return sample  # multi-out currently skipped; single-out only

        probed = _probe_out_shape_dtype(self._resolved_op, sample)
        if probed is None:
            return sample
        out_shape, out_dtype = probed

        arg = write_args[0]
        type_str = str(arg.type)
        if "Tensor[]" in type_str:
            # list length inferred from first tensor-list input, else 2
            n_list = 2
            for a in (sample.input, *sample.args):
                if isinstance(a, list) and a and isinstance(a[0], torch.Tensor):
                    n_list = len(a)
                    break
            extra = {arg.name: [torch.empty(out_shape, dtype=out_dtype) for _ in range(n_list)]}
        elif "Tensor" in type_str:
            extra = {arg.name: torch.empty(out_shape, dtype=out_dtype)}
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

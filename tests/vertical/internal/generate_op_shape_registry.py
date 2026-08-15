#!/usr/bin/env python3
"""Generate op_shape_registry.json from Neuron compilation artifacts or cache key files.

Parses cache keys from CacheKeyGenerator (torch_neuronx/csrc/core/cache/CacheKeyGenerator.cpp)
and maps parsed values to op schema arg names to produce a shape registry JSON.

Usage:
    # From compilation artifacts directory
    python generate_op_shape_registry.py /tmp/neuron_backend/neuronx-cc

    # From cache key text file (one cache key per line)
    python generate_op_shape_registry.py --cache-keys aggregate_cache_keys.txt

    # Merge into existing registry
    python generate_op_shape_registry.py /tmp/neuron_backend/neuronx-cc --merge existing.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from .neuron_op_info import _resolve_op

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_NON_ATEN = {"all_gather", "reduce_scatter", "prologue"}

# Regex to match a single tensor token: [dims]:Type:S(strides)
_TENSOR_RE = re.compile(r"\[([^\]]*)\]:([A-Za-z][A-Za-z0-9]*):S\(([^)]*)\)")

# Maps CacheKeyGenerator dtype strings to Python dtype names
_DTYPE_MAP = {
    "Float": "float32",
    "Double": "float64",
    "Half": "float16",
    "BFloat16": "bfloat16",
    "Long": "int64",
    "Int": "int32",
    "Short": "int16",
    "Byte": "uint8",
    "Char": "int8",
    "Bool": "bool",
    "ComplexFloat": "complex64",
    "ComplexDouble": "complex128",
}


@dataclass
class ParsedTensor:
    """A tensor parsed from a cache key token."""

    shape: list[int]
    dtype: str = "float32"
    contiguous: bool = True
    strides: list[int] | None = None


@dataclass
class ParsedCacheKey:
    """Result of parsing an operator cache key."""

    op_name: str
    positional: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)


def _parse_tensor_token(token: str) -> ParsedTensor | None:
    """Parse a tensor token like [1,2048,4096]:BFloat16:S(C) into ParsedTensor."""
    m = _TENSOR_RE.match(token)
    if not m:
        return None
    raw_dims, dtype_str, stride_str = m.group(1).strip(), m.group(2), m.group(3)
    shape = [int(d) for d in raw_dims.split(",") if d.strip().isdigit()] if raw_dims else []
    dtype = _DTYPE_MAP.get(dtype_str, dtype_str.lower())
    contiguous = stride_str == "C"
    strides = None if contiguous else [int(s) for s in stride_str.split(",") if s.strip().isdigit()]
    return ParsedTensor(shape=shape, dtype=dtype, contiguous=contiguous, strides=strides)


def _parse_token(token: str) -> Any:
    """Parse a single cache key token into a Python value.

    This is the inverse of CacheKeyGenerator's Append* methods in
    torch_neuronx/csrc/core/cache/CacheKeyGenerator.cpp.
    """
    if not token:
        return None

    # Tensor: [dims]:Type:S(...)
    parsed_tensor = _parse_tensor_token(token)
    if parsed_tensor is not None:
        return {"shape": parsed_tensor.shape, "dtype": parsed_tensor.dtype}

    if token in ("undef", "None"):
        return None
    if token in ("b0", "b1"):
        return token == "b1"
    # Static int: i<val>
    if token.startswith("i") and token != "int":
        try:
            return int(token[1:])
        except ValueError:
            pass
    # Dynamic int
    if token == "int":
        return None
    # Static float: d<val> (but not dev: or dtype:)
    if (
        len(token) > 1
        and token[0] == "d"
        and not token.startswith("dev:")
        and not token.startswith("dtype:")
    ):
        try:
            return float(token[1:])
        except ValueError:
            pass
    if token == "float":
        return None
    # Static scalar
    if token.startswith("sc:"):
        try:
            return float(token[3:])
        except ValueError:
            return None
    # String
    if token.startswith("s:"):
        return token[2:]
    # Dtype: dtype:BFloat16 → "bfloat16"
    if token.startswith("dtype:"):
        dtype_str = token[6:]
        return _DTYPE_MAP.get(dtype_str, dtype_str.lower())
    # Device: dev:neuron:0 → None (device indices are runtime-specific, not shape info)
    if token.startswith("dev:"):
        return None
    # Scalar dynamic types — the exact value is data-dependent and not
    # recoverable from the cache key. Return None so _fill_scalar_default
    # can apply op-specific defaults (lr=0.001, eps=1e-5, etc.) or fall back
    # to type-based defaults (Scalar→0.0) as a last resort.
    if token.startswith("scalar:"):
        return None
    # Complex, generator
    if token.startswith("cd:") or token in ("complex", "gen"):
        return None
    # IntList: IL[i1,i2,...]
    if token.startswith("IL[") and token.endswith("]"):
        inner = token[3:-1]
        return (
            [int(v[1:]) if v.startswith("i") else int(v) for v in inner.split(",") if v]
            if inner
            else []
        )
    # BoolList: BL[b0,b1,...]
    if token.startswith("BL[") and token.endswith("]"):
        inner = token[3:-1]
        return [v == "b1" for v in inner.split(",") if v] if inner else []
    # DoubleList: DL[d1.0,...]
    if token.startswith("DL[") and token.endswith("]"):
        inner = token[3:-1]
        return (
            [float(v[1:]) if v.startswith("d") else float(v) for v in inner.split(",") if v]
            if inner
            else []
        )
    # TensorList / GenericList: parse tensor shapes from inner content
    if token.startswith(("TL[", "GL[")):
        inner = token[3:-1]  # Strip TL[...] or GL[...] outer brackets
        # Empty bracket (e.g. "GL[]") is a legitimate empty list — return [],
        # not the literal string.
        if not inner:
            return []
        tensors = []
        for m in _TENSOR_RE.finditer(inner):
            raw_dims, dtype_str = m.group(1).strip(), m.group(2)
            shape = [int(d) for d in raw_dims.split(",") if d.strip().isdigit()] if raw_dims else []
            dtype = _DTYPE_MAP.get(dtype_str, dtype_str.lower())
            tensors.append({"shape": shape, "dtype": dtype})
        # Non-empty inner with no tensor matches: fall back to the raw token
        # so the caller can log/debug it.
        return tensors if tensors else token

    logger.debug(f"Unknown cache key token: {token!r}")
    return token


def _tokenize_signature(signature: str) -> list[tuple[str | None, str]]:
    """Split cache key signature into (kw_name_or_None, raw_token) pairs.

    Implements a character-by-character state machine that tracks bracket/paren
    depth to correctly split on '_' without breaking inside nested structures
    like tensor strides S(1024,1) or list brackets IL[i1,i2].

    Returns list of (kw_name, raw_token) where kw_name is None for positional args.
    """
    tokens: list[tuple[str | None, str]] = []
    depth = 0  # bracket + paren nesting depth
    start = 0  # start of current token
    in_kw = False  # True when accumulating a kw:name=value token

    for i, ch in enumerate(signature):
        if ch in ("[", "("):
            depth += 1
        elif ch in ("]", ")"):
            depth = max(0, depth - 1)
        elif ch == "_" and depth == 0 and not in_kw:
            token = signature[start:i]
            if token:
                if token.startswith("kw:") and "=" in token:
                    eq = token.index("=")
                    tokens.append((token[3:eq], token[eq + 1 :]))
                else:
                    tokens.append((None, token))
            start = i + 1

        # Track whether we're inside a kw: token that hasn't reached = yet
        if depth == 0:
            current = signature[start : i + 1]
            in_kw = current.startswith("kw:") and "=" not in current

    # Last token
    token = signature[start:]
    if token:
        if token.startswith("kw:") and "=" in token:
            eq = token.index("=")
            tokens.append((token[3:eq], token[eq + 1 :]))
        else:
            tokens.append((None, token))

    return tokens


def parse_cache_key(cache_key: str) -> ParsedCacheKey | None:
    """Parse operator_cache_key into op_name and all arg values.

    Cache key format: op_name--_arg0_arg1_..._kw:name=val_...:compilation_hash
    This is the inverse of CacheKeyGenerator::Generate() — it tokenizes the
    signature, parses each token, and separates positional args from kwargs.
    """
    if not cache_key or "--" not in cache_key:
        return None
    op_name, signature = cache_key.split("--", 1)
    # Skip optional int64 mode field
    if signature.startswith(("p64--", "n64--")):
        signature = signature.split("--", 1)[1]
    if not op_name or "alias:" in cache_key:
        return None
    if re.match(r"nki_\d+_", op_name) or re.match(r"[0-9a-f]+:[0-9a-f]+", op_name):
        return None

    tokens = _tokenize_signature(signature)
    positional: list[Any] = []
    kwargs: dict[str, Any] = {}
    for kw_name, raw in tokens:
        # Strip trailing compilation hash (64-char hex after last ':')
        if ":" in raw:
            last = raw.split(":")[-1]
            if len(last) == 64 and all(c in "0123456789abcdef" for c in last):
                raw = raw[: raw.rfind(":")]
        val = _parse_token(raw)
        if kw_name:
            kwargs[kw_name] = val
        else:
            positional.append(val)
    return ParsedCacheKey(op_name=op_name, positional=positional, kwargs=kwargs)


def _get_schema(op_name: str):
    """Get the op schema using NeuronOpInfo's resolver."""
    op = _resolve_op(op_name)
    return getattr(op, "_schema", None) if op else None


# Defaults for required non-Tensor schema args that cache keys encode as
# dynamic (null). The value is synthesized so the op can actually execute at
# test time — real models tend to train with these canonical scalars anyway.
# Matched first by (base_op, arg_name), then by arg_name alone, then by
# type_str heuristic.
_SCALAR_DEFAULT_BY_OP_ARG: dict[tuple[str, str], object] = {
    ("aten::_fused_adam_", "lr"): 0.001,
    ("aten::_fused_adam_", "beta1"): 0.9,
    ("aten::_fused_adam_", "beta2"): 0.999,
    ("aten::_fused_adam_", "weight_decay"): 0.01,
    ("aten::_fused_adam_", "eps"): 1e-8,
    ("aten::_fused_adam_", "amsgrad"): False,
    ("aten::_fused_adam_", "maximize"): False,
    ("aten::_fused_adamw_", "lr"): 0.001,
    ("aten::_fused_adamw_", "beta1"): 0.9,
    ("aten::_fused_adamw_", "beta2"): 0.999,
    ("aten::_fused_adamw_", "weight_decay"): 0.01,
    ("aten::_fused_adamw_", "eps"): 1e-8,
    ("aten::_fused_adamw_", "amsgrad"): False,
    ("aten::_fused_adamw_", "maximize"): False,
    ("aten::_fused_sgd_", "lr"): 0.001,
    ("aten::_fused_sgd_", "momentum"): 0.9,
    ("aten::_fused_sgd_", "dampening"): 0.0,
    ("aten::_fused_sgd_", "weight_decay"): 0.0,
    ("aten::_fused_sgd_", "nesterov"): False,
    ("aten::_fused_sgd_", "maximize"): False,
    ("aten::_fused_sgd_", "is_first_step"): False,
    # clamp rejects both bounds None; cache key drops the value. See _FORCE_FILL_OPTIONAL.
    ("aten::clamp", "max"): 1.0,
}

# (op, arg) Optional args that must be filled anyway — the op is invalid when the
# cache-key-dropped value defaults to None (e.g. clamp with both bounds None).
_FORCE_FILL_OPTIONAL: set[tuple[str, str]] = {
    ("aten::clamp", "max"),
}

# Arg-name-scoped defaults (any op). Applied after the op-specific table.
_SCALAR_DEFAULT_BY_NAME: dict[str, object] = {
    "eps": 1e-5,  # layer_norm / batch_norm / group_norm canonical eps
    "momentum": 0.1,
    "alpha": 1.0,
    "beta": 1.0,
    "dim": 0,
    "keepdim": False,
    "unbiased": True,
    "correction": 1,
}

# Type-based fallback: coarse but safe when we can't match by name.
_SCALAR_DEFAULT_BY_TYPE: dict[str, object] = {
    "float": 0.0,
    "int": 0,
    "SymInt": 0,
    "bool": False,
    "number": 0.0,
    "Scalar": 0.0,
    "str": "",
}


def _fill_scalar_default(op_name: str, arg) -> object | None:
    """Return a sensible default for a required non-Tensor arg, or None if
    we don't know. Looked up first by (op_name, arg_name), then arg_name
    alone, then schema type."""
    type_str = str(arg.type)
    base = op_name.split(".", 1)[0]  # strip overload so .out etc. match
    hit = _SCALAR_DEFAULT_BY_OP_ARG.get((base, arg.name))
    if hit is not None:
        return hit
    hit = _SCALAR_DEFAULT_BY_NAME.get(arg.name)
    if hit is not None:
        return hit
    return _SCALAR_DEFAULT_BY_TYPE.get(type_str)


def _build_arg_shapes(op_name: str, parsed: ParsedCacheKey) -> dict[str, Any]:
    """Map parsed cache key values to schema arg names.

    Walks the op schema in order, consuming positional args sequentially
    and matching kwargs by name. For missing args, uses schema defaults or None.

    Cache keys encode dynamic (SymFloat/SymInt) scalars as the tokens 'float'
    and 'int', which ``_parse_token`` translates to ``None``. For schema args
    that require a concrete non-None value at execution time (the aten
    dispatcher won't accept None for a ``float lr``), we fill a plausible
    default via ``_fill_scalar_default`` as a post-processing pass.
    """
    schema = _get_schema(op_name)
    if schema is None:
        return {f"arg{i}": v for i, v in enumerate(parsed.positional) if isinstance(v, list | dict)}

    arg_shapes: dict[str, Any] = {}
    pos_idx = 0
    for arg in schema.arguments:
        name = arg.name
        type_str = str(arg.type)

        if arg.kwarg_only:
            if name in parsed.kwargs:
                arg_shapes[name] = parsed.kwargs[name]
            else:
                has_default = getattr(arg, "has_default_value", lambda: False)
                if callable(has_default):
                    has_default = has_default()
                if has_default:
                    arg_shapes[name] = arg.default_value
                else:
                    arg_shapes[name] = None
            continue

        if pos_idx < len(parsed.positional):
            arg_shapes[name] = parsed.positional[pos_idx]
            pos_idx += 1
        elif name in parsed.kwargs:
            arg_shapes[name] = parsed.kwargs[name]
        else:
            has_default = getattr(arg, "has_default_value", lambda: False)
            if callable(has_default):
                has_default = has_default()
            if has_default:
                arg_shapes[name] = arg.default_value
            elif "?" in type_str or "Optional" in type_str:
                arg_shapes[name] = None

    # Convert 0-d tensor specs to None for Scalar-typed args: the cache key
    # encodes a scalar value as a 0-d tensor []:Type:S(C), but the schema
    # expects a number. The actual value is lost, so set to None and let
    # _fill_scalar_default apply op-specific or type-based defaults below.
    for arg in schema.arguments:
        type_str = str(arg.type)
        if "Scalar" in type_str and arg.name in arg_shapes:
            val = arg_shapes[arg.name]
            if isinstance(val, dict) and val.get("shape") == []:
                arg_shapes[arg.name] = None

    # Fill sensible defaults for required non-Tensor args captured as dynamic.
    for arg in schema.arguments:
        name = arg.name
        type_str = str(arg.type)
        if name not in arg_shapes or arg_shapes[name] is not None:
            continue
        # Only fill non-Tensor required args; Optional[Tensor]=None is legitimate,
        # and writable kwargs are populated by the test's _with_out_kwargs.
        if "Tensor" in type_str:
            continue
        if ("?" in type_str or "Optional" in type_str) and (
            op_name.split(".", 1)[0],
            name,
        ) not in _FORCE_FILL_OPTIONAL:
            continue
        default = _fill_scalar_default(op_name, arg)
        if default is not None:
            arg_shapes[name] = default
            logger.debug(
                "%s: filled %s=%r (was null; schema type=%s)",
                op_name,
                name,
                default,
                type_str,
            )

    for arg in schema.arguments:
        if arg.name not in arg_shapes:
            logger.warning(f"{op_name}: missing required arg '{arg.name}' (type={arg.type})")

    return arg_shapes


def _normalize_parsed(
    parsed: ParsedCacheKey, op_name_override: str | None = None
) -> ParsedCacheKey | None:
    """Apply op name override and filter non-aten ops."""
    op = parsed.op_name
    if op in _NON_ATEN:
        return None
    if op_name_override and op_name_override != "CompilableKernelExecution":
        op = op_name_override
    if op in _NON_ATEN:
        return None
    if not op.startswith("aten::"):
        op = f"aten::{op}"
    return ParsedCacheKey(op_name=op, positional=parsed.positional, kwargs=parsed.kwargs)


def collect_from_dir(root_dir: str) -> list[ParsedCacheKey]:
    """Collect from metadata.txt files under a directory."""
    results: list[ParsedCacheKey] = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "metadata.txt" not in filenames:
            continue
        try:
            with open(os.path.join(dirpath, "metadata.txt")) as f:
                lines = f.readlines()
            if len(lines) < 2:
                continue
            op_name_raw = lines[0].strip()
            parsed = parse_cache_key(lines[1].strip())
            if not parsed:
                continue
            # Skip prologue cache keys before overriding op name
            if parsed.op_name in _NON_ATEN:
                continue
            normalized = _normalize_parsed(parsed, op_name_raw)
            if normalized:
                results.append(normalized)
        except OSError as e:
            logger.warning(f"Failed to read {dirpath}/metadata.txt: {e}")
            continue
    logger.info(f"Collected {len(results)} entries from {root_dir}")
    return results


def collect_from_cache_keys_file(path: str) -> list[ParsedCacheKey]:
    """Collect from a text file with one cache key per line."""
    results: list[ParsedCacheKey] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = parse_cache_key(line)
            if not parsed:
                continue
            normalized = _normalize_parsed(parsed)
            if normalized:
                results.append(normalized)
    logger.info(f"Collected {len(results)} entries from {path}")
    return results


# ── Per-op annotators ───────────────────────────────────────────────────────
#
# Cache keys record only [shape]:dtype:S(strides) — never tensor values (see
# AppendTensor in torch_neuronx/csrc/core/cache/CacheKeyGenerator.cpp). Ops
# whose implementations impose value-level constraints (alignment, bounds,
# consistency) can't be safely sampled from a bare shape/dtype spec: the test
# would generate random values that violate the constraint.
#
# Annotators inject the missing value-generation metadata into arg_shapes so
# _tensor_from_spec produces valid samples. Keyed by base op name so one entry
# covers every overload. Register a new annotator here when an op fails the
# coverage sweep with an error the shape parser structurally cannot avoid.


def _annotate_grouped_mm(arg_shapes: dict) -> dict:
    """offs must be sorted + multiple of 128 (NKI tile alignment)."""
    offs = arg_shapes.get("offs")
    if not isinstance(offs, dict) or "multiple_of" in offs:
        return arg_shapes
    self_spec = arg_shapes.get("self")
    self_shape = self_spec.get("shape") if isinstance(self_spec, dict) else self_spec
    if not isinstance(self_shape, list | tuple) or not self_shape:
        return arg_shapes
    # `offs * 128` must land in [0, self.shape[0]]; generate uniform in
    # [0, self.shape[0] // 128 + 1) so the sorted, x128 result fits.
    t_dim = int(self_shape[0])
    return {
        **arg_shapes,
        "offs": {
            **offs,
            "multiple_of": 128,
            "sorted": True,
            "low": 0,
            "high": max(1, t_dim // 128 + 1),
        },
    }


def _annotate_index(arg_shapes: dict) -> dict:
    """Set each index's ``high`` to the corresponding self dim so random fill
    stays in bounds — critical for size-1 dims where the default ``high=4`` is
    always out of range."""
    self_spec = arg_shapes.get("self")
    self_shape = self_spec.get("shape") if isinstance(self_spec, dict) else self_spec
    indices = arg_shapes.get("indices")
    if not isinstance(self_shape, list | tuple) or not isinstance(indices, list | tuple):
        return arg_shapes
    new_indices = []
    for i, idx in enumerate(indices):
        if isinstance(idx, dict) and "high" not in idx and i < len(self_shape):
            new_indices.append({**idx, "low": 0, "high": max(1, int(self_shape[i]))})
        else:
            new_indices.append(idx)
    return {**arg_shapes, "indices": new_indices}


def _annotate_repeat_interleave(arg_shapes: dict) -> dict:
    """Drop the recorded ``output_size``: it was consistent with the model's
    real repeats at trace time, but the test generates random repeats whose
    sum won't match. Letting output_size be None makes the runtime derive it
    from the actual sample."""
    if arg_shapes.get("output_size") is None:
        return arg_shapes
    return {**arg_shapes, "output_size": None}


_ANNOTATORS = {
    "_grouped_mm": _annotate_grouped_mm,
    "index": _annotate_index,
    "repeat_interleave": _annotate_repeat_interleave,
}


def _annotate(op_name: str, arg_shapes: dict) -> dict:
    """Apply per-op annotator, if any. Returns arg_shapes unchanged when no
    annotator is registered for the base op name."""
    base = op_name.replace("aten::", "").split(".", 1)[0]
    fn = _ANNOTATORS.get(base)
    return fn(arg_shapes) if fn else arg_shapes


def build_registry(entries: list[ParsedCacheKey], existing: list | None = None) -> list[dict]:
    """Build flat registry with named arg_shapes, deduplicating."""
    seen: set[tuple[str, str]] = set()
    registry: list[dict] = []
    if existing:
        for e in existing:
            seen.add((e["op_name"], json.dumps(e["arg_shapes"], sort_keys=True)))
            registry.append(e)

    for entry in entries:
        arg_shapes = _annotate(entry.op_name, _build_arg_shapes(entry.op_name, entry))
        key = (entry.op_name, json.dumps(arg_shapes, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        registry.append({"op_name": entry.op_name, "arg_shapes": arg_shapes})

    registry.sort(key=lambda x: x["op_name"])
    return registry


def main():
    parser = argparse.ArgumentParser(
        description="Generate op_shape_registry.json from compilation artifacts or cache key files"
    )
    parser.add_argument(
        "dirs",
        nargs="*",
        help="Directories containing compilation artifacts with metadata.txt",
    )
    parser.add_argument(
        "--cache-keys",
        help="Text file with one cache key per line (e.g. aggregate_cache_keys.txt)",
    )
    parser.add_argument("--merge", help="Existing registry to merge into")
    parser.add_argument("-o", "--output", default="op_shape_registry.json")
    args = parser.parse_args()

    entries: list[ParsedCacheKey] = []
    if args.dirs:
        for d in args.dirs:
            entries.extend(collect_from_dir(d))
    if args.cache_keys:
        entries.extend(collect_from_cache_keys_file(args.cache_keys))

    if not entries:
        logger.warning("No entries collected from any source")

    existing = None
    if args.merge and os.path.exists(args.merge):
        with open(args.merge) as f:
            existing = json.load(f)
        logger.info(f"Merging into existing registry with {len(existing)} entries")

    registry = build_registry(entries, existing)

    with open(args.output, "w") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
        f.write("\n")

    ops = len({e["op_name"] for e in registry})
    logger.info(f"Wrote {args.output}: {ops} ops, {len(registry)} unique shape combinations")


if __name__ == "__main__":
    main()

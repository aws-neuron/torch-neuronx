"""Helpers for answering questions about ATen ops from their schema."""

import functools
import logging

import torch

logger = logging.getLogger(__name__)


def _strip_aten_prefix(aten_name: str) -> str:
    return aten_name[6:] if aten_name.startswith("aten::") else aten_name


def _inplace_from_name(op_name: str) -> bool:
    """Name-based in-place heuristic: a trailing underscore on the base name.

    Only used when the schema can't be resolved. This misfires on dunder ops
    (``__rshift__`` "ends with _"), which is why the schema is preferred.
    """
    base = op_name.split(".")[0] if "." in op_name else op_name
    return "_." in op_name or (base.endswith("_") and "." not in op_name)


@functools.cache
def is_inplace_op(aten_name: str) -> bool:
    """Return True if `aten_name` writes through to its first argument.

    Decided by the op's schema: an op is in-place when its first argument is
    annotated as written (``Tensor(a!) self``). That distinguishes the
    operator-form dunders, which are pure (``aten::__rshift__.Tensor``), from
    their genuinely mutating counterparts (``aten::__irshift__.Tensor``) —
    a distinction the trailing-underscore heuristic cannot make.

    Falls back to the name heuristic for names that don't resolve to a schema
    (unqualified aliases like ``aten::bitwise_and`` and internal helpers).
    """
    name = _strip_aten_prefix(aten_name)
    base, _, overload = name.partition(".")

    try:
        packet = getattr(torch.ops.aten, base)
        schema = getattr(packet, overload or "default")._schema
    except (AttributeError, RuntimeError):
        logger.debug("No schema for %s; using name-based in-place heuristic", aten_name)
        return _inplace_from_name(name)

    args = schema.arguments
    if not args:
        return False
    alias_info = args[0].alias_info
    return bool(alias_info and alias_info.is_write)

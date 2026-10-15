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
    """Return whether an ATen operation writes through its first argument.

    Args:
        aten_name (str): ATen name, with or without the ``"aten::"`` prefix.

    Returns:
        bool: ``True`` when the schema marks the first argument writable or the fallback
            heuristic identifies an in-place name.

    Note:
        Schema metadata takes precedence over the name heuristic. Names without a
        resolvable schema use the trailing-underscore heuristic, which can misclassify
        operator-form dunders as in-place.
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

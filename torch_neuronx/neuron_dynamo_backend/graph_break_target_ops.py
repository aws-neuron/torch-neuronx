"""Collect the ops that Neuron cannot lower, at ``import torch_neuronx`` time.

An op is uncovered when its ATen op is neither registered in torch-mlir's Torch
dialect nor decomposed by the Neuron backend. The collection runs on import, so
the list always matches the installed dependencies and needs no checked-in copy.

Sources, all through native introspection:

  * Python universe:       ``torch.overrides.get_overridable_functions()``.
                           Callables there operate on tensors by design.
  * ATen mapping:          ``torch.jit._builtins._builtin_ops``, inverted into
                           ``id(callable) -> name``.
  * torch-mlir coverage:   ``OPERATION_NAME`` on each generated op class in
                           ``torch_mlir.dialects._torch_ops_gen``.
  * Neuron decompositions: keys of ``get_compile_decomposition_table()``.

An op counts as covered when ANY overload of its base name appears in either
coverage set, so ``empty.memory_format`` covers the base ``empty``.

The collection then drops:

  * ``.out`` / ``.output`` variants, ``*_backward`` ops, and trailing-underscore
    in-place variants.
  * An op the dispatcher does not know, such as ``aten::sym_max``. Coverage has
    no meaning for an op that never reaches a backend.
  * A ``CompositeImplicitAutograd`` op such as ``aten::concat``. Its kernel runs
    above the backend dispatch key and redispatches to ``aten::cat``, so the
    backend never receives it. See the gap this leaves in ``graph_break_ops``.
  * ``torch.Tensor`` methods. Dynamo routes a method through
    ``TensorVariable.call_method``, so the second element of the result carries
    the method names for that hook.
"""

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _fmt_overload(schema_name: str, overload_name: str) -> str:
    base = schema_name.split("::", 1)[-1]
    return f"{base}.{overload_name}" if overload_name else base


def _torch_mlir_ops() -> set[str]:
    import torch_mlir.dialects._torch_ops_gen as td

    prefix = "torch.aten."
    ops: set[str] = set()
    for name in dir(td):
        op_name = getattr(getattr(td, name, None), "OPERATION_NAME", None)
        if isinstance(op_name, str) and op_name.startswith(prefix):
            ops.add(op_name[len(prefix) :])
    return ops


def _decomposed_ops() -> set[str]:
    from torch_neuronx.neuron_dynamo_backend.decompositions import (
        get_compile_decomposition_table,
    )

    return {
        _fmt_overload(op._schema.name, op._schema.overload_name)
        for op in get_compile_decomposition_table()
    }


@functools.cache
def _aten_names_by_id() -> dict[int, str]:
    """Map ``id(python_callable) -> "<base>[.<overload>]"`` (``aten::`` stripped)."""
    import torch.jit._builtins as jb

    return {
        id(py): qname[len("aten::") :]
        for py, qname in jb._builtin_ops
        if isinstance(qname, str) and qname.startswith("aten::")
    }


def _aten_base_name(fn: Any) -> str | None:
    """Return the base ATen op name of ``fn``, or None when it has none."""
    name = _aten_names_by_id().get(id(fn))
    return name.split(".", 1)[0] if name else None


def _free_function_candidates() -> list[tuple[Any, Any]]:
    """Return ``(callable, owning_module)`` pairs from ``torch.overrides``.

    Skips the Tensor-method entries. Their owning namespace is a type object.
    """
    import torch.overrides as ov

    pairs: list[tuple[Any, Any]] = []
    for owner, fns in ov.get_overridable_functions().items():
        if isinstance(owner, type):
            continue
        pairs.extend((fn, owner) for fn in fns)
    return pairs


def _is_out_variant(name: str) -> bool:
    return name.endswith(".out") or name.endswith(".output")


def _is_backward(name: str) -> bool:
    return name.split(".", 1)[0].endswith("_backward")


def _is_inplace(fn: Any) -> bool:
    name = getattr(fn, "__name__", "") or ""
    return name.endswith("_") and not name.endswith("__")


def _user_facing_name(fn: Any, owner: Any) -> str | None:
    """Return ``<owner_module>.<attr>``, the name the user calls ``fn`` by.

    ``get_overridable_functions()`` buckets each callable under its user-facing
    module. The callable's ``__name__`` is the pybind name
    (``linalg_cholesky``), which often differs from the attribute name on that
    module (``cholesky``), so scan the namespace for the matching attribute.
    """
    mod_name = getattr(owner, "__name__", None)
    if not mod_name:
        return None
    for attr in dir(owner):
        if attr.startswith("_"):
            continue
        try:
            if getattr(owner, attr) is fn:
                return f"{mod_name}.{attr}"
        except Exception:
            continue
    return None


def _overload_keys(base: str) -> list[str]:
    """Return the dispatcher keys of every overload of ``base``, for example
    ``["aten::sort", "aten::sort.stable"]``. Empty when the dispatcher does not
    know the op."""
    packet = getattr(torch.ops.aten, base, None)
    if packet is None:
        return []
    keys = []
    for overload in packet.overloads():
        schema = getattr(packet, overload)._schema
        keys.append(
            f"{schema.name}.{schema.overload_name}" if schema.overload_name else schema.name
        )
    return keys


def _is_decomposed_above_backend(base: str) -> bool:
    """True when every overload of ``base`` has a ``CompositeImplicitAutograd``
    kernel, which means the backend never receives this op."""
    keys = _overload_keys(base)
    if not keys:
        return False
    for key in keys:
        try:
            if not torch._C._dispatch_has_kernel_for_dispatch_key(key, "CompositeImplicitAutograd"):
                return False
        except Exception:
            return False
    return True


def _returns_tensor(aten_name: str) -> bool:
    """True when the ATen op returns at least one Tensor.

    A metadata query such as ``aten::is_complex`` returns a bool. Dynamo folds
    the method form of those to a constant, so a graph break there would be
    wrong. Only Tensor-returning ops get a method entry.
    """
    packet = getattr(torch.ops.aten, aten_name.split(".", 1)[0], None)
    if packet is None:
        return False
    try:
        schema = getattr(packet, packet.overloads()[0])._schema
    except Exception:
        return False
    return any("Tensor" in str(ret.type) for ret in schema.returns)


def collect_unsupported_ops() -> tuple[frozenset[Callable[..., Any]], frozenset[str]]:
    """Return operations that Neuron cannot lower.

    Returns:
        tuple: Uncovered free functions and uncovered tensor method names.

    Note:
        Method names are included only for uncovered tensor-returning ATen operations.
    """
    covered = _torch_mlir_ops() | _decomposed_ops()
    covered_bases = {name.split(".", 1)[0] for name in covered}
    aten_by_id = _aten_names_by_id()

    functions: set[Callable[..., Any]] = set()
    methods: set[str] = set()
    for fn, owner in _free_function_candidates():
        if not callable(fn) or inspect.isclass(fn) or _is_inplace(fn):
            continue
        aten_name = aten_by_id.get(id(fn))
        if aten_name is None:
            continue  # No known ATen op, so out of scope.
        if _is_out_variant(aten_name) or _is_backward(aten_name):
            continue
        base = aten_name.split(".", 1)[0]
        if aten_name in covered or base in covered_bases:
            continue
        if not _overload_keys(base) or _is_decomposed_above_backend(base):
            continue
        if _user_facing_name(fn, owner) is None:
            continue
        functions.add(fn)
        if hasattr(torch.Tensor, base) and _returns_tensor(aten_name):
            methods.add(base)

    logger.debug(
        "Neuron graph-break targets: %d free functions, %d tensor methods",
        len(functions),
        len(methods),
    )
    return frozenset(functions), frozenset(methods)

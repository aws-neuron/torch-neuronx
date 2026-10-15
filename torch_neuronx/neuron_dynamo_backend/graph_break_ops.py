"""Force ``torch.compile`` graph breaks on ops that Neuron cannot lower.

Three block lists feed one registry. ``CONDITIONAL_TARGET_OPS`` is configured by
hand: each entry carries a predicate that runs on the fake tensors and the
constant arguments of the call, so only the calls Neuron cannot serve break.
``UNCONDITIONAL_TARGET_OPS`` covers ops that always break, checked at
``trace_rules.lookup`` to catch C-extension built-ins that bypass the call-time
hooks (``torch._grouped_mm``). ``DEFAULT_TARGET_OPS`` holds ops with no Neuron
lowering at all, collected on import from the installed torch, torch-mlir, and
Neuron decompositions. An entry without a predicate always breaks, so all three
lists share one code path.

Known gap in the collected list: the coverage test cannot see through a
decomposition. ``torch.svd`` has a ``CompositeImplicitAutograd`` kernel, so the
collection skips it, but it decomposes to ``aten::_linalg_svd``, which Neuron
does not lower. Dynamo only sees the ``torch.svd`` call, so it gets no break.

Dynamo decides a graph break during tracing, upstream of the backend, so the
active backend comes from the ``InstructionTranslator``. All hooks return early
unless the traced frame is bound for ``backend="neuron"``.

Three entry points need a patch, because Dynamo routes a free function, a
method, and a re-classified callable differently:

* ``TorchInGraphFunctionVariable.call_function`` -- ``torch.topk(x, k)``. Only
  this hook sees the call arguments.
* ``TensorVariable.call_method`` -- ``x.topk(k)``, which the other hook misses.
* ``trace_rules.lookup`` -- classifies a callable at variable-build time,
  needed for C-extension built-ins that retrace through frame restart.

A break needs ``fullgraph=False`` (the default). ``fullgraph=True`` raises
``Unsupported`` on a target op.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch._dynamo.trace_rules as _trace_rules
import torch._dynamo.variables.tensor as _tensor_variables
import torch._dynamo.variables.torch as _torch_variables
from torch._dynamo.exc import unimplemented
from torch._dynamo.variables.functions import SkipFunctionVariable

from torch_neuronx.neuron_dynamo_backend.graph_break_target_ops import collect_unsupported_ops

logger = logging.getLogger(__name__)

SORT_SIZE_LIMIT = 16384

_NO_LOWERING = "Neuron has no lowering for this op."


@dataclass(frozen=True)
class BlockedOp:
    """Store one graph-break registry entry.

    Args:
        name (str): Tensor method name and name included in the break message.
        reason (str): Explanation included in the graph-break diagnostic.
        supported (Callable, optional): Predicate over resolved call arguments;
            ``None`` makes the operation always break. Defaults to ``None``.

    Note:
        Predicates receive fake tensors and resolved Python constants and must type-check
        every argument because Dynamo cannot always resolve a value.
    """

    name: str
    reason: str
    supported: Callable[..., bool] | None = None


def _above_limit(value: Any, limit: int) -> bool:
    """True only when ``value`` is known to exceed ``limit``.

    ``statically_known_true`` reports False for a ``SymInt`` that Dynamo cannot
    resolve, so a dynamic shape leaves the op in the graph and adds no guard.
    """
    from torch.fx.experimental.symbolic_shapes import statically_known_true

    if isinstance(value, torch.SymInt):
        return bool(statically_known_true(value > limit))
    return isinstance(value, int) and value > limit


def _sort_dim_supported(input: Any, dim: Any = -1, *_args: Any, **_kwargs: Any) -> bool:
    """False when the sorted dimension of ``input`` is too long for Neuron."""
    if not isinstance(input, torch.Tensor) or not isinstance(dim, int):
        return True
    return not _above_limit(input.shape[dim], SORT_SIZE_LIMIT)


def _topk_k_supported(_input: Any, k: Any = None, *_args: Any, **_kwargs: Any) -> bool:
    """False when ``k`` is too large for the Neuron topk kernel.

    The bound is exclusive: ``k == SORT_SIZE_LIMIT`` already fails to compile,
    unlike a sorted dimension of that length.
    """
    return not _above_limit(k, SORT_SIZE_LIMIT - 1)


_SORT_REASON = f"The sorted dimension is longer than the Neuron limit of {SORT_SIZE_LIMIT}."
_TOPK_REASON = f"k is not below the Neuron limit of {SORT_SIZE_LIMIT}."
_GROUPED_MM_REASON = "grouped_mm is not supported by the neuron dynamo backend"


CONDITIONAL_TARGET_OPS: dict[Callable[..., Any], BlockedOp] = {
    torch.sort: BlockedOp("sort", _SORT_REASON, _sort_dim_supported),
    torch.argsort: BlockedOp("argsort", _SORT_REASON, _sort_dim_supported),
    torch.topk: BlockedOp("topk", _TOPK_REASON, _topk_k_supported),
}


UNCONDITIONAL_TARGET_OPS: dict[Callable[..., Any], BlockedOp] = {
    torch.nn.functional.grouped_mm: BlockedOp("grouped_mm", _GROUPED_MM_REASON),
    torch._grouped_mm: BlockedOp("_grouped_mm", _GROUPED_MM_REASON),
}


def _collect_default_targets() -> tuple[frozenset[Any], frozenset[str]]:
    """Collect the full block list.

    A missing or changed dependency must degrade to "no full block list" instead
    of a failed ``import torch_neuronx``.
    """
    try:
        return collect_unsupported_ops()
    except Exception as e:
        logger.warning(
            f"Could not collect the Neuron graph-break block list; an op with no "
            f"Neuron lowering may reach the backend. Error: {e}"
        )
        return frozenset(), frozenset()


DEFAULT_TARGET_OPS, DEFAULT_TARGET_METHODS = _collect_default_targets()


def _build_registry() -> tuple[dict[Any, BlockedOp], dict[str, BlockedOp]]:
    """Merge both block lists into a by-callable and a by-method-name lookup."""
    by_fn: dict[Any, BlockedOp] = {
        fn: BlockedOp(getattr(fn, "__name__", str(fn)), _NO_LOWERING) for fn in DEFAULT_TARGET_OPS
    }
    by_method: dict[str, BlockedOp] = {
        name: BlockedOp(name, _NO_LOWERING) for name in DEFAULT_TARGET_METHODS
    }
    # A conditional entry wins, so a limit is never masked by an unconditional break.
    by_fn.update(CONDITIONAL_TARGET_OPS)
    by_method.update({op.name: op for op in CONDITIONAL_TARGET_OPS.values()})
    by_fn.update(UNCONDITIONAL_TARGET_OPS)
    by_method.update({op.name: op for op in UNCONDITIONAL_TARGET_OPS.values()})
    return by_fn, by_method


_BY_FN, _BY_METHOD = _build_registry()

_detection_warned = False


def _is_neuron_compile(tx: Any = None) -> bool:
    global _detection_warned
    try:
        if tx is None:
            import torch._dynamo.symbolic_convert as _sc

            tx = _sc.InstructionTranslator.current_tx()
        compiler_fn = tx.output.compiler_fn
    except Exception as e:
        if not _detection_warned:
            logger.warning(
                f"Neuron backend detection failed; a blocked op may lower to Neuron. Error: {e}"
            )
            _detection_warned = True
        return False
    return getattr(compiler_fn, "compiler_name", None) == "neuron"


def _resolve(var: Any) -> Any:
    """Resolve a Dynamo ``VariableTracker`` to a fake tensor or to a constant.

    Returns None when neither is available.
    """
    proxy = getattr(var, "proxy", None)
    if proxy is not None:
        example = proxy.node.meta.get("example_value")
        if example is not None:
            return example
    try:
        if var.is_python_constant():
            return var.as_python_constant()
    except Exception:
        pass
    return None


def _should_break(op: BlockedOp, args: list[Any], kwargs: dict[str, Any]) -> bool:
    """True when this call must leave the graph.

    An op with no predicate always breaks. A predicate that cannot judge the
    call reports supported, so an unexpected signature never breaks a graph that
    Neuron can serve.
    """
    if op.supported is None:
        return True
    try:
        return not op.supported(
            *(_resolve(arg) for arg in args),
            **{name: _resolve(value) for name, value in kwargs.items()},
        )
    except Exception as e:
        logger.debug(f"Neuron graph-break check skipped for {op.name}: {e}")
        return False


def _graph_break(op_name: str, reason: str) -> None:
    """Force a Dynamo graph break at the current instruction."""
    unimplemented(
        gb_type="Op not supported by the Neuron backend",
        context=f"{op_name}: {reason}",
        explanation=f"The Neuron backend cannot lower this call to {op_name}. {reason}",
        hints=["This op runs in eager mode. The rest of the function still compiles."],
    )


_patch_applied = False


def _apply_dynamo_graph_break_patch():
    """Install the three Dynamo hooks. A second call does nothing."""
    global _patch_applied
    if _patch_applied:
        return
    _patch_applied = True

    _orig_call_function = _torch_variables.TorchInGraphFunctionVariable.call_function
    _orig_call_method = _tensor_variables.TensorVariable.call_method
    _orig_lookup = _trace_rules.lookup

    def _patched_call_function(self, tx, args, kwargs):
        op = _BY_FN.get(self.value)
        if op is not None and _is_neuron_compile(tx) and _should_break(op, list(args), kwargs):
            _graph_break(f"torch.{op.name}", op.reason)
        return _orig_call_function(self, tx, args, kwargs)

    def _patched_call_method(self, tx, name, args, kwargs):
        op = _BY_METHOD.get(name)
        if op is not None and _is_neuron_compile(tx) and _should_break(op, [self, *args], kwargs):
            _graph_break(f"Tensor.{name}", op.reason)
        return _orig_call_method(self, tx, name, args, kwargs)

    def _patched_lookup(obj):
        op = UNCONDITIONAL_TARGET_OPS.get(obj)
        if op is not None and _is_neuron_compile():
            return SkipFunctionVariable
        return _orig_lookup(obj)

    _torch_variables.TorchInGraphFunctionVariable.call_function = _patched_call_function
    _tensor_variables.TensorVariable.call_method = _patched_call_method
    _trace_rules.lookup = _patched_lookup

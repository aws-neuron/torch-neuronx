"""Automatic torch.compile promotion for eligible Neuron leaf modules.

Module.to, Module.to_empty, and Module.neuron seed bounded discovery.
AutoCompile keeps module hooks and checkpoint owners outside compiled leaves.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import inspect
import os
import sys
import threading
import types
import warnings
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from torch.utils._pytree import tree_leaves
from torch.utils.weak import WeakIdKeyDictionary

_ENABLE_ENV = "TORCH_NEURONX_MODULE_AUTO_COMPILE"
_DEBUG_ENV = "TORCH_NEURONX_MODULE_AUTO_COMPILE_DEBUG"
_AUTO_FORWARD_HOLDER = "_torch_neuronx_auto_forward_holder"
_BACKEND_ATTEMPT_ATTR = "_torch_neuronx_autocompile_backend_attempt"
_REENTRANT_CONTEXT_TOKENS_ATTR = "_torch_neuronx_autocompile_lifecycle_tokens"
_TERMINAL_REASON_COVERAGE_RESTRICTED = "coverage_restricted"
_TERMINAL_REASON_COMPILE_SETUP_FAILED = "compile_setup_failed"
_TERMINAL_REASON_FIRST_COMPILED_CALL_FAILED = "first_compiled_call_failed"
_TERMINAL_REASON_PREEXISTING_COMPILED_CALL = "preexisting_compiled_call"
_TERMINAL_REASON_EXPLICIT_COMPILE = "explicit_compile"
_TERMINAL_REASON_COMPILER_OWNED_GRAPH = "compiler_owned_graph"
_TERMINAL_REASON_COMPILED_ANCESTOR = "compiled_ancestor"
_TERMINAL_REASON_UNSUPPORTED_COMPILE_CONTEXT = "unsupported_compile_context"
_TERMINAL_REASON_COMPILER_CAPABILITY_UNAVAILABLE = "compiler_capability_unavailable"
_TERMINAL_REASON_HOOK_SENSITIVE_ANCESTOR = "hook_sensitive_ancestor"
_TERMINAL_REASON_MODULE_GAINED_CHILDREN = "module_gained_children"
_TERMINAL_REASON_HIGHER_ORDER_AUTOGRAD = "higher_order_autograd"
_TERMINAL_REASON_BACKEND_FALLBACK = "backend_fallback"
_TERMINAL_REASON_OBSERVATION_BUDGET = "observation_budget_exhausted"
# Require enough reuse to amortize compilation. Apply one policy to every leaf.
_PROMOTION_INVOCATION_THRESHOLD = 32
_OBSERVATION_CALL_BUDGET = 64
_MIN_PROFITABLE_FORWARD_GRAPH_NODES = 2
# TransformerEncoderLayer changes fused-fastpath selection when any descendant
# has forward hooks. Keep its descendants eager until bootstrap discovery no
# longer requires hooks. This restriction preserves fused-fastpath behavior.
_HOOK_SENSITIVE_CONTAINER_TYPES = (torch.nn.TransformerEncoderLayer,)

# Terminal states that need no further processing
_TERMINAL_STATES = frozenset()  # populated after class definition


class ModuleAutoCompileState(str, Enum):
    """Enumerate lifecycle states for automatic module compilation."""

    COLD = "cold"
    BOOTSTRAP_PENDING = "bootstrap_pending"
    PROMOTION_READY = "promotion_ready"
    PROMOTING = "promoting"
    PROMOTED_PENDING_FIRST_COMPILED_CALL = "promoted_pending_first_compiled_call"
    COMPILED = "compiled"
    DRAINING_TO_EAGER = "draining_to_eager"
    BLOCKED = "blocked"
    USER_OWNED = "user_owned"


# Now populate _TERMINAL_STATES
_TERMINAL_STATES = frozenset(
    {
        ModuleAutoCompileState.COMPILED,
        ModuleAutoCompileState.DRAINING_TO_EAGER,
        ModuleAutoCompileState.BLOCKED,
        ModuleAutoCompileState.USER_OWNED,
    }
)


@dataclass(slots=True)
class ModuleAutoCompileRecord:
    """Track lifecycle, compilation, and autograd state for one module.

    Args:
        class_name (str): Name of the module class.
        state (ModuleAutoCompileState): Current lifecycle state. Defaults to ``COLD``.
        generation (int): Record invalidation generation. Defaults to ``0``.
        promotion_token (int): Token invalidating stale promotion attempts. Defaults to ``0``.
        discovery_count (int): Number of classification observations. Defaults to ``0``.
        successful_invocation_count (int): Successful top-level bootstrap calls. Defaults
            to ``0``.
        bootstrap_forward_pre_handle (object, optional): Discovery pre-hook handle. Defaults
            to ``None``.
        bootstrap_compiled_call_installed (bool): Whether bootstrap owns the module call slot.
            Defaults to ``False``.
        terminal_reason (str, optional): Recorded terminal transition reason. Defaults to ``None``.
        pending_promotion (bool): Whether promotion is queued. Defaults to ``False``.
        first_compiled_call_in_flight (bool): Whether the first compiled execution is active.
            Defaults to ``False``.
        active_bootstrap_invocations (int): Active discovery calls. Defaults to ``0``.
        pending_backward_count (int): Active backward lifecycle tokens. Defaults to ``0``.
        pending_compiled_backward_count (int): Active compiled backward tokens. Defaults to ``0``.
        pending_reentrant_checkpoint_count (int): Active reentrant-checkpoint tokens. Defaults
            to ``0``.
        unresolved_output_budget (int): Calls remaining while output resolution is pending.
            Defaults to ``64``.
        probation_invocation_count (int): Completed compiled probation calls. Defaults to ``0``.
        defer_first_compiled_call (bool): Whether to defer the first compiled execution. Defaults
            to ``False``.
        requires_backward_executor (bool): Whether observed use requires a backward executor.
            Defaults to ``False``.
        compile_attempt (object, optional): Backend attempt state. Defaults to ``None``.
        pending_terminal_reason (str, optional): Terminal reason deferred until lifecycle tokens
            drain. Defaults to ``None``.
        auto_wrapper_ref (weakref.ReferenceType, optional): Installed wrapper reference. Defaults
            to ``None``.
        active_transition_calls (int): Calls crossing lifecycle transitions. Defaults to ``0``.
        transition_lock (object): Per-record reentrant lock created by a default factory.
    """

    class_name: str
    state: ModuleAutoCompileState = ModuleAutoCompileState.COLD
    generation: int = 0
    promotion_token: int = 0
    discovery_count: int = 0
    successful_invocation_count: int = 0
    bootstrap_forward_pre_handle: object | None = None
    bootstrap_compiled_call_installed: bool = False
    terminal_reason: str | None = None
    pending_promotion: bool = False
    first_compiled_call_in_flight: bool = False
    active_bootstrap_invocations: int = 0
    pending_backward_count: int = 0
    pending_compiled_backward_count: int = 0
    pending_reentrant_checkpoint_count: int = 0
    unresolved_output_budget: int = _OBSERVATION_CALL_BUDGET
    probation_invocation_count: int = 0
    defer_first_compiled_call: bool = False
    requires_backward_executor: bool = False
    compile_attempt: object | None = None
    pending_terminal_reason: str | None = None
    auto_wrapper_ref: weakref.ReferenceType[Callable[..., object]] | None = None
    active_transition_calls: int = 0
    transition_lock: object = field(default_factory=threading.RLock, repr=False)


@dataclass(slots=True)
class _BootstrapInvocation:
    module_ref: weakref.ReferenceType[torch.nn.Module]
    record: ModuleAutoCompileRecord
    generation: int
    top_level: bool
    succeeded: bool = False

    @property
    def module(self) -> torch.nn.Module | None:
        """Return the live bootstrap module when it still exists.

        Returns:
            torch.nn.Module | None: Referenced module, or ``None`` after collection.
        """
        return self.module_ref()


class _AutogradLifecycleToken:
    __slots__ = (
        "active",
        "callback_queued",
        "compiled",
        "counter_attribute",
        "generation",
        "module_ref",
        "record",
        "released",
    )

    def __init__(
        self,
        module: torch.nn.Module,
        record: ModuleAutoCompileRecord,
        generation: int,
        *,
        counter_attribute: str,
        compiled: bool,
    ) -> None:
        self.module_ref = weakref.ref(module)
        self.record = record
        self.generation = generation
        self.counter_attribute = counter_attribute
        self.compiled = compiled
        self.active = False
        self.callback_queued = False
        self.released = False

    def activate(self) -> bool:
        """Activate this token for its current module generation.

        Returns:
            bool: Whether the matching record counter was incremented.
        """
        module = self.module_ref()
        with _STATE_LOCK:
            if (
                module is None
                or self.released
                or _MODULE_REGISTRY.get(module) is not self.record
                or self.record.generation != self.generation
            ):
                return False
            count = getattr(self.record, self.counter_attribute)
            setattr(self.record, self.counter_attribute, count + 1)
            if self.compiled and self.counter_attribute == "pending_backward_count":
                self.record.pending_compiled_backward_count += 1
            self.active = True
            return True

    def release(self) -> None:
        """Release this token and decrement its active lifecycle counters."""
        module = self.module_ref()
        with _STATE_LOCK:
            if self.released:
                return
            self.released = True
            self.callback_queued = False
            if not self.active:
                return
            self.active = False
            if (
                module is None
                or _MODULE_REGISTRY.get(module) is not self.record
                or self.record.generation != self.generation
            ):
                return
            count = getattr(self.record, self.counter_attribute)
            if count > 0:
                setattr(self.record, self.counter_attribute, count - 1)
            if (
                self.compiled
                and self.counter_attribute == "pending_backward_count"
                and self.record.pending_compiled_backward_count > 0
            ):
                self.record.pending_compiled_backward_count -= 1
            _finish_terminal_eager_if_drained_locked(module, self.record)

    def queue_graph_task_completion(self, *, retain_token: bool) -> bool:
        """Queue token handling after the current autograd graph task.

        Args:
            retain_token (bool): Keep the token active after the callback runs.

        Returns:
            bool: Whether a callback is queued or was already queued.
        """
        module = self.module_ref()
        with _STATE_LOCK:
            if self.released:
                return False
            if self.callback_queued:
                return True
            self.callback_queued = True

        def complete_graph_task() -> None:
            """Retain or release the token when the graph task completes."""
            if retain_token:
                with _STATE_LOCK:
                    if not self.released:
                        self.callback_queued = False
                return
            self.release()

        try:
            torch.autograd.Variable._execution_engine.queue_callback(complete_graph_task)
        except Exception:
            with _STATE_LOCK:
                self.callback_queued = False
                if (
                    module is not None
                    and _MODULE_REGISTRY.get(module) is self.record
                    and self.record.generation == self.generation
                ):
                    _request_terminal_eager_locked(
                        module,
                        self.record,
                        _TERMINAL_REASON_COMPILER_CAPABILITY_UNAVAILABLE,
                    )
            return False
        return True

    def _observe_backward(self, *, check_grad_mode: bool) -> bool:
        keep_graph = _current_graph_task_keeps_graph()
        if not self.compiled and ((check_grad_mode and torch.is_grad_enabled()) or keep_graph):
            module = self.module_ref()
            with _STATE_LOCK:
                if (
                    module is not None
                    and _MODULE_REGISTRY.get(module) is self.record
                    and self.record.generation == self.generation
                ):
                    _request_terminal_eager_locked(
                        module,
                        self.record,
                        _TERMINAL_REASON_HIGHER_ORDER_AUTOGRAD,
                    )
        return self.queue_graph_task_completion(retain_token=self.compiled and keep_graph)

    def observe_node_backward(self, _gradients: tuple[torch.Tensor, ...]) -> None:
        """Observe a backward-node prehook and queue token completion.

        Args:
            _gradients (tuple[torch.Tensor, ...]): Gradients accepted but not inspected.
        """
        self._observe_backward(check_grad_mode=True)

    def observe_reentrant_recompute(self) -> bool:
        """Observe reentrant checkpoint recomputation.

        Returns:
            bool: Whether graph-task completion handling is queued.
        """
        # Reentrant recompute enables grad mode before its nested backward.
        # Queue this callback on the outer GraphTask.
        return self._observe_backward(check_grad_mode=False)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.release()


def _discard_auto_forward_holder() -> None:
    return None


def _restore_inherited_bootstrap_call_impl() -> None:
    return None


def _module_call_requires_hook_path(module: torch.nn.Module) -> bool:
    module_namespace = torch.nn.modules.module
    return any(
        (
            getattr(module, "_backward_hooks", None),
            getattr(module, "_backward_pre_hooks", None),
            getattr(module, "_forward_hooks", None),
            getattr(module, "_forward_pre_hooks", None),
            getattr(module_namespace, "_global_backward_pre_hooks", None),
            getattr(module_namespace, "_global_backward_hooks", None),
            getattr(module_namespace, "_global_forward_hooks", None),
            getattr(module_namespace, "_global_forward_pre_hooks", None),
        )
    )


def _call_during_external_capture(
    module: torch.nn.Module,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    if _module_call_requires_hook_path(module):
        return module._call_impl(*args, **kwargs)
    return module.forward(*args, **kwargs)


class _BootstrapCallImpl:
    """Temporary owner of Module._compiled_call_impl during discovery."""

    __slots__ = ("module_ref",)

    def __init__(self, module: torch.nn.Module) -> None:
        self.module_ref = weakref.ref(module)

    def __call__(self, *args: object, **kwargs: object) -> object:
        if torch.compiler.is_compiling():
            owner = self.module_ref()
            if owner is None:
                raise RuntimeError("auto-compile bootstrap module was released")
            return _call_during_external_capture(owner, args, kwargs)

        owner = self.module_ref()
        active_module = _recover_innermost_active_module()
        if active_module is None:
            active_module = owner
        if active_module is None:
            raise RuntimeError("auto-compile bootstrap module was released")
        if active_module is not owner:
            return _run_inherited_bootstrap_call_impl(active_module, args, kwargs)

        context_flags = _runtime_context_flags(inspect_active_module_inputs=True)
        checkpoint_recompute = bool(context_flags & _RUNTIME_CONTEXT_RECOMPUTE)
        checkpoint_forward = bool(
            context_flags & _RUNTIME_CONTEXT_CHECKPOINT and not checkpoint_recompute
        )
        promotion_context_blocked = bool(
            context_flags
            & (
                _RUNTIME_CONTEXT_RECOMPUTE
                | _RUNTIME_CONTEXT_SYMBOLIC_TRACE
                | _RUNTIME_CONTEXT_AUTOGRAD_VALIDATION
                | _RUNTIME_CONTEXT_NON_CONTIGUOUS_ANCESTOR
            )
        ) or (
            _functorch_transform_active()
            or _call_contains_empty_neuron_tensor(args, kwargs)
            or _call_contains_non_contiguous_neuron_tensor(args, kwargs)
        )
        with _STATE_LOCK:
            record = _MODULE_REGISTRY.get(active_module)
        if record is None:
            return _run_inherited_bootstrap_call_impl(active_module, args, kwargs)

        reserved_transition = False
        with record.transition_lock:
            with _STATE_LOCK:
                should_promote = (
                    _MODULE_REGISTRY.get(active_module) is record
                    and record.state == ModuleAutoCompileState.PROMOTION_READY
                    and record.pending_promotion
                    and record.active_transition_calls == 0
                    and record.active_bootstrap_invocations == 0
                    and record.pending_backward_count == 0
                    and record.pending_reentrant_checkpoint_count == 0
                    and not promotion_context_blocked
                )
            if should_promote:
                _promote_module(
                    active_module,
                    record,
                    defer_first_call=not checkpoint_forward,
                )

            with _STATE_LOCK:
                live_record = _MODULE_REGISTRY.get(active_module)
                installed_call = active_module._compiled_call_impl
                if installed_call is not self:
                    delegate = installed_call
                elif live_record is record and record.state in _PRE_PROMOTION_STATES:
                    record.active_transition_calls += 1
                    reserved_transition = True
                    delegate = None
                else:
                    delegate = None

            if not reserved_transition:
                if delegate is not None:
                    return delegate(*args, **kwargs)
                return _call_class_descriptor(active_module, "_call_impl", args, kwargs)

        reentrant_token = None
        observation_active = True
        try:
            with _STATE_LOCK:
                if (
                    (
                        _MODULE_REGISTRY.get(active_module) is record
                        and record.state in _PRE_PROMOTION_STATES
                    )
                    and not checkpoint_recompute
                    and not _consume_call_budget_locked(
                        active_module,
                        record,
                    )
                ):
                    observation_active = False
            if observation_active and _is_reentrant_checkpoint_forward(context_flags):
                reentrant_token = _attach_reentrant_checkpoint_token(
                    active_module,
                    record,
                    record.generation,
                    compiled=False,
                )

            try:
                output = _run_bootstrap_call_impl(
                    active_module,
                    args,
                    kwargs,
                    track_output=not bool(context_flags & _RUNTIME_CONTEXT_REENTRANT_CHECKPOINT),
                )
            except BaseException:
                if reentrant_token is not None:
                    reentrant_token.release()
                raise

            if (
                observation_active
                and checkpoint_recompute
                and bool(context_flags & _RUNTIME_CONTEXT_REENTRANT_CHECKPOINT)
            ):
                _track_reentrant_checkpoint_backward(
                    active_module,
                    record,
                    record.generation,
                    output,
                    compiled=False,
                )
            return output
        finally:
            if reserved_transition:
                with record.transition_lock, _STATE_LOCK:
                    if record.active_transition_calls > 0:
                        record.active_transition_calls -= 1
                    _finish_terminal_eager_if_drained_locked(active_module, record)

    def __reduce__(self):
        # A deserialized module has no process-local registry record, so do not
        # retain temporary compiled-call ownership.
        return _restore_inherited_bootstrap_call_impl, ()


class _AutoForwardHolder:
    """Keeps a compiled forward alive without serializing it with the module."""

    __slots__ = ("module_ref", "original_forward", "wrapper")

    def __init__(
        self,
        module: torch.nn.Module,
        wrapper: Callable[..., object],
        original_forward: Callable[..., object] | None,
    ) -> None:
        self.module_ref = weakref.ref(module)
        self.wrapper = wrapper
        self.original_forward = original_forward

    def __reduce__(self):
        return _discard_auto_forward_holder, ()


def _bind_class_descriptor(
    module: torch.nn.Module,
    attribute: str,
) -> Callable[..., object]:
    descriptor = inspect.getattr_static(type(module), attribute)
    return descriptor.__get__(module, type(module))


def _call_class_descriptor(
    module: torch.nn.Module,
    attribute: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    return _bind_class_descriptor(module, attribute)(*args, **kwargs)


def forward(module: torch.nn.Module, *args: object, **kwargs: object) -> object:
    """Dispatch an FSDP-managed module through its process-local wrapper.

    Args:
        module (torch.nn.Module): Module whose forward path is dispatched.
        *args (object): Positional forward arguments.
        **kwargs (object): Keyword forward arguments.

    Returns:
        object: Result of the compiled wrapper or class ``forward`` method.

    Note:
        The function name makes copied or unpickled bound methods resolve the class
        forward instead of retaining a process-local compiled callable.
    """
    holder = module.__dict__.get(_AUTO_FORWARD_HOLDER)
    holder_is_owned = isinstance(holder, _AutoForwardHolder) and holder.module_ref() is module
    if torch.compiler.is_compiling():
        if holder_is_owned:
            return holder.wrapper(*args, **kwargs)
        return _call_class_descriptor(module, "forward", args, kwargs)

    active_module = _recover_innermost_active_module() or module
    holder = active_module.__dict__.get(_AUTO_FORWARD_HOLDER)
    if not isinstance(holder, _AutoForwardHolder) or holder.module_ref() is not active_module:
        active_module.__dict__.pop("forward", None)
        active_module.__dict__.pop(_AUTO_FORWARD_HOLDER, None)
        return _call_class_descriptor(active_module, "forward", args, kwargs)
    return holder.wrapper(*args, **kwargs)


_MODULE_REGISTRY: WeakIdKeyDictionary = WeakIdKeyDictionary()
_INTERNAL_COMPILE_STATE = threading.local()
_BOOTSTRAP_INVOCATION_STATE = threading.local()
_STATE_LOCK = threading.RLock()
_RESET_GENERATION = 0

_MODULE_CALL_FRAME_CODES = frozenset(
    function.__code__
    for function in (torch.nn.Module._call_impl, torch.nn.Module._wrapped_call_impl)
)

_PRE_PROMOTION_STATES = frozenset(
    {
        ModuleAutoCompileState.BOOTSTRAP_PENDING,
        ModuleAutoCompileState.PROMOTION_READY,
    }
)
_INSTALLATION_COMPLETE = False


def _env_flag(name: str, *, default: bool) -> bool:
    # This gate runs from feature-owned module paths. Invalid values use the
    # supplied default instead of changing module behavior.
    value = (os.environ.get(name) or "").strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


# Hard kill-switch. Set to True (via ``_disable_feature``) when any part of the
# feature fails to wire up against the running torch — e.g. a private torch API
# it depends on moved or changed signature across a version bump. Once set, the
# feature no-ops for the rest of the process regardless of the env var, so a
# torch-compat break degrades to "auto-compile off" instead of crashing every
# importer of torch_neuronx. Checked first in module_auto_compile_enabled(),
# which is the single runtime gate that feature-owned patches and bootstrap
# hooks consult.
_FEATURE_HARD_DISABLED = False


def _auto_wrapper(record: ModuleAutoCompileRecord) -> Callable[..., object] | None:
    if record.auto_wrapper_ref is None:
        return None
    return record.auto_wrapper_ref()


def _clear_auto_wrapper_locked(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
) -> bool:
    wrapper = _auto_wrapper(record)
    if wrapper is not None and module._compiled_call_impl is wrapper:
        module._compiled_call_impl = None
        record.auto_wrapper_ref = None
        return True
    holder = module.__dict__.get(_AUTO_FORWARD_HOLDER)
    if wrapper is not None and module.__dict__.get("forward") is wrapper:
        if isinstance(holder, _AutoForwardHolder) and holder.original_forward is not None:
            module.__dict__["forward"] = holder.original_forward
        else:
            module.__dict__.pop("forward", None)
        if isinstance(holder, _AutoForwardHolder):
            module.__dict__.pop(_AUTO_FORWARD_HOLDER, None)
        record.auto_wrapper_ref = None
        return True
    if isinstance(holder, _AutoForwardHolder):
        if holder.original_forward is not None:
            module.__dict__["forward"] = holder.original_forward
        elif _has_auto_forward_dispatcher(module):
            module.__dict__.pop("forward", None)
        module.__dict__.pop(_AUTO_FORWARD_HOLDER, None)
        record.auto_wrapper_ref = None
        return True
    record.auto_wrapper_ref = None
    return False


def _auto_execution_wrapper_is_installed(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
    execution_wrapper: Callable[..., object],
) -> bool:
    installed_wrapper = _auto_wrapper(record)
    if installed_wrapper is None:
        return False
    if module._compiled_call_impl is installed_wrapper:
        return installed_wrapper is execution_wrapper
    if module.__dict__.get("forward") is not installed_wrapper:
        return False
    holder = module.__dict__.get(_AUTO_FORWARD_HOLDER)
    return (
        isinstance(holder, _AutoForwardHolder)
        and holder.module_ref() is module
        and holder.wrapper is execution_wrapper
    )


def _install_auto_wrapper_locked(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
    execution_wrapper: Callable[..., object],
    *,
    forward_only: bool,
    original_forward: Callable[..., object] | None = None,
) -> None:
    if not forward_only:
        record.auto_wrapper_ref = weakref.ref(execution_wrapper)
        module._compiled_call_impl = execution_wrapper
        return

    dispatcher = types.MethodType(forward, module)
    module.__dict__[_AUTO_FORWARD_HOLDER] = _AutoForwardHolder(
        module,
        execution_wrapper,
        original_forward,
    )
    module.forward = dispatcher
    record.auto_wrapper_ref = weakref.ref(dispatcher)


def _sanitize_module_getstate(
    module: torch.nn.Module,
    state: dict[str, object],
) -> dict[str, object]:
    """Remove feature-owned callables from serialized module state."""
    holder = state.get(_AUTO_FORWARD_HOLDER)
    with _STATE_LOCK:
        record = _MODULE_REGISTRY.get(module)
        wrapper = None if record is None else _auto_wrapper(record)
    owns_forward = wrapper is not None and state.get("forward") is wrapper
    owns_holder = isinstance(holder, _AutoForwardHolder)
    if not owns_forward and not owns_holder:
        return state

    state.pop(_AUTO_FORWARD_HOLDER, None)
    instance_forward = state.get("forward")
    if owns_forward or (
        isinstance(instance_forward, types.MethodType) and instance_forward.__func__ is forward
    ):
        if isinstance(holder, _AutoForwardHolder) and holder.original_forward is not None:
            state["forward"] = holder.original_forward
        else:
            state.pop("forward", None)
    return state


def _clear_compile_attempt_locked(record: ModuleAutoCompileRecord) -> None:
    attempt = record.compile_attempt
    setter = getattr(attempt, "set_failure_callback", None)
    if setter is not None:
        setter(None)
    record.compile_attempt = None


def _invalidate_record_locked(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
) -> None:
    _clear_auto_wrapper_locked(module, record)
    _discard_bootstrap_invocations(module)
    _remove_bootstrap_hooks(module, record)
    record.generation += 1
    record.promotion_token += 1
    record.pending_promotion = False
    record.first_compiled_call_in_flight = False
    record.active_bootstrap_invocations = 0
    record.pending_backward_count = 0
    record.pending_compiled_backward_count = 0
    record.pending_reentrant_checkpoint_count = 0
    record.unresolved_output_budget = _OBSERVATION_CALL_BUDGET
    record.probation_invocation_count = 0
    record.defer_first_compiled_call = False
    record.requires_backward_executor = False
    record.pending_terminal_reason = None
    _clear_compile_attempt_locked(record)


def _reset_state_locked() -> None:
    global _BOOTSTRAP_INVOCATION_STATE, _RESET_GENERATION
    _RESET_GENERATION += 1
    for module, record in list(_MODULE_REGISTRY.items()):
        _invalidate_record_locked(module, record)
    _MODULE_REGISTRY.clear()
    _BOOTSTRAP_INVOCATION_STATE = threading.local()


def _disable_feature(reason: str) -> None:
    """Permanently disable auto-compile for this process (torch-compat guard)."""
    global _FEATURE_HARD_DISABLED
    should_warn = False
    with _STATE_LOCK:
        if not _FEATURE_HARD_DISABLED:
            _FEATURE_HARD_DISABLED = True
            _reset_state_locked()
            should_warn = True
    if should_warn:
        with contextlib.suppress(Exception):
            warnings.warn(
                f"torch_neuronx module auto-compile disabled for this process: {reason}. "
                "Eager execution is unaffected; "
                "set TORCH_NEURONX_MODULE_AUTO_COMPILE=0 to silence.",
                stacklevel=2,
            )


def _reset_feature_disabled() -> None:
    """Clear the sticky kill-switch. Test-only: conftests call it between tests
    so one tripped switch can't force later xdist tests to run feature-off."""
    global _FEATURE_HARD_DISABLED
    with _STATE_LOCK:
        _FEATURE_HARD_DISABLED = False


def _reset_state() -> None:
    with _STATE_LOCK:
        _reset_state_locked()


def _after_fork(*, hard_disable: bool) -> None:
    global _BOOTSTRAP_INVOCATION_STATE, _FEATURE_HARD_DISABLED, _INTERNAL_COMPILE_STATE
    global _MODULE_REGISTRY, _STATE_LOCK
    inherited_hard_disable = _FEATURE_HARD_DISABLED
    _STATE_LOCK = threading.RLock()
    with _STATE_LOCK:
        _reset_state_locked()
        _MODULE_REGISTRY = WeakIdKeyDictionary()
        _FEATURE_HARD_DISABLED = hard_disable or inherited_hard_disable
        _INTERNAL_COMPILE_STATE = threading.local()
        _BOOTSTRAP_INVOCATION_STATE = threading.local()


def _module_auto_compile_requested() -> bool:
    return not _FEATURE_HARD_DISABLED and _env_flag(_ENABLE_ENV, default=False)


def _set_installation_complete(complete: bool) -> None:
    global _INSTALLATION_COMPLETE
    with _STATE_LOCK:
        _INSTALLATION_COMPLETE = complete


def module_auto_compile_enabled() -> bool:
    """Return whether automatic module compilation is installed and requested.

    Returns:
        bool: Whether the feature may observe and promote modules.
    """
    return _INSTALLATION_COMPLETE and _module_auto_compile_requested()


def module_auto_compile_debug_enabled() -> bool:
    """Return whether automatic module compilation debug logging is enabled.

    Returns:
        bool: Parsed value of ``TORCH_NEURONX_MODULE_AUTO_COMPILE_DEBUG``.
    """
    return _env_flag(_DEBUG_ENV, default=False)


def _compile_context_active() -> bool:
    # is_compiling() moved from torch._dynamo to torch.compiler; try both.
    for _getter in (
        getattr(getattr(torch, "compiler", None), "is_compiling", None),
        getattr(getattr(torch, "_dynamo", None), "is_compiling", None),
    ):
        try:
            if _getter is not None and _getter():
                return True
        except Exception:
            continue
    return False


_RUNTIME_CONTEXT_CHECKPOINT = 1 << 0
_RUNTIME_CONTEXT_RECOMPUTE = 1 << 1
_RUNTIME_CONTEXT_SYMBOLIC_TRACE = 1 << 2
_RUNTIME_CONTEXT_AUTOGRAD_VALIDATION = 1 << 3
_RUNTIME_CONTEXT_NON_CONTIGUOUS_ANCESTOR = 1 << 4
_RUNTIME_CONTEXT_SAC_FORWARD = 1 << 5
_RUNTIME_CONTEXT_SAC_RECOMPUTE = 1 << 6
_RUNTIME_CONTEXT_REENTRANT_CHECKPOINT = 1 << 7
_RUNTIME_CONTEXT_NON_REENTRANT_CHECKPOINT = 1 << 8


def _active_sac_forward_policy() -> Callable[..., object] | None:
    try:
        from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

        modes = _get_current_dispatch_mode_stack()
    except Exception:
        return None

    for mode in reversed(modes):
        mode_type = type(mode)
        if (
            mode_type.__module__ == "torch.utils.checkpoint"
            and mode_type.__name__ == "_CachingTorchDispatchMode"
        ):
            policy = getattr(mode, "policy_fn", None)
            return policy if callable(policy) else None
    return None


def _active_sac_policy_signature() -> tuple[object, ...]:
    from torch_neuronx.neuron_dynamo_backend.backend import (
        _autocompile_sac_policy_identity,
    )

    try:
        from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

        modes = _get_current_dispatch_mode_stack()
    except Exception:
        return ()

    signature = []
    for mode in modes:
        mode_type = type(mode)
        if (
            mode_type.__module__ == "torch.utils.checkpoint"
            and mode_type.__name__ == "_CachingTorchDispatchMode"
        ):
            signature.append(_autocompile_sac_policy_identity(getattr(mode, "policy_fn", None)))
    return tuple(signature)


def _autocompile_context_signature(context_flags: int) -> tuple[object, ...]:
    if context_flags & _RUNTIME_CONTEXT_REENTRANT_CHECKPOINT:
        checkpoint_kind = 1
    elif context_flags & _RUNTIME_CONTEXT_NON_REENTRANT_CHECKPOINT:
        checkpoint_kind = 2
    else:
        checkpoint_kind = 0
    if context_flags & _RUNTIME_CONTEXT_SAC_RECOMPUTE:
        from torch_neuronx.neuron_dynamo_backend.backend import (
            _AUTOCOMPILE_SAC_RECOMPUTE_POLICY,
        )

        return (checkpoint_kind, _AUTOCOMPILE_SAC_RECOMPUTE_POLICY)
    return (checkpoint_kind, *_active_sac_policy_signature())


@contextlib.contextmanager
def _sac_policy_compile_context(attempt: object | None, context_flags: int):
    policy = _active_sac_forward_policy()
    signature = _autocompile_context_signature(context_flags)
    setter = getattr(attempt, "set_pending_autocompile_context", None)
    clearer = getattr(attempt, "clear_pending_autocompile_context", None)
    if setter is not None:
        setter(policy, signature)
    try:
        yield signature
    finally:
        if clearer is not None:
            clearer(policy, signature)


@contextlib.contextmanager
def _compiled_invocation_context(attempt: object | None, context_flags: int):
    with (
        _sac_policy_compile_context(attempt, context_flags) as signature,
        contextlib.ExitStack() as stack,
    ):
        invocation_context = getattr(attempt, "autocompile_invocation_context", None)
        if invocation_context is not None:
            stack.enter_context(invocation_context(signature))
        if context_flags & (_RUNTIME_CONTEXT_SAC_FORWARD | _RUNTIME_CONTEXT_SAC_RECOMPUTE):
            from torch_neuronx.neuron_dynamo_backend.backend import (
                _suspend_autocompile_sac_mode,
            )

            stack.enter_context(_suspend_autocompile_sac_mode(True))
        yield


def _runtime_context_flags(*, inspect_active_module_inputs: bool = False) -> int:
    """Collect stack-derived call context in one pass."""
    flags = 0

    try:
        if torch.jit.is_tracing() or torch.jit.is_scripting():
            flags |= _RUNTIME_CONTEXT_SYMBOLIC_TRACE
    except Exception:
        # Tracing detection is best-effort across torch versions and runtimes.
        pass
    try:
        if torch._C._get_tracing_state() is not None:
            flags |= _RUNTIME_CONTEXT_SYMBOLIC_TRACE
    except Exception:
        # This private API may be unavailable; continue with the remaining probes.
        pass

    try:
        from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

        for mode in _get_current_dispatch_mode_stack():
            mode_type = type(mode)
            name = mode_type.__name__
            module_name = mode_type.__module__
            if module_name == "torch.utils.checkpoint":
                if name == "_CachingTorchDispatchMode":
                    flags |= _RUNTIME_CONTEXT_SAC_FORWARD
                    flags |= _RUNTIME_CONTEXT_CHECKPOINT | _RUNTIME_CONTEXT_NON_REENTRANT_CHECKPOINT
                elif name == "_CachedTorchDispatchMode":
                    flags |= _RUNTIME_CONTEXT_SAC_RECOMPUTE | _RUNTIME_CONTEXT_RECOMPUTE
                    flags |= _RUNTIME_CONTEXT_CHECKPOINT | _RUNTIME_CONTEXT_NON_REENTRANT_CHECKPOINT
            if "Proxy" in name or "Fake" in name:
                flags |= _RUNTIME_CONTEXT_SYMBOLIC_TRACE
    except (AttributeError, ImportError):
        pass

    frame = sys._getframe(1)
    checkpoint_suffix = os.path.normpath("torch/utils/checkpoint.py")
    gradcheck_suffix = os.path.normpath("torch/autograd/gradcheck.py")
    jit_path = os.path.normpath("torch/jit/")
    while frame is not None:
        module_name = frame.f_globals.get("__name__")
        filename = os.path.normpath(frame.f_code.co_filename)
        checkpoint_frame = module_name == "torch.utils.checkpoint" or filename.endswith(
            checkpoint_suffix
        )
        if checkpoint_frame:
            qualname = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
            if qualname.startswith("CheckpointFunction."):
                flags |= _RUNTIME_CONTEXT_REENTRANT_CHECKPOINT
                flags |= _RUNTIME_CONTEXT_CHECKPOINT
                if qualname.startswith("CheckpointFunction.backward"):
                    flags |= _RUNTIME_CONTEXT_RECOMPUTE
            elif frame.f_code.co_name == "checkpoint":
                flags |= _RUNTIME_CONTEXT_CHECKPOINT
                use_reentrant = frame.f_locals.get("use_reentrant")
                if use_reentrant is False:
                    flags |= _RUNTIME_CONTEXT_NON_REENTRANT_CHECKPOINT
                elif use_reentrant is True:
                    flags |= _RUNTIME_CONTEXT_REENTRANT_CHECKPOINT
            if frame.f_code.co_name == "recompute_fn":
                flags |= _RUNTIME_CONTEXT_CHECKPOINT | _RUNTIME_CONTEXT_RECOMPUTE
                flags |= _RUNTIME_CONTEXT_NON_REENTRANT_CHECKPOINT

        if module_name == "torch.autograd.gradcheck" or filename.endswith(gradcheck_suffix):
            flags |= _RUNTIME_CONTEXT_AUTOGRAD_VALIDATION

        if (
            isinstance(module_name, str)
            and (module_name == "torch.jit" or module_name.startswith("torch.jit."))
        ) or jit_path in filename:
            flags |= _RUNTIME_CONTEXT_SYMBOLIC_TRACE

        if (
            inspect_active_module_inputs
            and frame.f_code in _MODULE_CALL_FRAME_CODES
            and isinstance(frame.f_locals.get("self"), torch.nn.Module)
            and _call_contains_non_contiguous_neuron_tensor(
                frame.f_locals.get("args", ()),
                frame.f_locals.get("kwargs"),
            )
        ):
            flags |= _RUNTIME_CONTEXT_NON_CONTIGUOUS_ANCESTOR

        frame = frame.f_back
    return flags


def _inside_checkpoint_context() -> bool:
    """Return true during a reentrant checkpoint forward or recompute."""
    return bool(_runtime_context_flags() & _RUNTIME_CONTEXT_REENTRANT_CHECKPOINT)


def _inside_non_reentrant_checkpoint_recompute() -> bool:
    flags = _runtime_context_flags()
    return bool(
        flags & _RUNTIME_CONTEXT_NON_REENTRANT_CHECKPOINT and flags & _RUNTIME_CONTEXT_RECOMPUTE
    )


def _is_reentrant_checkpoint_forward(context_flags: int) -> bool:
    return bool(
        context_flags & _RUNTIME_CONTEXT_REENTRANT_CHECKPOINT
        and not context_flags & _RUNTIME_CONTEXT_RECOMPUTE
        and not torch.is_grad_enabled()
    )


def _active_reentrant_checkpoint_context() -> object | None:
    frame = sys._getframe(1)
    checkpoint_suffix = os.path.normpath("torch/utils/checkpoint.py")
    while frame is not None:
        module_name = frame.f_globals.get("__name__")
        filename = os.path.normpath(frame.f_code.co_filename)
        if module_name == "torch.utils.checkpoint" or filename.endswith(checkpoint_suffix):
            qualname = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
            if qualname.startswith(("CheckpointFunction.forward", "CheckpointFunction.backward")):
                return frame.f_locals.get("ctx")
        frame = frame.f_back
    return None


def _attach_reentrant_checkpoint_token(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
    generation: int,
    *,
    compiled: bool,
) -> _AutogradLifecycleToken | None:
    context = _active_reentrant_checkpoint_context()
    if context is None:
        return None
    with _STATE_LOCK:
        allowed_states = {
            *_PRE_PROMOTION_STATES,
            ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL,
            ModuleAutoCompileState.COMPILED,
            ModuleAutoCompileState.DRAINING_TO_EAGER,
        }
        if (
            _MODULE_REGISTRY.get(module) is not record
            or record.generation != generation
            or record.state not in allowed_states
        ):
            return None
    token = _AutogradLifecycleToken(
        module,
        record,
        generation,
        counter_attribute="pending_reentrant_checkpoint_count",
        compiled=compiled,
    )
    if not token.activate():
        return None
    try:
        tokens = getattr(context, _REENTRANT_CONTEXT_TOKENS_ATTR, None)
        if tokens is None:
            tokens = []
            setattr(context, _REENTRANT_CONTEXT_TOKENS_ATTR, tokens)
        if not isinstance(tokens, list):
            token.release()
            return None
        tokens.append(token)
    except Exception:
        token.release()
        return None
    with _STATE_LOCK:
        if _MODULE_REGISTRY.get(module) is record and record.generation == generation:
            record.requires_backward_executor = True
    return token


def _matching_reentrant_checkpoint_token(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
    generation: int,
) -> _AutogradLifecycleToken | None:
    context = _active_reentrant_checkpoint_context()
    if context is None:
        return None
    tokens = getattr(context, _REENTRANT_CONTEXT_TOKENS_ATTR, None)
    if not isinstance(tokens, list):
        return None
    with _STATE_LOCK:
        return next(
            (
                token
                for token in tokens
                if isinstance(token, _AutogradLifecycleToken)
                and token.module_ref() is module
                and token.record is record
                and token.generation == generation
                and token.active
                and not token.released
                and not token.callback_queued
            ),
            None,
        )


def _inside_sac_forward() -> bool:
    return bool(_runtime_context_flags() & _RUNTIME_CONTEXT_SAC_FORWARD)


def _inside_sac_recompute() -> bool:
    return bool(_runtime_context_flags() & _RUNTIME_CONTEXT_SAC_RECOMPUTE)


def _current_graph_task_keeps_graph() -> bool:
    try:
        from torch_neuronx import _C

        return bool(_C._current_graph_task_keep_graph())
    except Exception:
        return True


def _is_checkpoint_early_stop(exc: BaseException) -> bool:
    exception_type = type(exc)
    return (
        exception_type.__module__ == "torch.utils.checkpoint"
        and exception_type.__name__ == "_StopRecomputationError"
    )


def _inside_autograd_validation_context() -> bool:
    """True while PyTorch gradcheck/gradgradcheck is driving the module call."""
    return bool(_runtime_context_flags() & _RUNTIME_CONTEXT_AUTOGRAD_VALIDATION)


def _functorch_transform_active() -> bool:
    getter = getattr(torch._C, "_are_functorch_transforms_active", None)
    if getter is None:
        return False
    try:
        return bool(getter())
    except Exception:
        return False


def _inside_symbolic_trace() -> bool:
    """True inside make_fx/proxy/fake tensor or JIT tracing."""
    return bool(_runtime_context_flags() & _RUNTIME_CONTEXT_SYMBOLIC_TRACE)


def _observation_context_blocked() -> bool:
    if _internal_module_compile_active() or _compile_context_active():
        return True
    context_flags = _runtime_context_flags()
    return (
        bool(
            context_flags & (_RUNTIME_CONTEXT_SYMBOLIC_TRACE | _RUNTIME_CONTEXT_AUTOGRAD_VALIDATION)
        )
        or _functorch_transform_active()
    )


def _compiler_owns_graph_module(module: torch.nn.Module) -> bool:
    """Whether a GraphModule belongs to a compile, export, or AOT runtime."""
    if not isinstance(module, torch.fx.GraphModule):
        return False

    # Exported callable modules use torch.export subclasses. Inspect the MRO
    # instead of importing private classes whose locations have changed across
    # supported torch versions.
    for base in type(module).__mro__:
        base_module = getattr(base, "__module__", "")
        if base_module.startswith("torch.export."):
            return True
    # Dynamo attaches this reason to the graph passed to a backend.
    if hasattr(module, "compile_subgraph_reason"):
        return True

    # AOTAutograd and custom backends may materialize a plain or lazy
    # GraphModule. Keep it under the enclosing compiler runtime rather than
    # treating it as a new eager leaf. LazyGraphModule alone is not sufficient:
    # torch.fx uses it for ordinary eager-created graphs in some torch versions.
    frame = sys._getframe(1)
    while frame is not None:
        module_name = frame.f_globals.get("__name__")
        if isinstance(module_name, str) and (
            module_name.startswith("torch._dynamo.backends.")
            or module_name == "torch._functorch.aot_autograd"
            or module_name.startswith("torch._functorch._aot_autograd.")
            or module_name.startswith("torch.export.")
        ):
            return True
        frame = frame.f_back
    return False


def _tensor_is_neuron(value: object) -> bool:
    if not isinstance(value, torch.Tensor):
        return False
    return value.device.type == "neuron"


def _value_contains_matching_tensor(
    value: object,
    predicate: Callable[[torch.Tensor], bool],
    seen: set[int] | None = None,
) -> bool:
    if isinstance(value, torch.Tensor):
        return predicate(value)
    if not isinstance(value, list | tuple | dict):
        return False

    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return False
    seen.add(value_id)

    values = value.values() if isinstance(value, dict) else value
    return any(_value_contains_matching_tensor(item, predicate, seen) for item in values)


def _value_contains_neuron_tensor(value: object) -> bool:
    return _value_contains_matching_tensor(value, _tensor_is_neuron)


def _value_contains_empty_neuron_tensor(value: object) -> bool:
    return _value_contains_matching_tensor(
        value,
        lambda tensor: _tensor_is_neuron(tensor) and tensor.numel() == 0,
    )


def _call_contains_empty_neuron_tensor(
    args: object,
    kwargs: object | None,
) -> bool:
    return _value_contains_empty_neuron_tensor(args) or _value_contains_empty_neuron_tensor(
        kwargs or {}
    )


def _value_contains_non_contiguous_neuron_tensor(value: object) -> bool:
    return _value_contains_matching_tensor(
        value,
        lambda tensor: _tensor_is_neuron(tensor) and not tensor.is_contiguous(),
    )


def _call_contains_non_contiguous_neuron_tensor(
    args: object,
    kwargs: object | None,
) -> bool:
    return _value_contains_non_contiguous_neuron_tensor(
        args
    ) or _value_contains_non_contiguous_neuron_tensor(kwargs or {})


def _active_module_call_contains_non_contiguous_neuron_tensor() -> bool:
    """True when any live nn.Module call was entered with non-contiguous Neuron tensors.

    Composite modules can receive non-contiguous public inputs and pass
    contiguous intermediates to their leaves. Those leaves still execute inside
    an invocation whose eager semantics depend on non-contiguous inputs, so keep
    the whole active module stack in eager for that invocation.
    """
    return bool(
        _runtime_context_flags(inspect_active_module_inputs=True)
        & _RUNTIME_CONTEXT_NON_CONTIGUOUS_ANCESTOR
    )


def _recover_active_module_stack() -> tuple[torch.nn.Module, ...]:
    """Active nn.Module instances from the live frames, innermost-first.

    Identify Module call frames by code object. Avoid reading arbitrary
    ``f_locals`` mappings because they can retain deleted local values.
    """
    modules: list[torch.nn.Module] = []
    seen_ids: set[int] = set()
    frame = sys._getframe(0)
    while frame is not None:
        if frame.f_code not in _MODULE_CALL_FRAME_CODES:
            frame = frame.f_back
            continue
        self_obj = frame.f_locals.get("self")
        if isinstance(self_obj, torch.nn.Module):
            module_id = id(self_obj)
            if module_id not in seen_ids:
                seen_ids.add(module_id)
                modules.append(self_obj)
        frame = frame.f_back
    return tuple(modules)


def _recover_innermost_active_module() -> torch.nn.Module | None:
    """Return the first active module without walking the full Python stack."""
    frame = sys._getframe(0)
    while frame is not None:
        if frame.f_code in _MODULE_CALL_FRAME_CODES:
            self_obj = frame.f_locals.get("self")
            if isinstance(self_obj, torch.nn.Module):
                return self_obj
        frame = frame.f_back
    return None


def _debug_log(message: str) -> None:
    if not module_auto_compile_debug_enabled():
        return
    sys.stderr.write(f"[torch-neuronx:module-auto-compile] {message}\n")


@contextlib.contextmanager
def _internal_module_compile_scope():
    depth = getattr(_INTERNAL_COMPILE_STATE, "depth", 0)
    _INTERNAL_COMPILE_STATE.depth = depth + 1
    try:
        yield
    finally:
        _INTERNAL_COMPILE_STATE.depth = depth


def _internal_module_compile_active() -> bool:
    return getattr(_INTERNAL_COMPILE_STATE, "depth", 0) > 0


@contextlib.contextmanager
def _isolated_leaf_compile_scope():
    depth = getattr(_INTERNAL_COMPILE_STATE, "isolated_leaf_depth", 0)
    _INTERNAL_COMPILE_STATE.isolated_leaf_depth = depth + 1
    try:
        yield
    finally:
        _INTERNAL_COMPILE_STATE.isolated_leaf_depth = depth


def _isolated_leaf_compile_active() -> bool:
    return getattr(_INTERNAL_COMPILE_STATE, "isolated_leaf_depth", 0) > 0


def _compile_isolated_callable(
    callable_: Callable[..., object],
):
    try:
        from torch._dynamo import external_utils

        wrapped = external_utils.wrap_inline(callable_)
        # Dynamo caches guards per code object. Give each promoted module an
        # isolated cache instead of exhausting wrap_inline.inner's global limit.
        compile_target = types.FunctionType(
            wrapped.__code__.replace(),
            wrapped.__globals__,
            name=wrapped.__name__,
            argdefs=wrapped.__defaults__,
            closure=wrapped.__closure__,
        )
        compile_target.__kwdefaults__ = wrapped.__kwdefaults__
    except Exception:
        compile_target = callable_

    from torch_neuronx import synchronize as runtime_synchronize
    from torch_neuronx.neuron_dynamo_backend.backend import (
        _AutoCompileBackendAttempt,
        _create_autocompile_backend,
    )

    attempt = _AutoCompileBackendAttempt(
        minimum_forward_graph_node_count=_MIN_PROFITABLE_FORWARD_GRAPH_NODES,
        runtime_validator=runtime_synchronize,
    )
    backend_options = {
        "fallback_execution": False,
        "autocompile_compile_fallback": True,
    }
    neuron_backend = _create_autocompile_backend(attempt)

    @functools.wraps(neuron_backend)
    def autocompile_backend(gm, example_inputs):
        """Invoke the AutoCompile backend with fallback execution disabled.

        Args:
            gm (torch.fx.GraphModule): Captured graph module.
            example_inputs (list[torch.Tensor]): Example graph inputs.

        Returns:
            Callable: Executable returned by the Neuron backend.
        """
        return neuron_backend(gm, example_inputs, options=backend_options)

    def guard_fail(_failure) -> None:
        """Record a forward fallback after a Dynamo guard miss.

        Args:
            _failure (object): Guard-failure details accepted but not inspected.
        """
        attempt.record_fallback("forward", "guard_miss")

    compiled = torch._dynamo.optimize(
        autocompile_backend,
        dynamic=False,
        recompile_limit=1,
        guard_fail_fn=guard_fail,
    )(compile_target)
    setattr(compiled, _BACKEND_ATTEMPT_ATTR, attempt)
    return compiled


def _compile_module_call_impl(module: torch.nn.Module):
    """Compile the module call path using a non-skip Dynamo frame."""
    return _compile_isolated_callable(module._call_impl)


def _compile_module_forward(forward_: Callable[..., object]):
    """Compile only forward so composable FSDP hooks remain eager."""
    return _compile_isolated_callable(forward_)


def _module_is_leaf(module: torch.nn.Module) -> bool:
    return not any(
        isinstance(child, torch.nn.Module) for child in getattr(module, "_modules", {}).values()
    )


def _module_is_compile_region(module: torch.nn.Module) -> bool:
    return _module_is_leaf(module)


def _callable_disables_dynamo(value: object) -> bool:
    pending = [value]
    seen: set[int] = set()
    while pending:
        candidate = pending.pop()
        candidate_id = id(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        if isinstance(candidate, functools.partial):
            pending.append(candidate.func)
            continue
        function = getattr(candidate, "__func__", candidate)
        if bool(getattr(function, "_torchdynamo_disable", False)):
            return True
        code = getattr(function, "__code__", None)
        if code is not None and "_dynamo" in code.co_names and "disable" in code.co_names:
            return True
        wrapped = getattr(function, "__wrapped__", None)
        if wrapped is not None:
            pending.append(wrapped)
    return False


def _module_has_dynamo_incompatible_hooks(module: torch.nn.Module) -> bool:
    if getattr(module, "_backward_pre_hooks", None) or getattr(module, "_backward_hooks", None):
        return True
    auto_hooks = {
        _bootstrap_forward_pre_hook,
    }
    forward_hooks = (
        *getattr(module, "_forward_pre_hooks", {}).values(),
        *getattr(module, "_forward_hooks", {}).values(),
    )
    return any(hook not in auto_hooks and _callable_disables_dynamo(hook) for hook in forward_hooks)


def _is_composable_fsdp_module(module: torch.nn.Module) -> bool:
    return any(
        getattr(base, "__name__", "") == "FSDPModule"
        and getattr(base, "__module__", "").startswith("torch.distributed.fsdp")
        for base in type(module).__mro__
    )


def _has_auto_forward_dispatcher(module: torch.nn.Module) -> bool:
    instance_forward = module.__dict__.get("forward")
    return isinstance(instance_forward, types.MethodType) and instance_forward.__func__ is forward


def _has_serialized_class_forward(module: torch.nn.Module) -> bool:
    instance_forward = module.__dict__.get("forward")
    return (
        isinstance(instance_forward, types.MethodType)
        and instance_forward.__self__ is module
        and instance_forward.__func__ is type(module).forward
    )


def _is_fsdp_managed_module(module: torch.nn.Module) -> bool:
    return bool(getattr(module, "_is_fsdp_managed_module", False))


def _is_distributed_data_parallel_module(module: torch.nn.Module) -> bool:
    for base in type(module).__mro__:
        class_name = getattr(base, "__name__", "")
        class_module = getattr(base, "__module__", "")
        if class_name == "DistributedDataParallel" and class_module.startswith("torch.nn.parallel"):
            return True
        if class_name == "DDP" and class_module == "torch.distributed._composable.replicate":
            return True
    return False


def _active_module_stack_contains_ddp() -> bool:
    return any(
        _is_distributed_data_parallel_module(active) for active in _recover_active_module_stack()
    )


def _module_has_direct_meta_state(module: torch.nn.Module) -> bool:
    direct_state = (
        *getattr(module, "_parameters", {}).values(),
        *getattr(module, "_buffers", {}).values(),
    )
    return any(isinstance(value, torch.Tensor) and value.is_meta for value in direct_state)


def _requires_forward_only_compile(module: torch.nn.Module) -> bool:
    return (
        _is_composable_fsdp_module(module)
        or _is_fsdp_managed_module(module)
        or _active_composable_fsdp_owns_module(module)
    )


def _compiler_can_capture_module(
    module: torch.nn.Module,
    *,
    check_parameter_protocols: bool = True,
) -> bool:
    del check_parameter_protocols
    cumulative_batch_norm_training = (
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        and module.training
        and module.track_running_stats
        and module.momentum is None
    )
    dtensor_embedding = False
    if isinstance(module, torch.nn.Embedding):
        from torch.distributed.tensor import DTensor

        # AOT mutates the sharded embedding mask before fallback can run.
        dtensor_embedding = isinstance(module.weight, DTensor)
    composable_fsdp = _is_composable_fsdp_module(module)
    fsdp_managed = _is_fsdp_managed_module(module)
    forward_only = _requires_forward_only_compile(module)
    return (
        not cumulative_batch_norm_training
        and not dtensor_embedding
        and "_call_impl" not in module.__dict__
        and (
            forward_only
            or (
                not composable_fsdp
                and not fsdp_managed
                and not _module_has_dynamo_incompatible_hooks(module)
            )
        )
    )


def _tensor_leaves(value: object) -> tuple[torch.Tensor, ...]:
    try:
        leaves = tree_leaves(value)
    except Exception:
        return ()
    return tuple(leaf for leaf in leaves if isinstance(leaf, torch.Tensor))


def _shares_parameter_with_registered_module(module: torch.nn.Module) -> bool:
    parameter_ids = {
        id(parameter)
        for parameter in getattr(module, "_parameters", {}).values()
        if parameter is not None
    }
    if not parameter_ids:
        return False
    return any(
        other is not module
        and any(
            parameter is not None and id(parameter) in parameter_ids
            for parameter in getattr(other, "_parameters", {}).values()
        )
        for other in _MODULE_REGISTRY
    )


def _get_composable_fsdp_state(module: torch.nn.Module) -> object | None:
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_state import (
            _get_module_fsdp_state,
        )
    except (AttributeError, ImportError):
        return None
    try:
        return _get_module_fsdp_state(module)
    except Exception:
        return None


def _iter_owning_composable_fsdp_parameter_tensors(
    module: torch.nn.Module,
) -> Iterator[torch.Tensor]:
    for fsdp_parameter in _iter_owning_composable_fsdp_parameters(module):
        for attribute in ("sharded_param", "_sharded_post_forward_param"):
            try:
                parameter = getattr(fsdp_parameter, attribute, None)
            except Exception:
                continue
            if isinstance(parameter, torch.Tensor):
                yield parameter


def _iter_owning_composable_fsdp_parameters(module: torch.nn.Module) -> Iterator[object]:
    for active_module in _recover_active_module_stack():
        state = _get_composable_fsdp_state(active_module)
        for parameter_group in getattr(state, "_fsdp_param_groups", ()):
            for fsdp_parameter in getattr(parameter_group, "fsdp_params", ()):
                module_info = getattr(fsdp_parameter, "_module_info", None)
                owned_modules = (
                    getattr(module_info, "module", None),
                    *getattr(module_info, "shared_modules", ()),
                )
                if not any(owned_module is module for owned_module in owned_modules):
                    continue
                yield fsdp_parameter


def _active_composable_fsdp_owns_module(module: torch.nn.Module) -> bool:
    return next(_iter_owning_composable_fsdp_parameters(module), None) is not None


def _iter_parameter_tensor_chains(module: torch.nn.Module) -> Iterator[torch.Tensor]:
    seen: set[int] = set()
    parameters = (
        *getattr(module, "_parameters", {}).values(),
        *_iter_owning_composable_fsdp_parameter_tensors(module),
    )
    for parameter in parameters:
        if not isinstance(parameter, torch.Tensor):
            continue
        pending = [parameter]
        while pending:
            tensor = pending.pop()
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            yield tensor
            for attribute in ("_base", "_local_tensor"):
                try:
                    child = getattr(tensor, attribute, None)
                except Exception:
                    continue
                if isinstance(child, torch.Tensor):
                    pending.append(child)


def _direct_parameters_are_ready_for_capture(module: torch.nn.Module) -> bool:
    """Whether a first Dynamo capture can safely fake parameter view metadata."""
    for tensor in _iter_parameter_tensor_chains(module):
        try:
            if tensor.is_leaf and tensor.grad is not None:
                return False
        except Exception:
            return False
    return True


def _call_contains_neuron_tensor(
    args: tuple[object, ...],
    kwargs: dict[str, object] | None,
) -> bool:
    return _value_contains_neuron_tensor(args) or _value_contains_neuron_tensor(kwargs or {})


def _get_or_create_record(
    module: torch.nn.Module,
) -> tuple[ModuleAutoCompileRecord, bool]:
    with _STATE_LOCK:
        record = _MODULE_REGISTRY.get(module)
        is_new = record is None
        if record is not None:
            return record, is_new

        record = ModuleAutoCompileRecord(class_name=type(module).__name__)
        _MODULE_REGISTRY[module] = record
        _debug_log(
            f"discover module={type(module).__name__} id={id(module)} state={record.state.value}"
        )
        return record, is_new


def _bootstrap_compiled_call_is_owned(module: torch.nn.Module) -> bool:
    value = module._compiled_call_impl
    return isinstance(value, _BootstrapCallImpl) and value.module_ref() is module


def _remove_bootstrap_hooks(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
    *,
    clear_compiled_call_impl: bool = True,
) -> None:
    handle = record.bootstrap_forward_pre_handle
    if handle is not None:
        with contextlib.suppress(Exception):
            handle.remove()
        record.bootstrap_forward_pre_handle = None

    if (
        clear_compiled_call_impl
        and record.bootstrap_compiled_call_installed
        and _bootstrap_compiled_call_is_owned(module)
    ):
        module._compiled_call_impl = None
        record.bootstrap_compiled_call_installed = False


def _remove_inherited_bootstrap_hooks(module: torch.nn.Module) -> None:
    """Drop AutoCompile state copied without its registry record."""
    with _STATE_LOCK:
        record = _MODULE_REGISTRY.get(module)
        if (
            record is not None
            and record.bootstrap_compiled_call_installed
            and _bootstrap_compiled_call_is_owned(module)
        ):
            return

    compiled_call = module._compiled_call_impl
    hook_specs = (
        (
            "_forward_pre_hooks",
            {_bootstrap_forward_pre_hook},
            ("_forward_pre_hooks_with_kwargs",),
        ),
    )
    for hooks_attribute, owned_hooks, metadata_attributes in hook_specs:
        hooks = getattr(module, hooks_attribute, {})
        owned_hook_ids = [hook_id for hook_id, hook in hooks.items() if hook in owned_hooks]
        if not owned_hook_ids:
            continue
        hooks = copy.copy(hooks)
        setattr(module, hooks_attribute, hooks)
        for hook_id in owned_hook_ids:
            hooks.pop(hook_id, None)
            for metadata_attribute in metadata_attributes:
                metadata = getattr(module, metadata_attribute, None)
                if metadata is None:
                    continue
                metadata = copy.copy(metadata)
                setattr(module, metadata_attribute, metadata)
                discard = getattr(metadata, "discard", None)
                if discard is not None:
                    discard(hook_id)
                else:
                    metadata.pop(hook_id, None)

    if isinstance(compiled_call, _BootstrapCallImpl):
        module._compiled_call_impl = None


def _move_bootstrap_forward_pre_hook_to_end(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
) -> None:
    """Let pre-hooks registered after discovery transform inputs first."""
    hooks = module._forward_pre_hooks
    hook_id = getattr(record.bootstrap_forward_pre_handle, "id", None)
    if hook_id not in hooks:
        return
    move_to_end = getattr(hooks, "move_to_end", None)
    if move_to_end is not None:
        move_to_end(hook_id)
    else:
        hook = hooks.pop(hook_id)
        hooks[hook_id] = hook


def _mark_promotion_ready_locked(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
) -> None:
    record.state = ModuleAutoCompileState.PROMOTION_READY
    record.pending_promotion = True
    record.promotion_token += 1
    _debug_log(f"promotion queued module={record.class_name} id={id(module)}")


def _request_terminal_eager_locked(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
    reason: str,
    *,
    force_drain: bool = False,
) -> bool:
    with _STATE_LOCK:
        if record.pending_terminal_reason is None:
            record.pending_terminal_reason = reason
        record.terminal_reason = record.pending_terminal_reason
        record.pending_promotion = False
        _discard_bootstrap_invocations(module)
        _remove_bootstrap_hooks(module, record)
        if (
            force_drain
            or record.first_compiled_call_in_flight
            or record.pending_backward_count != 0
            or record.pending_reentrant_checkpoint_count != 0
        ):
            record.state = ModuleAutoCompileState.DRAINING_TO_EAGER
            return False
        _mark_blocked(
            module,
            record,
            reason=record.pending_terminal_reason,
            clear_compiled_call_impl=True,
        )
        return True


def _finish_terminal_eager_if_drained_locked(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
) -> bool:
    reason = record.pending_terminal_reason
    if (
        record.state != ModuleAutoCompileState.DRAINING_TO_EAGER
        or reason is None
        or record.first_compiled_call_in_flight
        or record.pending_backward_count != 0
        or record.pending_reentrant_checkpoint_count != 0
    ):
        return False
    _mark_blocked(
        module,
        record,
        reason=reason,
        clear_compiled_call_impl=True,
    )
    return True


def _mark_blocked(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
    *,
    reason: str,
    clear_compiled_call_impl: bool,
) -> None:
    with _STATE_LOCK:
        if (
            record.first_compiled_call_in_flight
            or record.pending_backward_count != 0
            or record.pending_reentrant_checkpoint_count != 0
        ):
            raise RuntimeError(
                "AutoCompile cannot block this module. A lifecycle token is still active."
            )
        if clear_compiled_call_impl:
            _clear_auto_wrapper_locked(module, record)
        _discard_bootstrap_invocations(module)
        _remove_inherited_bootstrap_hooks(module)
        _remove_bootstrap_hooks(module, record)
        record.generation += 1
        record.promotion_token += 1
        record.pending_promotion = False
        record.first_compiled_call_in_flight = False
        record.active_bootstrap_invocations = 0
        record.unresolved_output_budget = _OBSERVATION_CALL_BUDGET
        record.probation_invocation_count = 0
        record.defer_first_compiled_call = False
        record.requires_backward_executor = False
        record.pending_terminal_reason = None
        _clear_compile_attempt_locked(record)
        record.state = ModuleAutoCompileState.BLOCKED
        record.terminal_reason = reason
        _debug_log(f"blocked module={record.class_name} id={id(module)} reason={reason}")


def _mark_user_owned(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
    *,
    reason: str,
) -> None:
    with _STATE_LOCK:
        _clear_auto_wrapper_locked(module, record)
        _discard_bootstrap_invocations(module)
        _remove_inherited_bootstrap_hooks(module)
        _remove_bootstrap_hooks(module, record)
        record.generation += 1
        record.promotion_token += 1
        record.pending_promotion = False
        record.first_compiled_call_in_flight = False
        record.active_bootstrap_invocations = 0
        record.pending_backward_count = 0
        record.pending_compiled_backward_count = 0
        record.pending_reentrant_checkpoint_count = 0
        record.unresolved_output_budget = _OBSERVATION_CALL_BUDGET
        record.probation_invocation_count = 0
        record.defer_first_compiled_call = False
        record.requires_backward_executor = False
        record.pending_terminal_reason = None
        _clear_compile_attempt_locked(record)
        record.state = ModuleAutoCompileState.USER_OWNED
        record.terminal_reason = reason
        _debug_log(f"user owned module={record.class_name} id={id(module)} reason={reason}")


def _mark_module_tree_user_owned(module: torch.nn.Module, *, reason: str) -> None:
    with _STATE_LOCK:
        for owned in _iter_module_tree(module):
            record, _ = _get_or_create_record(owned)
            _mark_user_owned(owned, record, reason=reason)


def record_explicit_module_compile(module: torch.nn.Module) -> None:
    """Mark an explicitly compiled module tree as user owned.

    Args:
        module (torch.nn.Module): Root module compiled outside AutoCompile.

    Note:
        This is a no-op when AutoCompile is hard-disabled or the compile originated in
        AutoCompile's internal compilation scope.
    """
    if _FEATURE_HARD_DISABLED:
        return
    if _internal_module_compile_active():
        return
    with _STATE_LOCK:
        record, _ = _get_or_create_record(module)
        if record.discovery_count == 0 and record.state == ModuleAutoCompileState.COLD:
            reason = _TERMINAL_REASON_PREEXISTING_COMPILED_CALL
        else:
            reason = _TERMINAL_REASON_EXPLICIT_COMPILE
        _mark_module_tree_user_owned(module, reason=reason)


def _compiled_execution_eligible(
    module: torch.nn.Module,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    context_flags: int | None = None,
    check_parameter_protocols: bool = True,
) -> bool:
    if context_flags is None:
        context_flags = _runtime_context_flags(inspect_active_module_inputs=True)
    blocked_context = bool(
        context_flags
        & (
            _RUNTIME_CONTEXT_SYMBOLIC_TRACE
            | _RUNTIME_CONTEXT_AUTOGRAD_VALIDATION
            | _RUNTIME_CONTEXT_NON_CONTIGUOUS_ANCESTOR
        )
    )
    return (
        module_auto_compile_enabled()
        and not blocked_context
        and not _functorch_transform_active()
        and not _module_has_direct_meta_state(module)
        and _call_contains_neuron_tensor(args, kwargs)
        and _module_is_compile_region(module)
        and _compiler_can_capture_module(
            module,
            check_parameter_protocols=check_parameter_protocols,
        )
        and not _call_contains_empty_neuron_tensor(args, kwargs)
        and not _call_contains_non_contiguous_neuron_tensor(args, kwargs)
    )


def _attempt_has_profitable_forward(attempt: object | None) -> bool:
    if attempt is None:
        return False
    check = getattr(attempt, "has_profitable_forward", None)
    if check is not None:
        return bool(check())
    return bool(
        not attempt.failed()
        and attempt.forward_executor_count == 1
        and attempt.forward_graph_node_count >= _MIN_PROFITABLE_FORWARD_GRAPH_NODES
    )


def _consume_call_budget_locked(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
) -> bool:
    record.unresolved_output_budget -= 1
    if record.unresolved_output_budget > 0:
        return True
    _request_terminal_eager_locked(
        module,
        record,
        _TERMINAL_REASON_OBSERVATION_BUDGET,
    )
    return False


def _mark_compiled(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
    generation: int,
    execution_wrapper: Callable[..., object],
    *,
    require_backward: bool,
) -> bool:
    with _STATE_LOCK:
        if (
            _MODULE_REGISTRY.get(module) is not record
            or record.generation != generation
            or record.state != ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
            or record.pending_backward_count != 0
            or record.pending_reentrant_checkpoint_count != 0
            or record.pending_terminal_reason is not None
        ):
            return False
        attempt = record.compile_attempt
        seal = getattr(attempt, "seal", None)
        if seal is not None:
            seal_result = seal()
            if seal_result is False:
                _request_terminal_eager_locked(
                    module,
                    record,
                    _TERMINAL_REASON_BACKEND_FALLBACK,
                )
                return False
        if not _attempt_has_profitable_forward(attempt) or (
            require_backward and not attempt.has_real_backward_executor()
        ):
            _request_terminal_eager_locked(
                module,
                record,
                _TERMINAL_REASON_BACKEND_FALLBACK,
            )
            return False
        if not _auto_execution_wrapper_is_installed(
            module,
            record,
            execution_wrapper,
        ):
            _request_terminal_eager_locked(
                module,
                record,
                _TERMINAL_REASON_COMPILER_CAPABILITY_UNAVAILABLE,
            )
            return False
        record.state = ModuleAutoCompileState.COMPILED
        record.terminal_reason = None
        record.pending_terminal_reason = None
        _debug_log(f"compiled module={record.class_name} id={id(module)}")
        return True


def _track_output_backward(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
    generation: int,
    output: object,
    *,
    compiled: bool,
) -> bool:
    if not torch.is_grad_enabled():
        return False

    get_root = getattr(torch.autograd.graph, "_get_grad_fn_or_grad_acc", None)
    if get_root is None:
        with _STATE_LOCK:
            if _MODULE_REGISTRY.get(module) is record and record.generation == generation:
                _request_terminal_eager_locked(
                    module,
                    record,
                    _TERMINAL_REASON_COMPILER_CAPABILITY_UNAVAILABLE,
                )
        return False

    roots: dict[int, torch.autograd.graph.Node] = {}
    for tensor in _tensor_leaves(output):
        if not tensor.requires_grad:
            continue
        try:
            root = get_root(tensor)
        except Exception:
            with _STATE_LOCK:
                if _MODULE_REGISTRY.get(module) is record and record.generation == generation:
                    _request_terminal_eager_locked(
                        module,
                        record,
                        _TERMINAL_REASON_COMPILER_CAPABILITY_UNAVAILABLE,
                    )
            return False
        if root is not None:
            roots.setdefault(id(root), root)
    if not roots:
        return False

    with _STATE_LOCK:
        allowed_states = {
            *_PRE_PROMOTION_STATES,
            ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL,
            ModuleAutoCompileState.COMPILED,
            ModuleAutoCompileState.DRAINING_TO_EAGER,
        }
        if (
            _MODULE_REGISTRY.get(module) is not record
            or record.generation != generation
            or record.state not in allowed_states
        ):
            return False
        record.requires_backward_executor = True

    registered = False
    for root in roots.values():
        token = _AutogradLifecycleToken(
            module,
            record,
            generation,
            counter_attribute="pending_backward_count",
            compiled=compiled,
        )
        if not token.activate():
            continue
        try:
            root.register_prehook(token.observe_node_backward)
        except Exception:
            token.release()
            with _STATE_LOCK:
                if _MODULE_REGISTRY.get(module) is record and record.generation == generation:
                    _request_terminal_eager_locked(
                        module,
                        record,
                        _TERMINAL_REASON_COMPILER_CAPABILITY_UNAVAILABLE,
                    )
            continue
        registered = True
    return registered


def _code_contains_text(code: types.CodeType, text: str) -> bool:
    for constant in code.co_consts:
        if isinstance(constant, str) and text in constant:
            return True
        if isinstance(constant, types.CodeType) and _code_contains_text(constant, text):
            return True
    return False


def _autocompile_double_backward(
    ctx: object,
    impl_fn: Callable[[object], object],
    _all_args: list[object],
) -> object:
    # AutoCompile's backward executor uses the eager FX graph when grad mode is
    # enabled, so its result remains differentiable for higher-order autograd.
    return impl_fn(ctx)


def _enable_higher_order_compatibility(output: object) -> bool:
    """Enable differentiable backward only on AOT classes owned by this output."""
    unsupported_message = "does not currently support double backward"
    patched_marker = "_torch_neuronx_autocompile_double_backward"
    pending = [
        tensor.grad_fn
        for tensor in _tensor_leaves(output)
        if tensor.requires_grad and tensor.grad_fn is not None
    ]
    visited: set[int] = set()
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        forward_cls = getattr(node, "_forward_cls", None)
        double_backward = getattr(forward_cls, "_double_backward", None)
        if isinstance(forward_cls, type) and callable(double_backward):
            if getattr(forward_cls, patched_marker, False):
                continue
            code = getattr(double_backward, "__code__", None)
            if not isinstance(code, types.CodeType):
                return False
            if _code_contains_text(code, unsupported_message):
                try:
                    forward_cls._double_backward = staticmethod(_autocompile_double_backward)
                    setattr(forward_cls, patched_marker, True)
                except Exception:
                    return False
        for next_node, _output_index in getattr(node, "next_functions", ()):
            if next_node is not None:
                pending.append(next_node)
    return True


def _track_reentrant_checkpoint_backward(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
    generation: int,
    output: object,
    *,
    compiled: bool,
) -> bool:
    del output
    if not torch.is_grad_enabled():
        return False
    token = _matching_reentrant_checkpoint_token(module, record, generation)
    if token is None or token.compiled != compiled:
        return False
    return token.observe_reentrant_recompute()


def _make_compiled_wrapper(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
    raw_compiled_callable: Callable[..., object],
    *,
    original_forward: Callable[..., object] | None = None,
) -> Callable[..., object]:
    module_ref = weakref.ref(module)

    def is_installed(live_module: torch.nn.Module) -> bool:
        """Check whether this compiled wrapper remains installed.

        Args:
            live_module (torch.nn.Module): Module whose wrapper ownership is checked.

        Returns:
            bool: Whether ``wrapped_compiled_call`` still owns execution.
        """
        return _auto_execution_wrapper_is_installed(
            live_module,
            record,
            wrapped_compiled_call,
        )

    def run_eager(
        live_module: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        *,
        track_output: bool,
        context_flags: int | None = None,
    ) -> object:
        """Run the eager module path and optionally track output autograd.

        Args:
            live_module (torch.nn.Module): Module to execute.
            args (tuple[object, ...]): Positional module arguments.
            kwargs (dict[str, object]): Keyword module arguments.
            track_output (bool): Whether to attach backward lifecycle tracking.
            context_flags (int, optional): Precomputed runtime flags. Defaults to ``None``.

        Returns:
            object: Eager module output.

        Raises:
            BaseException: Re-raises an exception from the eager module call.
        """
        if context_flags is None:
            context_flags = _runtime_context_flags(inspect_active_module_inputs=True)
        generation = record.generation
        reentrant = bool(context_flags & _RUNTIME_CONTEXT_REENTRANT_CHECKPOINT)
        checkpoint_recompute = bool(context_flags & _RUNTIME_CONTEXT_RECOMPUTE)
        token = None
        if reentrant:
            if checkpoint_recompute:
                token = _matching_reentrant_checkpoint_token(
                    live_module,
                    record,
                    generation,
                )
            else:
                token = _attach_reentrant_checkpoint_token(
                    live_module,
                    record,
                    generation,
                    compiled=False,
                )
        try:
            output = (
                original_forward(*args, **kwargs)
                if original_forward is not None
                else live_module._call_impl(*args, **kwargs)
            )
        except BaseException:
            if token is not None and not checkpoint_recompute:
                token.release()
            raise
        if token is not None and checkpoint_recompute:
            token.observe_reentrant_recompute()
        elif track_output and not reentrant:
            _track_output_backward(
                live_module,
                record,
                generation,
                output,
                compiled=False,
            )
        return output

    @contextlib.contextmanager
    def isolated_leaf_compile_context():
        """Isolate leaf compilation only while a DDP call is active.

        Yields:
            None: Control within the conditional isolated-leaf scope.
        """
        if not _active_module_stack_contains_ddp():
            yield
            return
        with _isolated_leaf_compile_scope():
            yield

    def run_compiled(
        args: tuple[object, ...],
        kwargs: dict[str, object],
        *,
        context_flags: int,
    ) -> object:
        """Run the compiled callable within feature-owned scopes.

        Args:
            args (tuple[object, ...]): Positional module arguments.
            kwargs (dict[str, object]): Keyword module arguments.
            context_flags (int): Runtime-context bit mask.

        Returns:
            object: Compiled callable output.
        """
        with (
            _internal_module_compile_scope(),
            isolated_leaf_compile_context(),
            _compiled_invocation_context(record.compile_attempt, context_flags),
        ):
            return raw_compiled_callable(*args, **kwargs)

    def run_terminal_compiled(
        live_module: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        """Run a terminally compiled module with autograd lifecycle tracking.

        Args:
            live_module (torch.nn.Module): Compiled module to execute.
            args (tuple[object, ...]): Positional module arguments.
            kwargs (dict[str, object]): Keyword module arguments.

        Returns:
            object: Compiled output, or eager output when reentrant checkpoint tracking
                cannot be attached.

        Raises:
            BaseException: Re-raises an exception from the selected module call.
        """
        context_flags = _runtime_context_flags(inspect_active_module_inputs=True)
        generation = record.generation
        checkpoint_recompute = bool(context_flags & _RUNTIME_CONTEXT_RECOMPUTE)
        reentrant_checkpoint = bool(context_flags & _RUNTIME_CONTEXT_REENTRANT_CHECKPOINT)
        token = None
        if reentrant_checkpoint:
            if checkpoint_recompute:
                token = _matching_reentrant_checkpoint_token(
                    live_module,
                    record,
                    generation,
                )
            else:
                token = _attach_reentrant_checkpoint_token(
                    live_module,
                    record,
                    generation,
                    compiled=True,
                )
                if token is None:
                    with _STATE_LOCK:
                        if (
                            _MODULE_REGISTRY.get(live_module) is record
                            and record.generation == generation
                        ):
                            _request_terminal_eager_locked(
                                live_module,
                                record,
                                _TERMINAL_REASON_COMPILER_CAPABILITY_UNAVAILABLE,
                            )
                    return run_eager(
                        live_module,
                        args,
                        kwargs,
                        track_output=False,
                        context_flags=context_flags,
                    )
        try:
            with _compiled_invocation_context(record.compile_attempt, context_flags):
                output = raw_compiled_callable(*args, **kwargs)
        except BaseException:
            if token is not None and not checkpoint_recompute:
                token.release()
            with _STATE_LOCK:
                if _MODULE_REGISTRY.get(live_module) is record:
                    _finish_terminal_eager_if_drained_locked(live_module, record)
            raise

        if token is not None and checkpoint_recompute:
            token.observe_reentrant_recompute()
        elif not checkpoint_recompute and not reentrant_checkpoint:
            _track_output_backward(
                live_module,
                record,
                generation,
                output,
                compiled=True,
            )
        with _STATE_LOCK:
            if _MODULE_REGISTRY.get(live_module) is record:
                _finish_terminal_eager_if_drained_locked(live_module, record)
        return output

    def run_transition_call(*args: object, **kwargs: object) -> object:
        """Dispatch a call while promotion or eager draining is in progress.

        Args:
            *args (object): Positional module arguments.
            **kwargs (object): Keyword module arguments.

        Returns:
            object: Eager or compiled output selected by lifecycle state.

        Raises:
            RuntimeError: If the auto-compiled module has been released.
        """
        live_module = module_ref()
        if live_module is None:
            raise RuntimeError("auto-compiled module was released")
        with _STATE_LOCK:
            pending_terminal_reason = record.pending_terminal_reason
            attempt = record.compile_attempt
            generation = record.generation
            has_compiled_backward_obligation = record.pending_compiled_backward_count != 0
            has_live_obligation = (
                record.first_compiled_call_in_flight
                or record.pending_backward_count != 0
                or record.pending_reentrant_checkpoint_count != 0
            )
        if pending_terminal_reason is not None:
            if not has_live_obligation:
                wait_for_drain = getattr(attempt, "wait_for_executor_drain", None)
                if wait_for_drain is not None:
                    wait_for_drain()
                with _STATE_LOCK:
                    if (
                        _MODULE_REGISTRY.get(live_module) is record
                        and record.pending_terminal_reason == pending_terminal_reason
                    ):
                        _finish_terminal_eager_if_drained_locked(live_module, record)
                return run_eager(
                    live_module,
                    args,
                    kwargs,
                    track_output=False,
                    context_flags=0,
                )

            context_flags = _runtime_context_flags(inspect_active_module_inputs=True)
            checkpoint_recompute = bool(context_flags & _RUNTIME_CONTEXT_RECOMPUTE)
            reentrant_recompute = bool(
                checkpoint_recompute and context_flags & _RUNTIME_CONTEXT_REENTRANT_CHECKPOINT
            )
            token = (
                _matching_reentrant_checkpoint_token(
                    live_module,
                    record,
                    generation,
                )
                if reentrant_recompute
                else None
            )
            if (token is not None and token.compiled) or (
                checkpoint_recompute and has_compiled_backward_obligation
            ):
                output = run_compiled(
                    args,
                    kwargs,
                    context_flags=context_flags,
                )
                if token is not None:
                    token.observe_reentrant_recompute()
                return output
            return run_eager(
                live_module,
                args,
                kwargs,
                track_output=False,
                context_flags=context_flags,
            )

        context_flags = _runtime_context_flags(inspect_active_module_inputs=True)
        checkpoint_recompute = bool(context_flags & _RUNTIME_CONTEXT_RECOMPUTE)
        reentrant_recompute = bool(
            checkpoint_recompute and context_flags & _RUNTIME_CONTEXT_REENTRANT_CHECKPOINT
        )
        with _STATE_LOCK:
            first_capture_pending = (
                is_installed(live_module)
                and record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
            )
            defer_first_compiled_call = first_capture_pending and record.defer_first_compiled_call
            terminalize_training = (
                first_capture_pending
                and record.probation_invocation_count > 0
                and record.pending_backward_count == 0
                and record.pending_reentrant_checkpoint_count == 0
                and record.requires_backward_executor
                and record.compile_attempt is not None
                and record.compile_attempt.has_real_backward_executor()
            )
            generation = record.generation
            eager_backward_pending = (
                first_capture_pending
                and record.probation_invocation_count == 0
                and (
                    record.pending_backward_count != 0
                    or record.pending_reentrant_checkpoint_count != 0
                )
            )
        if defer_first_compiled_call:
            with _STATE_LOCK:
                if is_installed(live_module):
                    record.defer_first_compiled_call = False
                    keep_tracking = _consume_call_budget_locked(
                        live_module,
                        record,
                    )
                else:
                    keep_tracking = False
            return run_eager(
                live_module,
                args,
                kwargs,
                track_output=keep_tracking,
            )
        if terminalize_training:
            if _mark_compiled(
                live_module,
                record,
                generation,
                wrapped_compiled_call,
                require_backward=True,
            ):
                return run_terminal_compiled(live_module, args, kwargs)
            return run_eager(live_module, args, kwargs, track_output=False)
        if eager_backward_pending:
            keep_tracking = False
            if not checkpoint_recompute:
                with _STATE_LOCK:
                    keep_tracking = is_installed(live_module) and _consume_call_budget_locked(
                        live_module,
                        record,
                    )
            return run_eager(
                live_module,
                args,
                kwargs,
                track_output=keep_tracking,
            )
        if first_capture_pending and not checkpoint_recompute:
            with _STATE_LOCK:
                if is_installed(live_module) and not _consume_call_budget_locked(
                    live_module,
                    record,
                ):
                    first_capture_pending = False
            if not first_capture_pending:
                return run_eager(live_module, args, kwargs, track_output=False)
        if first_capture_pending and not _direct_parameters_are_ready_for_capture(live_module):
            return run_eager(live_module, args, kwargs, track_output=True)
        if original_forward is None and _requires_forward_only_compile(live_module):
            with _STATE_LOCK:
                if is_installed(live_module):
                    _request_terminal_eager_locked(
                        live_module,
                        record,
                        _TERMINAL_REASON_COMPILER_CAPABILITY_UNAVAILABLE,
                    )
            return live_module._call_impl(*args, **kwargs)

        checkpoint_forward = (
            bool(context_flags & (_RUNTIME_CONTEXT_CHECKPOINT | _RUNTIME_CONTEXT_SAC_FORWARD))
            and not checkpoint_recompute
        )
        reentrant_checkpoint_forward = _is_reentrant_checkpoint_forward(context_flags)
        if checkpoint_forward:
            with _STATE_LOCK:
                if is_installed(live_module):
                    record.requires_backward_executor = True
        compiler_owned_graph = _compiler_owns_graph_module(live_module)
        eligible = _compiled_execution_eligible(
            live_module,
            args,
            kwargs,
            context_flags=context_flags,
            check_parameter_protocols=first_capture_pending,
        )
        if compiler_owned_graph or not eligible:
            if compiler_owned_graph:
                with _STATE_LOCK:
                    if is_installed(live_module):
                        _mark_user_owned(
                            live_module,
                            record,
                            reason=_TERMINAL_REASON_COMPILER_OWNED_GRAPH,
                        )
            elif not _module_is_compile_region(live_module):
                with _STATE_LOCK:
                    if is_installed(live_module):
                        _request_terminal_eager_locked(
                            live_module,
                            record,
                            _TERMINAL_REASON_MODULE_GAINED_CHILDREN,
                        )
            elif not _compiler_can_capture_module(live_module):
                with _STATE_LOCK:
                    if is_installed(live_module):
                        _request_terminal_eager_locked(
                            live_module,
                            record,
                            _TERMINAL_REASON_COMPILER_CAPABILITY_UNAVAILABLE,
                        )
            else:
                with _STATE_LOCK:
                    if is_installed(live_module):
                        _request_terminal_eager_locked(
                            live_module,
                            record,
                            _TERMINAL_REASON_UNSUPPORTED_COMPILE_CONTEXT,
                        )
            return run_eager(live_module, args, kwargs, track_output=False)

        run_eager_after_unlock = False
        with _STATE_LOCK:
            if (
                not is_installed(live_module)
                or record.state != ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
            ):
                run_eager_after_unlock = True
            elif record.first_compiled_call_in_flight:
                _request_terminal_eager_locked(
                    live_module,
                    record,
                    _TERMINAL_REASON_COMPILER_CAPABILITY_UNAVAILABLE,
                )
                run_eager_after_unlock = True
            else:
                record.first_compiled_call_in_flight = True
        if run_eager_after_unlock:
            return run_eager(
                live_module,
                args,
                kwargs,
                track_output=False,
                context_flags=context_flags,
            )

        reentrant_token = None
        if reentrant_checkpoint_forward:
            reentrant_token = _attach_reentrant_checkpoint_token(
                live_module,
                record,
                generation,
                compiled=True,
            )
            if reentrant_token is None:
                with _STATE_LOCK:
                    if is_installed(live_module):
                        _request_terminal_eager_locked(
                            live_module,
                            record,
                            _TERMINAL_REASON_COMPILER_CAPABILITY_UNAVAILABLE,
                        )
                        record.first_compiled_call_in_flight = False
                        _finish_terminal_eager_if_drained_locked(live_module, record)
                return run_eager(
                    live_module,
                    args,
                    kwargs,
                    track_output=False,
                    context_flags=context_flags,
                )

        try:
            output = run_compiled(
                args,
                kwargs,
                context_flags=context_flags,
            )
        except BaseException as exc:
            if reentrant_token is not None:
                reentrant_token.release()
            with _STATE_LOCK:
                if is_installed(live_module):
                    if not _is_checkpoint_early_stop(exc):
                        _request_terminal_eager_locked(
                            live_module,
                            record,
                            _TERMINAL_REASON_FIRST_COMPILED_CALL_FAILED,
                        )
                    record.first_compiled_call_in_flight = False
                    _finish_terminal_eager_if_drained_locked(live_module, record)
            raise

        if not _enable_higher_order_compatibility(output):
            with _STATE_LOCK:
                if is_installed(live_module):
                    _request_terminal_eager_locked(
                        live_module,
                        record,
                        _TERMINAL_REASON_COMPILER_CAPABILITY_UNAVAILABLE,
                    )

        backend_failed = False
        with _STATE_LOCK:
            if is_installed(live_module):
                generation = record.generation
                record.probation_invocation_count += 1
                attempt = record.compile_attempt
                backend_failed = (
                    attempt is None or attempt.failed() or not attempt.has_real_forward_executor()
                )
                if backend_failed:
                    _request_terminal_eager_locked(
                        live_module,
                        record,
                        _TERMINAL_REASON_BACKEND_FALLBACK,
                    )

        if reentrant_token is not None:
            tracked = True
        elif checkpoint_recompute and bool(context_flags & _RUNTIME_CONTEXT_REENTRANT_CHECKPOINT):
            tracked = _track_reentrant_checkpoint_backward(
                live_module,
                record,
                generation,
                output,
                compiled=True,
            )
        elif checkpoint_recompute:
            tracked = True
        else:
            tracked = _track_output_backward(
                live_module,
                record,
                generation,
                output,
                compiled=True,
            )

        with _STATE_LOCK:
            if _MODULE_REGISTRY.get(live_module) is record and record.generation == generation:
                record.first_compiled_call_in_flight = False
                _finish_terminal_eager_if_drained_locked(live_module, record)
                requires_backward = record.requires_backward_executor
            else:
                requires_backward = True

        if not backend_failed and not tracked and not requires_backward:
            _mark_compiled(
                live_module,
                record,
                generation,
                wrapped_compiled_call,
                require_backward=False,
            )
        return output

    def wrapped_compiled_call(*args: object, **kwargs: object) -> object:
        """Dispatch an installed wrapper across capture and runtime states.

        Args:
            *args (object): Positional module arguments.
            **kwargs (object): Keyword module arguments.

        Returns:
            object: Captured, terminal, or transitional module output.

        Raises:
            RuntimeError: If the auto-compiled module has been released.
        """
        if torch.compiler.is_compiling():
            live_module = module_ref()
            if live_module is None:
                raise RuntimeError("auto-compiled module was released")
            if original_forward is not None:
                return original_forward(*args, **kwargs)
            return _call_during_external_capture(live_module, args, kwargs)

        with record.transition_lock:
            live_module = module_ref()
            if live_module is None:
                raise RuntimeError("auto-compiled module was released")
            with _STATE_LOCK:
                terminal_compiled = (
                    _MODULE_REGISTRY.get(live_module) is record
                    and record.state == ModuleAutoCompileState.COMPILED
                )
            if terminal_compiled:
                return run_terminal_compiled(live_module, args, kwargs)
            return run_transition_call(*args, **kwargs)

    return wrapped_compiled_call


def _promote_module(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
    *,
    defer_first_call: bool = False,
) -> None:
    with record.transition_lock:
        _promote_module_under_transition_lock(
            module,
            record,
            defer_first_call=defer_first_call,
        )


def _promote_module_under_transition_lock(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
    *,
    defer_first_call: bool,
) -> None:
    with _STATE_LOCK:
        if (
            _MODULE_REGISTRY.get(module) is not record
            or record.state != ModuleAutoCompileState.PROMOTION_READY
            or not record.pending_promotion
            or record.active_transition_calls != 0
            or record.active_bootstrap_invocations != 0
            or record.pending_backward_count != 0
            or record.pending_reentrant_checkpoint_count != 0
            or not module_auto_compile_enabled()
        ):
            return
        if _compiler_owns_graph_module(module):
            _mark_user_owned(
                module,
                record,
                reason=_TERMINAL_REASON_COMPILER_OWNED_GRAPH,
            )
            return
        if _module_has_direct_meta_state(module):
            return
        if not _compiler_can_capture_module(module):
            _request_terminal_eager_locked(
                module,
                record,
                _TERMINAL_REASON_COMPILER_CAPABILITY_UNAVAILABLE,
            )
            return
        if _shares_parameter_with_registered_module(module):
            _request_terminal_eager_locked(
                module,
                record,
                _TERMINAL_REASON_COMPILER_CAPABILITY_UNAVAILABLE,
            )
            return
        forward_only = _requires_forward_only_compile(module)
        current_forward = module.forward if forward_only else None
        original_instance_forward = (
            current_forward
            if current_forward is not None
            and "forward" in module.__dict__
            and not _has_auto_forward_dispatcher(module)
            and not _has_serialized_class_forward(module)
            else None
        )
        compile_forward = current_forward
        if current_forward is not None and (
            _has_auto_forward_dispatcher(module) or _has_serialized_class_forward(module)
        ):
            compile_forward = _bind_class_descriptor(module, "forward")
        bootstrap_owned = (
            record.bootstrap_compiled_call_installed and _bootstrap_compiled_call_is_owned(module)
        )
        if module._compiled_call_impl is not None and not bootstrap_owned:
            _mark_user_owned(
                module,
                record,
                reason=_TERMINAL_REASON_PREEXISTING_COMPILED_CALL,
            )
            return
        generation = record.generation
        promotion_token = record.promotion_token
        record.state = ModuleAutoCompileState.PROMOTING
        record.pending_promotion = False
        _remove_bootstrap_hooks(
            module,
            record,
            clear_compiled_call_impl=False,
        )

    try:
        with _internal_module_compile_scope():
            raw_compiled_callable = (
                _compile_module_forward(compile_forward)
                if compile_forward is not None
                else _compile_module_call_impl(module)
            )
    except Exception:
        with _STATE_LOCK:
            if (
                _MODULE_REGISTRY.get(module) is record
                and record.state == ModuleAutoCompileState.PROMOTING
                and record.generation == generation
                and record.promotion_token == promotion_token
            ):
                _request_terminal_eager_locked(
                    module,
                    record,
                    _TERMINAL_REASON_COMPILE_SETUP_FAILED,
                )
        return
    except BaseException:
        with _STATE_LOCK:
            if (
                _MODULE_REGISTRY.get(module) is record
                and record.state == ModuleAutoCompileState.PROMOTING
                and record.generation == generation
                and record.promotion_token == promotion_token
            ):
                record.state = ModuleAutoCompileState.PROMOTION_READY
                record.pending_promotion = True
                try:
                    _install_bootstrap_hooks(module, record)
                except Exception:
                    _request_terminal_eager_locked(
                        module,
                        record,
                        _TERMINAL_REASON_COMPILE_SETUP_FAILED,
                    )
        raise

    if raw_compiled_callable is None:
        with _STATE_LOCK:
            if (
                _MODULE_REGISTRY.get(module) is record
                and record.state == ModuleAutoCompileState.PROMOTING
                and record.generation == generation
                and record.promotion_token == promotion_token
            ):
                _request_terminal_eager_locked(
                    module,
                    record,
                    _TERMINAL_REASON_COMPILE_SETUP_FAILED,
                )
        return

    attempt = getattr(raw_compiled_callable, _BACKEND_ATTEMPT_ATTR, None)
    module_ref = weakref.ref(module)

    def _backend_failed(_phase: str, _reason: str) -> None:
        live_module = module_ref()
        if live_module is None:
            return
        with _STATE_LOCK:
            if _MODULE_REGISTRY.get(live_module) is not record or record.state not in {
                ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL,
                ModuleAutoCompileState.COMPILED,
            }:
                return
            if not _auto_execution_wrapper_is_installed(
                live_module,
                record,
                wrapper,
            ):
                _mark_user_owned(
                    live_module,
                    record,
                    reason=_TERMINAL_REASON_PREEXISTING_COMPILED_CALL,
                )
                return
            _request_terminal_eager_locked(
                live_module,
                record,
                _TERMINAL_REASON_BACKEND_FALLBACK,
                force_drain=True,
            )

    wrapper = _make_compiled_wrapper(
        module,
        record,
        raw_compiled_callable,
        original_forward=compile_forward,
    )
    with _STATE_LOCK:
        if (
            _MODULE_REGISTRY.get(module) is not record
            or record.state != ModuleAutoCompileState.PROMOTING
            or record.generation != generation
            or record.promotion_token != promotion_token
        ):
            return
        if not module_auto_compile_enabled():
            record.state = ModuleAutoCompileState.PROMOTION_READY
            record.pending_promotion = True
            _install_bootstrap_hooks(module, record)
            return
        bootstrap_gate_owned = (
            record.bootstrap_compiled_call_installed and _bootstrap_compiled_call_is_owned(module)
        )
        if module._compiled_call_impl is not None and not bootstrap_gate_owned:
            _mark_user_owned(
                module,
                record,
                reason=_TERMINAL_REASON_PREEXISTING_COMPILED_CALL,
            )
            return
        if current_forward is not None and module.forward != current_forward:
            _mark_user_owned(
                module,
                record,
                reason=_TERMINAL_REASON_PREEXISTING_COMPILED_CALL,
            )
            return

        record.state = ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
        record.first_compiled_call_in_flight = False
        record.probation_invocation_count = 0
        record.defer_first_compiled_call = defer_first_call
        record.compile_attempt = attempt
        if attempt is not None:
            attempt.set_failure_callback(_backend_failed)
        record.terminal_reason = None
        if bootstrap_gate_owned:
            module._compiled_call_impl = None
            record.bootstrap_compiled_call_installed = False
        _install_auto_wrapper_locked(
            module,
            record,
            wrapper,
            forward_only=forward_only,
            original_forward=original_instance_forward,
        )
        _debug_log(f"promoted module={record.class_name} id={id(module)}")


# This module-level hook and temporary compiled-call dispatcher keep bootstrap
# state picklable. Explicit user compile removes both before Dynamo guards them.
def _bootstrap_forward_pre_hook(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> None:
    if torch.compiler.is_compiling():
        return

    stack = getattr(_BOOTSTRAP_INVOCATION_STATE, "stack", None)
    if stack:
        stack[:] = [entry for entry in stack if entry.module is not None]
        outer = next((entry for entry in reversed(stack) if entry.module is mod), None)
        if outer is not None:
            stack.append(
                _BootstrapInvocation(
                    module_ref=weakref.ref(mod),
                    record=outer.record,
                    generation=outer.generation,
                    top_level=False,
                )
            )
            return

    # Runtime gate: these per-module hooks stay installed after the feature is
    # turned off (env=0 or kill-switch), so re-check here or a mid-bootstrap
    # module would still get promoted while "disabled".
    if not module_auto_compile_enabled():
        return
    context_flags = _runtime_context_flags()
    checkpoint_recompute = bool(context_flags & _RUNTIME_CONTEXT_RECOMPUTE)
    if (
        checkpoint_recompute
        or bool(
            context_flags & (_RUNTIME_CONTEXT_SYMBOLIC_TRACE | _RUNTIME_CONTEXT_AUTOGRAD_VALIDATION)
        )
        or _functorch_transform_active()
    ):
        return
    with _STATE_LOCK:
        record = _MODULE_REGISTRY.get(mod)
        if record is None:
            _remove_inherited_bootstrap_hooks(mod)
            return
    _move_bootstrap_forward_pre_hook_to_end(mod, record)
    if not _call_contains_neuron_tensor(args, kwargs):
        return
    if (
        _call_contains_empty_neuron_tensor(args, kwargs)
        or _call_contains_non_contiguous_neuron_tensor(args, kwargs)
        or _active_module_call_contains_non_contiguous_neuron_tensor()
    ):
        return
    if record.state in _PRE_PROMOTION_STATES and not _module_is_compile_region(mod):
        with _STATE_LOCK:
            if _MODULE_REGISTRY.get(mod) is record and record.state in _PRE_PROMOTION_STATES:
                _request_terminal_eager_locked(
                    mod,
                    record,
                    _TERMINAL_REASON_COVERAGE_RESTRICTED,
                )
        return
    if stack is None:
        stack = []
        _BOOTSTRAP_INVOCATION_STATE.stack = stack

    with _STATE_LOCK:
        if _MODULE_REGISTRY.get(mod) is not record:
            return
        if context_flags & (_RUNTIME_CONTEXT_CHECKPOINT | _RUNTIME_CONTEXT_SAC_FORWARD):
            record.requires_backward_executor = True
        if record.state in _PRE_PROMOTION_STATES:
            record.active_bootstrap_invocations += 1
            stack.append(
                _BootstrapInvocation(
                    module_ref=weakref.ref(mod),
                    record=record,
                    generation=record.generation,
                    top_level=True,
                )
            )


def _latest_bootstrap_invocation(mod: torch.nn.Module) -> _BootstrapInvocation | None:
    stack = getattr(_BOOTSTRAP_INVOCATION_STATE, "stack", ())
    return next((entry for entry in reversed(stack) if entry.module is mod), None)


def _pop_bootstrap_invocation(mod: torch.nn.Module) -> _BootstrapInvocation | None:
    stack = getattr(_BOOTSTRAP_INVOCATION_STATE, "stack", None)
    if not stack:
        return None
    for index in range(len(stack) - 1, -1, -1):
        if stack[index].module is mod:
            return stack.pop(index)
    return None


def _discard_bootstrap_invocations(mod: torch.nn.Module) -> None:
    stack = getattr(_BOOTSTRAP_INVOCATION_STATE, "stack", None)
    if stack:
        stack[:] = [entry for entry in stack if entry.module is not mod]


def _complete_bootstrap_invocation(mod: torch.nn.Module) -> None:
    entry = _pop_bootstrap_invocation(mod)
    if entry is None or not entry.top_level:
        return
    record = entry.record

    with _STATE_LOCK:
        if _MODULE_REGISTRY.get(mod) is not record or record.generation != entry.generation:
            return
        if record.active_bootstrap_invocations > 0:
            record.active_bootstrap_invocations -= 1
        if (
            not module_auto_compile_enabled()
            or _inside_symbolic_trace()
            or _inside_autograd_validation_context()
            or _functorch_transform_active()
        ):
            return
        if entry.succeeded:
            record.successful_invocation_count += 1
            if record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING:
                if mod._compiled_call_impl is not None and not _bootstrap_compiled_call_is_owned(
                    mod
                ):
                    _mark_user_owned(
                        mod,
                        record,
                        reason=_TERMINAL_REASON_PREEXISTING_COMPILED_CALL,
                    )
                elif record.successful_invocation_count >= _PROMOTION_INVOCATION_THRESHOLD:
                    _mark_promotion_ready_locked(mod, record)


def _run_bootstrap_call_impl(
    module: torch.nn.Module,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    track_output: bool = True,
) -> object:
    with _STATE_LOCK:
        record = _MODULE_REGISTRY.get(module)
        installed = (
            record is not None
            and record.bootstrap_compiled_call_installed
            and _bootstrap_compiled_call_is_owned(module)
        )
    if not installed:
        _remove_inherited_bootstrap_hooks(module)
        with _STATE_LOCK:
            record = _MODULE_REGISTRY.get(module)
            wrapper = None if record is None else _auto_wrapper(record)
        if wrapper is not None and module._compiled_call_impl is wrapper:
            return wrapper(*args, **kwargs)
        return _call_class_descriptor(module, "_call_impl", args, kwargs)

    try:
        output = _call_class_descriptor(module, "_call_impl", args, kwargs)
        entry = _latest_bootstrap_invocation(module)
        if entry is not None:
            entry.succeeded = True
            if entry.top_level and track_output:
                _track_output_backward(
                    module,
                    entry.record,
                    entry.generation,
                    output,
                    compiled=False,
                )
        return output
    finally:
        _complete_bootstrap_invocation(module)


def _run_inherited_bootstrap_call_impl(
    module: torch.nn.Module,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    _remove_inherited_bootstrap_hooks(module)
    return _call_class_descriptor(module, "_call_impl", args, kwargs)


def _install_bootstrap_hooks(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
) -> None:
    bootstrap_gate_owned = (
        record.bootstrap_compiled_call_installed and _bootstrap_compiled_call_is_owned(module)
    )
    if record.bootstrap_forward_pre_handle is not None and bootstrap_gate_owned:
        return

    _remove_inherited_bootstrap_hooks(module)
    if not bootstrap_gate_owned:
        if module._compiled_call_impl is not None:
            raise RuntimeError("cannot bootstrap a module with a compiled call override")
        module._compiled_call_impl = _BootstrapCallImpl(module)
        record.bootstrap_compiled_call_installed = True
    try:
        try:
            record.bootstrap_forward_pre_handle = module.register_forward_pre_hook(
                _bootstrap_forward_pre_hook,
                with_kwargs=True,
            )
        except TypeError:
            record.bootstrap_forward_pre_handle = module.register_forward_pre_hook(
                _bootstrap_forward_pre_hook
            )
    except Exception:
        _remove_bootstrap_hooks(module, record)
        raise


def _active_stack_has_hook_sensitive_ancestor(
    active_stack: tuple[torch.nn.Module, ...],
) -> bool:
    return any(
        isinstance(ancestor, _HOOK_SENSITIVE_CONTAINER_TYPES) for ancestor in active_stack[1:]
    )


def _classify_and_transition(
    module: torch.nn.Module,
    record: ModuleAutoCompileRecord,
    active_stack: tuple[torch.nn.Module, ...],
) -> None:
    if _compiler_owns_graph_module(module):
        _mark_user_owned(
            module,
            record,
            reason=_TERMINAL_REASON_COMPILER_OWNED_GRAPH,
        )
        return
    if record.state in _TERMINAL_STATES:
        return

    _remove_inherited_bootstrap_hooks(module)
    record.discovery_count += 1

    if _active_stack_has_hook_sensitive_ancestor(active_stack):
        if record.state == ModuleAutoCompileState.COLD or record.state in _PRE_PROMOTION_STATES:
            _request_terminal_eager_locked(
                module,
                record,
                _TERMINAL_REASON_HOOK_SENSITIVE_ANCESTOR,
            )
        return

    if any(ancestor._compiled_call_impl is not None for ancestor in active_stack[1:]):
        if record.state == ModuleAutoCompileState.COLD:
            _request_terminal_eager_locked(
                module,
                record,
                _TERMINAL_REASON_COMPILED_ANCESTOR,
            )
        return

    if record.state != ModuleAutoCompileState.COLD:
        return
    if module._compiled_call_impl is not None:
        _mark_user_owned(
            module,
            record,
            reason=_TERMINAL_REASON_PREEXISTING_COMPILED_CALL,
        )
    elif not _compiler_can_capture_module(module):
        _request_terminal_eager_locked(
            module,
            record,
            _TERMINAL_REASON_COMPILER_CAPABILITY_UNAVAILABLE,
        )
    elif _module_is_compile_region(module):
        _install_bootstrap_hooks(module, record)
        record.state = ModuleAutoCompileState.BOOTSTRAP_PENDING
        _debug_log(
            f"discover module={record.class_name} id={id(module)} state={record.state.value}"
        )
    else:
        _request_terminal_eager_locked(
            module,
            record,
            _TERMINAL_REASON_COVERAGE_RESTRICTED,
        )


def _observe_module_forward(module: torch.nn.Module) -> None:
    """Legacy/test entry point for direct module observation.

    Product discovery uses Module.to, Module.to_empty, and Module.neuron. This
    helper supports focused state-machine tests.
    """
    if _observation_context_blocked():
        return

    record, _ = _get_or_create_record(module)
    if record.state in _TERMINAL_STATES:
        return

    # module at [0] + its live-frame ancestors (drop the self-duplicate).
    ancestors = tuple(m for m in _recover_active_module_stack() if m is not module)
    active_stack = (module, *ancestors)
    _classify_and_transition(module, record, active_stack)


def _module_has_direct_neuron_state(module: torch.nn.Module) -> bool:
    parameters = getattr(module, "_parameters", {})
    buffers = getattr(module, "_buffers", {})
    return any(_tensor_is_neuron(tensor) for tensor in (*parameters.values(), *buffers.values()))


def _iter_module_tree(root: torch.nn.Module):
    pending = [root]
    seen: set[int] = set()
    while pending:
        module = pending.pop()
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        yield module
        children = [
            child
            for child in getattr(module, "_modules", {}).values()
            if isinstance(child, torch.nn.Module)
        ]
        pending.extend(reversed(children))


def _module_parent_map(root: torch.nn.Module) -> dict[int, torch.nn.Module]:
    parents: dict[int, torch.nn.Module] = {}
    for parent in _iter_module_tree(root):
        for child in getattr(parent, "_modules", {}).values():
            if not isinstance(child, torch.nn.Module):
                continue
            parents.setdefault(id(child), parent)
    return parents


def _modules_with_shared_parameters(root: torch.nn.Module) -> set[int]:
    owners: dict[int, int] = {}
    shared_modules: set[int] = set()
    for module in _iter_module_tree(root):
        module_id = id(module)
        for parameter in getattr(module, "_parameters", {}).values():
            if parameter is None:
                continue
            parameter_id = id(parameter)
            owner_id = owners.setdefault(parameter_id, module_id)
            if owner_id != module_id:
                shared_modules.update((owner_id, module_id))
    return shared_modules


def _module_stack_from_parent_map(
    module: torch.nn.Module,
    parents: dict[int, torch.nn.Module],
) -> tuple[torch.nn.Module, ...]:
    stack = [module]
    seen = {id(module)}
    current = module
    while (parent := parents.get(id(current))) is not None:
        if id(parent) in seen:
            break
        seen.add(id(parent))
        stack.append(parent)
        current = parent
    return tuple(stack)


def observe_module_tree_after_to(
    root: torch.nn.Module,
    *,
    target_device: torch.device | None,
    recurse: bool = True,
) -> None:
    """Discover eligible modules after a module device conversion.

    Args:
        root (torch.nn.Module): Root module that was converted.
        target_device (torch.device | None): Explicit conversion target, if available.
        recurse (bool, optional): Inspect descendants of ``root``. Defaults to ``True``.
    """
    if not module_auto_compile_enabled() or _observation_context_blocked():
        return

    with _STATE_LOCK:
        reset_generation = _RESET_GENERATION

    target_is_neuron = target_device is not None and target_device.type == "neuron"
    parents = _module_parent_map(root) if recurse else {}
    shared_parameter_modules = _modules_with_shared_parameters(root) if recurse else set()
    modules = _iter_module_tree(root) if recurse else (root,)
    for module in modules:
        if not _module_is_compile_region(module):
            continue
        if not target_is_neuron and not _module_has_direct_neuron_state(module):
            continue

        with _STATE_LOCK:
            if reset_generation != _RESET_GENERATION or not module_auto_compile_enabled():
                return
            record, _ = _get_or_create_record(module)
            if record.state in _TERMINAL_STATES:
                continue
            if id(module) in shared_parameter_modules:
                _request_terminal_eager_locked(
                    module,
                    record,
                    _TERMINAL_REASON_COMPILER_CAPABILITY_UNAVAILABLE,
                )
                continue
            active_stack = _module_stack_from_parent_map(module, parents)
            _classify_and_transition(module, record, active_stack)

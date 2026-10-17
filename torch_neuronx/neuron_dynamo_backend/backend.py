"""
Neuron backend implementation for torch.compile
"""

import contextlib
import functools
import hashlib
import io
import logging
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.utils import dynamo_timed
from torch._functorch._aot_autograd.logging_utils import get_aot_graph_name
from torch._functorch._aot_autograd.schemas import SerializableAOTDispatchCompiler
from torch._guards import TracingContext
from torch.utils._filelock import FileLock

from torch_neuronx.kernels.cache_utils import CompilationJob, compilation_job, get_lock_timeout
from torch_neuronx.neuron_dynamo_backend.compile import CompileGraph
from torch_neuronx.neuron_dynamo_backend.compile_only import build_compile_only_handle

# Do not call get_model_name() directly in this file; use _resolve_model_name(options) instead.
from torch_neuronx.neuron_dynamo_backend.config import (
    get_model_name,
)
from torch_neuronx.neuron_dynamo_backend.decompositions import get_compile_decomposition_table
from torch_neuronx.neuron_dynamo_backend.fx.fx_transform import (
    convert_fx_to_stablehlo,
)
from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
    IOMutation,
    MutationKind,
    PassMutations,
    apply_input_mutations,
    apply_output_mutations,
    get_alias_output_buffer_map,
    get_mutation_copy_back_inputs,
    translate_input_indices_to_neff,
)
from torch_neuronx.neuron_dynamo_backend.fx.pipelines.post_aot_processing_pipeline import (
    post_aot_processing,
)
from torch_neuronx.neuron_dynamo_backend.fx.pipelines.preprocessing_pipeline import preprocess_graph
from torch_neuronx.neuron_dynamo_backend.hilo_infergoldens_executor import HiloInferGoldensExecutor
from torch_neuronx.neuron_dynamo_backend.metrics import (
    NeuronCompilationMetrics,
    record_compilation,
)
from torch_neuronx.neuron_dynamo_backend.settings import _getenv_bool
from torch_neuronx.neuron_dynamo_backend.stablehlo_executor import StableHLOInterpreterExecutor

logger = logging.getLogger(__name__)
_PRESERVE_ACT_INPUT_GRAD_LOCK = threading.RLock()
_AUTOCOMPILE_SAC_RECOMPUTE_POLICY = -1


def _autocompile_sac_policy_identity(policy: object) -> tuple[object, ...]:
    code = getattr(policy, "__code__", None)
    if code is None:
        return (id(policy),)

    captures = []
    for cell in getattr(policy, "__closure__", None) or ():
        try:
            captures.append(id(cell.cell_contents))
        except ValueError:
            captures.append(0)
    defaults = tuple(id(value) for value in (getattr(policy, "__defaults__", None) or ()))
    keyword_defaults = tuple(
        (name, id(value))
        for name, value in sorted((getattr(policy, "__kwdefaults__", None) or {}).items())
    )
    return (id(code), tuple(captures), defaults, keyword_defaults)


def _fallback_execution_disabled(options) -> bool:
    if options is not None and "fallback_execution" in options:
        return not bool(options["fallback_execution"])
    return _getenv_bool("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", False)


def _autocompile_compile_fallback_enabled(options, autocompile_attempt=None) -> bool:
    return bool(
        autocompile_attempt is not None and options and options.get("autocompile_compile_fallback")
    )


@dataclass
class _AutoCompileBackendSegment:
    phase: str
    graph_node_count: int
    cache_key: str
    sac_policy_trace: tuple[tuple[str, str], ...] = ()
    execution_count: int = 0


@dataclass
class _AutoCompileBackendAttempt:
    minimum_forward_graph_node_count: int = 1
    on_failure: Callable[[str, str], None] | None = None
    runtime_validator: Callable[[], None] | None = None
    segments: list[_AutoCompileBackendSegment] = field(default_factory=list)
    forward_executor_count: int = 0
    backward_executor_count: int = 0
    forward_graph_node_count: int = 0
    backward_graph_node_count: int = 0
    cache_keys: list[str] = field(default_factory=list)
    fallback_reasons: list[tuple[str, str]] = field(default_factory=list)
    failure_reasons: list[tuple[str, str]] = field(default_factory=list)
    compatibility_reasons: list[tuple[str, str]] = field(default_factory=list)
    late_guard_miss_count: int = 0
    _active_executor_count: int = 0
    _executor_execution_blocked: bool = False
    _sealed: bool = False
    _seal_generation: int = 0
    _pending_sac_policy: Callable[..., object] | None = field(default=None, repr=False)
    _pending_context_signature: tuple[object, ...] | None = field(default=None, repr=False)
    _expected_context_signature: tuple[object, ...] | None = field(default=None, repr=False)
    _invocation_context: threading.local = field(
        default_factory=threading.local,
        repr=False,
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _executor_condition: threading.Condition = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._executor_condition = threading.Condition(self._lock)

    def set_failure_callback(self, callback: Callable[[str, str], None] | None) -> None:
        """Set the callback for terminal AutoCompile failures.

        Args:
            callback (Callable[[str, str], None] | None): Phase-and-reason callback, or
                ``None`` to clear it.
        """
        with self._lock:
            self.on_failure = callback

    def set_pending_autocompile_context(
        self,
        policy: Callable[..., object] | None,
        signature: tuple[object, ...],
    ) -> bool:
        """Store a pending selective-activation-checkpoint context on an unsealed attempt.

        Args:
            policy (Callable | None): Active selective-checkpoint policy, if any.
            signature (tuple[object, ...]): Runtime context signature to associate with it.

        Returns:
            bool: ``True`` after storing the context or when a sealed attempt already
                expects ``signature``; ``False`` for any other sealed or mismatched context.
        """
        with self._lock:
            if self._expected_context_signature not in {None, signature}:
                return False
            if self._sealed:
                return self._expected_context_signature == signature
            self._pending_sac_policy = policy
            self._pending_context_signature = signature
            return True

    def take_pending_autocompile_context(
        self,
    ) -> tuple[Callable[..., object] | None, tuple[object, ...] | None]:
        """Consume the pending selective-checkpoint context.

        Returns:
            tuple[Callable | None, tuple | None]: Pending policy and signature, each
                cleared from the attempt after retrieval.

        Note:
            The first consumed signature becomes the expected invocation signature.
        """
        with self._lock:
            policy = self._pending_sac_policy
            signature = self._pending_context_signature
            self._pending_sac_policy = None
            self._pending_context_signature = None
            if self._expected_context_signature is None:
                self._expected_context_signature = signature
            return policy, signature

    def clear_pending_autocompile_context(
        self,
        policy: Callable[..., object] | None,
        signature: tuple[object, ...],
    ) -> None:
        """Clear a pending context when it still matches the caller.

        Args:
            policy (Callable | None): Policy object originally stored by the caller.
            signature (tuple[object, ...]): Signature originally stored by the caller.
        """
        with self._lock:
            if self._pending_sac_policy is policy and self._pending_context_signature == signature:
                self._pending_sac_policy = None
                self._pending_context_signature = None

    def autocompile_context_matches(self, signature: tuple[object, ...]) -> bool:
        """Check whether an invocation matches the expected compile context.

        Args:
            signature (tuple[object, ...]): Active runtime context signature.

        Returns:
            bool: ``True`` when no expected signature is set, when ``signature`` matches
                it, or for a compatible selective-checkpoint recomputation signature.
        """
        with self._lock:
            expected = self._expected_context_signature
            if expected in {None, signature}:
                return True
            return (
                signature == (2, _AUTOCOMPILE_SAC_RECOMPUTE_POLICY)
                and expected is not None
                and len(expected) > 1
                and expected[0] == 2
            )

    @contextlib.contextmanager
    def autocompile_invocation_context(self, signature: tuple[object, ...]):
        """Track a nested AutoCompile invocation signature.

        Args:
            signature (tuple[object, ...]): Signature to push for the context lifetime.

        Yields:
            None: Keeps ``signature`` active until the context exits.
        """
        stack = getattr(self._invocation_context, "stack", None)
        if stack is None:
            stack = []
            self._invocation_context.stack = stack
        stack.append(signature)
        try:
            yield
        finally:
            stack.pop()
            if not stack:
                del self._invocation_context.stack

    def active_autocompile_context_signature(self) -> tuple[object, ...] | None:
        """Return the innermost active AutoCompile invocation signature.

        Returns:
            tuple[object, ...] | None: Top thread-local signature, or ``None`` outside
                an invocation.
        """
        stack = getattr(self._invocation_context, "stack", ())
        return stack[-1] if stack else None

    def begin_segment(self, phase: str) -> int | None:
        """Reserve the current generation for an AutoCompile segment.

        Args:
            phase (str): Compilation phase associated with the segment.

        Returns:
            int | None: Current seal generation, or ``None`` after the attempt is sealed.

        Note:
            A late reservation records a guard miss, blocks executor execution, and invokes
            the failure callback.
        """
        with self._lock:
            if not self._sealed:
                return self._seal_generation
            self.late_guard_miss_count += 1
            self.fallback_reasons.append((phase, "late_guard_miss"))
            self._executor_execution_blocked = True
            callback = self.on_failure
        if callback is not None:
            callback(phase, "late_guard_miss")
        return None

    def register_executor(
        self,
        phase: str,
        graph_node_count: int,
        cache_key: str,
        reservation_generation: int | None = None,
        sac_policy_trace: tuple[tuple[str, str], ...] = (),
    ) -> int | None:
        """Register a compiled executor segment when its reservation is current.

        Args:
            phase (str): Compilation phase associated with the segment.
            graph_node_count (int): Number of compute nodes in the compiled graph.
            cache_key (str): NEFF cache key for the executor.
            reservation_generation (int | None, optional): Generation returned by
                ``begin_segment``. Defaults to ``None``.
            sac_policy_trace (tuple[tuple[str, str], ...], optional): Selective-checkpoint
                decisions. Defaults to ``()``.

        Returns:
            int | None: New segment index, or ``None`` for a sealed or stale reservation.

        Note:
            Rejection records a late guard miss, blocks executor execution, and invokes
            a configured failure callback.
        """
        with self._lock:
            if self._sealed or (
                reservation_generation is not None
                and reservation_generation != self._seal_generation
            ):
                self.late_guard_miss_count += 1
                self.fallback_reasons.append((phase, "late_guard_miss"))
                self._executor_execution_blocked = True
                callback = self.on_failure
                segment_id = None
            else:
                segment_id = len(self.segments)
                self.segments.append(
                    _AutoCompileBackendSegment(
                        phase=phase,
                        graph_node_count=graph_node_count,
                        cache_key=cache_key,
                        sac_policy_trace=sac_policy_trace,
                    )
                )
                self.cache_keys.append(cache_key)
                callback = None
        if callback is not None:
            callback(phase, "late_guard_miss")
        return segment_id

    def record_executor_success(self, segment_id: int) -> None:
        """Record a successful execution of a registered segment.

        Args:
            segment_id (int): Index of the segment that executed.

        Note:
            The optional runtime validator runs before the first success. Executor and graph
            counts increase only for that first success; every success increments the segment count.
        """
        with self._lock:
            needs_validation = self.segments[segment_id].execution_count == 0
            validator = self.runtime_validator
        if needs_validation and validator is not None:
            validator()
        with self._lock:
            segment = self.segments[segment_id]
            segment.execution_count += 1
            if segment.execution_count != 1:
                return
            if segment.phase == "backward":
                self.backward_executor_count += 1
                self.backward_graph_node_count += segment.graph_node_count
            else:
                self.forward_executor_count += 1
                self.forward_graph_node_count += segment.graph_node_count

    def record_executor(
        self,
        phase: str,
        graph_node_count: int,
        cache_key: str,
        *,
        sac_policy_trace: tuple[tuple[str, str], ...] = (),
    ) -> None:
        """Register and mark an executor segment successful when accepted.

        Args:
            phase (str): Compilation phase associated with the executor.
            graph_node_count (int): Number of compute nodes in the compiled graph.
            cache_key (str): NEFF cache key for the executor.
            sac_policy_trace (tuple[tuple[str, str], ...], optional): Selective-checkpoint
                decisions. Defaults to ``()``.

        Note:
            A sealed or stale reservation records a late guard miss instead of a success.
            Runtime validation runs before the first success is recorded.
        """
        reservation_generation = self.begin_segment(phase)
        if reservation_generation is None:
            return
        segment_id = self.register_executor(
            phase,
            graph_node_count,
            cache_key,
            reservation_generation,
            sac_policy_trace,
        )
        if segment_id is not None:
            self.record_executor_success(segment_id)

    def begin_executor_execution(self) -> bool:
        """Begin one executor execution unless execution is blocked.

        Returns:
            bool: ``True`` after incrementing the active count, or ``False`` when blocked.
        """
        with self._executor_condition:
            if self._executor_execution_blocked:
                return False
            self._active_executor_count += 1
            return True

    def end_executor_execution(self) -> None:
        """Finish one active executor execution.

        Raises:
            RuntimeError: If no executor execution is active.
        """
        with self._executor_condition:
            if self._active_executor_count <= 0:
                raise RuntimeError("AutoCompile executor activity is unbalanced")
            self._active_executor_count -= 1
            if self._active_executor_count == 0:
                self._executor_condition.notify_all()

    def wait_for_executor_drain(self) -> None:
        """Wait until all active executor executions finish."""
        with self._executor_condition:
            while self._active_executor_count != 0:
                self._executor_condition.wait()

    def seal(self) -> bool:
        """Seal the attempt against additional executor segments.

        Returns:
            bool: ``True`` when every registered segment has executed, or ``False`` when
                an unexecuted segment triggers a late guard miss.

        Note:
            A late guard miss blocks executor execution and invokes the failure callback.
        """
        with self._lock:
            self._sealed = True
            self._seal_generation += 1
            pending_phase = next(
                (segment.phase for segment in self.segments if segment.execution_count == 0),
                None,
            )
            if pending_phase is None:
                return True
            self.late_guard_miss_count += 1
            self.fallback_reasons.append((pending_phase, "late_guard_miss"))
            self._executor_execution_blocked = True
            callback = self.on_failure
        if callback is not None:
            callback(pending_phase, "late_guard_miss")
        return False

    def record_fallback(self, phase: str, reason: str) -> None:
        """Record a fallback and block subsequent executor execution.

        Args:
            phase (str): Phase that fell back.
            reason (str): Machine-readable fallback reason.

        Note:
            Invokes the failure callback after updating the attempt.
        """
        with self._lock:
            self.fallback_reasons.append((phase, reason))
            self._executor_execution_blocked = True
            callback = self.on_failure
        if callback is not None:
            callback(phase, reason)

    def record_failure(self, phase: str, reason: str) -> None:
        """Record a runtime failure and block subsequent executor execution.

        Args:
            phase (str): Phase that failed.
            reason (str): Machine-readable failure reason.

        Note:
            Invokes the failure callback after updating the attempt.
        """
        with self._lock:
            self.failure_reasons.append((phase, reason))
            self._executor_execution_blocked = True
            callback = self.on_failure
        if callback is not None:
            callback(phase, reason)

    def record_compatibility(self, phase: str, reason: str) -> None:
        """Record a nonterminal compatibility observation.

        Args:
            phase (str): Phase where compatibility handling occurred.
            reason (str): Machine-readable compatibility reason.
        """
        with self._lock:
            self.compatibility_reasons.append((phase, reason))

    def has_real_forward_executor(self) -> bool:
        """Check whether a forward executor has completed successfully.

        Returns:
            bool: ``True`` after at least one forward executor's first success.
        """
        with self._lock:
            return self.forward_executor_count > 0

    def has_real_backward_executor(self) -> bool:
        """Check whether a backward executor has completed successfully.

        Returns:
            bool: ``True`` after at least one backward executor's first success.
        """
        with self._lock:
            return self.backward_executor_count > 0

    def has_profitable_forward(self) -> bool:
        """Check whether every forward segment is successful and profitable.

        Returns:
            bool: ``True`` when forward segments exist, all have executed, each meets the
                minimum node count, and no fallback or failure was recorded.
        """
        with self._lock:
            forward_segments = tuple(
                segment for segment in self.segments if segment.phase != "backward"
            )
            return (
                not self.fallback_reasons
                and not self.failure_reasons
                and bool(forward_segments)
                and self.forward_executor_count == len(forward_segments)
                and all(
                    segment.graph_node_count >= self.minimum_forward_graph_node_count
                    for segment in forward_segments
                )
            )

    def forward_sac_policy_trace(self) -> tuple[tuple[str, str], ...]:
        """Collect selective-checkpoint decisions from forward segments.

        Returns:
            tuple[tuple[str, str], ...]: Forward operation names and policy decisions in
                segment order.
        """
        with self._lock:
            return tuple(
                entry
                for segment in self.segments
                if segment.phase == "forward"
                for entry in segment.sac_policy_trace
            )

    def failed(self) -> bool:
        """Check whether the attempt recorded a fallback or failure.

        Returns:
            bool: ``True`` when either terminal-reason list is nonempty.
        """
        with self._lock:
            return bool(self.fallback_reasons or self.failure_reasons)


def _record_autocompile_fallback(
    attempt: _AutoCompileBackendAttempt | None,
    phase: str,
    reason: str,
) -> None:
    if attempt is not None:
        attempt.record_fallback(phase, reason)


def _resolve_model_name(options):
    """Resolve the model name from options dict, falling back to get_model_name().

    Priority: options["model_name"] > get_model_name() <-- returns global/default
    """
    model_name = options.get("model_name") if options else None
    if not model_name:
        model_name = get_model_name()
    return model_name


def _compile_fx_to_stablehlo(
    gm: torch.fx.GraphModule,
    example_inputs,
    model_name: str,
    segment_id: str,
    job: CompilationJob,
    alias_buffer_map: dict[int, int] | None = None,
    static_input_indices: list[int] | None = None,
):
    """Convert an FX GraphModule to StableHLO MLIR.

    Args:
        gm (torch.fx.GraphModule): FX GraphModule with ATen operations.
        example_inputs: Example input tensors.
        model_name (str): Model name for logging and artifact naming.
        segment_id (str): Segment ID for statistics tracking.
        job (CompilationJob): Compilation job for artifact management.
        alias_buffer_map (dict[int, int] | None): NEFF output→input aliasing hints.
        static_input_indices (list[int] | None): WLO — indices of static weights the
            compiler may transpose (stamped as transposable_weight_idx). None disables WLO.

    Returns:
        tuple: (stablehlo_mlir, artifacts_stablehlo, io_spec, cast_spec)
            - stablehlo_mlir: Compiled StableHLO MLIR module.
            - artifacts_stablehlo (Path): Path to saved StableHLO artifacts.
            - io_spec (FunctionIO): Input/output specifications.
            - cast_spec (list[TensorSpec]): Output dtype specifications.
    """
    logger.debug("Starting FX Graph to StableHLO conversion")

    stablehlo_mlir, io_spec, cast_spec = convert_fx_to_stablehlo(
        gm,
        example_inputs,
        alias_buffer_map,
        job=job,
        static_input_indices=static_input_indices,
    )
    logger.debug("FX Graph to StableHLO conversion completed")

    # Save stablehlo to artifacts dir
    artifacts_stablehlo = job.save_stablehlo(model_name, stablehlo_mlir)

    return stablehlo_mlir, artifacts_stablehlo, io_spec, cast_spec


def _has_collectives_in_graph(gm: torch.fx.GraphModule) -> bool:
    """Detect if an FX graph contains collective operations.

    Detects torch._c10d_functional collectives and NKI HOPs whose specialized
    frontend config reports embedded collective communication.

    Args:
        gm (torch.fx.GraphModule): FX GraphModule to analyze.

    Returns:
        bool: True if graph contains any collective operations.
    """
    from torch_neuronx.nki_hop import nki_kernel_wrapper

    collective_ops = {"all_reduce", "all_gather", "reduce_scatter", "all_to_all"}

    for node in gm.graph.nodes:
        if node.op == "call_function":
            if node.target is nki_kernel_wrapper and node.kwargs.get("has_collectives", False):
                logger.info("Detected collective NKI kernel")
                return True

            func_name = (
                node.target.__name__ if hasattr(node.target, "__name__") else str(node.target)
            )
            if any(collective in func_name for collective in collective_ops):
                logger.info(f"Detected collective operation: {func_name}")
                return True

    return False


def _count_compute_nodes(gm: torch.fx.GraphModule) -> int:
    """Count FX nodes that execute computation."""
    compute_ops = {"call_function", "call_method", "call_module"}
    return sum(node.op in compute_ops for node in gm.graph.nodes)


_SAC_POLICY_TARGET_ALIASES = {
    # Tracing replaces the eager async wrapper with an explicit wait.
    "_c10d_functional._wrap_tensor_autograd.default": "_c10d_functional.wait_tensor.default",
}


def _canonical_sac_policy_target(target: object) -> str:
    """Return a stable SAC target name across eager and traced forms."""
    target_name = str(target)
    return _SAC_POLICY_TARGET_ALIASES.get(target_name, target_name)


def _apply_sac_policy_to_nodes(
    nodes: list[torch.fx.Node],
    policy_fn: Callable[..., object],
) -> None:
    from torch.fx.node import map_arg
    from torch.utils import checkpoint as checkpoint_utils

    checkpoint_policy = checkpoint_utils.CheckpointPolicy
    context_type = checkpoint_utils.SelectiveCheckpointContext
    ignored_ops = getattr(checkpoint_utils, "SAC_IGNORED_OPS", ())

    def node_value(node: torch.fx.Node) -> object:
        """Return the example value stored on an FX node.

        Args:
            node (torch.fx.Node): Node whose metadata to inspect.

        Returns:
            object: ``node.meta['val']``, or ``None`` when absent.
        """
        return node.meta.get("val")

    for node in nodes:
        if node.target in ignored_ops:
            continue
        args = map_arg(node.args, node_value)
        kwargs = map_arg(node.kwargs, node_value)
        context = context_type(is_recompute=False, op_output=node.meta.get("val"))
        policy = policy_fn(context, node.target, *args, **kwargs)
        if isinstance(policy, bool):
            policy = checkpoint_policy.MUST_SAVE if policy else checkpoint_policy.PREFER_RECOMPUTE
        if not isinstance(policy, checkpoint_policy):
            raise TypeError(
                "SAC policy returned an invalid value. Return CheckpointPolicy or bool."
            )
        node.meta["recompute"] = policy


def _extract_sac_policy_trace(
    gm: torch.fx.GraphModule,
) -> tuple[tuple[str, str], ...]:
    trace = []
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        policy = node.meta.get("recompute")
        if policy is None:
            continue
        policy_name = getattr(policy, "name", str(policy))
        trace.append((_canonical_sac_policy_target(node.target), str(policy_name)))
    return tuple(trace)


def _is_autocompile_sac_mode(mode: object) -> bool:
    mode_type = type(mode)
    return mode_type.__module__ == "torch.utils.checkpoint" and mode_type.__name__ in {
        "_CachingTorchDispatchMode",
        "_CachedTorchDispatchMode",
    }


def _active_autocompile_checkpoint_kind(modes: tuple[object, ...]) -> int:
    if any(_is_autocompile_sac_mode(mode) for mode in modes):
        return 2

    try:
        frame = sys._getframe(1)
    except Exception:
        return 0
    checkpoint_suffix = os.path.normpath("torch/utils/checkpoint.py")
    while frame is not None:
        module_name = frame.f_globals.get("__name__")
        filename = os.path.normpath(frame.f_code.co_filename)
        if module_name == "torch.utils.checkpoint" or filename.endswith(checkpoint_suffix):
            qualname = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
            if qualname.startswith("CheckpointFunction."):
                return 1
            if frame.f_code.co_name == "recompute_fn":
                return 2
            if frame.f_code.co_name == "checkpoint":
                return 2 if frame.f_locals.get("use_reentrant") is False else 1
        frame = frame.f_back
    return 0


def _active_autocompile_context_signature() -> tuple[object, ...]:
    try:
        from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

        modes = tuple(_get_current_dispatch_mode_stack())
    except Exception:
        modes = ()
    if any(
        _is_autocompile_sac_mode(mode) and type(mode).__name__ == "_CachedTorchDispatchMode"
        for mode in modes
    ):
        policy_signature = (_AUTOCOMPILE_SAC_RECOMPUTE_POLICY,)
    else:
        policy_signature = tuple(
            _autocompile_sac_policy_identity(mode.policy_fn)
            for mode in modes
            if _is_autocompile_sac_mode(mode) and type(mode).__name__ == "_CachingTorchDispatchMode"
        )
    return (_active_autocompile_checkpoint_kind(modes), *policy_signature)


@contextlib.contextmanager
def _suspend_autocompile_sac_mode(enabled: bool):
    if not enabled:
        yield
        return

    try:
        from torch.utils._python_dispatch import (
            _disable_current_modes,
            _get_current_dispatch_mode_stack,
        )

        modes = _get_current_dispatch_mode_stack()
    except (AttributeError, ImportError):
        yield
        return

    if not any(_is_autocompile_sac_mode(mode) for mode in modes):
        yield
        return

    with _disable_current_modes() as disabled_modes, contextlib.ExitStack() as stack:
        for mode in reversed(disabled_modes):
            if not _is_autocompile_sac_mode(mode):
                stack.enter_context(mode)
        yield


def _is_dynamo_graph_break(gm: torch.fx.GraphModule) -> bool:
    reason = getattr(gm, "compile_subgraph_reason", None)
    if reason is None:
        return False
    return getattr(reason, "graph_break", False) is True


def _is_trivial_graph(gm: torch.fx.GraphModule) -> bool:
    """Check if graph has no computation (only placeholders, output, get_attr).

    Graph breaks (e.g. from dist.all_gather_object using pickle) produce trivial
    passthrough subgraphs that don't need NEFF compilation.
    """
    return all(node.op in ("placeholder", "output", "get_attr") for node in gm.graph.nodes)


def _is_compile_only() -> bool:
    """Compile-only mode: generate the NEFF but do not execute it on device.

    Used to compile (incl. backward graphs) on a host with no usable Neuron
    device — e.g. compiler CI sweeps. The NEFF is still produced by
    CompileGraph.compile(); only the device execution is skipped.

    Reads the single C++ flag (set by `torch_neuronx.compiler.compile_only()` /
    `TORCH_NEURONX_COMPILE_ONLY`) so allocation (empty_strided_neuron), the
    device guard, and this execution skip all agree.
    """
    import torch_neuronx

    return torch_neuronx._C._is_compile_only()


class _StableHLOError(Exception):
    """Raised by _lower_fx_to_neff when FX -> StableHLO conversion fails, so the
    caller (the dynamo backend) can fall back to running the original graph. The
    standalone compile API lets it propagate."""

    def __init__(self, original, lowering):
        super().__init__(str(original))
        self.original = original
        self.lowering = lowering  # _Lowering with the analysis info gathered so far


@dataclass
class _Lowering:
    """Result of the shared FX -> StableHLO -> NEFF core (`_lower_fx_to_neff`).

    This is the single lowering path used by BOTH `torch.compile(backend="neuron")`
    (via `neuron_backend_fx_compiler`) and the public compile API
    (`torch_neuronx.compiler.fx_to_neff`), so the two cannot drift: same FX passes,
    same trace, same compile, same cache key.
    """

    gm: torch.fx.GraphModule  # post-FX-pass graph
    # A live MLIR module from a fresh lowering, or serialized MLIR bytes when
    # resumed from the StableHLO cache.
    stablehlo_mlir: object
    io_spec: object
    cast_spec: object
    cache_key: str
    graph_name: str
    segment_id: str
    has_collectives: bool
    # post-AOT analysis (needed to build the executor / fallback)
    # Ordered per-pass IO mutations; alias_buffer_map derived from these.
    pass_mutations: list[PassMutations] = field(default_factory=list)
    alias_buffer_map: dict[int, int] = field(default_factory=dict)


def _make_graph_name(model_name: str) -> str:
    """Per-run graph name: `{model_name}_{compile_id}` from the AOT graph name."""
    aot_name = get_aot_graph_name()  # e.g., "model__0_inference_0"
    compile_id = aot_name.split("__", 1)[-1] if "__" in aot_name else aot_name
    return f"{model_name}_{compile_id}"


def _lower_fx_to_stablehlo(
    gm: torch.fx.GraphModule,
    example_inputs,
    job,
    *,
    model_name: str,
    pre_processing_analysis_info=None,
    run_fx_passes: bool = True,
    options=None,
) -> _Lowering:
    """Front half of the shared lowering core: FX passes -> FX→StableHLO.

    Runs `post_aot_processing` (the FX passes) and `_compile_fx_to_stablehlo`,
    returning a `_Lowering` with `stablehlo_mlir`/`io_spec` populated and
    `cache_key=""` (the NEFF stage has not run yet). This is the ONE place the FX
    passes run, so every consumer — `backend="neuron"`, `fx_to_neff`, and
    `fx_to_stablehlo` — sees the identical transformed graph. Raises
    `_StableHLOError` if FX→StableHLO conversion fails.

    With `run_fx_passes=False` the graph is taken as already post-pass; the
    alias/None bookkeeping is read from `pre_processing_analysis_info` instead
    of being recomputed (absent keys mean none).
    """
    analysis_info = pre_processing_analysis_info or {}
    pre_aot_pass_mutations = analysis_info.get("pre_aot_pass_mutations")
    if run_fx_passes:
        logger.debug("Applying FX graph transformations...")
        with dynamo_timed("torch_neuronx_fx_passes"):
            post_aot_processing_result = post_aot_processing(
                gm,
                job=job,
                model_name=model_name,
                pre_aot_pass_mutations=pre_aot_pass_mutations,
            )
        gm = post_aot_processing_result.graph_module
        pass_analysis = post_aot_processing_result.analysis
    else:
        logger.debug("Skipping FX graph transformations (run_fx_passes=False)")
        # Caller-supplied analysis stands in for the pass outputs.
        pass_analysis = dict(analysis_info)
        # Public contract: client backends supply aliasing as "mutation_info"
        # (an AliasingInfo). Translate it into the unified pass_mutations form
        # so it reaches alias_buffer_map / io_spec.aliases.
        supplied = pass_analysis.get("mutation_info")
        if supplied is not None and supplied.aliases and not pass_analysis.get("pass_mutations"):
            pass_analysis["pass_mutations"] = [
                PassMutations(
                    pass_name="caller_supplied_mutation_info",
                    output_mutations=[
                        IOMutation(
                            kind=MutationKind.ALIAS,
                            indices=[a.output_index for a in supplied.aliases],
                            alias_input_indices=[a.parameter_number for a in supplied.aliases],
                            is_mutation=True,
                            pass_name="caller_supplied_mutation_info",
                        )
                    ],
                )
            ]

    # Per-pass IO mutations; alias_buffer_map is the NEFF output->input hint.
    pass_mutations = pass_analysis.get("pass_mutations", [])
    alias_buffer_map = get_alias_output_buffer_map(pass_mutations)

    graph_name = _make_graph_name(model_name)
    segment_id = job.timestamp

    has_collectives = _has_collectives_in_graph(gm)
    logger.debug(f"Graph contains collectives: {has_collectives}")

    analysis = _Lowering(
        gm=gm,
        stablehlo_mlir=None,
        io_spec=None,
        cast_spec=None,
        cache_key="",
        graph_name=graph_name,
        segment_id=segment_id,
        has_collectives=has_collectives,
        pass_mutations=pass_mutations,
        alias_buffer_map=alias_buffer_map,
    )

    # WLO: when weight_layout_optimization is enabled, mark the static weight
    # inputs the compiler may transpose.
    static_input_indices = _wlo_static_input_indices(
        options, example_inputs, get_aot_graph_name(), pass_mutations, alias_buffer_map
    )

    try:
        stablehlo_mlir, _, io_spec, cast_spec = _compile_fx_to_stablehlo(
            gm,
            example_inputs,
            model_name,
            segment_id,
            job,
            alias_buffer_map,
            static_input_indices=static_input_indices,
        )
    except Exception as stablehlo_err:
        raise _StableHLOError(stablehlo_err, analysis) from stablehlo_err

    analysis.stablehlo_mlir = stablehlo_mlir
    analysis.io_spec = io_spec
    analysis.cast_spec = cast_spec
    return analysis


def _lower_stablehlo_to_neff(analysis: _Lowering, *, model_name: str, options=None) -> _Lowering:
    """Back half of the shared lowering core: StableHLO -> NEFF.

    Consumes the `_Lowering` produced by `_lower_fx_to_stablehlo` (with
    `stablehlo_mlir` populated), runs `CompileGraph.compile()`, and sets
    `cache_key` in place. Shared by `backend="neuron"`, `fx_to_neff`, and
    `stablehlo_to_neff` so they submit the identical StableHLO to neuronx-cc.

    `stablehlo_mlir` may be a live module or the serialized bytes a cache hit
    carries; `CompileGraph` accepts either.
    """
    use_tensorizer_backend = options.get("use_tensorizer_backend", False) if options else False
    compile_graph = CompileGraph(
        analysis.stablehlo_mlir,
        model_name,
        analysis.segment_id,
        analysis.graph_name,
        analysis.has_collectives,
        use_tensorizer=use_tensorizer_backend,
    )
    analysis.cache_key = compile_graph.compile()
    # WLO weight transforms are not recovered here: for a WLO graph compile() submits async and
    # returns a "|wlo" cache_key; the wrapped_neff.hlo sidecar is fetched + parsed lazily when the
    # executor is first built (cache_artifact.NeuronOutputCode._get_executor).
    return analysis


def _lower_fx_to_neff(
    gm: torch.fx.GraphModule,
    example_inputs,
    job,
    *,
    model_name: str,
    pre_processing_analysis_info=None,
    options=None,
    run_fx_passes: bool = True,
) -> _Lowering:
    """Shared lowering core: FX passes -> FX→StableHLO -> StableHLO→NEFF.

    The ONE path both consumers go through. `neuron_backend_fx_compiler` calls
    this and then wraps the result in an executor; `fx_to_neff` calls it and
    returns the artifact. Composed from `_lower_fx_to_stablehlo` +
    `_lower_stablehlo_to_neff`, the same two halves `fx_to_stablehlo` /
    `stablehlo_to_neff` use — so all entry points share one implementation and
    produce the same NEFF (same passes, same cache key). Raises `_StableHLOError`
    if FX→StableHLO conversion fails.

    The front half goes through `_lower_fx_to_stablehlo_cached`: with
    `TORCH_NEURONX_HLO_COMPILE_CACHE=1` a cache hit skips FX→StableHLO and resumes
    directly at `_lower_stablehlo_to_neff`.
    """
    analysis = _lower_fx_to_stablehlo_cached(
        gm,
        example_inputs,
        job,
        model_name=model_name,
        pre_processing_analysis_info=pre_processing_analysis_info,
        run_fx_passes=run_fx_passes,
        use_tensorizer=(options or {}).get("use_tensorizer_backend", False),
    )
    return _lower_stablehlo_to_neff(analysis, model_name=model_name, options=options)


def _pass_mutations_fingerprint(pass_mutations) -> str:
    """Stable text for a caller-supplied `pass_mutations` list.

    Order is preserved rather than sorted: `get_alias_output_buffer_map` translates
    each alias through the mutations that follow it, so the same entries in a
    different order can yield a different alias map. ADD/REMOVE entries are included
    for the same reason -- they shift the indices an alias resolves to.
    """
    entries = []
    for pm in pass_mutations or ():
        for direction, mutations in (
            ("in", getattr(pm, "input_mutations", None) or ()),
            ("out", getattr(pm, "output_mutations", None) or ()),
        ):
            for m in mutations:
                kind = getattr(getattr(m, "kind", None), "value", None)
                entries.append(
                    (
                        direction,
                        kind,
                        tuple(getattr(m, "indices", None) or ()),
                        tuple(getattr(m, "alias_input_indices", None) or ()),
                        getattr(m, "is_mutation", None),
                        getattr(m, "transform_key", None),
                    )
                )
    return repr(entries)


def _alias_fingerprint(pre_processing_analysis_info) -> str:
    """Stable text for the aliasing a caller supplies via analysis info.

    Each key is serialized by its declared shape; an unrecognized value contributes
    only its type name.
    """
    # pre_aot_pass_mutations is omitted: it is derived from gm by AliasingAnalysis,
    # so the graph hash already discriminates it.
    info = pre_processing_analysis_info or {}
    parts = []
    for name in (
        "aliasing_analysis",
        "mutation_info",
        "pass_mutations",
    ):
        value = info.get(name)
        if value is None:
            continue
        if name == "pass_mutations":
            parts.append(f"{name}={_pass_mutations_fingerprint(value)}")
        elif (aliases := getattr(value, "aliases", None)) is not None:
            pairs = sorted(
                (getattr(a, "parameter_number", None), getattr(a, "output_index", None))
                for a in aliases
            )
            parts.append(f"{name}={pairs}")
        else:
            # Unknown shape: contribute the type, not repr's memory address.
            parts.append(f"{name}=<{type(value).__name__}>")
    return "|".join(parts)


def _lower_fx_to_stablehlo_cached(
    gm: torch.fx.GraphModule,
    example_inputs,
    job,
    *,
    model_name: str,
    pre_processing_analysis_info=None,
    run_fx_passes: bool = True,
    use_tensorizer: bool = False,
) -> _Lowering:
    """`_lower_fx_to_stablehlo` with the FX-keyed StableHLO cache in front.

    Returns an uncached lowering when the cache is off, the tensorizer is in use, or
    the key cannot be computed.
    On a cache hit the result's `stablehlo_mlir` holds serialized bytes rather than a live module.
    On a miss, a per-key file lock lets one rank lower and store while the others wait
    and resume from the stored entry, so co-launched ranks do not all repeat the work.
    """
    from torch_neuronx.python_ops.compilation import hlo_cache

    def lower():
        """Lower the captured FX graph to StableHLO without using the cache.

        Returns:
            _Lowering: Fresh StableHLO lowering and its analysis metadata.
        """
        return _lower_fx_to_stablehlo(
            gm,
            example_inputs,
            job,
            model_name=model_name,
            pre_processing_analysis_info=pre_processing_analysis_info,
            run_fx_passes=run_fx_passes,
        )

    # Caching disabled
    if not hlo_cache.cache_enabled(compile_mode=True):
        return lower()

    # TODO: support caching the tensorizer path.
    if use_tensorizer:
        return lower()

    try:
        key = hlo_cache.compute_fx_cache_key(
            gm,
            run_fx_passes=run_fx_passes,
            alias_fingerprint=_alias_fingerprint(pre_processing_analysis_info),
            use_tensorizer=use_tensorizer,
        )
    except Exception as exc:
        logger.warning("StableHLO cache key failed (%s); lowering uncached", exc)
        return lower()

    def resume(cached: hlo_cache.CachedHLO) -> _Lowering:
        """Reconstruct lowering metadata from a cached StableHLO entry.

        Args:
            cached (hlo_cache.CachedHLO): Cached StableHLO and analysis metadata.

        Returns:
            _Lowering: Lowering state ready for NEFF compilation.
        """
        return _Lowering(
            gm=None,
            stablehlo_mlir=cached.hlo_bytes,
            io_spec=cached.io_spec,
            cast_spec=cached.cast_spec,
            cache_key="",
            graph_name=_make_graph_name(model_name),
            segment_id=job.timestamp,
            has_collectives=cached.has_collectives,
            alias_buffer_map=cached.alias_buffer_map,
            pass_mutations=cached.pass_mutations,
        )

    def lower_and_store() -> _Lowering:
        """Lower the captured graph and attempt to cache its StableHLO form.

        Returns:
            _Lowering: Fresh lowering state, even if the cache write fails.

        Note:
            Cache serialization or storage failures are logged and do not fail lowering.
        """
        analysis = lower()
        try:
            hlo_bytes = CompileGraph(
                analysis.stablehlo_mlir,
                model_name,
                analysis.segment_id,
                analysis.graph_name,
                analysis.has_collectives,
            ).serialize()
            hlo_cache.put(
                key,
                hlo_cache.CachedHLO(
                    hlo_bytes=hlo_bytes,
                    io_spec=analysis.io_spec,
                    cast_spec=analysis.cast_spec,
                    has_collectives=analysis.has_collectives,
                    alias_buffer_map=analysis.alias_buffer_map,
                    pass_mutations=analysis.pass_mutations,
                ),
                compile_mode=True,
            )
        except Exception as exc:
            logger.warning("StableHLO cache store failed (%s)", exc)
        return analysis

    cached = hlo_cache.get(key, compile_mode=True)
    if cached is not None:
        logger.info("StableHLO cache HIT (key=%s); skipping FX->StableHLO", key[:12])
        return resume(cached)

    # Miss. Serialize lowering across ranks so only one does the FX->StableHLO work
    # for a given key; the rest wait on the lock and resume from the stored entry.
    # Mirrors the eager path (torch_mlir/kernel.py) and nki_kernel.py.
    lock_path = os.path.join(
        hlo_cache.CACHE_DIR, hashlib.sha256(key.encode()).hexdigest() + ".fx.lock"
    )
    with FileLock(lock_path, timeout=get_lock_timeout()):
        # Another rank may have stored the entry while we waited for the lock.
        cached = hlo_cache.get(key, compile_mode=True)
        if cached is not None:
            logger.info("StableHLO cache HIT after lock (key=%s)", key[:12])
            return resume(cached)
        return lower_and_store()


def _make_fallback_fn(
    gm,
    pass_mutations: list[PassMutations],
):
    """Create a boxed fallback function that runs the original GraphModule."""

    def fallback_fn(inputs):
        """Run the captured graph while preserving input and output mutations.

        Args:
            inputs (Sequence): Boxed graph inputs.

        Returns:
            tuple: Graph outputs after fallback mutation handling.
        """
        original_inputs = list(inputs)
        mutated_inputs = apply_input_mutations(original_inputs, pass_mutations, "cpu")
        result = gm.forward(*mutated_inputs)
        outputs = list(result) if isinstance(result, tuple) else [result]
        for output_index, input_index in get_mutation_copy_back_inputs(pass_mutations).items():
            original_inputs[input_index].copy_(outputs[output_index])
        outputs = apply_output_mutations(outputs, pass_mutations, "cpu", original_inputs)
        return tuple(outputs)

    fallback_fn._boxed_call = True
    return fallback_fn


def _filter_wlo_indices(static_input_indices, example_inputs):
    """Keep only the floating-point weight inputs (the ones the compiler transposes)."""
    if not static_input_indices:
        return None

    filtered = [
        i
        for i in static_input_indices
        if isinstance(example_inputs[i], torch.Tensor) and example_inputs[i].dtype.is_floating_point
    ]
    dropped = set(static_input_indices) - set(filtered)
    if dropped:
        logger.debug(
            f"WLO: excluding non-floating-point inputs {sorted(dropped)} from weight "
            "layout optimization (only floating-point weights are transposed)."
        )
    return filtered or None


def _wlo_static_input_indices(options, example_inputs, aot_name, pass_mutations, alias_buffer_map):
    """Compute the NEFF-space static-weight indices eligible for WLO, or None if disabled.

    WLO is opt-in (options["weight_layout_optimization"]) and applies to inference and
    forward graphs (not backward/joint).
    """
    wlo_enabled = options is not None and options.get("weight_layout_optimization", False)
    if not wlo_enabled:
        return None

    graph_type = aot_name.rsplit("_", 1)[0].rsplit("_", 1)[-1] if "__" in aot_name else ""
    if graph_type not in ("inference", "forward"):
        logger.warning(
            f"WLO: weight_layout_optimization=True but graph_type='{graph_type}' "
            f"(name={aot_name}); WLO applies to inference and forward graphs only and is "
            "being SKIPPED (it does not apply to backward/joint graphs)."
        )
        return None

    ctx = TracingContext.try_get()
    fw_metadata = ctx.fw_metadata if ctx else None
    static_input_indices = (
        list(fw_metadata.static_input_indices)
        if fw_metadata and fw_metadata.static_input_indices
        else None
    )
    caller_indices = _filter_wlo_indices(static_input_indices, example_inputs)
    if not caller_indices:
        return None

    neff_indices = translate_input_indices_to_neff(caller_indices, pass_mutations)
    removed = len(caller_indices) - len(neff_indices)
    if removed:
        logger.debug(f"WLO: {removed} weight(s) were removed by FX passes and dropped from WLO.")

    # A relaid-out buffer holds permuted bytes under the caller's shape, so it can be neither
    # end of an alias. Static and aliased is a real combination: a cache registered as a
    # module buffer is both.
    aliased = set((alias_buffer_map or {}).values())
    eligible = [i for i in neff_indices if i not in aliased]
    if len(eligible) != len(neff_indices):
        logger.debug(
            f"WLO: {len(neff_indices) - len(eligible)} weight(s) are alias targets and were "
            "dropped from WLO; their layout cannot be permuted."
        )
    return eligible or None


def neuron_backend_fx_compiler(
    gm: torch.fx.GraphModule,
    example_inputs,
    pre_processing_analysis_info=None,
    options=None,
    *,
    autocompile_attempt: _AutoCompileBackendAttempt | None = None,
    autocompile_phase: str = "forward",
):
    """Compile an FX graph to a NEFF-backed boxed execution function.

    Args:
        gm (torch.fx.GraphModule): AOTAutograd FX graph containing ATen operations.
        example_inputs (Sequence): Example inputs used for shape inference.
        pre_processing_analysis_info (dict, optional): Preprocessing results. Defaults to ``None``.
        options (dict, optional): ``torch.compile`` options. Defaults to ``None``.
        autocompile_attempt (_AutoCompileBackendAttempt | None, optional): Attempt tracker.
            Defaults to ``None``.
        autocompile_phase (str, optional): Segment phase. Defaults to ``"forward"``.

    Returns:
        Callable: Boxed NEFF executor, debug executor, compile-only handle, or
            original-graph fallback.

    Raises:
        RuntimeError: If unexpected compilation or executor setup fails.

    Note:
        The pipeline lowers FX to StableHLO, compiles StableHLO to NEFF, and wraps execution.
    """
    reservation_generation = None
    if autocompile_attempt is not None:
        reservation_generation = autocompile_attempt.begin_segment(autocompile_phase)
    if autocompile_attempt is not None and reservation_generation is None:
        logger.debug(
            "AutoCompile guard miss uses the captured FX graph for phase=%s",
            autocompile_phase,
        )
        return make_boxed_func(gm.forward)

    if autocompile_attempt is not None and _is_dynamo_graph_break(gm):
        logger.debug("AutoCompile rejected a graph-break segment")
        _record_autocompile_fallback(autocompile_attempt, autocompile_phase, "graph_break")
        return make_boxed_func(gm.forward)

    # Short-circuit trivial graphs produced by graph breaks. When dynamo can't trace
    # an op (e.g. dist.all_gather_object uses pickle), it breaks the graph and may
    # produce empty passthrough subgraphs that have no real computation.
    if _is_trivial_graph(gm):
        logger.debug("Trivial graph detected (graph break artifact), using passthrough")
        _record_autocompile_fallback(autocompile_attempt, autocompile_phase, "trivial_graph")
        return make_boxed_func(gm.forward)

    from torch_neuronx.neuron_dynamo_backend.fx.passes.complex_decomposition import (
        maybe_decompose_complex,
    )

    gm = maybe_decompose_complex(gm, example_inputs)
    if (
        autocompile_attempt is not None
        and autocompile_phase == "forward"
        and _count_compute_nodes(gm) < autocompile_attempt.minimum_forward_graph_node_count
    ):
        logger.debug("AutoCompile rejected an unprofitable forward graph")
        _record_autocompile_fallback(
            autocompile_attempt,
            autocompile_phase,
            "unprofitable_graph",
        )
        return make_boxed_func(gm.forward)

    model_name = _resolve_model_name(options)
    pre_processing_analysis_info = (
        pre_processing_analysis_info if pre_processing_analysis_info else {}
    )

    retain_device = os.environ.get("TORCH_NEURONX_RETAIN_DEVICE_MODE", "0") == "1"

    # Create a CompilationJob for this compilation using context manager for automatic cleanup
    with compilation_job(op_name=model_name) as job:
        # Save the FX graph before any transformations
        job.save_fx_graph(model_name, gm)

        # Shared two-stage core (_lower_fx_to_stablehlo / _lower_stablehlo_to_neff).
        logger.debug(
            f"Example inputs: "
            f"{[inp.shape if hasattr(inp, 'shape') else inp for inp in example_inputs]}"
        )

        low = None
        try:
            # Two-stage lowering; debug paths exit between stages.
            # WLO (weight_layout_optimization) is handled inside
            # _lower_fx_to_stablehlo, which is why options is threaded through.
            try:
                low = _lower_fx_to_stablehlo(
                    gm,
                    example_inputs,
                    job,
                    model_name=model_name,
                    pre_processing_analysis_info=pre_processing_analysis_info,
                    options=options,
                )
            except _StableHLOError as err:
                if _fallback_execution_disabled(
                    options,
                ) and not _autocompile_compile_fallback_enabled(
                    options,
                    autocompile_attempt,
                ):
                    raise err.original from err
                logger.error(f"StableHLO conversion failed: {err.original}")
                logger.warning("Falling back to original GraphModule execution")
                _record_autocompile_fallback(
                    autocompile_attempt,
                    autocompile_phase,
                    "stablehlo_conversion",
                )
                a = err.lowering
                return _make_fallback_fn(a.gm, a.pass_mutations)

            # Compile-only: return CompilationArtifact handle (no execution).
            if os.environ.get("TORCH_NEURONX_DYNAMO_STABLEHLO_EXECUTOR") == "1":
                logger.debug("Bypassing compilation -> returning stablehlo executor")
                _record_autocompile_fallback(
                    autocompile_attempt,
                    autocompile_phase,
                    "stablehlo_interpreter",
                )
                executor = StableHLOInterpreterExecutor(
                    low.stablehlo_mlir, low.io_spec, low.cast_spec
                )

                def interp_wrapper(inputs):
                    """Run boxed inputs through the StableHLO reference interpreter.

                    Args:
                        inputs (Sequence): Boxed graph inputs.

                    Returns:
                        list: Interpreter outputs.
                    """
                    return executor(inputs)

                interp_wrapper._boxed_call = True
                return interp_wrapper

            if os.environ.get("TORCH_NEURONX_DYNAMO_HILO_INFERGOLDENS") == "1":
                logger.debug("Bypassing compilation -> returning hilo infergoldens executor")
                _record_autocompile_fallback(
                    autocompile_attempt,
                    autocompile_phase,
                    "hilo_infergoldens",
                )
                executor = HiloInferGoldensExecutor(low.stablehlo_mlir, low.io_spec, low.cast_spec)

                def hilo_ig_wrapper(inputs):
                    """Run boxed inputs through the HILO infer-goldens executor.

                    Args:
                        inputs (Sequence): Boxed graph inputs.

                    Returns:
                        list[torch.Tensor]: Golden output tensors in graph output order.
                    """
                    return executor(inputs)

                hilo_ig_wrapper._boxed_call = True
                return hilo_ig_wrapper

            # Stage 2: StableHLO -> NEFF (async submit; sets low.cache_key. For WLO the cache_key
            # carries a "|wlo" suffix and the sidecar/weight-transforms are fetched lazily in the
            # executor — see NeuronOutputCode._get_executor).
            low = _lower_stablehlo_to_neff(low, model_name=model_name, options=options)

            # Compile-only: return CompilationArtifact handle (no execution).
            if _is_compile_only():
                from torch_neuronx.compiler.api import _make_artifact

                artifact = _make_artifact(low, target=None, neff=None)
                return build_compile_only_handle(gm, low, artifact)

            # Serialize HLO bytes for embedding in NeuronOutputCode.
            # The embedded bytes make the Dynamo cache self-contained —
            # no dependency on the ephemeral /tmp/hlo_cache/ directory.
            bytecode_buf = io.BytesIO()
            low.stablehlo_mlir.operation.write_bytecode(bytecode_buf)
            hlo_bytes_raw = bytecode_buf.getvalue()

            # Record compilation metrics (stablehlo_to_neff timing populated lazily at sync)
            record_compilation(
                NeuronCompilationMetrics(
                    graph_id=low.cache_key,
                    graph_name=low.graph_name,
                    model_name=model_name,
                    has_collectives=low.has_collectives,
                    graph_node_count=(
                        _count_compute_nodes(low.gm)
                        if autocompile_attempt is not None
                        else len(low.gm.graph.nodes)
                    ),
                    timestamp=low.segment_id,
                )
            )

            from torch_neuronx.neuron_dynamo_backend.cache_artifact import NeuronOutputCode

            output_code = NeuronOutputCode(
                graph_name=model_name,
                cache_key=low.cache_key,
                hlo_bytes=hlo_bytes_raw,
                io_spec=low.io_spec,
                cast_spec=low.cast_spec,
                has_collectives=low.has_collectives,
                retain_device=retain_device,
                pass_mutations=low.pass_mutations,
                alias_buffer_map=low.alias_buffer_map,
                dump_dir=job.get_dump_dir(),
            )
            if autocompile_attempt is not None:
                segment_id = autocompile_attempt.register_executor(
                    autocompile_phase,
                    _count_compute_nodes(low.gm),
                    low.cache_key,
                    reservation_generation,
                    _extract_sac_policy_trace(low.gm),
                )
                if segment_id is None:
                    logger.debug(
                        "AutoCompile guard sealed during compilation for phase=%s",
                        autocompile_phase,
                    )
                    return _make_fallback_fn(low.gm, low.pass_mutations)
            else:
                segment_id = None

            # Wrap with CPU fallback for cold path debugging.
            # __wrapped__ lets AOTAutograd's unwrap_output_code find the
            # OutputCode for serialization.
            def fallback_wrapper(inputs):
                """Execute the NEFF and fall back to the original graph on failure.

                Args:
                    inputs (Sequence): Boxed graph inputs.

                Returns:
                    tuple: NEFF outputs or original-graph fallback outputs.

                Raises:
                    Exception: If execution fails while fallback execution is disabled.
                """
                if (
                    autocompile_attempt is not None
                    and autocompile_phase == "backward"
                    and torch.is_grad_enabled()
                ):
                    autocompile_attempt.record_compatibility(
                        autocompile_phase,
                        "higher_order_autograd",
                    )
                    return _make_fallback_fn(low.gm, low.pass_mutations)(inputs)
                if (
                    autocompile_attempt is not None
                    and autocompile_phase == "forward"
                    and not autocompile_attempt.autocompile_context_matches(
                        autocompile_attempt.active_autocompile_context_signature()
                        or _active_autocompile_context_signature()
                    )
                ):
                    autocompile_attempt.record_fallback(
                        autocompile_phase,
                        "autocompile_context_mismatch",
                    )
                    return _make_fallback_fn(low.gm, low.pass_mutations)(inputs)
                execution_started = False
                if autocompile_attempt is not None and segment_id is not None:
                    execution_started = autocompile_attempt.begin_executor_execution()
                    if not execution_started:
                        autocompile_attempt.wait_for_executor_drain()
                        return _make_fallback_fn(low.gm, low.pass_mutations)(inputs)
                try:
                    with _suspend_autocompile_sac_mode(autocompile_attempt is not None):
                        result = output_code(inputs)
                    if autocompile_attempt is not None and segment_id is not None:
                        autocompile_attempt.record_executor_success(segment_id)
                except Exception as e:
                    if autocompile_attempt is not None:
                        autocompile_attempt.record_failure(
                            autocompile_phase,
                            "runtime_execution",
                        )
                    if _fallback_execution_disabled(options):
                        raise
                    logger.error(f"Execution failed: {e}")
                    logger.warning("Falling back to original GraphModule execution")
                    return _make_fallback_fn(low.gm, low.pass_mutations)(inputs)
                finally:
                    if execution_started:
                        autocompile_attempt.end_executor_execution()
                return result

            fallback_wrapper._boxed_call = True
            fallback_wrapper.__wrapped__ = output_code
            return fallback_wrapper

        except Exception as e:
            if (
                _autocompile_compile_fallback_enabled(options, autocompile_attempt)
                and low is not None
            ):
                logger.warning(
                    "Neuron compilation failed; AutoCompile is using the captured FX graph: %s",
                    e,
                )
                _record_autocompile_fallback(
                    autocompile_attempt,
                    autocompile_phase,
                    "neff_setup",
                )
                return _make_fallback_fn(low.gm, low.pass_mutations)
            raise RuntimeError(
                f"Neuron backend NEFF execution setup failed with unexpected error: {e}"
            ) from e


def make_compiler(
    analysis_info: dict,
    options=None,
    *,
    autocompile_attempt: _AutoCompileBackendAttempt | None = None,
    autocompile_phase: str = "forward",
) -> Callable[[torch.fx.GraphModule, list[torch.Tensor]], any]:
    """Create an FX compiler bound to preprocessing analysis and options.

    Args:
        analysis_info (dict): FX graph preprocessing and aliasing analysis.
        options (dict, optional): ``torch.compile`` options. Defaults to ``None``.
        autocompile_attempt (_AutoCompileBackendAttempt | None, optional): Attempt tracker.
            Defaults to ``None``.
        autocompile_phase (str, optional): Segment phase. Defaults to ``"forward"``.

    Returns:
        Callable: Compiler that invokes the Dynamo backend with the bound values.
    """

    def wrapper(gm, example_inputs):
        """Compile an FX graph with the bound analysis and options.

        Args:
            gm (torch.fx.GraphModule): FX graph to compile.
            example_inputs (Sequence): Example inputs used for shape inference.

        Returns:
            Callable: Boxed compiled execution function.
        """
        return neuron_backend_fx_compiler(
            gm,
            example_inputs,
            analysis_info,
            options,
            autocompile_attempt=autocompile_attempt,
            autocompile_phase=autocompile_phase,
        )

    return wrapper


@contextlib.contextmanager
def _preserve_act_input_grad():
    """Preserve requires_grad through the eager->compile boundary for
    grad-requiring top-level AsyncCollectiveTensor inputs.

    AOTAutograd's process_inputs resolves an ACT input via trigger_wait(), which
    returns the grad-less inner .elem; the fake built from it reads
    requires_grad=False, so AOTAutograd builds a non-differentiable graph and
    silently drops the gradient to the upstream eager producer (e.g. a
    RowwiseParallel nn.Embedding feeding a torch.compile'd TP block). Record
    grad-requiring top-level ACT inputs before the unwrap and re-sync
    requires_grad onto their fakes afterward. Scoped to one Neuron compile;
    process_inputs is restored on exit. Fail-open on any AOTAutograd/collectives
    API drift so a torch version change can never break the compile path.

    This is a temporary bridge. Remove it once the fix lands upstream
    (pytorch AsyncCollectiveTensor.trigger_wait, tracked in pytorch#191113) and
    the pinned torch version includes it.
    """
    import torch._functorch.aot_autograd as _aot_autograd

    with _PRESERVE_ACT_INPUT_GRAD_LOCK:
        if not hasattr(_aot_autograd, "process_inputs"):
            yield
            return

        original = _aot_autograd.process_inputs

        def wrapped(flat_args, *args, **kwargs):
            """Preserve gradient requirements while processing AOTAutograd inputs.

            Args:
                flat_args (Sequence): Flattened inputs passed to AOTAutograd.
                *args (Any): Additional positional arguments forwarded to ``process_inputs``.
                **kwargs (Any): Keyword arguments forwarded to ``process_inputs``.

            Returns:
                object: Original ``process_inputs`` result.

            Note:
                Fail-open handling restores ``requires_grad`` on fake tensors corresponding to
                gradient-bearing top-level asynchronous collective inputs.
            """
            try:
                from torch.distributed._functional_collectives import AsyncCollectiveTensor

                grad_act_indices = [
                    index
                    for index, value in enumerate(flat_args)
                    if isinstance(value, AsyncCollectiveTensor) and value.requires_grad
                ]
            except Exception:
                grad_act_indices = []
            result = original(flat_args, *args, **kwargs)
            try:
                fake_flat_args = result[0] if isinstance(result, tuple) else result
                for index in grad_act_indices:
                    fake_arg = fake_flat_args[index]
                    if isinstance(fake_arg, torch.Tensor) and not fake_arg.requires_grad:
                        fake_arg.requires_grad_(True)
            except Exception:
                return result
            return result

        _aot_autograd.process_inputs = wrapped
        try:
            yield
        finally:
            if _aot_autograd.process_inputs is wrapped:
                _aot_autograd.process_inputs = original


def _autocompile_partition(
    attempt: _AutoCompileBackendAttempt,
    joint_module: torch.fx.GraphModule,
    joint_inputs,
    *,
    num_fwd_outputs: int,
    static_lifetime_input_indices: list[int] | None = None,
):
    from torch._functorch.partitioners import classify_nodes, default_partition

    policy_fn, _signature = attempt.take_pending_autocompile_context()
    if policy_fn is not None:
        node_info = classify_nodes(
            joint_module,
            static_lifetime_input_indices or [],
            num_fwd_outputs,
        )
        forward_nodes = [
            node
            for node in joint_module.graph.nodes
            if node.op == "call_function" and node_info.is_required_fw(node)
        ]
        _apply_sac_policy_to_nodes(forward_nodes, policy_fn)
    return default_partition(
        joint_module,
        joint_inputs,
        num_fwd_outputs=num_fwd_outputs,
        static_lifetime_input_indices=static_lifetime_input_indices,
    )


def _create_neuron_backend(
    autocompile_attempt: _AutoCompileBackendAttempt | None = None,
):
    """Create a neuron backend for torch.compile.

    Returns:
        Callable: Neuron backend function for torch.compile.

    Example:
        >>> compiled_model = torch.compile(model, backend="neuron")
    """

    # TODO (NF-24): add explicit bw_compiler
    decomposition_table = get_compile_decomposition_table()

    def neuron_backend_wrapper(
        gm: torch.fx.GraphModule, example_inputs, *, options=None, mode="default"
    ):
        """Compile an FX graph through the Neuron AOTAutograd backend.

        Args:
            gm (torch.fx.GraphModule): FX graph module to preprocess and compile.
            example_inputs (Sequence): Example inputs supplied to AOTAutograd.
            options (dict | None, optional): Backend options. Defaults to ``None``.
            mode (str, optional): Requested compile mode; unsupported values use
                ``"default"``. Defaults to ``"default"``.

        Returns:
            Callable: Compiled callable or original graph forward function after
                AutoCompile fallback.

        Raises:
            Exception: If AOT setup fails and AutoCompile fallback is disabled.
        """
        # There are 4 torch.compile mode: "default", "reduce-overhead",
        #   "max-autotune" or "max-autotune-no-cudagraphs"
        # Currently, we only support mode "default".
        if mode != "default":
            logging.warning(
                f"Neuron backend does not support mode='{mode}', falling back to default mode."
            )
            mode = "default"

        # Force dynamic=False for neuron backend
        if options is None:
            options = {}
        options.setdefault("dynamic", False)

        if autocompile_attempt is not None and _is_dynamo_graph_break(gm):
            logger.debug("AutoCompile rejected a graph-break segment before AOT")
            _record_autocompile_fallback(autocompile_attempt, "forward", "graph_break")
            return gm.forward

        fallback_gm = gm
        try:
            gm, analysis_results = preprocess_graph(gm)

            fw_compiler = make_compiler(
                analysis_results,
                options=options,
                autocompile_attempt=autocompile_attempt,
                autocompile_phase="forward",
            )
            bw_compiler = make_compiler(
                {},
                options=options,
                autocompile_attempt=autocompile_attempt,
                autocompile_phase="backward",
            )
            if autocompile_attempt is None:
                # Explicit compilation can use the bundled AOT cache. AutoCompile
                # needs execution callbacks that cache rehydration cannot restore.
                from torch_neuronx.neuron_dynamo_backend.cache_artifact import NeuronOutputCode

                fw_compiler = SerializableAOTDispatchCompiler(NeuronOutputCode, fw_compiler)
                bw_compiler = SerializableAOTDispatchCompiler(NeuronOutputCode, bw_compiler)

            aot_kwargs = {
                "fw_compiler": fw_compiler,
                "bw_compiler": bw_compiler,
                "keep_inference_input_mutations": True,
                "decompositions": decomposition_table,
            }
            if autocompile_attempt is not None:
                aot_kwargs["partition_fn"] = functools.partial(
                    _autocompile_partition,
                    autocompile_attempt,
                )
            aot_backend = aot_autograd(**aot_kwargs)
            with _preserve_act_input_grad():
                return aot_backend(gm, example_inputs)
        except Exception as error:
            if not _autocompile_compile_fallback_enabled(options, autocompile_attempt):
                raise
            logger.warning(
                "AOT compilation failed; AutoCompile is using the captured FX graph: %s",
                error,
            )
            _record_autocompile_fallback(
                autocompile_attempt,
                "forward",
                "aot_setup",
            )
            return fallback_gm.forward

    return neuron_backend_wrapper


def _create_autocompile_backend(
    attempt: _AutoCompileBackendAttempt,
):
    return _create_neuron_backend(attempt)


def create_neuron_backend():
    """Create the Dynamo backend used by ``torch.compile`` for Neuron devices.

    Returns:
        Callable: Backend function that compiles FX graphs for Neuron devices.

    Example::

        >>> compiled_model = torch.compile(model, backend="neuron")  # doctest: +SKIP

    Note:
        Only ``mode="default"`` is supported, and dynamic shapes default to disabled.
    """
    return _create_neuron_backend()


# Create default backend instance with selective decomposition
neuron_backend = create_neuron_backend()


logger.debug("Neuron backend module successfully loaded")

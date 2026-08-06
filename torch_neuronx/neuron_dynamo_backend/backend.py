"""
Neuron backend implementation for torch.compile
"""

import contextlib
import io
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.utils import dynamo_timed
from torch._functorch._aot_autograd.logging_utils import get_aot_graph_name
from torch._functorch._aot_autograd.schemas import SerializableAOTDispatchCompiler

from torch_neuronx.kernels.cache_utils import CompilationJob, compilation_job
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
):
    """Convert an FX GraphModule to StableHLO MLIR.

    Args:
        gm (torch.fx.GraphModule): FX GraphModule with ATen operations.
        example_inputs: Example input tensors.
        model_name (str): Model name for logging and artifact naming.
        segment_id (str): Segment ID for statistics tracking.
        job (CompilationJob): Compilation job for artifact management.
        alias_buffer_map (dict[int, int] | None): NEFF output→input aliasing hints.

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
    )
    logger.debug("FX Graph to StableHLO conversion completed")

    # Save stablehlo to artifacts dir
    artifacts_stablehlo = job.save_stablehlo(model_name, stablehlo_mlir)

    return stablehlo_mlir, artifacts_stablehlo, io_spec, cast_spec


def _has_collectives_in_graph(gm: torch.fx.GraphModule) -> bool:
    """Detect if an FX graph contains collective operations.

    Scans the FX graph for torch._c10d_functional collective operations.

    Args:
        gm (torch.fx.GraphModule): FX GraphModule to analyze.

    Returns:
        bool: True if graph contains any collective operations.
    """
    collective_ops = {"all_reduce", "all_gather", "reduce_scatter", "all_to_all"}

    for node in gm.graph.nodes:
        if node.op == "call_function":
            func_name = (
                node.target.__name__ if hasattr(node.target, "__name__") else str(node.target)
            )
            if any(collective in func_name for collective in collective_ops):
                logger.info(f"Detected collective operation: {func_name}")
                return True

    return False


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


def _lower_fx_to_stablehlo(
    gm: torch.fx.GraphModule,
    example_inputs,
    job,
    *,
    model_name: str,
    pre_processing_analysis_info=None,
    run_fx_passes: bool = True,
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

    aot_name = get_aot_graph_name()  # e.g., "model__0_inference_0"
    compile_id = aot_name.split("__", 1)[-1] if "__" in aot_name else aot_name
    graph_name = f"{model_name}_{compile_id}"
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

    try:
        stablehlo_mlir, _, io_spec, cast_spec = _compile_fx_to_stablehlo(
            gm, example_inputs, model_name, segment_id, job, alias_buffer_map
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
    """
    analysis = _lower_fx_to_stablehlo(
        gm,
        example_inputs,
        job,
        model_name=model_name,
        pre_processing_analysis_info=pre_processing_analysis_info,
        run_fx_passes=run_fx_passes,
    )
    return _lower_stablehlo_to_neff(analysis, model_name=model_name, options=options)


def _make_fallback_fn(
    gm,
    pass_mutations: list[PassMutations],
):
    """Create a boxed fallback function that runs the original GraphModule."""

    def fallback_fn(inputs):
        mutated_inputs = apply_input_mutations(list(inputs), pass_mutations, "cpu")
        result = gm.forward(*mutated_inputs)
        outputs = list(result) if isinstance(result, tuple) else [result]
        outputs = apply_output_mutations(outputs, pass_mutations, "cpu", list(inputs))
        return tuple(outputs)

    fallback_fn._boxed_call = True
    return fallback_fn


def neuron_backend_fx_compiler(
    gm: torch.fx.GraphModule, example_inputs, pre_processing_analysis_info=None, options=None
):
    """Compile an FX GraphModule to NEFF for Neuron execution.

    Implements the compilation pipeline:
    1. FX Graph -> StableHLO conversion
    2. StableHLO -> NEFF compilation
    3. NEFF execution wrapper creation

    Args:
        gm (torch.fx.GraphModule): FX GraphModule with ATen operations from AOTAutograd.
        example_inputs: Example input tensors for shape inference.
        pre_processing_analysis_info (dict | None): Analysis results from preprocessing.
        options (dict | None): Compilation options from torch.compile.

    Returns:
        Callable: Boxed function for execution.

    Raises:
        RuntimeError: If compilation fails at any stage.
    """

    # Short-circuit trivial graphs produced by graph breaks. When dynamo can't trace
    # an op (e.g. dist.all_gather_object uses pickle), it breaks the graph and may
    # produce empty passthrough subgraphs that have no real computation.
    if _is_trivial_graph(gm):
        logger.debug("Trivial graph detected (graph break artifact), using passthrough")
        return make_boxed_func(gm.forward)

    from torch_neuronx.neuron_dynamo_backend.fx.passes.complex_decomposition import (
        maybe_decompose_complex,
    )

    gm = maybe_decompose_complex(gm, example_inputs)

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

        try:
            # Two-stage lowering; debug paths exit between stages.
            try:
                low = _lower_fx_to_stablehlo(
                    gm,
                    example_inputs,
                    job,
                    model_name=model_name,
                    pre_processing_analysis_info=pre_processing_analysis_info,
                )
            except _StableHLOError as err:
                if _getenv_bool("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", False):
                    raise err.original from err
                logger.error(f"StableHLO conversion failed: {err.original}")
                logger.warning("Falling back to original GraphModule execution")
                a = err.lowering
                return _make_fallback_fn(a.gm, a.pass_mutations)

            # Compile-only: return CompilationArtifact handle (no execution).
            if os.environ.get("TORCH_NEURONX_DYNAMO_STABLEHLO_EXECUTOR") == "1":
                logger.debug("Bypassing compilation -> returning stablehlo executor")
                executor = StableHLOInterpreterExecutor(
                    low.stablehlo_mlir, low.io_spec, low.cast_spec
                )

                def interp_wrapper(inputs):
                    return executor(inputs)

                interp_wrapper._boxed_call = True
                return interp_wrapper

            if os.environ.get("TORCH_NEURONX_DYNAMO_HILO_INFERGOLDENS") == "1":
                logger.debug("Bypassing compilation -> returning hilo infergoldens executor")
                executor = HiloInferGoldensExecutor(low.stablehlo_mlir, low.io_spec, low.cast_spec)

                def hilo_ig_wrapper(inputs):
                    return executor(inputs)

                hilo_ig_wrapper._boxed_call = True
                return hilo_ig_wrapper

            # Stage 2: StableHLO -> NEFF (async submit; sets low.cache_key).
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
                    graph_node_count=len(low.gm.graph.nodes),
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
            )

            # Wrap with CPU fallback for cold path debugging.
            # __wrapped__ lets AOTAutograd's unwrap_output_code find the
            # OutputCode for serialization.
            def fallback_wrapper(inputs):
                try:
                    return output_code(inputs)
                except Exception as e:
                    if _getenv_bool("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", False):
                        raise
                    logger.error(f"Execution failed: {e}")
                    logger.warning("Falling back to original GraphModule execution")
                    return _make_fallback_fn(low.gm, low.pass_mutations)(inputs)

            fallback_wrapper._boxed_call = True
            fallback_wrapper.__wrapped__ = output_code
            return fallback_wrapper

        except Exception as e:
            raise RuntimeError(
                f"Neuron backend NEFF execution setup failed with unexpected error: {e}"
            ) from e


def make_compiler(
    analysis_info: dict,
    options=None,
) -> Callable[[torch.fx.GraphModule, list[torch.Tensor]], any]:
    """Create a forward compiler wrapper with custom aliasing information.

    Args:
        analysis_info (dict): FX graph pre-processing info from aliasing analysis.
        options (dict | None): Compilation options from torch.compile.

    Returns:
        Callable: Wrapper function that invokes neuron_compiler with aliasing info.
    """

    def wrapper(gm, example_inputs):
        return neuron_backend_fx_compiler(gm, example_inputs, analysis_info, options)

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

    if not hasattr(_aot_autograd, "process_inputs"):
        yield
        return

    _original = _aot_autograd.process_inputs

    def _wrapped(flat_args, *args, **kwargs):
        try:
            from torch.distributed._functional_collectives import AsyncCollectiveTensor

            grad_act_indices = [
                i
                for i, a in enumerate(flat_args)
                if isinstance(a, AsyncCollectiveTensor) and a.requires_grad
            ]
        except Exception:
            grad_act_indices = []
        result = _original(flat_args, *args, **kwargs)
        try:
            fake_flat_args = result[0] if isinstance(result, tuple) else result
            for i in grad_act_indices:
                fi = fake_flat_args[i]
                if isinstance(fi, torch.Tensor) and not fi.requires_grad:
                    fi.requires_grad_(True)
        except Exception:
            return result  # fail-open on any shape/API drift
        return result

    _aot_autograd.process_inputs = _wrapped
    try:
        yield
    finally:
        _aot_autograd.process_inputs = _original


def create_neuron_backend():
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

        gm, analysis_results = preprocess_graph(gm)

        # Wrap in SerializableAOTDispatchCompiler so AOTAutograd's bundled
        # cache can serialize the NeuronOutputCode alongside its own metadata
        # (ViewAndMutationMeta, subclass info, etc.).  This enables
        # caching_precompile to skip both Dynamo and AOTAutograd on warm starts.
        from torch_neuronx.neuron_dynamo_backend.cache_artifact import NeuronOutputCode

        fw_compiler = SerializableAOTDispatchCompiler(
            NeuronOutputCode, make_compiler(analysis_results, options=options)
        )
        bw_compiler = SerializableAOTDispatchCompiler(
            NeuronOutputCode, make_compiler({}, options=options)
        )

        aot_backend = aot_autograd(
            # pass pre-AOT aliasing info to functionalized post-AOT FWD FX graph
            fw_compiler=fw_compiler,
            # functionalized post-AOT BWD FX graph does not contain aliasing hints
            bw_compiler=bw_compiler,
            keep_inference_input_mutations=True,  # Prevent Copy-back.
            decompositions=decomposition_table,
        )
        with _preserve_act_input_grad():
            return aot_backend(gm, example_inputs)

    return neuron_backend_wrapper


# Create default backend instance with selective decomposition
neuron_backend = create_neuron_backend()


logger.debug("Neuron backend module successfully loaded")

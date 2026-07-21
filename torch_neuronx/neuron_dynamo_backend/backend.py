"""
Neuron backend implementation for torch.compile
"""

import io
import logging
import os
from collections.abc import Callable

import torch
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.utils import dynamo_timed
from torch._functorch._aot_autograd.logging_utils import get_aot_graph_name
from torch._functorch._aot_autograd.schemas import SerializableAOTDispatchCompiler

from torch_neuronx.kernels.cache_utils import CompilationJob, compilation_job
from torch_neuronx.neuron_dynamo_backend.compile import CompileGraph

# Do not call get_model_name() directly in this file; use _resolve_model_name(options) instead.
from torch_neuronx.neuron_dynamo_backend.config import (
    get_model_name,
)
from torch_neuronx.neuron_dynamo_backend.decompositions import get_compile_decomposition_table
from torch_neuronx.neuron_dynamo_backend.fx.fx_transform import (
    convert_fx_to_stablehlo,
)
from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
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

    model_name = _resolve_model_name(options)
    pre_processing_analysis_info = (
        pre_processing_analysis_info if pre_processing_analysis_info else {}
    )

    # Create a CompilationJob for this compilation using context manager for automatic cleanup
    with compilation_job(op_name=model_name) as job:
        # Save the FX graph before any transformations
        job.save_fx_graph(model_name, gm)

        logger.debug("Applying FX graph transformations...")
        pre_aot_pass_mutations = pre_processing_analysis_info.get("pre_aot_pass_mutations")
        with dynamo_timed("torch_neuronx_fx_passes"):
            post_aot_processing_result = post_aot_processing(
                gm,
                job=job,
                model_name=model_name,
                pre_aot_pass_mutations=pre_aot_pass_mutations,
            )
        gm = post_aot_processing_result.graph_module
        pass_mutations = post_aot_processing_result.analysis.get("pass_mutations", [])
        alias_buffer_map = get_alias_output_buffer_map(pass_mutations)

        logger.debug(f"Graph nodes: {len(gm.graph.nodes)}")
        logger.debug(
            f"Example inputs: "
            f"{[inp.shape if hasattr(inp, 'shape') else inp for inp in example_inputs]}"
        )

        # Set retain device mode
        retain_device = os.environ.get("TORCH_NEURONX_RETAIN_DEVICE_MODE", "0") == "1"

        # Construct human-readable graph name from model_name and AOT compile ID
        # get_aot_graph_name() returns: model__{aot_id}_{graph_type}_{nth}
        # We replace generic "model" prefix with user's model_name
        aot_name = get_aot_graph_name()  # e.g., "model__0_inference_0"
        compile_id = aot_name.split("__", 1)[-1] if "__" in aot_name else aot_name
        graph_name = f"{model_name}_{compile_id}"  # e.g., "LlamaForCausalLM_0_inference_0"

        segment_id = job.timestamp  # Use job timestamp as segment ID

        logger.debug(f"Model name: {model_name}")
        logger.debug(f"Segment ID: {segment_id}")

        try:
            # Detect collectives early in the pipeline
            has_collectives = _has_collectives_in_graph(gm)
            logger.debug(f"Graph contains collectives: {has_collectives}")

            try:
                stablehlo_mlir, _, io_spec, cast_spec = _compile_fx_to_stablehlo(
                    gm,
                    example_inputs,
                    model_name,
                    segment_id,
                    job,
                    alias_buffer_map,
                )
            except Exception as stablehlo_err:
                if _getenv_bool("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", False):
                    raise
                logger.error(f"StableHLO conversion failed: {stablehlo_err}")
                logger.warning("Falling back to original GraphModule execution")
                return _make_fallback_fn(gm, pass_mutations)

            if os.environ.get("TORCH_NEURONX_DYNAMO_STABLEHLO_EXECUTOR") == "1":
                logger.debug("Bypassing compilation -> returning stablehlo executor")
                executor = StableHLOInterpreterExecutor(stablehlo_mlir, io_spec, cast_spec)

                def interp_wrapper(inputs):
                    return executor(inputs)

                interp_wrapper._boxed_call = True
                return interp_wrapper

            if os.environ.get("TORCH_NEURONX_DYNAMO_HILO_INFERGOLDENS") == "1":
                logger.debug("Bypassing compilation -> returning hilo infergoldens executor")
                executor = HiloInferGoldensExecutor(stablehlo_mlir, io_spec, cast_spec)

                def hilo_ig_wrapper(inputs):
                    return executor(inputs)

                hilo_ig_wrapper._boxed_call = True
                return hilo_ig_wrapper

            use_tensorizer_backend = (
                options.get("use_tensorizer_backend", False) if options else False
            )
            compile_graph = CompileGraph(
                stablehlo_mlir,
                model_name,
                segment_id,
                graph_name,
                has_collectives,
                use_tensorizer=use_tensorizer_backend,
            )
            final_cache_key = compile_graph.compile()

            # Serialize HLO bytes for embedding in NeuronOutputCode.
            # The embedded bytes make the Dynamo cache self-contained —
            # no dependency on the ephemeral /tmp/hlo_cache/ directory.
            bytecode_buf = io.BytesIO()
            stablehlo_mlir.operation.write_bytecode(bytecode_buf)
            hlo_bytes_raw = bytecode_buf.getvalue()

            # Record compilation metrics (stablehlo_to_neff timing populated lazily at sync)
            record_compilation(
                NeuronCompilationMetrics(
                    graph_id=final_cache_key,
                    graph_name=graph_name,
                    model_name=model_name,
                    has_collectives=has_collectives,
                    graph_node_count=len(gm.graph.nodes),
                    timestamp=segment_id,
                )
            )

            from torch_neuronx.neuron_dynamo_backend.cache_artifact import NeuronOutputCode

            output_code = NeuronOutputCode(
                graph_name=model_name,
                cache_key=final_cache_key,
                hlo_bytes=hlo_bytes_raw,
                io_spec=io_spec,
                cast_spec=cast_spec,
                has_collectives=has_collectives,
                retain_device=retain_device,
                pass_mutations=pass_mutations,
                alias_buffer_map=alias_buffer_map,
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
                    return _make_fallback_fn(gm, pass_mutations)(inputs)

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
        return aot_backend(gm, example_inputs)

    return neuron_backend_wrapper


# Create default backend instance with selective decomposition
neuron_backend = create_neuron_backend()


logger.debug("Neuron backend module successfully loaded")

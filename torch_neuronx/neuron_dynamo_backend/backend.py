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

# Import process group variable tracking (needed for registration side effect)
from torch_neuronx.kernels.cache_utils import CompilationJob, compilation_job
from torch_neuronx.neuron_dynamo_backend import process_group_variable
from torch_neuronx.neuron_dynamo_backend.compile import CompileGraph
from torch_neuronx.neuron_dynamo_backend.config import (
    get_model_name,
)
from torch_neuronx.neuron_dynamo_backend.decompositions import get_compile_decomposition_table
from torch_neuronx.neuron_dynamo_backend.fx.fx_transform import (
    convert_fx_to_stablehlo,
)
from torch_neuronx.neuron_dynamo_backend.fx.pipelines.post_aot_processing_pipeline import (
    post_aot_processing,
)
from torch_neuronx.neuron_dynamo_backend.fx.pipelines.preprocessing_pipeline import preprocess_graph
from torch_neuronx.neuron_dynamo_backend.metrics import (
    NeuronCompilationMetrics,
    record_compilation,
)
from torch_neuronx.neuron_dynamo_backend.settings import _getenv_bool
from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo
from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import (
    ShardInputInfo,
)

logger = logging.getLogger(__name__)

# Ensure process group registration happens
_ = process_group_variable


def _compile_fx_to_stablehlo(
    gm: torch.fx.GraphModule,
    example_inputs,
    model_name: str,
    segment_id: str,
    job: CompilationJob,
    aliasing_info: AliasingInfo = None,
    shard_input_info: ShardInputInfo | None = None,
):
    """Convert an FX GraphModule to StableHLO MLIR.

    Args:
        gm (torch.fx.GraphModule): FX GraphModule with ATen operations.
        example_inputs: Example input tensors.
        model_name (str): Model name for logging and artifact naming.
        segment_id (str): Segment ID for statistics tracking.
        job (CompilationJob): Compilation job for artifact management.
        aliasing_info (AliasingInfo): Input-output aliasing information.

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
        aliasing_info,
        job=job,
        shard_input_info=shard_input_info,
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

    logger.debug("Applying FX graph transformations...")
    with dynamo_timed("torch_neuronx_fx_passes"):
        post_aot_processing_result = post_aot_processing(gm)
    gm = post_aot_processing_result.graph_module
    mutation_info = post_aot_processing_result.analysis.get("mutation_info", None)
    none_output_info = post_aot_processing_result.analysis.get("none_output_info", None)
    shard_input_info = post_aot_processing_result.analysis.get("shard_input_info", None)
    removed_device_mesh_indices = post_aot_processing_result.analysis.get(
        "removed_device_mesh_indices", None
    )
    passthrough_info = post_aot_processing_result.analysis.get("passthrough_info", None)
    aliasing_info = (
        None
        if pre_processing_analysis_info is None
        else pre_processing_analysis_info.get("aliasing_analysis", None)
    )

    # Merge pass-through aliasing into aliasing_info for StableHLO input_output_alias.
    # Pass-throughs are inputs returned unchanged — the compiler treats these as no-op
    # when input_output_alias is set. Mutations are not merged here because they
    # require the compiler to execute the computation and write into the aliased buffer.
    if passthrough_info is not None and len(passthrough_info) > 0:
        if aliasing_info is None:
            aliasing_info = AliasingInfo()
        for alias in passthrough_info.aliases:
            aliasing_info.add(alias.parameter_number, alias.parameter_index, alias.output_index)

    logger.debug(f"Graph nodes: {len(gm.graph.nodes)}")
    logger.debug(
        f"Example inputs: {[inp.shape if hasattr(inp, 'shape') else inp for inp in example_inputs]}"
    )

    # Get model name from global variable
    model_name = get_model_name() if options is None else options.get("model_name", "model_default")

    # Set retain device mode
    retain_device = os.environ.get("TORCH_NEURONX_RETAIN_DEVICE_MODE", "0") == "1"

    # Construct human-readable graph name from model_name and AOT compile ID
    # get_aot_graph_name() returns: model__{aot_id}_{graph_type}_{nth}
    # We replace generic "model" prefix with user's model_name
    aot_name = get_aot_graph_name()  # e.g., "model__0_inference_0"
    compile_id = aot_name.split("__", 1)[-1] if "__" in aot_name else aot_name
    graph_name = f"{model_name}_{compile_id}"  # e.g., "LlamaForCausalLM_0_inference_0"

    # Create a CompilationJob for this compilation using context manager for automatic cleanup
    with compilation_job(op_name=model_name) as job:
        segment_id = job.timestamp  # Use job timestamp as segment ID

        logger.debug(f"Model name: {model_name}")
        logger.debug(f"Segment ID: {segment_id}")

        try:
            # Detect collectives early in the pipeline
            has_collectives = _has_collectives_in_graph(gm)
            logger.debug(f"Graph contains collectives: {has_collectives}")

            # Save the FX graph we received
            job.save_fx_graph(model_name, gm)

            stablehlo_mlir, _, io_spec, cast_spec = _compile_fx_to_stablehlo(
                gm,
                example_inputs,
                model_name,
                segment_id,
                job,
                aliasing_info,
                shard_input_info,
            )

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

            # Save HLO bytes to the persistent HLO cache so warm loads can
            # retrieve them and feed them back through compile_graph (which
            # will hit the NEFF persistent cache).
            hlo_cache_key = final_cache_key
            try:
                from torch_neuronx.python_ops.compilation.hlo_cache import put as hlo_cache_put

                bytecode_buf = io.BytesIO()
                stablehlo_mlir.operation.write_bytecode(bytecode_buf)
                hlo_cache_put(
                    hlo_cache_key,
                    bytecode_buf.getvalue(),
                    io_spec,
                    list(range(len(io_spec.inputs))),  # kept_input_indices: all
                    list(range(len(io_spec.outputs))),  # kept_output_indices: all
                    None,  # downcasted_outputs
                    None,  # original_outputs
                    {},  # passthrough_mappings
                )
            except Exception:
                logger.debug("Failed to save HLO to cache", exc_info=True)
                hlo_cache_key = ""

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
                hlo_cache_key=hlo_cache_key,
                io_spec=io_spec,
                cast_spec=cast_spec,
                has_collectives=has_collectives,
                retain_device=retain_device,
                mutation_info=mutation_info,
                none_output_info=none_output_info,
                removed_device_mesh_indices=removed_device_mesh_indices,
                passthrough_info=passthrough_info,
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
                    fallback_inputs = list(inputs)
                    if (
                        shard_input_info
                        and shard_input_info.new_input_count > shard_input_info.original_input_count
                    ):
                        fallback_inputs.append(
                            torch.tensor(shard_input_info.shard_index, dtype=torch.int32)
                        )
                    return gm.forward(*fallback_inputs)

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

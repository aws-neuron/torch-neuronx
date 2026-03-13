"""
Neuron backend implementation for torch.compile
"""

import logging
import os
from collections.abc import Callable

import torch
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.utils import dynamo_timed
from torch._functorch._aot_autograd.logging_utils import get_aot_graph_name

# Import process group variable tracking (needed for registration side effect)
from torch_neuronx.neuron_dynamo_backend import process_group_variable
from torch_neuronx.neuron_dynamo_backend.compile import CompileGraph
from torch_neuronx.neuron_dynamo_backend.config import (
    get_current_timestamp,
    get_fx_graph_path,
    get_model_name,
    get_stablehlo_path,
    managed_artifacts_directory,
    reset_timestamp,
)
from torch_neuronx.neuron_dynamo_backend.decompositions import get_compile_decomposition_table
from torch_neuronx.neuron_dynamo_backend.executor import Executor
from torch_neuronx.neuron_dynamo_backend.fx.fx_transform import (
    convert_fx_to_stablehlo,
    save_mlir_bytecode,
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
from torch_neuronx.neuron_dynamo_backend.utils.io_utils import save_fx_graph_txt
from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import RandomInputInfo

logger = logging.getLogger(__name__)

# Ensure process group registration happens
_ = process_group_variable


def _compile_fx_to_stablehlo(
    gm: torch.fx.GraphModule,
    example_inputs,
    model_name: str,
    segment_id: str,
    preserve_artifacts: bool,
    aliasing_info: AliasingInfo = None,
    random_input_info: RandomInputInfo = None,
):
    """Convert an FX GraphModule to StableHLO MLIR.

    Args:
        gm (torch.fx.GraphModule): FX GraphModule with ATen operations.
        example_inputs: Example input tensors.
        model_name (str): Model name for logging and artifact naming.
        segment_id (str): Segment ID for statistics tracking.
        preserve_artifacts (bool): Whether to save intermediate artifacts.
        aliasing_info (AliasingInfo): Input-output aliasing information.
        random_input_info: Optional metadata about random inputs added to the graph

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
        preserve_artifacts=preserve_artifacts,
        random_input_info=random_input_info,
    )
    logger.debug("FX Graph to StableHLO conversion completed")

    # Save stablehlo to artifacts dir
    artifacts_stablehlo = get_stablehlo_path(model_name)
    save_mlir_bytecode(stablehlo_mlir, artifacts_stablehlo)
    logger.debug(f"StableHLO saved to artifacts directory: {artifacts_stablehlo}")

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

    logger.debug("Applying FX graph transformations...")
    with dynamo_timed("torch_neuronx_fx_passes"):
        post_aot_processing_result = post_aot_processing(gm)
    gm = post_aot_processing_result.graph_module
    mutation_info = post_aot_processing_result.analysis.get("mutation_info", None)
    none_output_info = post_aot_processing_result.analysis.get("none_output_info", None)
    random_input_info = post_aot_processing_result.analysis.get("random_input_info", None)
    aliasing_info = (
        None
        if pre_processing_analysis_info is None
        else pre_processing_analysis_info.get("aliasing_analysis", None)
    )

    logger.debug(f"Graph nodes: {len(gm.graph.nodes)}")
    logger.debug(
        f"Example inputs: "
        f"{[inp.shape if hasattr(inp, 'shape') else inp for inp in example_inputs]}"
    )

    # Reset timestamp for new compilation (each graph segment gets unique timestamp)
    # TODO (NF-23): Improve graph unique ID generation logic (use unique graph/rank info)
    reset_timestamp()
    segment_id = get_current_timestamp()  # Use timestamp as segment ID

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

    logger.debug(f"Model name: {model_name}")
    logger.debug(f"Segment ID: {segment_id}")

    # Use managed artifacts directory context manager for all compilation artifacts
    with managed_artifacts_directory() as (_, preserve_artifacts):
        try:
            # Detect collectives early in the pipeline
            has_collectives = _has_collectives_in_graph(gm)
            logger.debug(f"Graph contains collectives: {has_collectives}")

            # Save the FX graph we received
            graph_path = get_fx_graph_path(model_name)
            save_fx_graph_txt(gm, graph_path)

            stablehlo_mlir, _, io_spec, cast_spec = _compile_fx_to_stablehlo(
                gm,
                example_inputs,
                model_name,
                segment_id,
                preserve_artifacts,
                aliasing_info,
                random_input_info,
            )
            compile_graph = CompileGraph(stablehlo_mlir, model_name, segment_id, has_collectives)
            final_cache_key = compile_graph.compile()

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

            executor = Executor(
                model_name,
                final_cache_key,
                io_spec,
                cast_spec,
                has_collectives,
                retain_device=retain_device,
                mutation_info=mutation_info,
                none_output_info=none_output_info,
            )

            # Create wrapper function that matches torch.compile expectations
            def execution_wrapper(*inputs):
                """Wrapper function for execution"""
                try:
                    return executor(*inputs)
                except Exception as e:
                    if _getenv_bool("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", False):
                        raise e from e
                    logger.error(f"Execution failed: {e}")
                    # Fallback to original execution for debugging
                    logger.warning("Falling back to original GraphModule execution")
                    return gm.forward(*inputs)

            return make_boxed_func(execution_wrapper)

        except Exception as e:
            raise RuntimeError(
                f"Neuron backend NEFF execution setup failed with unexpected error: {e}"
            ) from e


def make_fw_compiler(
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

    def neuron_backend_wrapper(gm, example_inputs, *, options=None, mode="default"):
        # There are 4 torch.compile mode: “default”, “reduce-overhead”,
        #   “max-autotune” or “max-autotune-no-cudagraphs”
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
        aot_backend = aot_autograd(
            fw_compiler=make_fw_compiler(analysis_results, options=options),
            keep_inference_input_mutations=True,  # Prevent Copy-back.
            decompositions=decomposition_table,
        )
        return aot_backend(gm, example_inputs)

    return neuron_backend_wrapper


# Create default backend instance with selective decomposition
neuron_backend = create_neuron_backend()


logger.debug("Neuron backend module successfully loaded")

"""
Neuron backend implementation for torch.compile
"""

import logging

import torch
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

from torch_neuronx.neuron_dynamo_backend.compile import CompileGraph

# Import local utilities
from torch_neuronx.neuron_dynamo_backend.config import (
    get_current_timestamp,
    get_fx_graph_path,
    get_model_name,
    get_stablehlo_path,
    managed_artifacts_directory,
    reset_timestamp,
)

# Import local decompositions
from torch_neuronx.neuron_dynamo_backend.decompositions import get_decomposition_table
from torch_neuronx.neuron_dynamo_backend.executor import Executor
from torch_neuronx.neuron_dynamo_backend.fx.fx_transform import (
    convert_fx_to_stablehlo,
    save_mlir_bytecode,
)
from torch_neuronx.neuron_dynamo_backend.fx_passes import legalize_flex_attention_pass
from torch_neuronx.neuron_dynamo_backend.utils.io_utils import save_fx_graph_txt

logger = logging.getLogger(__name__)


def _compile_fx_to_stablehlo(
    gm: torch.fx.GraphModule,
    example_inputs,
    model_name: str,
    segment_id: str,
    preserve_artifacts: bool,
):
    """
    FX Graph to StableHLO conversion helper function.

    Args:
        gm: FX GraphModule with Aten operations
        example_inputs: Example input tensors
        model_name: Model name for logging
        segment_id: Segment ID for stats
        preserve_artifacts: Save artifacts

    Returns:
        tuple: (stablehlo_mlir, stablehlo_time, artifacts_stablehlo)
            - stablehlo_mlir: Compiled StableHLO MLIR module
            - artifacts_stablehlo: Path to saved StableHLO artifacts
    """
    logger.debug("--- Starting FX Graph -> StableHLO conversion ---")

    stablehlo_mlir, io_spec, cast_spec = convert_fx_to_stablehlo(
        gm, example_inputs, preserve_artifacts=preserve_artifacts
    )
    logger.debug("FX Graph -> StableHLO completed")

    # Save stablehlo to artifacts dir
    artifacts_stablehlo = get_stablehlo_path(model_name)
    save_mlir_bytecode(stablehlo_mlir, artifacts_stablehlo)
    logger.debug(f"StableHLO saved to artifacts directory: {artifacts_stablehlo}")

    return stablehlo_mlir, artifacts_stablehlo, io_spec, cast_spec


def _has_collectives_in_graph(gm: torch.fx.GraphModule) -> bool:
    """
    Detect if FX graph contains collective operations.

    Scans the FX graph for torch._c10d_functional collective operations.

    Args:
        gm: FX GraphModule to analyze

    Returns:
        True if graph contains any collective operations, False otherwise
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


# TODO (NF-19): move all FX->NEFF compilation into a single function
def neuron_backend_fx_compiler(gm: torch.fx.GraphModule, example_inputs, options=None):
    """
    Neuron compiler function that receives core Aten ops from AOTAutograd

    This implements the complete compilation pipeline:
    1. FX Graph -> StableHLO conversion
    2. StableHLO validation
    3. StableHLO -> NEFF compilation
    4. NEFF execution wrapper creation

    Args:
        gm: FX GraphModule with Aten operations (from AOTAutograd)
        example_inputs: Example input tensors
        options: Compilation options from torch.compile

    Returns:
        Boxed function for execution

    Raises:
        RuntimeError: If compilation fails at any stage
    """

    logger.debug("Applying FX graph transformations...")
    gm = legalize_flex_attention_pass(gm, example_inputs)

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
                gm, example_inputs, model_name, segment_id, preserve_artifacts
            )

            compile_graph = CompileGraph(stablehlo_mlir, model_name, segment_id, has_collectives)
            final_cache_key = compile_graph.compile()
            executor = Executor(model_name, final_cache_key, io_spec, cast_spec, has_collectives)

            # Create wrapper function that matches torch.compile expectations
            def execution_wrapper(*inputs):
                """Wrapper function for execution"""
                try:
                    return executor(*inputs)
                except Exception as e:
                    logger.error(f"Execution failed: {e}")
                    # Fallback to original execution for debugging
                    logger.warning("Falling back to original GraphModule execution")
                    return gm.forward(*inputs)

            return make_boxed_func(execution_wrapper)

        except Exception as e:
            raise RuntimeError(
                f"Neuron backend NEFF execution setup failed with unexpected error: {e}"
            ) from e


def create_neuron_backend(decompose_all: bool = False):
    """
    Create a new neuron backend with specified decomposition settings

    Args:
        decompose_all: If True, decompose all operations. If False (default),
            only decompose ops in DECOMP_NAMES.

    Returns:
        Neuron backend for torch.compile

    Example:
        compiled_model = torch.compile(model, backend="neuron")
    """

    # TODO (NF-24): add explicit bw_compiler
    decomposition_table = get_decomposition_table(decompose_all)

    def neuron_backend_wrapper(gm, example_inputs, *, options=None):
        # Force dynamic=False for neuron backend
        if options is None:
            options = {}
        options.setdefault("dynamic", False)
        aot_backend = aot_autograd(
            fw_compiler=lambda gm, inputs: neuron_backend_fx_compiler(gm, inputs, options),
            decompositions=decomposition_table,
        )
        return aot_backend(gm, example_inputs)

    return neuron_backend_wrapper


# Create default backend instance with selective decomposition
neuron_backend = create_neuron_backend()


logger.debug("Neuron backend module loaded successfully")

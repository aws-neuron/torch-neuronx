"""
Compilation phase for torch.compile backend.
Dispatches to compilation worker.
"""

import io
import logging

import torch_neuronx
from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import compute_cache_key

logger = logging.getLogger(__name__)


class CompileGraph:
    """
    Takes StableHLO MLIR, compiles it to NEFF, and returns cache key.
    """

    def __init__(
        self, stablehlo_mlir, model_name: str, segment_id: str, has_collectives: bool = False
    ):
        """
        Initialize compilation context.

        Args:
            stablehlo_mlir: StableHLO MLIR module
            model_name: Model name for logging
            segment_id: Segment ID for stats
            has_collectives: Whether graph contains collective operations
        """
        self.stablehlo_mlir = stablehlo_mlir
        self.model_name = model_name
        self.segment_id = segment_id
        self.has_collectives = has_collectives

    def compile(self) -> str:
        """
        Compile StableHLO to NEFF.

        Returns:
            str: execution handle
        """
        logger.debug(f"CompileGraph.compile() called: model={self.model_name}")
        base_cache_key = compute_cache_key(self.stablehlo_mlir)

        bytecode_buffer = io.BytesIO()
        self.stablehlo_mlir.operation.write_bytecode(bytecode_buffer)
        stablehlo_bytes = bytecode_buffer.getvalue()

        exec_handle = torch_neuronx._C.compile_graph(
            base_cache_key, stablehlo_bytes, self.has_collectives
        )

        logger.debug(f"Compilation complete: execution_cache_key={exec_handle}")
        return exec_handle

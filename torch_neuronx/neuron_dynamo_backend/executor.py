"""
NEFF execution for TorchNeuronEager async runtime
"""

import logging
from typing import Final

import torch

import torch_neuronx
from torch_neuronx.neuron_dynamo_backend.exceptions import NEFFExecutionError
from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import FunctionIO, TensorSpec

logger = logging.getLogger(__name__)


class Executor:
    """
    Executor for separated compile/execute flow using cache-based execution.

    Uses final_cache_key to retrieve NEFF from cache and execute via execute_compiled_graph.
    """

    INT_64_TO_32: Final = {
        torch.int64: torch.int32,
        torch.uint64: torch.uint32,
    }

    def __init__(
        self,
        graph_name: str,
        cache_key: str,
        io_spec: FunctionIO,
        cast_spec: list[TensorSpec],
        has_collectives: bool = False,
    ):
        """
        Initialize executor with cache key and I/O specifications.

        Args:
            graph_name: Name of graph
            cache_key: Cache key from compilation phase. Handle for executor.
            io_spec: Input/output specifications from StableHLO.
            cast_spec: Output specifications for final dtype.
            has_collectives: Whether graph contains collective operations (default: False)
        """
        self.graph_name = graph_name
        self.cache_key = cache_key
        self.io_spec = io_spec
        self.cast_spec = cast_spec
        self.has_collectives = has_collectives

        logger.debug(
            f"Executor initialized: cache_key={cache_key}, " f"has_collectives={has_collectives}"
        )
        logger.debug(
            f"Inputs: {len(self.io_spec.inputs)}, " f"Outputs: {len(self.io_spec.outputs)}"
        )

    def _prepare_inputs(self, inputs, device_id):
        """
        Prepare inputs for execution, handling device transfer and dtype conversion.

        Neuron hardware doesn't support 64-bit dtypes natively. This method:
        - Converts Python scalars to scalar tensors
        - Transfers CPU tensors to neuron device
        - Downcasts 64-bit dtypes (int64, uint64, float64) to 32-bit equivalents
        - Handles scalar tensors by reshaping before view operation

        Args:
            inputs: Iterable of input tensors and scalars
            device_id: Target neuron device ID

        Yields:
            torch.Tensor: Prepared inputs as tensors
        """
        device_str = f"neuron:{device_id}"

        for inp in inputs:
            # Convert Python scalars to scalar tensors
            if not isinstance(inp, torch.Tensor):
                tensor = torch.tensor(inp, device=device_str)
            else:
                tensor = inp.to(device_str) if inp.device.type == "cpu" else inp

            if tensor.dtype in self.INT_64_TO_32:
                if tensor.dim() == 0:
                    tensor = tensor.reshape(1).view(self.INT_64_TO_32[tensor.dtype])
                else:
                    tensor = tensor.view(self.INT_64_TO_32[tensor.dtype])
            elif tensor.dtype == torch.float64:
                tensor = tensor.to(torch.float32)
            yield tensor

    def _process_outputs(self, outputs):
        """
        Process outputs after execution, converting back to original dtypes if needed.

        Upcasts 32-bit outputs back to 64-bit dtypes when the original spec requires it,
        maintaining the illusion that execution happened in 64-bit precision.

        Args:
            outputs: List of output tensors from execution (32-bit dtypes)

        Yields:
            torch.Tensor: Output tensors with dtypes matching original specs
        """
        for tensor, cast_spec in zip(outputs, self.cast_spec, strict=False):
            if cast_spec.dtype in self.INT_64_TO_32:
                # view() reinterprets bits, then reshape() fixes shape
                tensor = tensor.view(cast_spec.dtype).reshape(cast_spec.shape)
            elif cast_spec.dtype == torch.float64:
                tensor = tensor.to(cast_spec.dtype)
            yield tensor

    def __call__(self, *inputs) -> tuple[torch.Tensor, ...]:
        """
        Execute the compiled graph using cache-based execution.

        Workflow:
        1. Validate input count matches specs
        2. Prepare inputs (device transfer + dtype downcast)
        3. Allocate output buffers with 32-bit dtypes
        4. Execute compiled graph via C++ runtime
        5. Process outputs (dtype upcast to match specs)

        Args:
            *inputs: Input tensors matching io_spec.inputs

        Returns:
            tuple[torch.Tensor, ...]: Output tensors on neuron device with dtypes
                                      matching cast_spec.

        Raises:
            NEFFExecutionError: If input count doesn't match expected count
        """
        logger.debug(f"Executor invoked: cache_key={self.cache_key}")

        if len(inputs) != len(self.io_spec.inputs):
            raise NEFFExecutionError(
                f"Expected {len(self.io_spec.inputs)} inputs, " f"got {len(inputs)}"
            )

        device_id = torch.neuron.current_device()
        device_str = f"neuron:{device_id}"
        neuron_inputs = list(self._prepare_inputs(inputs, device_id))

        neuron_outputs = [
            torch.empty(out.shape, dtype=out.dtype, device=device_str)
            for out in self.io_spec.outputs
        ]

        logger.debug(f"Calling execute_compiled_graph: cache_key={self.cache_key}")
        torch_neuronx._C.execute_compiled_graph(
            self.graph_name,
            self.cache_key,
            neuron_inputs,
            neuron_outputs,
            self.has_collectives,
        )

        return tuple(self._process_outputs(neuron_outputs))

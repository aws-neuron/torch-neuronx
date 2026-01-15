"""
NEFF execution for TorchNeuronEager async runtime
"""

import logging
from typing import Final

import torch

import torch_neuronx
from torch_neuronx.neuron_dynamo_backend.exceptions import NEFFExecutionError
from torch_neuronx.neuron_dynamo_backend.fx.passes.remove_none_outputs import NoneOutputInfo
from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo
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
        mutation_info: AliasingInfo | None = None,
        none_output_info: NoneOutputInfo | None = None,
    ):
        """
        Initialize executor with cache key and I/O specifications.

        Args:
            graph_name: Name of graph
            cache_key: Cache key from compilation phase. Handle for executor.
            io_spec: Input/output specifications from StableHLO.
            cast_spec: Output specifications for final dtype.
            has_collectives: Whether graph contains collective operations (default: False)
            none_output_info: metadata on FX graph output spec
        """
        self.graph_name = graph_name
        self.cache_key = cache_key
        self.io_spec = io_spec
        self.cast_spec = cast_spec
        self.has_collectives = has_collectives
        self.mutation_info = mutation_info
        self.none_output_info = none_output_info

        logger.debug(
            f"Executor initialized: cache_key={cache_key}, has_collectives={has_collectives}"
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
        Process execution outputs, handling dtype upcasting and alias filtering.

        Neuron hardware operates on 32-bit dtypes, so this method restores
        original 64-bit dtypes where needed. Aliased outputs are skipped since
        they were written directly to input buffers during execution.

        Args:
            outputs: List of output tensors from graph execution.

        Yields:
            torch.Tensor: Processed tensors with dtypes matching cast_spec.
                - int32/uint32 tensors are reinterpreted as int64/uint64 via view()
                - float32 tensors are converted to float64 via to()
                - Aliased outputs are excluded from the results.
                  Combination of AOTAutograd and Compiler handle mapping of
                  outputs to inputs.
        """
        # Apply casting
        tensors = []
        for tensor, cast_spec in zip(outputs, self.cast_spec, strict=False):
            if cast_spec.dtype in self.INT_64_TO_32:
                tensor = tensor.view(cast_spec.dtype).reshape(cast_spec.shape)
            elif cast_spec.dtype == torch.float64:
                tensor = tensor.to(cast_spec.dtype)
            tensors.append(tensor)

        # Re-insert None outputs into list of outputs
        if self.none_output_info:
            assert len(tensors) == len(
                self.none_output_info.non_none_positions
            ), "Length of outputs does not match non_none_positions:"
            f"{len(tensors)=} {len(self.none_output_info.non_none_positions)=}"
            tensors_with_nones = [None for _ in range(self.none_output_info.original_output_count)]
            for i in range(len(tensors)):
                index = self.none_output_info.non_none_positions[i]
                tensors_with_nones[index] = tensors[i]
            tensors = tensors_with_nones

        # Filter aliased outputs
        if self.mutation_info:
            aliased_output_indices = {alias.output_index for alias in self.mutation_info.aliases}
            tensors = (
                tensor for i, tensor in enumerate(tensors) if i not in aliased_output_indices
            )
        yield from tensors

    def _allocate_output(self, i, out, neuron_inputs, device_str):
        """
        Allocate a single output buffer for graph execution.

        For aliased outputs (in-place mutations), reuses the corresponding input
        buffer so mutations are visible to the caller. For non-aliased outputs,
        allocates a new empty tensor.

        Args:
            i: Output index in io_spec.outputs.
            out: TensorSpec for this output (shape, dtype).
            neuron_inputs: List of prepared input tensors.
            device_str: Target device string (e.g., "neuron:0").

        Returns:
            torch.Tensor: Either a reused input buffer (aliased) or a newly
                allocated empty tensor (non-aliased).
        """
        if self.mutation_info is not None:
            mutated_output_to_input = dict(
                sorted(
                    (alias.output_index, alias.parameter_number)
                    for alias in self.mutation_info.aliases
                )
            )
        else:
            mutated_output_to_input = {}

        if i in mutated_output_to_input:
            return neuron_inputs[mutated_output_to_input[i]]
        return torch.empty(out.shape, dtype=out.dtype, device=device_str)

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
            self._allocate_output(i, out, neuron_inputs, device_str)
            for i, out in enumerate(self.io_spec.outputs)
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

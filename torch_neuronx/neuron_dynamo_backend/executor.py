"""
NEFF execution for TorchNeuronEager async runtime
"""

import logging
from typing import Final

import torch

import torch_neuronx
from torch_neuronx.neuron_dynamo_backend.exceptions import NEFFExecutionError
from torch_neuronx.neuron_dynamo_backend.fx.passes.remove_none_outputs import NoneOutputInfo
from torch_neuronx.neuron_dynamo_backend.settings import _getenv_bool
from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo
from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import (
    FunctionIO,
    RandomInputInfo,
    TensorSpec,
)

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
        retain_device: bool = False,
        mutation_info: AliasingInfo | None = None,
        none_output_info: NoneOutputInfo | None = None,
    ):
        """Initialize executor with cache key and I/O specifications.

        Args:
            graph_name (str): Human-readable name for the graph.
            cache_key (str): Cache key from compilation phase, used as execution handle.
            io_spec (FunctionIO): Input/output specifications from StableHLO.
            cast_spec (list[TensorSpec]): Output specifications for final dtype casting.
            has_collectives (bool): Whether graph contains collective operations.
            retain_device (bool): Whether to cast results back to the input device (default: False)
            mutation_info (AliasingInfo | None): Input-output mutation relationships.
            none_output_info (NoneOutputInfo | None): Metadata for None output restoration.
        """
        self.graph_name = graph_name
        self.cache_key = cache_key
        self.io_spec = io_spec
        self.cast_spec = cast_spec
        self.has_collectives = has_collectives
        self.mutation_info = mutation_info
        self.none_output_info = none_output_info

        self.retain_device = retain_device

        logger.debug(
            f"Executor initialized: cache_key={cache_key}, has_collectives={has_collectives}"
        )
        logger.debug(
            f"Inputs: {len(self.io_spec.inputs)}, " f"Outputs: {len(self.io_spec.outputs)}"
        )

    def _prepare_inputs(self, inputs, device_id):
        """Prepare inputs for execution with device transfer and dtype conversion.

        Neuron hardware doesn't support 64-bit dtypes natively. This method:
        - Converts Python scalars to scalar tensors
        - Transfers CPU tensors to neuron device
          (unless TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY=1)
        - Downcasts 64-bit dtypes (int64, uint64, float64) to 32-bit equivalents
        - Handles scalar tensors by reshaping before view operation

        Args:
            inputs (Iterable): Input tensors and scalars.
            device_id (int): Target neuron device ID.

        Yields:
            torch.Tensor: Prepared inputs as tensors on neuron device.

        Raises:
            RuntimeError: If CPU tensors are provided and
                          TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY=1.
        """
        device_str = f"neuron:{device_id}"
        disable_cpu_autocopy = _getenv_bool("TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY", False)

        for i, inp in enumerate(inputs):
            # Convert Python scalars to scalar tensors
            if not isinstance(inp, torch.Tensor):
                tensor = torch.tensor(inp, device=device_str)
            elif inp.device.type == "cpu":
                if disable_cpu_autocopy:
                    raise RuntimeError(
                        f"Input tensor at index {i} is on cpu device, expected neuron. "
                        "Move tensor to Neuron device first with tensor.to('neuron')"
                    )
                tensor = inp.to(device_str)
            else:
                tensor = inp

            if tensor.dtype in self.INT_64_TO_32:
                if tensor.dim() == 0:
                    tensor = tensor.reshape(1).view(self.INT_64_TO_32[tensor.dtype])
                else:
                    tensor = tensor.view(self.INT_64_TO_32[tensor.dtype])
            elif tensor.dtype == torch.float64:
                tensor = tensor.to(torch.float32)
            yield tensor

    def _process_outputs(self, outputs):
        """Process execution outputs with dtype upcasting and alias filtering.

        Neuron hardware operates on 32-bit dtypes, so this method restores
        original 64-bit dtypes where needed. Aliased outputs are skipped since
        they were written directly to input buffers during execution.

        Args:
            outputs (list[torch.Tensor]): Output tensors from graph execution.

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
        """Allocate a single output buffer for graph execution.

        For aliased outputs (in-place mutations), reuses the corresponding input
        buffer so mutations are visible to the caller. For non-aliased outputs,
        allocates a new empty tensor.

        Args:
            i (int): Output index in io_spec.outputs.
            out (TensorSpec): Tensor specification for this output.
            neuron_inputs (list[torch.Tensor]): List of prepared input tensors.
            device_str (str): Target device string (e.g., "neuron:0").

        Returns:
            torch.Tensor: Reused input buffer or newly allocated empty tensor.
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

    def _cast_output_device(self, original_inputs, outputs):
        """
        Cast output tensors to the same device as the original inputs when
        retain_device=True is set.

        Args:
            original_inputs: The original input tensors passed to the executor.
            outputs: The output tensors from the graph execution.

        Yields:
            torch.Tensor: Output tensors cast to the same device as the inputs.
        """
        output_device = None

        # Identify the input device type
        for inp in original_inputs:
            if inp is None or not isinstance(inp, torch.Tensor):
                continue
            if output_device is None:
                output_device = inp.device.type
            elif inp.device.type != output_device:
                raise NEFFExecutionError(
                    f"cannot mix input devices with TORCH_NEURONX_RETAIN_DEVICE_MODE=1, "
                    f"got {output_device} and {inp.device.type}"
                )

        for i, output in enumerate(outputs):
            if (
                output is not None
                and output_device is not None
                and output.device.type != output_device
            ):
                if self.mutation_info:
                    # mutations need to be propagated back to cpu inputs
                    # that were automatically cast to neuron
                    # SEE: test_retain_device_copies_mutated_output_to_original_input in
                    # test_aliasing_executor.py
                    input_idx = self.mutation_info.get_input_index(i)
                    if input_idx is not None:
                        original_input = original_inputs[input_idx]
                        yield original_input.copy_(output.to(original_input.device))
                        continue
                yield output.to(output_device)
            else:
                yield output

    def _generate_random_inputs(
        self, random_input_info: RandomInputInfo, device_str: str
    ) -> list[torch.Tensor]:
        """Generate random mask inputs for dropout operations.

        Args:
            random_input_info: Metadata about random inputs needed
            device_str: Target device string (e.g., "neuron:0")

        Returns:
            List of random tensor inputs
        """
        return [op.sample(device_str) for op in random_input_info.ops]

    def __call__(self, *inputs) -> tuple[torch.Tensor, ...]:
        """Execute the compiled graph using cache-based execution.

        Workflow:
        1. Validate input count matches io_spec.inputs
        2. Prepare inputs (device transfer + dtype downcast)
        3. Generate random inputs for dropout masks if needed
        4. Allocate output buffers with 32-bit dtypes
        5. Execute compiled graph via C++ runtime
        6. Process outputs (dtype upcast to match specs)

        Args:
            *inputs: Input tensors matching io_spec.inputs.

        Returns:
            tuple[torch.Tensor, ...]: Output tensors on Neuron device with dtypes
                matching cast_spec.

        Raises:
            NEFFExecutionError: If input count doesn't match expected count.
        """
        logger.debug(f"Executor invoked: cache_key={self.cache_key}")

        random_input_count = 0
        if self.io_spec.random_input_info:
            random_input_count = (
                self.io_spec.random_input_info.new_input_count
                - self.io_spec.random_input_info.original_input_count
            )
        if len(inputs) != (len(self.io_spec.inputs) - random_input_count):
            raise NEFFExecutionError(
                f"Expected {len(self.io_spec.inputs)} inputs, got {len(inputs)}"
            )

        device_id = torch.neuron.current_device()
        device_str = f"neuron:{device_id}"
        neuron_inputs = list(self._prepare_inputs(inputs, device_id))

        # Generate and append random inputs for dropout masks
        if self.io_spec.random_input_info:
            random_inputs = self._generate_random_inputs(self.io_spec.random_input_info, device_str)
            neuron_inputs.extend(random_inputs)
            logger.debug(f"Added {len(random_inputs)} random inputs for dropout masks")

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

        outputs = tuple(self._process_outputs(neuron_outputs))

        if self.retain_device:
            return tuple(self._cast_output_device(inputs, outputs))
        else:
            return outputs

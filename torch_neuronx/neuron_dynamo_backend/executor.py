"""
NEFF execution for TorchNeuronEager async runtime
"""

import logging
import math
from typing import Final

import torch

import torch_neuronx
from torch_neuronx.neuron_dynamo_backend.exceptions import NEFFExecutionError
from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
    PassMutations,
    apply_input_mutations,
    apply_output_mutations,
    get_mutation_copy_back_inputs,
)
from torch_neuronx.neuron_dynamo_backend.settings import _getenv_bool
from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import (
    FunctionIO,
    TensorSpec,
)
from torch_neuronx.neuron_dynamo_backend.wlo_contract import (
    WLOContractError,
    validate_transform,
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
        pass_mutations: list[PassMutations] | None = None,
        alias_buffer_map: dict[int, int] | None = None,
        wlo_transforms: list | None = None,
    ):
        """Initialize executor with cache key and I/O specifications.

        Args:
            graph_name (str): Human-readable name for the graph.
            cache_key (str): Cache key from compilation phase, used as execution handle.
            io_spec (FunctionIO): Input/output specifications from StableHLO.
            cast_spec (list[TensorSpec]): Output specifications for final dtype casting.
            has_collectives (bool): Whether graph contains collective operations.
            retain_device (bool): Whether to cast results back to the input device (default: False)
            pass_mutations (list[PassMutations] | None): Per-pass input/output mutations.
            alias_buffer_map (dict[int, int] | None): NEFF output→input index map for buffer reuse.
            wlo_transforms (list | None): Weight-layout relayout rules discovered at compile
                time. Each rule follows the shared contract in wlo_contract (reshape = target
                dims, transpose = permutation of the reshape rank); see _prepare_inputs.
        """
        self.graph_name = graph_name
        self.cache_key = cache_key
        self.io_spec = io_spec
        self.cast_spec = cast_spec
        self.has_collectives = has_collectives
        self.pass_mutations = pass_mutations or []
        self.alias_buffer_map = alias_buffer_map
        self.copy_back_map = get_mutation_copy_back_inputs(self.pass_mutations)

        self.retain_device = retain_device

        self.wlo_transforms = wlo_transforms or []
        self._wlo_cache: dict[int, tuple[int, torch.Tensor]] = {}

        for t in self.wlo_transforms:
            validate_transform(t)
            if not 0 <= t["input_index"] < len(self.io_spec.inputs):
                raise WLOContractError(
                    f"WLO transform input_index {t['input_index']} out of range for a graph "
                    f"with {len(self.io_spec.inputs)} input(s); the relayout rules do not match "
                    "this graph (likely a stale or mismatched cache entry)"
                )

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

            if not tensor.is_contiguous():
                contiguous_tensor = torch.empty(
                    tensor.shape, dtype=tensor.dtype, device=tensor.device
                )
                torch_neuronx._C._nki_copy_strided_neuron_to_neuron(tensor, contiguous_tensor)
                tensor = contiguous_tensor
            if tensor.dtype in self.INT_64_TO_32:
                if tensor.dim() == 0:
                    tensor = tensor.reshape(1).view(self.INT_64_TO_32[tensor.dtype])
                else:
                    tensor = tensor.view(self.INT_64_TO_32[tensor.dtype])
            elif tensor.dtype == torch.float64:
                tensor = tensor.to(torch.float32)

            if self.wlo_transforms:
                for t in self.wlo_transforms:
                    if t["input_index"] == i:
                        ptr = inp.data_ptr() if isinstance(inp, torch.Tensor) else id(inp)
                        cached = self._wlo_cache.get(i)
                        if cached is not None and cached[0] == ptr:
                            tensor = cached[1]
                        else:
                            original_shape = tensor.shape
                            reshape_numel = math.prod(t["reshape"])
                            if reshape_numel != tensor.numel():
                                raise RuntimeError(
                                    f"WLO: transform for input {i} expects {reshape_numel} "
                                    f"elements (reshape={t['reshape']}) but the runtime tensor "
                                    f"has {tensor.numel()} (shape={tuple(original_shape)}, "
                                    f"dtype={tensor.dtype}). The weight-layout rules do not "
                                    "match this input; refusing to relayout."
                                )
                            # Apply the wlo_contract relayout: reshape to the target
                            # dims, permute, then materialize the optimal byte layout
                            # with contiguous(). The final reshape is the compiler's
                            # reshape-back -- reconstructed from original_shape, which
                            # always equals the weight's original shape -- so the NEFF
                            # still sees the original shape with only the bytes relaid out.
                            tensor = (
                                tensor.reshape(t["reshape"])
                                .permute(t["transpose"])
                                .contiguous()
                                .reshape(original_shape)
                            )
                            self._wlo_cache[i] = (ptr, tensor)
                        break

            yield tensor

    def _process_outputs(self, outputs):
        """Process execution outputs with dtype upcasting and shape correction.

        Neuron hardware operates on 32-bit dtypes, so this method restores
        original 64-bit dtypes where needed. For aliased passthrough outputs
        that reuse an input buffer, inserts view ops to match expected shapes.

        Args:
            outputs (list[torch.Tensor]): Output tensors from graph execution.

        Returns:
            list[torch.Tensor]: Tensors with dtypes matching cast_spec.
        """
        tensors = []
        for tensor, cast_spec in zip(outputs, self.cast_spec, strict=False):
            if cast_spec.dtype in self.INT_64_TO_32:
                tensor = tensor.view(cast_spec.dtype).reshape(cast_spec.shape)
            elif cast_spec.dtype == torch.float64:
                tensor = tensor.to(cast_spec.dtype)
            tensors.append(tensor)

        return tensors

    def _allocate_output(self, i, out, neuron_inputs, device_str):
        """Allocate a single output buffer for graph execution.

        For aliased outputs (in-place mutations) or pass-through outputs,
        reuses the corresponding input buffer. For other outputs,
        allocates a new empty tensor.

        Args:
            i (int): Output index in io_spec.outputs.
            out (TensorSpec): Tensor specification for this output.
            neuron_inputs (list[torch.Tensor]): List of prepared input tensors.
            device_str (str): Target device string (e.g., "neuron:0").

        Returns:
            torch.Tensor: Reused input buffer or newly allocated empty tensor.
        """
        if self.alias_buffer_map and i in self.alias_buffer_map:
            in_idx = self.alias_buffer_map[i]
            assert in_idx < len(neuron_inputs), (
                f"alias_buffer_map[{i}]={in_idx} is out of range for neuron_inputs "
                f"(len={len(neuron_inputs)})"
            )
            return neuron_inputs[in_idx]

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
                if i in self.copy_back_map:
                    # Mutation result: copy back into the original caller input tensor
                    original_input = original_inputs[self.copy_back_map[i]]
                    yield original_input.copy_(output.to(original_input.device))
                    continue
                yield output.to(output_device)
            else:
                yield output

    def _writeback_downcast_aliases(self, inputs, neuron_outputs, tensors):
        """Write fp64 aliased-out results back into the caller's buffer.

        An fp64 out= input is downcast to fp32 by a copy, so the NEFF writes the
        shadow, not the caller's tensor. int64 downcasts via a storage-sharing
        view and is already correct, so it's skipped.
        """
        for out_idx, caller_in_idx in self.copy_back_map.items():
            orig = inputs[caller_in_idx]
            shadow = neuron_outputs[out_idx]
            if (
                isinstance(orig, torch.Tensor)
                and orig.dtype == torch.float64
                and shadow.dtype == torch.float32
            ):
                orig.copy_(shadow)
                tensors[out_idx] = orig
        return tensors

    def __call__(self, *inputs) -> tuple[torch.Tensor, ...]:
        """Execute the compiled graph using cache-based execution.

        Workflow:
        1. Apply input mutations (remove/add inputs as specified by FX passes)
        2. Validate input count matches io_spec.inputs
        3. Prepare inputs (device transfer + dtype downcast)
        4. Allocate output buffers with 32-bit dtypes
        5. Execute compiled graph via C++ runtime
        6. Process outputs (dtype upcast to match specs)
        7. Apply output mutations (restore original signature)

        Args:
            *inputs: Input tensors matching the original graph signature.

        Returns:
            tuple[torch.Tensor, ...]: Output tensors matching the original graph signature.

        Raises:
            NEFFExecutionError: If input count doesn't match expected count after mutations.
        """
        logger.debug(f"Executor invoked: cache_key={self.cache_key}")

        device_id = torch.neuron.current_device()
        device_str = f"neuron:{device_id}"

        # Apply input mutations to transform caller inputs into NEFF inputs
        mutated_inputs = apply_input_mutations(list(inputs), self.pass_mutations, device_str)

        if len(mutated_inputs) != len(self.io_spec.inputs):
            raise NEFFExecutionError(
                f"Expected {len(self.io_spec.inputs)} inputs after mutations, "
                f"got {len(mutated_inputs)}"
            )

        neuron_inputs = list(self._prepare_inputs(mutated_inputs, device_id))

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

        tensors = self._process_outputs(neuron_outputs)
        tensors = self._writeback_downcast_aliases(inputs, neuron_outputs, tensors)

        if self.retain_device:
            # Apply copy-back from NEFF outputs directly back to caller inputs
            tensors = list(self._cast_output_device(inputs, tensors))

        # Apply output mutations to restore original graph signature
        tensors = apply_output_mutations(tensors, self.pass_mutations, device_str, list(inputs))

        return tuple(tensors)

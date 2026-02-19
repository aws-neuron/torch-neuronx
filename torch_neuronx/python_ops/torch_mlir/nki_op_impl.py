"""Torch-MLIR implementation for NKI kernels."""

import torch

from torch_neuronx.nki_hop import nki_kernel_wrapper
from torch_neuronx.python_ops import io_tensor
from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel
from torch_neuronx.python_ops.torch_mlir.op_impl import TorchMlirOpImpl


def nki_torch_fn(*args, **nki_params):
    """Global torch function for all NKI kernels."""
    return nki_kernel_wrapper(**nki_params, args=args)


class NkiKernel(TorchMlirKernel):
    """Specialized kernel for NKI that handles aliasing in execution tensor preparation."""

    def _generate_unified_cache_key(self, inputs: tuple, kwargs: dict, context=None) -> str:
        """Use kernel_idx and constant_args hash_key for NKI caching."""
        kernel_idx = kwargs.get("kernel_idx", 0)
        constant_args_key = kwargs.get("constant_args_key", 0)
        # Include tensor shapes/dtypes for cache differentiation
        tensor_info = tuple((t.shape, t.dtype) for t in inputs if isinstance(t, torch.Tensor))
        return f"nki_{kernel_idx}_{constant_args_key}_{hash(tensor_info)}"

    def _pad_outputs_for_aliasing(
        self,
        provided_output,
        downcasted_outputs,
        original_outputs=None,
        operand_output_aliases=None,
    ):
        """Pad downcasted and original outputs to match provided_output length for aliasing."""
        if not isinstance(provided_output, tuple) or len(provided_output) <= 1:
            return downcasted_outputs, original_outputs

        def _pad_single_output(outputs, target_length):
            """Helper to pad a single output tuple to target length."""
            if outputs is None:
                return None

            # Convert to tuple if needed
            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            if len(provided_output) > len(outputs):
                padded = list(outputs)
                # Pad by repeating outputs at aliased indices
                if operand_output_aliases:
                    for i in range(len(outputs), target_length):
                        # Find which output this aliased position should repeat
                        aliased_output_idx = None
                        for input_id, output_id in operand_output_aliases.items():
                            if output_id == i:
                                aliased_output_idx = input_id % len(outputs)
                                break

                        if aliased_output_idx is not None:
                            padded.append(outputs[aliased_output_idx])
                        else:
                            padded.append(outputs[0])  # Fallback
                else:
                    # Fallback: repeat first output
                    for _ in range(target_length - len(outputs)):
                        padded.append(outputs[0])
                outputs = tuple(padded)

            return outputs

        # Pad both downcasted and original outputs using the same logic
        target_length = len(provided_output)
        padded_downcasted = _pad_single_output(downcasted_outputs, target_length)
        padded_original = _pad_single_output(original_outputs, target_length)

        return padded_downcasted, padded_original

    def _prepare_execution_tensors(self, provided_output, downcasted_outputs):
        """Handle aliasing by padding outputs first, then use parent logic."""
        operand_output_aliases = getattr(self, "operand_output_aliases", {})
        downcasted_outputs, _ = self._pad_outputs_for_aliasing(
            provided_output, downcasted_outputs, operand_output_aliases=operand_output_aliases
        )
        return super()._prepare_execution_tensors(provided_output, downcasted_outputs)

    def _upcast_execution_results(
        self, execution_tensors, downcasted_outputs, original_outputs, kept_output_indices
    ):
        """Handle aliasing by padding outputs first, then use parent logic."""
        operand_output_aliases = getattr(self, "operand_output_aliases", {})
        downcasted_outputs, original_outputs = self._pad_outputs_for_aliasing(
            execution_tensors if isinstance(execution_tensors, tuple) else (execution_tensors,),
            downcasted_outputs,
            original_outputs,
            operand_output_aliases=operand_output_aliases,
        )
        return super()._upcast_execution_results(
            execution_tensors, downcasted_outputs, original_outputs, kept_output_indices
        )

    def _copy_to_provided_output(self, upcast_tensors, provided_output):
        """Handle aliasing - only copy to actual outputs, skip aliased inputs."""
        if not isinstance(provided_output, tuple):
            return super()._copy_to_provided_output(upcast_tensors, provided_output)

        if not isinstance(upcast_tensors, tuple):
            upcast_tensors = (upcast_tensors,)

        # Only copy to actual outputs (first len(upcast_tensors) of provided_output)
        actual_provided = provided_output[: len(upcast_tensors)]
        super()._copy_to_provided_output(upcast_tensors, tuple(actual_provided))


class NKITorchMlirOpImpl(TorchMlirOpImpl):
    """Single torch-mlir op implementation for all NKI kernels."""

    def __init__(self):
        # Use the global NKI torch function
        super().__init__(
            aten_op_name="nki_kernel_global",
            torch_fn=nki_torch_fn,
            output_params=None,
            static_argnums=(),
            static_argnames=(
                "kernel_idx",
                "grid",
                "backend_config",
                "operand_output_aliases",
                "arg_names",
                "constant_args_key",
            ),
        )
        # Replace with specialized kernel that handles aliasing
        self.kernel = NkiKernel(
            op_name=self.aten_op_name,
            torch_fn=self.torch_fn,
            output_params=self.output_params,
            static_argnums=self.static_argnums,
            static_argnames=self.static_argnames,
        )

    def __call__(self, *args, **kwargs):
        """Handle aliasing by passing proper output tensors via out= parameter."""
        operand_output_aliases = kwargs.get("operand_output_aliases", {})
        return_types = kwargs.pop("return_types", None)

        # Attach aliasing info to kernel for use in internal methods
        self.kernel.operand_output_aliases = operand_output_aliases

        if operand_output_aliases:
            tensor_inputs = [arg for arg in args if isinstance(arg, torch.Tensor)]
            device = tensor_inputs[0].device if tensor_inputs else "neuron"

            # Create output tensors from return_types (avoids expensive meta call)
            from torch_neuronx.utils import map_external_dtype_to_torch

            output_tensors = []
            for dtype, shape in return_types:
                output_tensors.append(
                    io_tensor.empty(shape, dtype=map_external_dtype_to_torch(dtype), device=device)
                )

            # Add aliased input tensors (these will be mutated in-place)
            for input_id in operand_output_aliases:
                if input_id < len(tensor_inputs):
                    output_tensors.append(tensor_inputs[input_id])

            # Execute with proper output tensors
            kwargs["out"] = tuple(output_tensors)
            result = self.execute(*args, **kwargs)

            if result.success:
                # Return only the actual outputs (first len(return_types) tensors)
                actual_outputs = output_tensors[: len(return_types)]
                return actual_outputs[0] if len(actual_outputs) == 1 else tuple(actual_outputs)
            else:
                raise RuntimeError(f"NKI kernel execution failed: {result.error_msg}")

        # No aliasing - normal execution
        result = self.execute(*args, **kwargs)
        if result.success:
            return result.output
        else:
            raise RuntimeError(f"NKI kernel execution failed: {result.error_msg}")

    def can_handle(self, *args, **kwargs) -> bool:
        """Check if this implementation can handle the given arguments."""
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.device.type != "neuron":
                return False
        return True


# Global NKI op instance (like aten ops)
_global_nki_op = NKITorchMlirOpImpl()


def get_nki_torch_mlir_op(
    kernel_idx: int,
    grid,
    backend_config: str,
    operand_output_aliases: dict,
    args: list,
    arg_names: list,
    constant_args_key: int,
    return_types: list,
):
    """Get the global NKI op and call it with the specific kernel parameters."""
    # Create the NKI parameters
    nki_params = {
        "kernel_idx": kernel_idx,
        "grid": grid,
        "backend_config": backend_config,
        "operand_output_aliases": operand_output_aliases,
        "arg_names": arg_names,
        "constant_args_key": constant_args_key,
        "return_types": return_types,
    }

    # Use the global op but with specific kernel parameters
    return lambda *args: _global_nki_op(*args, **nki_params)

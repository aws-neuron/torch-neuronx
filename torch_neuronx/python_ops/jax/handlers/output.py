"""Handler for output tensor management."""

import logging

import jax
import torch

from torch_neuronx.python_ops.handlers import BaseOutputHandler
from torch_neuronx.python_ops.shared import MetadataOps

from ..type_converter import JaxTypeConverter

logger = logging.getLogger(__name__)


class OutputHandler(BaseOutputHandler):
    """Manages output tensor creation and in-place operations."""

    def __init__(self):
        """Initialize the handler."""
        self.type_converter = JaxTypeConverter()

    def create_output_tensors(
        self,
        output_specs: tuple,
        device: torch.device,
        expected_dtypes: list,
    ) -> tuple[torch.Tensor, ...]:
        """Create output tensors based on JAX specifications.

        Note: We create tensors with JAX dtypes (e.g., int32) because that's what JAX will write.
        Casting to expected PyTorch dtypes happens after execution.

        Args:
            output_specs: Tuple of JAX ShapeDtypeStruct specifications for shapes and JAX dtypes
            device: Target device for tensors
            expected_dtypes: List of expected PyTorch dtypes (from meta tensor
                evaluation) - stored for later

        Returns:
            Tuple of output tensors (with JAX-compatible dtypes)
        """
        output_tensors = []

        for i, spec in enumerate(output_specs):
            # Use JAX dtype for tensor creation (what JAX will actually write)
            jax_dtype = self.type_converter.jax_to_torch_dtype(spec.dtype)

            # Special handling: JAX writes int32 but PyTorch may expect int64
            # We create int32 tensors that will be cast to int64 post-execution
            if (
                i < len(expected_dtypes)
                and expected_dtypes[i] == torch.int64
                and jax_dtype == torch.int32
            ):
                # Create as int32 (what JAX writes), will cast to int64 later
                tensor = torch.empty(spec.shape, dtype=torch.int32, device=device)
            else:
                tensor = torch.empty(spec.shape, dtype=jax_dtype, device=device)
            output_tensors.append(tensor)

        logger.debug(f"Created {len(output_tensors)} output tensors with JAX dtypes")
        return tuple(output_tensors)

    def infer_output_specs(
        self,
        jax_fn: callable,
        inputs: tuple,
        op_name: str,
        static_argnums: tuple = (),
        meta_inputs: tuple | None = None,
        expected_dtypes_override: list | None = None,
        kwargs: dict | None = None,
        convert_to_jax_spec=True,
    ) -> tuple[tuple, bool, list, list]:
        """Infer output specifications using PyTorch meta tensors for both shape and dtype.

        Args:
            jax_fn: JAX function
            inputs: Preprocessed inputs
            op_name: ATen operation name (required)
            static_argnums: Indices of static arguments

        Returns:
            Tuple of (output_specs, is_single_output, expected_torch_dtypes, none_mask)
        """
        # In-place operations: infer shape/dtype directly from the mutated tensor (or out=)
        full_name = op_name.replace("aten::", "")
        packet_name = full_name.split(".")[0]
        is_inplace = packet_name.endswith("_")

        if is_inplace:
            source_tensors: list[torch.Tensor] = []
            # Prefer explicit 'out=' if provided
            if kwargs and kwargs.get("out") is not None:
                out_val = kwargs.get("out")
                if isinstance(out_val, tuple):
                    source_tensors.extend([t for t in out_val if isinstance(t, torch.Tensor)])
                elif isinstance(out_val, torch.Tensor):
                    source_tensors.append(out_val)

            # Fall back to the mutated input tensor (self): first tensor in inputs
            if not source_tensors:
                original = meta_inputs if meta_inputs is not None else inputs
                for inp in original:
                    if isinstance(inp, torch.Tensor):
                        source_tensors.append(inp)
                        break

            if not source_tensors:
                raise RuntimeError(
                    f"Could not infer output for in-place op {op_name}: no tensor inputs found"
                )

            shapes = [tuple(t.shape) for t in source_tensors]
            dtypes = [t.dtype for t in source_tensors]
            pytorch_dtypes = (
                expected_dtypes_override if expected_dtypes_override is not None else dtypes
            )

            specs = []
            for shape, dtype in zip(shapes, pytorch_dtypes, strict=False):
                if convert_to_jax_spec:
                    jax_dtype = self.type_converter.to_execution_jax_dtype(dtype)
                    specs.append(jax.ShapeDtypeStruct(shape, jax_dtype))
                else:
                    dtype = self._maybe_downcast_dtype(dtype)
                    specs.append([shape, dtype])
                # jax_dtype = self.type_converter.to_execution_jax_dtype(dtype)
                # specs.append(jax.ShapeDtypeStruct(shape, jax_dtype))

            is_single = len(specs) == 1
            return tuple(specs), is_single, pytorch_dtypes, []

        # Default path: use meta tensors to get shape and dtype information
        shapes, dtypes, is_single, none_mask = self._get_output_info_from_meta(
            meta_inputs if meta_inputs is not None else inputs,
            op_name,
            static_argnums,
            kwargs,
        )

        # Store original PyTorch expected dtypes (for post-processing)
        pytorch_dtypes = dtypes.copy()

        # If user provided output dtypes, use those instead of meta-inferred ones
        if expected_dtypes_override is not None:
            pytorch_dtypes = expected_dtypes_override

        # Convert to JAX specs
        specs = []
        for shape, dtype in zip(shapes, pytorch_dtypes, strict=False):
            if convert_to_jax_spec:
                jax_dtype = self.type_converter.to_execution_jax_dtype(dtype)
                specs.append(jax.ShapeDtypeStruct(shape, jax_dtype))
            else:
                dtype = self._maybe_downcast_dtype(dtype)
                specs.append([shape, dtype])

        # Return JAX specs for execution, but PyTorch dtypes for final casting

        return tuple(specs), is_single, pytorch_dtypes, none_mask

    def _maybe_downcast_dtype(self, dtype):
        if dtype == torch.int64:
            dtype = torch.int32
        elif dtype == torch.float64:
            dtype = torch.float32
        return dtype

    def _get_tensor_dtypes(self, inputs: tuple) -> list:
        """Get dtypes of tensor inputs.

        Args:
            inputs: Input values

        Returns:
            List of tensor dtypes
        """
        dtypes = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                dtypes.append(inp.dtype)
        return dtypes

    def _get_output_info_from_meta(
        self, inputs: tuple, op_name: str, static_argnums: tuple, kwargs: dict | None = None
    ) -> tuple[list, list, bool, list]:
        """Get output shapes and dtypes using PyTorch meta tensors.

        Args:
            inputs: Original input tensors
            op_name: ATen operation name
            static_argnums: Indices of static arguments
            kwargs: Keyword arguments including potential 'out' parameter

        Returns:
            Tuple of (shapes, dtypes, is_single_output, none_mask)
        """
        if not op_name or not op_name.startswith("aten::"):
            raise ValueError(
                f"Cannot use meta tensor evaluation for non-ATen op: {op_name}. "
                "Meta tensor evaluation requires an ATen operation name."
            )

        # Resolve the ATen operator for meta evaluation using the exact schema name.
        full_name = op_name.replace("aten::", "")
        parts = full_name.split(".")
        packet_name = parts[0]
        packet = getattr(torch.ops.aten, packet_name, None)
        if packet is None:
            raise RuntimeError(
                f"Could not resolve ATen op packet for {op_name} via torch.ops.aten.{packet_name}"
            )

        # Build callable pytorch op by following explicit overload path, if any
        pytorch_op = packet
        for overload in parts[1:]:
            if not hasattr(pytorch_op, overload):
                raise RuntimeError(
                    f"Could not resolve ATen overload for {op_name}: missing attribute '{overload}'"
                )
            pytorch_op = getattr(pytorch_op, overload)

        # Convert inputs to meta tensors
        meta_inputs = []
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                # Create meta tensor with same shape and dtype
                meta_tensor = torch.empty(inp.shape, dtype=inp.dtype, device="meta")
                meta_inputs.append(meta_tensor)
            elif i in (static_argnums or ()):
                # Static arguments pass through
                meta_inputs.append(inp)
            elif isinstance(inp, list):
                meta_accum = []
                for it in inp:
                    if isinstance(it, torch.Tensor):
                        meta_tensor = torch.empty(it.shape, dtype=it.dtype, device="meta")
                        meta_accum.append(meta_tensor)
                    else:
                        meta_accum.append(it)
                meta_inputs.append(meta_accum)
            else:
                meta_inputs.append(inp)

        # Prepare kwargs for meta evaluation, converting any output tensors to meta
        meta_kwargs = {}
        if kwargs:
            for key, value in kwargs.items():
                if key == "out" and value is not None:
                    # Convert out tensors to meta tensors
                    if isinstance(value, tuple):
                        meta_kwargs[key] = tuple(
                            torch.empty(t.shape, dtype=t.dtype, device="meta") for t in value
                        )
                    elif isinstance(value, torch.Tensor):
                        meta_kwargs[key] = torch.empty(
                            value.shape, dtype=value.dtype, device="meta"
                        )
                    else:
                        meta_kwargs[key] = value
                elif isinstance(value, torch.Tensor):
                    # Convert any other tensor arguments to meta
                    meta_kwargs[key] = torch.empty(value.shape, dtype=value.dtype, device="meta")
                elif key == "device":
                    # Override device to meta to ensure meta dispatch
                    meta_kwargs[key] = "meta"
                else:
                    # Pass non-tensor arguments through unchanged
                    meta_kwargs[key] = value

        # Run the operation on meta tensors to get output shapes and dtypes
        try:
            logger.debug(f"Running meta evaluation for {op_name} with {len(meta_inputs)} inputs")
            meta_input_shapes = [inp.shape if hasattr(inp, "shape") else inp for inp in meta_inputs]
            logger.debug(f"Meta inputs: {meta_input_shapes}")
            if meta_kwargs:
                logger.debug("Using out parameter for meta evaluation")

            meta_output = pytorch_op(*meta_inputs, **meta_kwargs)

            logger.debug(f"Meta output type: {type(meta_output)}")

            # Extract shapes and dtypes
            is_single = not isinstance(meta_output, tuple | list)
            outputs = (meta_output,) if is_single else meta_output

            shapes = []
            dtypes = []
            none_mask = []
            for out in outputs:
                if out is None:
                    none_mask.append(True)
                elif hasattr(out, "shape") and hasattr(out, "dtype"):
                    shapes.append(tuple(out.shape))
                    dtypes.append(out.dtype)
                    none_mask.append(False)
                else:
                    raise ValueError(f"Non-tensor output from meta evaluation: {type(out)}")

            logger.debug(f"Meta shapes: {shapes}, dtypes: {dtypes}, none_mask: {none_mask}")

            dtypes = self._apply_meta_dtype_corrections(op_name, inputs, dtypes)

            return shapes, dtypes, is_single, none_mask

        except Exception as e:
            # Log error with useful context
            try:
                inputs_summary = []
                for inp in inputs:
                    if isinstance(inp, torch.Tensor):
                        inputs_summary.append(
                            f"Tensor(shape={tuple(inp.shape)}, dtype={inp.dtype})"
                        )
                    else:
                        inputs_summary.append(repr(inp))
            except Exception:
                inputs_summary = [type(inp).__name__ for inp in inputs]

            logger.error(
                "Meta tensor evaluation failed for %s. Reason: %s: %s; inputs=%s",
                op_name,
                type(e).__name__,
                e,
                inputs_summary,
            )
            raise

    def _apply_meta_dtype_corrections(self, op_name: str, inputs: tuple, dtypes: list) -> list:
        """Apply dtype corrections for PyTorch meta tensor bugs.

        Args:
            op_name: ATen operation name
            inputs: Input tensors
            dtypes: Inferred dtypes from meta evaluation

        Returns:
            Corrected dtypes list
        """
        correction_fn = MetadataOps.get_correction(op_name)
        if correction_fn is None:
            return dtypes

        return correction_fn(inputs, dtypes)

    def _infer_scalar_dtype(self, scalar: int | float | bool, tensor_dtypes: list) -> any:
        """Infer appropriate dtype for scalar.

        Args:
            scalar: Scalar value
            tensor_dtypes: Tensor dtypes in operation

        Returns:
            Inferred JAX dtype
        """
        return self.type_converter.infer_scalar_dtype(scalar, tensor_dtypes)

    def reconstruct_with_none_values(
        self, result: torch.Tensor | tuple, none_mask: list[bool], is_single: bool
    ) -> torch.Tensor | tuple | None:
        """Reconstruct full result with None values at specified positions.

        Args:
            result: Computed result tensor(s)
            none_mask: Boolean mask indicating which positions should be None
            is_single: Whether the original output was a single tensor

        Returns:
            Result with None values inserted at masked positions
        """
        full_result = []
        real_idx = 0
        for is_none in none_mask:
            if is_none:
                full_result.append(None)
            else:
                result_val = result[real_idx] if isinstance(result, tuple) else result
                full_result.append(result_val)
                real_idx += 1

        return tuple(full_result) if not is_single else full_result[0]

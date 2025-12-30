"""Torch-MLIR kernel implementation."""

import logging
from collections.abc import Callable

import torch

from torch_neuronx.kernels.base import BaseNeuronKernel
from torch_neuronx.kernels.compiler_config import CompilerConfig
from torch_neuronx.neuron_dynamo_backend.backend import neuron_backend
from torch_neuronx.python_ops.compilation import CompilationCache, HloCompiler
from torch_neuronx.utils import cast_dtype_if_needed

from ..jax.config import CompilationConfig
from ..jax.context import ExecutionContext
from ..jax.handlers import OutputHandler
from ..processors import ArgumentProcessor
from .compilation import TorchMlirCompiler

logger = logging.getLogger(__name__)


class TorchMlirKernel(BaseNeuronKernel):
    """Torch-MLIR kernel implementation.

    Compiles PyTorch operations to StableHLO via torch-mlir, then to NEFF.
    """

    def __init__(
        self,
        torch_fn: Callable,
        op_name: str,
        static_argnums: tuple[int, ...] | None = None,
        static_argnames: tuple[str, ...] | None = None,
        compiler_config: CompilerConfig | None = None,
        output_params: tuple[str, ...] | None = None,
    ):
        super().__init__()

        self.torch_fn = torch_fn
        self.op_name = op_name
        self.static_argnums = static_argnums or ()
        self.static_argnames = static_argnames or ()
        self.output_params = output_params or ()

        # Initialize components
        self.mlir_compiler = TorchMlirCompiler()
        self.hlo_compiler = HloCompiler(compiler_config)
        self.cache = CompilationCache()
        self.arg_processor = ArgumentProcessor(static_argnames)
        self.arg_processor.static_argnums = ()
        self.output_handler = OutputHandler()

        self._output_spec_cache = {}

        assert isinstance(self.static_argnums, tuple)
        assert isinstance(self.static_argnames, tuple)

    def _generate_unified_cache_key(
        self, inputs: tuple, kwargs: dict, context: ExecutionContext | None = None
    ) -> str:
        """Generate unified cache key (reuses JAX logic).

        Args:
            inputs: Input tensors
            kwargs: Keyword arguments
            context: Execution context

        Returns:
            Cache key string
        """
        # Get base cache key from parent class
        filtered_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in CompilationConfig.PYTORCH_SPECIFIC_PARAMS
            and key not in self.static_argnames
        }
        normalized_static_argnums = self._normalize_static_argnums(len(inputs))
        base_key = self.get_cache_key(
            self.op_name,
            *inputs,
            kwargs=filtered_kwargs,
            static_indices=normalized_static_argnums,
        )
        # Add kwargs
        key_with_kwargs = self.arg_processor.add_static_to_cache_key(base_key, kwargs)

        return key_with_kwargs

    def _get_cached_output_specs(
        self, inputs: tuple, kwargs: dict, context: ExecutionContext | None = None
    ) -> tuple:
        """Get cached output specs or compute them."""
        cache_key = self._generate_unified_cache_key(
            context.original_inputs if context else inputs,
            kwargs,
            context,
        )

        if cache_key in self._output_spec_cache:
            return self._output_spec_cache[cache_key]

        # For non-aten ops, run torch_fn directly with meta tensors
        if not self.op_name.startswith("aten::"):
            meta_inputs = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    meta_inputs.append(torch.empty(inp.shape, dtype=inp.dtype, device="meta"))
                else:
                    meta_inputs.append(inp)

            meta_output = self.torch_fn(*meta_inputs, **kwargs)
            outputs = (meta_output,) if not isinstance(meta_output, tuple) else meta_output

            output_specs = []
            expected_dtypes = []
            for out in outputs:
                output_specs.append([tuple(out.shape), out.dtype])
                expected_dtypes.append(out.dtype)

            result = (tuple(output_specs), expected_dtypes, [])
            self._output_spec_cache[cache_key] = result
            return result

        # Infer output specs using output handler for aten ops
        output_specs, _, expected_dtypes, none_mask = self.output_handler.infer_output_specs(
            self.torch_fn,
            inputs,
            self.op_name,
            static_argnums=None,
            meta_inputs=context.original_inputs if context else None,
            expected_dtypes_override=None,
            kwargs=kwargs,
            convert_to_jax_spec=False,
        )

        result = (output_specs, expected_dtypes, none_mask)
        self._output_spec_cache[cache_key] = result
        return result

    def _get_compilation_kwargs(self, kwargs):
        """Process kwargs for lowering."""
        # Exclude unsupported kwargs
        compilation_kwargs = {}

        # Extract dtype from 'out' tensor for ops that need it
        if (
            "dtype" in self.static_argnames
            and "dtype" not in kwargs
            and "out" in kwargs
            and kwargs["out"] is not None
        ):
            out_tensor = kwargs["out"]
            if isinstance(out_tensor, torch.Tensor):
                compilation_kwargs["dtype"] = out_tensor.dtype

        for key, value in kwargs.items():
            if key not in CompilationConfig.PYTORCH_SPECIFIC_PARAMS:
                compilation_kwargs[key] = value

        return compilation_kwargs

    def _get_or_compile_neff(self, inputs: tuple, kwargs: dict) -> bytes:
        """Get NEFF from cache or compile if needed.

        Args:
            inputs: Preprocessed inputs
            kwargs: Keyword arguments

        Returns:
            Compiled NEFF bytes
        """
        cache_key = self._generate_unified_cache_key(inputs, kwargs)

        # Try cache first
        neff_bytes = self.cache.get_neff(cache_key)
        if neff_bytes is not None:
            return neff_bytes

        # Compile: PyTorch -> MLIR (StableHLO) -> NEFF
        compilation_kwargs = self._get_compilation_kwargs(kwargs)
        stablehlo_module = self.mlir_compiler.compile(
            self.torch_fn, self.op_name, inputs, compilation_kwargs
        )
        stablehlo_text = self.mlir_compiler.export_to_hlo(stablehlo_module).encode()
        neff_bytes = self.hlo_compiler.compile_to_neff(stablehlo_text, ir_type="StableHLO")

        # Store in cache
        self.cache.store_neff(cache_key, neff_bytes)

        return neff_bytes

    def _prepare_outputs(
        self,
        output_specs: list,
        expected_dtypes: list,
        device: torch.device,
        kwargs: dict,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, ...], list, torch.Tensor | tuple | None]:
        """Prepare output tensors from specs.

        Args:
            output_specs: Output specifications
            expected_dtypes: Expected dtypes
            device: Device for tensors
            kwargs: Keyword arguments

        Returns:
            Tuple of (execution_tensors, expected_dtypes, provided_output)
        """
        # Check for provided output tensors
        provided_output = self.output_handler.extract_output_params(kwargs, self.output_params)

        execution_tensors = []
        if provided_output is not None:
            # Use provided tensors or create temporaries if needed
            provided_list = (
                provided_output if isinstance(provided_output, tuple) else (provided_output,)
            )

            for spec, provided_tensor in zip(output_specs, provided_list, strict=False):
                # Check if we can use provided tensor directly
                if (
                    tuple(provided_tensor.shape) == tuple(spec[0])
                    and provided_tensor.dtype == spec[1]
                ):
                    execution_tensors.append(provided_tensor)
                else:
                    # Need temporary
                    execution_tensors.append(torch.empty(spec[0], dtype=spec[1], device=device))

            execution_tensors = tuple(execution_tensors)
        else:
            # Create new output tensors
            for spec in output_specs:
                execution_tensors.append(torch.empty(spec[0], dtype=spec[1], device=device))
            execution_tensors = tuple(execution_tensors)

        return execution_tensors, expected_dtypes, provided_output

    def is_empty_output(self, output_specs):
        def _is_zero_shape(shape):
            # Zero-sized if any dimension is zero
            return any(dim == 0 for dim in shape)

        return output_specs and all(_is_zero_shape(spec[0]) for spec in output_specs)

    def __call__(self, *args, context: ExecutionContext | None = None, **kwargs):
        """Execute the kernel."""
        # Preprocess inputs
        processed_inputs = self.arg_processor.preprocess_inputs(args)

        # Get output specs
        output_specs, expected_dtypes, none_mask = self._get_cached_output_specs(
            processed_inputs, kwargs, context
        )

        # Handle all None outputs early
        if none_mask and all(none_mask):
            return tuple(None for _ in none_mask)

        device = self.arg_processor.extract_device(processed_inputs)

        # Prepare outputs
        execution_tensors, expected_dtypes, provided_output = self._prepare_outputs(
            output_specs, expected_dtypes, device, kwargs
        )

        if not self.is_empty_output(output_specs):
            if getattr(self.torch_fn, "use_torch_compile", True):
                # Additional processing to convert scalar value into tensor so that
                # non-static scalar value is not compiled into HLO and cause incorrect results
                # TODO do this in argument_processor.py once Jax is removed
                torch_fn = self.torch_fn
                if getattr(self.torch_fn, "uses_preprocessing", False):
                    torch_fn, processed_inputs, kwargs = self.torch_fn(*processed_inputs, **kwargs)

                compile_inputs = []
                for idx, inp in enumerate(processed_inputs):
                    if idx not in self.static_argnums and isinstance(inp, int | float | bool):
                        if isinstance(inp, bool):
                            compile_inputs.append(
                                torch.tensor(inp, dtype=torch.bool, device=device)
                            )
                        elif isinstance(inp, int):
                            compile_inputs.append(
                                torch.tensor(inp, dtype=torch.int32, device=device)
                            )
                        elif isinstance(inp, float):
                            compile_inputs.append(
                                torch.tensor(inp, dtype=torch.float32, device=device)
                            )
                    else:
                        compile_inputs.append(inp)

                compile_inputs = tuple(compile_inputs)
                compile_kwargs = self._get_compilation_kwargs(kwargs)
                compiled_model = torch.compile(
                    torch_fn,
                    backend=neuron_backend,
                    fullgraph=True,
                    dynamic=False,
                    options={"model_name": self.op_name},
                )
                with torch._dynamo.config.patch(cache_size_limit=float("inf")):
                    execution_tensors = compiled_model(*compile_inputs, **compile_kwargs)

                if not isinstance(execution_tensors, tuple):
                    execution_tensors = (execution_tensors,)
            else:
                # Get or compile NEFF
                neff_bytes = self._get_or_compile_neff(processed_inputs, kwargs)

                # Prepare execution inputs/outputs
                input_dict = self.arg_processor.prepare_execution_inputs(processed_inputs, device)
                output_dict = {f"output{i}": out for i, out in enumerate(execution_tensors)}

                # Execute NEFF (uses BaseNeuronKernel.execute_neff)
                self.execute_neff(neff_bytes, input_dict, output_dict, op_name=self.op_name)

        # Handle provided outputs
        if provided_output is not None:
            exec_list = (
                execution_tensors if isinstance(execution_tensors, tuple) else (execution_tensors,)
            )
            corrected = []
            for i, exec_tensor in enumerate(exec_list):
                target_dtype = expected_dtypes[i] if i < len(expected_dtypes) else None
                corrected.append(cast_dtype_if_needed(exec_tensor, target_dtype))
            corrected_result = tuple(corrected) if len(corrected) > 1 else corrected[0]
            return self.output_handler.handle_output_parameter(corrected_result, provided_output)

        # No provided outputs
        corrected_outputs = []
        for i, tensor in enumerate(execution_tensors):
            # torch.compile outputs will automatically add None values
            if tensor is None:
                corrected_outputs.append(tensor)
            else:
                target_dtype = expected_dtypes[i] if i < len(expected_dtypes) else None
                corrected_outputs.append(cast_dtype_if_needed(tensor, target_dtype))

        # Handle None reconstruction if needed
        if not getattr(self.torch_fn, "use_torch_compile", True) and none_mask and any(none_mask):
            return self.output_handler.reconstruct_with_none_values(
                tuple(corrected_outputs), none_mask, False
            )

        return corrected_outputs[0] if len(corrected_outputs) == 1 else tuple(corrected_outputs)

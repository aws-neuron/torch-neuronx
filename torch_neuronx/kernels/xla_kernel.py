"""XLA/JAX kernel implementation for torch-neuronx."""

import os
import traceback
from collections.abc import Callable

import jax
import jax.numpy as jnp
import torch

from torch_neuronx.python_ops.jax.compilation import JaxCompiler
from torch_neuronx.python_ops.jax.handlers import OutputHandler
from torch_neuronx.python_ops.jax.type_converter import JaxTypeConverter
from torch_neuronx.python_ops.processors import ArgumentProcessor

from ..utils import (
    is_neff_cache_disabled,
    is_sync_mode_enabled,
    log_neff_cache_hit,
    log_neff_cache_miss,
    log_neff_cache_store,
)
from .base import BaseNeuronKernel
from .compiler_config import CompilerConfig
from .compiler_subprocess import CompilerSubprocess
from .type_converter import TypeConverter


class TorchNeuronXLAKernel(BaseNeuronKernel):
    """XLA/JAX kernel implementation.

    This kernel takes a JAX function, compiles it to HLO, then to NEFF,
    and executes it on Neuron devices.
    """

    def __init__(
        self,
        jax_fn: Callable,
        op_name: str,
        static_argnums: tuple[int, ...] | None = None,
        compiler_config: CompilerConfig | None = None,
    ):
        """Initialize with a JAX function.

        Args:
            jax_fn: JAX function to compile and execute
            op_name: Operation name for cache key generation
            static_argnums: Indices of static positional arguments
            compiler_config: Compiler configuration
        """
        super().__init__()

        # Core function and configuration
        self.jax_fn = jax_fn
        self.op_name = op_name
        self.static_argnums = static_argnums or ()
        self.compiler_config = compiler_config or CompilerConfig()
        self.compiler_subprocess = CompilerSubprocess()
        self.arg_processor = ArgumentProcessor()
        # Initialize components
        self.jax_compiler = JaxCompiler()
        self.output_handler = OutputHandler()
        self.type_converter = JaxTypeConverter()

    def compile_jax_to_hlo(
        self,
        jax_fn: Callable,
        *sample_inputs: torch.Tensor,
        donate_argnums: tuple[int, ...] | None = None,
    ):
        """Compile JAX function to HLO module (legacy path).

        Args:
            jax_fn: JAX function to compile
            sample_inputs: Sample PyTorch tensors for shape/dtype inference
            donate_argnums: Optional tuple of argument indices to donate

        Returns:
            HLO module object
        """

        normalized_static_argnums = self._normalize_static_argnums(len(sample_inputs))

        # Convert torch tensors to JAX arrays for tracing
        jax_inputs = []
        for i, inp in enumerate(sample_inputs):
            if normalized_static_argnums and i in normalized_static_argnums:
                # Static arguments must remain as Python values
                if isinstance(inp, torch.dtype):
                    jax_inputs.append(TypeConverter.torch_to_jax(inp))
                else:
                    jax_inputs.append(inp)
            elif isinstance(inp, torch.Tensor):
                # Convert torch dtype to JAX dtype directly
                # NumPy doesn't support bfloat16, so we need to handle it specially
                jax_dtype = TypeConverter.torch_to_jax(inp.dtype)
                # Create JAX array directly with zeros
                jax_array = jnp.zeros(inp.shape, dtype=jax_dtype)
                jax_inputs.append(jax_array)
            elif isinstance(inp, list | tuple):
                # Handle list of tensors
                jax_list = []
                for tensor in inp:
                    if isinstance(tensor, torch.Tensor):
                        jax_dtype = TypeConverter.torch_to_jax(tensor.dtype)
                        jax_array = jnp.zeros(tensor.shape, dtype=jax_dtype)
                        jax_list.append(jax_array)
                    else:
                        jax_list.append(tensor)
                jax_inputs.append(jax_list)
            elif isinstance(inp, int | float | bool):
                jax_inputs.append(inp)
            elif inp is None:
                jax_inputs.append(None)
            else:
                raise ValueError(f"Unsupported input type: {type(inp)}, inp value is {inp}")

        # JIT compile with static_argnums and lower the function
        donate_argnums = donate_argnums or ()
        jitted_fn = jax.jit(
            jax_fn, static_argnums=normalized_static_argnums, donate_argnums=donate_argnums
        )
        lowered = jitted_fn.lower(*jax_inputs)
        # Get HLO representation (not StableHLO)
        hlo_computation = lowered.compiler_ir(dialect="hlo")
        # Return HLO module object
        hlo = hlo_computation.as_hlo_module()
        return hlo

    def _compile_hlo_to_neff(self, hlo) -> bytes:
        """Compile HLO to NEFF (legacy path).

        Args:
            hlo: HLO module object

        Returns:
            Compiled NEFF bytes
        """
        lnc = self.compiler_config.lnc

        # Use the subprocess handler for compilation
        return self.compiler_subprocess.get_or_compile(
            hlo.as_serialized_hlo_module_proto(), self.compiler_config, lnc
        )

    def get_cache_key(self, *inputs, donate_argnums: tuple[int, ...] | None = None) -> str:
        """Generate cache key including static arguments and donate_argnums.

        Args:
            inputs: Input tensors and scalars
            donate_argnums: Optional tuple of argument indices to donate

        Returns:
            Cache key string that includes static argument values and donate_argnums
        """
        normalized_static_argnums = self._normalize_static_argnums(len(inputs))
        base_key = super().get_cache_key(
            self.op_name, *inputs, static_indices=normalized_static_argnums
        )

        # Include donate_argnums in cache key to ensure different donation
        # configurations produce separate cache entries
        if donate_argnums:
            donate_str = f"_donate_{','.join(map(str, donate_argnums))}"
            return base_key + donate_str
        return base_key

    def _infer_output_specs(self, inputs: tuple) -> tuple:
        """Infer output specifications using JAX eval_shape.

        Args:
            inputs: Preprocessed inputs

        Returns:
            Tuple of (output_specs, is_single, none_mask)
        """
        normalized_static_argnums = self._normalize_static_argnums(len(inputs))

        def get_input_specs(inputs, check_static_argnums=True):
            # Create shape/dtype specifications for inputs
            input_specs = []

            for i, inp in enumerate(inputs):
                if isinstance(inp, torch.Tensor):
                    jax_dtype = TypeConverter.torch_to_jax(inp.dtype)
                    spec = jax.ShapeDtypeStruct(inp.shape, jax_dtype)
                    input_specs.append(spec)
                elif i in normalized_static_argnums:
                    # For static arguments, pass the actual value (can be scalar, tuple, list, etc.)
                    input_specs.append(inp)
                elif isinstance(inp, list | tuple):
                    # Handle list or tuple of tensors
                    input_specs.append(get_input_specs(inp, check_static_argnums=False))
                elif isinstance(inp, int | float | bool):
                    # Convert Python type to JAX dtype
                    if isinstance(inp, bool):
                        jax_dtype = jnp.bool_
                    elif isinstance(inp, int):
                        jax_dtype = jnp.int32  # Neuron doesn't support int64
                    else:  # float
                        jax_dtype = jnp.float32
                    # Create scalar spec (empty shape)
                    spec = jax.ShapeDtypeStruct((), jax_dtype)
                    input_specs.append(spec)
                else:
                    # For other types (like None), just append as is
                    input_specs.append(inp)

            return input_specs

        input_specs = get_input_specs(inputs)

        # Evaluate output shape without computation
        if normalized_static_argnums:
            # Handle static arguments for eval_shape
            partial_args = [None] * len(input_specs)
            eval_args = []

            # Fill in static arguments
            for i in normalized_static_argnums:
                if i < len(input_specs):
                    partial_args[i] = input_specs[i]

            # Collect non-static specs in order
            for i, spec in enumerate(input_specs):
                if i not in normalized_static_argnums:
                    eval_args.append(spec)

            # Create a wrapper function that takes only non-static args
            def wrapper(*args):
                full_args = partial_args.copy()
                arg_idx = 0
                for i in range(len(full_args)):
                    if i not in normalized_static_argnums:
                        full_args[i] = args[arg_idx]
                        arg_idx += 1
                return self.jax_fn(*full_args)

            output_spec = jax.eval_shape(wrapper, *eval_args)
        else:
            output_spec = jax.eval_shape(self.jax_fn, *input_specs)

        # Normalize to tuple
        is_single = not isinstance(output_spec, tuple | list)
        output_tuple = (output_spec,) if is_single else output_spec

        flattened_output = []
        for spec in output_tuple:
            if isinstance(spec, list | tuple):
                flattened_output.extend(spec)
            else:
                flattened_output.append(spec)

        # Create none_mask to track which outputs are None
        none_mask = [spec is None for spec in flattened_output]

        # Filter out None specs for actual tensor creation
        filtered_specs = [spec for spec in flattened_output if spec is not None]

        return filtered_specs, is_single, none_mask

    def _create_output_tensors(self, output_specs: tuple, device) -> tuple[torch.Tensor, ...]:
        """Create output tensors based on specifications.

        Args:
            output_specs: Tuple of JAX ShapeDtypeStruct specifications (None values filtered out)
            device: Target device for tensors

        Returns:
            Tuple of output tensors
        """
        output_tensors = []
        for spec in output_specs:
            torch_dtype = TypeConverter.jax_to_torch(spec.dtype)
            out_tensor = torch.empty(spec.shape, dtype=torch_dtype, device=device)
            output_tensors.append(out_tensor)
        return tuple(output_tensors)

    def _reconstruct_with_none_values(self, result, none_mask: list[bool], is_single: bool):
        """Reconstruct output with None values in correct positions.

        Args:
            result: Computed tensor result(s)
            none_mask: Boolean list indicating which positions should be None
            is_single: Whether original output was single tensor

        Returns:
            Result with None values inserted at correct positions
        """
        if not any(none_mask):
            return result

        # Convert result to list for easier manipulation
        result_list = [result] if not isinstance(result, tuple) else list(result)

        # Reconstruct full output with None values
        full_output = []
        result_idx = 0

        for is_none in none_mask:
            if is_none:
                full_output.append(None)
            else:
                full_output.append(result_list[result_idx])
                result_idx += 1

        # Return in original format
        if is_single:
            return full_output[0]
        return tuple(full_output)

    def _get_or_compile_neff(
        self,
        cache_key: str,
        inputs: tuple,
        donate_argnums: tuple[int, ...] | None = None,
    ) -> bytes:
        """Get NEFF from cache or compile if not present (legacy path).

        Args:
            cache_key: Cache key for NEFF lookup
            inputs: Preprocessed inputs for compilation

        Returns:
            Compiled NEFF bytes
        """
        # Check if caching is disabled
        cache_disabled = is_neff_cache_disabled()

        # Check cache unless disabled
        if not cache_disabled and cache_key in self._neff_cache:
            log_neff_cache_hit(cache_key)
            _, neff_bytes = self._neff_cache[cache_key]
            return neff_bytes

        # Log cache miss
        log_neff_cache_miss(cache_key)

        # Compile JAX -> HLO -> NEFF
        hlo = self.compile_jax_to_hlo(self.jax_fn, *inputs, donate_argnums=donate_argnums)
        neff_bytes = self._compile_hlo_to_neff(hlo)

        # Store in cache unless disabled
        if not cache_disabled:
            self._neff_cache[cache_key] = ("xla.neff", neff_bytes)
            log_neff_cache_store(cache_key)

        return neff_bytes

    def postprocess_output_tensors(self, output_tensors, original_outputs):
        """Postprocess output tensors."""
        for out, processed_out in zip(output_tensors, original_outputs, strict=False):
            if out.dtype != processed_out.dtype:
                out.copy_(processed_out.to(out.dtype))

    def __call__(
        self,
        *inputs: torch.Tensor,
        output: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        donate_argnums: tuple[int, ...] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Execute the kernel.

        Args:
            inputs: Input tensors
            output: Optional pre-allocated output tensor(s). Can be a single tensor
                   or tuple of tensors for multi-output operations.

        Returns:
            Output tensor(s). Returns a single tensor or tuple of tensors based on
            the JAX function's output structure.
        """
        normalized_static_argnums = self._normalize_static_argnums(len(inputs))
        self.arg_processor.static_argnums = normalized_static_argnums
        self.jax_compiler.static_argnums = normalized_static_argnums

        processed_inputs = self.arg_processor.preprocess_inputs(inputs)

        device = self.arg_processor.extract_device(processed_inputs)
        # Prepare outputs
        if output is None:
            # Infer output specs to detect None outputs
            output_specs, is_single_output, none_mask = self._infer_output_specs(processed_inputs)

            # Handle all None outputs early
            if none_mask and all(none_mask):
                return None if is_single_output else tuple(None for _ in none_mask)

            # Skip execution if no actual tensors to compute
            if not output_specs:
                return self._reconstruct_with_none_values((), none_mask, is_single_output)

            # Create tensors only for non-None outputs
            output_tensors = self._create_output_tensors(output_specs, device)
        else:
            is_single_output = not isinstance(output, tuple)
            output_tensors = (output,) if is_single_output else output
            none_mask = [False]

        # Generate cache key and get/compile NEFF
        cache_key = self.get_cache_key(*processed_inputs, donate_argnums=donate_argnums)
        # Prepare execution dictionaries
        input_dict = self.arg_processor.prepare_execution_inputs(processed_inputs, device)
        # Prepare outputs
        original_output_dict = {}
        for i, out in enumerate(output_tensors):
            original_output_dict[f"output{i}"] = TypeConverter.convert_for_neuron(out)

        # Get CPU fallback context stored directly on kernel
        cpu_fallback_context = getattr(self, "_cpu_fallback_context", None)
        # need to handle the case where the kernel is created after impl.execute call
        if cpu_fallback_context is None:
            cpu_fallback_context = {"original_inputs": list(inputs), "original_kwargs": {}}
        # TODO -  remove sync flow once async CRs are merged
        if is_sync_mode_enabled():
            self._execute_sync(
                cache_key,
                processed_inputs,
                input_dict,
                original_output_dict,
                donate_argnums,
            )
        else:
            self._execute_async(
                cache_key,
                processed_inputs,
                tuple(input_dict.values()),
                tuple(original_output_dict.values()),
                cpu_fallback_context,
                donate_argnums,
            )

        return self.finalize_outputs(
            output_tensors, original_output_dict, none_mask, is_single_output
        )

    def _execute_sync(
        self,
        cache_key: str,
        processed_inputs: tuple,
        input_dict,
        original_output_dict,
        donate_argnums: tuple[int, ...] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Execute the kernel using the legacy synchronous path.

        Args:
            processed_inputs: Preprocessed input tensors
            output: Optional pre-allocated output tensor(s)

        Returns:
            Output tensor(s)
        """
        neff_bytes = self._get_or_compile_neff(cache_key, processed_inputs, donate_argnums)

        # Execute NEFF
        self.execute_neff(neff_bytes, input_dict, original_output_dict, op_name=self.op_name)

    def _execute_async(
        self,
        cache_key: str,
        processed_inputs: tuple,
        non_static_inputs,
        output_tensors,
        cpu_fallback_context,
        donate_argnums: tuple[int, ...] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Execute the kernel using the modern C++ binding path.

        Args:
            processed_inputs: Preprocessed inputs
            output: Optional pre-allocated output tensor(s)

        Returns:
            Output tensor(s)
        """
        lowered_ir_bytes, _ = self.jax_compiler.get_or_compile_hlo(
            cache_key,
            self.jax_fn,
            processed_inputs,
            {},
            donate_argnums,
        )

        # Capture stack trace to tie to the running operation.
        if os.environ.get("TORCH_NEURONX_ENABLE_STACK_TRACE", "0") == "1":
            stack_trace = "".join(traceback.format_list(traceback.extract_stack()[:-2]))
        else:
            stack_trace = ""

        # Submit to C++ XLA pipeline
        import torch_neuronx._C as _C

        try:
            _C._submit_xla_task_to_pipeline(
                self.op_name,
                non_static_inputs,
                output_tensors,
                lowered_ir_bytes,
                None,
                cache_key,
                stack_trace,
                False,
                cpu_fallback_context,
            )
        except RuntimeError as e:
            raise RuntimeError(f"XLA kernel execution failed: {e}") from e

    def finalize_outputs(self, output_tensors, original_output_dict, none_mask, is_single_output):
        """Finalize output tensors."""
        # Upcasting to original int64/float64, if they were downcasted during
        # execution
        self.postprocess_output_tensors(output_tensors, original_output_dict.values())
        # Get result in original format
        result = output_tensors[0] if len(output_tensors) == 1 else output_tensors

        # Reconstruct with None values if needed
        if none_mask and any(none_mask):
            return self._reconstruct_with_none_values(result, none_mask, is_single_output)

        return result

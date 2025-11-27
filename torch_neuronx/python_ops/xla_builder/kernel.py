"""Refactored JAX kernel implementation with single responsibility."""

import logging
import os
import traceback
from collections.abc import Callable

import torch

from torch_neuronx import _C
from torch_neuronx.kernels.base import BaseNeuronKernel
from torch_neuronx.kernels.compiler_config import CompilerConfig
from torch_neuronx.kernels.type_converter import TypeConverter
from torch_neuronx.utils import is_sync_mode_enabled

from ..compilation import CompilationCache, HloCompiler
from ..processors import ArgumentProcessor
from .compilation import XLABuilderCompiler

logger = logging.getLogger(__name__)


class XLABuilderKernel(BaseNeuronKernel):
    """XLA kernel implementation with clean separation of concerns.

    This kernel compiles HLO (High Level Optimizer) functions to NEFF
    and executes them on Neuron devices. It provides a modular architecture
    where each component has a specific responsibility in the compilation
    and execution pipeline.

    Key Components:
        - XLABuilderCompiler: Handles compilation from HLO function to HLO
        - HloCompiler: Manages compilation from HLO to NEFF
        - CompilationCache: Provides caching functionality for compiled NEFF
        - ArgumentProcessor: Handles input preprocessing
        - OutputHandler: Manages output tensor operations

    Attributes:
        hlo_fn (Callable): The HLO function to be compiled and executed
        op_name (str): Operation name used for cache key generation
        output_params (tuple[str, ...]): Names of output tensor parameters
        static_argnums (tuple[int, ...]): Indices of static positional arguments
        static_argnames (tuple[str, ...]): Names of static keyword arguments
        has_collectives (bool): Flag indicating presence of collective operations

    Example:
        >>> kernel = XLABuilderKernel(
        ...     hlo_fn=my_function,
        ...     op_name="custom_op",
        ...     static_argnums=(0,),
        ...     compiler_config=config
        ... )
        >>> output = kernel(input_tensor, out=output)
    """

    def __init__(
        self,
        hlo_fn: Callable,
        op_name: str,
        static_argnums: tuple[int, ...] | None = None,
        static_argnames: tuple[str, ...] | None = None,
        compiler_config: CompilerConfig | None = None,
        output_params: tuple[str, ...] | None = None,
        has_collectives: bool = False,
    ):
        super().__init__()

        # Core function and configuration
        self.hlo_fn = hlo_fn
        self.op_name = op_name
        self.output_params = output_params or ()

        # Initialize components
        self.xla_builder_compiler = XLABuilderCompiler(static_argnames)
        self.hlo_compiler = HloCompiler(compiler_config)
        self.cache = CompilationCache()
        self.arg_processor = ArgumentProcessor(static_argnames)

        # Store static argument configuration
        self.static_argnums = static_argnums or ()
        self.static_argnames = static_argnames or ()
        self.has_collectives = has_collectives

    def __call__(self, *inputs: torch.Tensor, **kwargs) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Execute the kernel.

        Args:
            inputs: Input tensors
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor(s)
        """
        normalized_static_argnums = self._normalize_static_argnums(len(inputs))
        self.arg_processor.static_argnums = normalized_static_argnums
        self.xla_builder_compiler.static_argnums = normalized_static_argnums

        processed_inputs = self.arg_processor.preprocess_inputs(inputs)
        output_tensors = self._prepare_outputs(kwargs)
        is_single_output = not isinstance(output_tensors, tuple)
        if is_single_output:
            output_tensors = (output_tensors,)

        cache_key = self._generate_cache_key(self.op_name, processed_inputs, kwargs)
        device = self.arg_processor.extract_device(inputs)
        input_dict, list_arg_positions, list_arg_lengths = (
            self.arg_processor.prepare_execution_inputs(processed_inputs, device)
        )

        #  TODO (kvshbg) - remove sync flow once async CRs are merged
        if is_sync_mode_enabled():
            self._execute_sync(cache_key, processed_inputs, input_dict, kwargs, output_tensors)
        else:
            self._execute_async(
                cache_key,
                processed_inputs,
                tuple(input_dict.values()),
                kwargs,
                output_tensors,
                list_arg_positions,
                list_arg_lengths,
            )

        # Apply dtype corrections and return
        return self._finalize_outputs(output_tensors, is_single_output)

    def _get_or_compile_neff(self, cache_key: str, inputs: tuple, kwargs: dict) -> bytes:
        """Get NEFF from cache or compile if needed.

        Args:
            inputs: Preprocessed inputs
            kwargs: Keyword arguments

        Returns:
            Compiled NEFF bytes
        """
        # Try cache first
        neff_bytes = self.cache.get_neff(cache_key)
        if neff_bytes is not None:
            return neff_bytes

        hlo = self.xla_builder_compiler.compile_to_hlo(self.hlo_fn, inputs, kwargs)
        # Collectives always use XLA (HLO protobuf), never StableHLO
        ir_type = "XLA"  # xla_builder only support XLA IR
        neff_bytes = self.hlo_compiler.compile_to_neff(
            hlo.module_proto.SerializeToString(), ir_type=ir_type
        )

        # Store in cache
        self.cache.store_neff(cache_key, neff_bytes)

        return neff_bytes

    def _execute_sync(
        self,
        cache_key: str,
        processed_inputs: tuple,
        input_dict: dict,
        kwargs: dict,
        output_tensors: tuple,
    ) -> None:
        """Execute kernel in blocking mode."""
        neff_bytes = self._get_or_compile_neff(cache_key, processed_inputs, kwargs)
        output_dict = {f"output{i}": out for i, out in enumerate(output_tensors)}

        self.execute_neff(
            neff_bytes,
            input_dict,
            output_dict,
            op_name=self.op_name,
            has_collectives=self.has_collectives,
        )

    def _execute_async(
        self,
        cache_key: str,
        processed_inputs: tuple,
        non_static_inputs: tuple,
        kwargs: dict,
        output_tensors: tuple,
        list_arg_positions: list,
        list_arg_lengths: list,
    ) -> None:
        """Execute kernel in async mode."""
        hlo = self.xla_builder_compiler.get_or_compile_hlo(
            cache_key, self.hlo_fn, processed_inputs, kwargs
        )
        hlo_bytes = hlo.module_proto.SerializeToString()

        if os.environ.get("TORCH_NEURONX_ENABLE_STACK_TRACE", "0") == "1":
            stack_trace = "".join(traceback.format_list(traceback.extract_stack()[:-2]))
        else:
            stack_trace = ""

        # Create CPU fallback context using ArgumentProcessor
        cpu_fallback_context = self.arg_processor.create_cpu_fallback_context(
            processed_inputs,
            list_arg_positions or [],
            list_arg_lengths or [],
        )

        try:
            _C._submit_xla_task_to_pipeline(
                self.op_name,
                non_static_inputs,
                output_tensors,
                hlo_bytes,
                None,
                cache_key,
                stack_trace,
                self.has_collectives,
                cpu_fallback_context,
            )
        except RuntimeError as e:
            raise RuntimeError(f"XLA kernel execution failed: {e}") from e

    def _finalize_outputs(
        self, output_tensors: tuple, is_single_output: bool
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Apply dtype corrections and return outputs in original format."""
        corrected_outputs = []
        for tensor in output_tensors:
            if hasattr(tensor, "_expected_dtype") and tensor.dtype != tensor._expected_dtype:
                # Convert to expected dtype (e.g., int32 -> int64)
                tensor = tensor.to(tensor._expected_dtype)
            corrected_outputs.append(tensor)

        output_tensors = tuple(corrected_outputs)
        return output_tensors[0] if is_single_output else output_tensors

    def _prepare_outputs(self, kwargs: dict) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Prepare output tensors.

        Args:
            kwargs: Keyword arguments

        Returns:
            Output tensor(s)
        """
        # Check for provided output tensors
        output = self.extract_output_params(kwargs, self.output_params)
        # Here we only handle the case where the output is provided by
        # the kernel. We don't create the outputs on the fly based on
        # inputs. TODO: Handle the case where the output tensor is not
        # provided by the kernel
        assert output is not None, "output tensor not created by kernel"
        # Here we do the type conversion to handle the int64/float64 tensors
        converted = []
        is_single_output = False
        if isinstance(output, torch.Tensor):
            output = (output,)
            is_single_output = True

        for out in output:
            converted.append(TypeConverter.convert_for_neuron(out))
            converted[-1]._expected_dtype = out.dtype
        return converted[0] if is_single_output else tuple(converted)

    def extract_output_params(
        self, kwargs: dict, output_param_names: tuple
    ) -> torch.Tensor | tuple | None:
        """Extract output tensors from kwargs.

        Args:
            kwargs: Keyword arguments
            output_param_names: Names of output parameters

        Returns:
            Output tensor(s) if found, None otherwise
        """
        if not output_param_names:
            # Check for universal 'out' parameter
            return kwargs.get("out")

        # Collect named output tensors
        output_tensors = []
        for param in output_param_names:
            if param in kwargs:
                output_tensors.append(kwargs[param])

        if not output_tensors:
            return kwargs.get("out")  # Fall back to 'out'
        if len(output_tensors) == 1:
            return output_tensors[0]
        return tuple(output_tensors)

    def _generate_cache_key(self, op_name: str, inputs: tuple, kwargs: dict) -> str:
        """Generate cache key for NEFF lookup.

        Args:
            op_name: Operation name
            inputs: Input tensors
            kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        # Get base cache key
        normalized_static_argnums = self._normalize_static_argnums(len(inputs))
        base_key = self.get_cache_key(op_name, *inputs, static_indices=normalized_static_argnums)

        # Add static kwargs
        return self.arg_processor.add_static_to_cache_key(base_key, kwargs)

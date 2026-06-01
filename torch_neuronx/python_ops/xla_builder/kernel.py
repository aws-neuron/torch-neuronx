"""XLA Builder kernel implementation with single responsibility."""

import logging
import os
import traceback
from collections.abc import Callable

import torch
from torch._C._distributed_c10d import OpType, ReduceOp

from torch_neuronx import _C
from torch_neuronx.kernels.base import BaseNeuronKernel
from torch_neuronx.kernels.compiler_config import CompilerConfig
from torch_neuronx.neuron import current_stream, default_stream
from torch_neuronx.utils import convert_for_neuron, is_host_cc_enabled

from ..compilation import CompilationCache, HloCompiler
from ..processors import ArgumentProcessor
from .compilation import XLABuilderCompiler

logger = logging.getLogger(__name__)

# Mapping from op_name to torch.distributed OpType
_OP_NAME_TO_COLLECTIVE_TYPE = {
    "all_gather": OpType.ALLGATHER,
    "all_reduce": OpType.ALLREDUCE,
    "reduce_scatter": OpType.REDUCE_SCATTER,
    "all_to_all": OpType.ALLTOALL,
}

# Mapping from reduce string to torch.distributed ReduceOp
_REDUCE_STR_TO_OP = {
    "SUM": ReduceOp.SUM,
    "AVG": ReduceOp.AVG,
    "MIN": ReduceOp.MIN,
    "MAX": ReduceOp.MAX,
    "PRODUCT": ReduceOp.PRODUCT,
}


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
        device = self.arg_processor.extract_device(inputs, kwargs)
        input_dict = self.arg_processor.prepare_execution_inputs(processed_inputs, device)
        # Get CPU fallback context stored directly on kernel
        cpu_fallback_context = getattr(self, "_cpu_fallback_context", None)
        # need to handle the case where the kernel is created after impl.execute call
        if cpu_fallback_context is None:
            cpu_fallback_context = {"original_inputs": list(inputs), "original_kwargs": {}}
        # For collective operations on non-default stream, use direct host API path
        # which doesn't require HLO compilation
        if self.has_collectives and not self._is_default_stream() and is_host_cc_enabled():
            self._execute_collective_direct(
                tuple(input_dict.values()),
                kwargs,
                output_tensors,
                cpu_fallback_context,
            )
        else:
            self._execute(
                cache_key,
                processed_inputs,
                tuple(input_dict.values()),
                kwargs,
                output_tensors,
                cpu_fallback_context,
            )

        # Apply dtype corrections and return
        return self._finalize_outputs(output_tensors, is_single_output)

    def _execute(
        self,
        cache_key: str,
        processed_inputs: tuple,
        non_static_inputs: tuple,
        kwargs: dict,
        output_tensors: tuple,
        cpu_fallback_context,
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
            converted.append(convert_for_neuron(out))
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

        Uses C++ CacheKeyGenerator for the base key (strides, offsets, typed lists).
        XLA builder handles kwargs separately via add_static_to_cache_key (no version hash,
        no persistent HLO disk cache).

        Args:
            op_name: Operation name
            inputs: Input tensors
            kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        from torch_neuronx import _C

        normalized_static_argnums = list(self._normalize_static_argnums(len(inputs)))

        # C++ generates the base key (positional args only)
        base_key = _C._generate_cache_key(
            op_name,
            tuple(inputs),
            {},  # kwargs handled below via add_static_to_cache_key
            static_argnums=normalized_static_argnums,
            static_argnames=[],
        )

        # Append static kwargs (XLA builder's existing mechanism, unchanged)
        return self.arg_processor.add_static_to_cache_key(base_key, kwargs)

    def _is_default_stream(self) -> bool:
        """Check if current stream is the default stream.

        Returns:
            True if current stream is the default stream, False otherwise.
        """
        return current_stream() == default_stream()

    def _get_collective_op_type(self) -> OpType:
        """Map op_name to c10d::OpType enum value.

        Returns:
            c10d::OpType enum value corresponding to this operation.

        Raises:
            ValueError: If op_name is not a recognized collective operation.
        """
        op_type = _OP_NAME_TO_COLLECTIVE_TYPE.get(self.op_name)
        if op_type is None:
            raise ValueError(f"Unknown collective operation: {self.op_name}")
        return op_type

    def _get_reduce_op(self, kwargs: dict) -> ReduceOp:
        """Get reduce operation from kwargs, mapped to c10d::ReduceOp enum value.

        Args:
            kwargs: Keyword arguments containing 'reduce' key for reduce operations.

        Returns:
            c10d::ReduceOp enum value. Defaults to SUM if not specified.
        """
        reduce_str = kwargs.get("reduce", "SUM")
        return _REDUCE_STR_TO_OP.get(reduce_str, ReduceOp.SUM)

    def _execute_collective_direct(
        self,
        non_static_inputs: tuple,
        kwargs: dict,
        output_tensors: tuple,
        cpu_fallback_context,
    ) -> None:
        """Execute collective operation directly via host API (no HLO compilation).

        Bypasses XLA compilation and submits collective directly to runtime.
        Used for collective operations on non-default streams.
        """
        stack_trace = ""
        if os.environ.get("TORCH_NEURONX_ENABLE_STACK_TRACE", "0") == "1":
            stack_trace = "".join(traceback.format_list(traceback.extract_stack()[:-2]))

        op_type = self._get_collective_op_type()
        reduce_op = self._get_reduce_op(kwargs)
        replica_groups = kwargs.get("replica_groups", [])

        # AVG not supported by NRT - convert to SUM + post-divide
        needs_avg_divide = reduce_op == ReduceOp.AVG
        if needs_avg_divide:
            reduce_op = ReduceOp.SUM
            # Strip the divisor tensor appended by prepare_avg_inputs()
            non_static_inputs = non_static_inputs[:-1]
            # Get world_size from replica_groups directly to avoid device-to-host sync
            world_size = len(replica_groups[0]) if replica_groups else 1

        def submit(inputs, outputs):
            """Submit collective task to pipeline."""
            try:
                _C._submit_collective_task_to_pipeline(
                    self.op_name,
                    inputs,
                    outputs,
                    op_type,
                    reduce_op,
                    replica_groups,
                    None,
                    stack_trace,
                    cpu_fallback_context,
                )
            except RuntimeError as e:
                raise RuntimeError(f"Collective direct execution failed: {e}") from e

        submit(non_static_inputs, output_tensors)
        if needs_avg_divide:
            for out in output_tensors:
                out.div_(world_size)

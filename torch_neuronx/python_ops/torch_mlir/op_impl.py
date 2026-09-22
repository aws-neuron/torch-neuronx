"""Torch-MLIR operation implementation."""

import logging
import traceback
from collections.abc import Callable

import torch

from torch_neuronx.python_ops.base import CompilableOpImpl, ExecutionResult
from torch_neuronx.python_ops.dtype_autocast import autocast_neuron
from torch_neuronx.python_ops.shared import ExecutionContext

from ..handlers.output import BaseOutputHandler
from .kernel import TorchMlirKernel

logger = logging.getLogger(__name__)


class TorchMlirOpImpl(CompilableOpImpl):
    """Torch-MLIR based operation implementation.

    Executes PyTorch operations by compiling to StableHLO via torch-mlir,
    then to NEFF for Neuron execution.
    """

    def __init__(
        self,
        aten_op_name: str,
        torch_fn: callable,
        output_params: tuple[str, ...] | None = None,
        static_argnums: tuple[int, ...] | None = None,
        static_argnames: tuple[str, ...] | None = None,
        validate_static_inputs_func: Callable | None = None,
    ):
        super().__init__()

        self.aten_op_name = aten_op_name
        self.torch_fn = torch_fn
        self.output_params = output_params
        self.static_argnums = static_argnums or ()
        self.static_argnames = static_argnames or ()
        # Stored on the impl (not the kernel) because for preprocessing ops
        # the kernel is lazily created in execute(), after registration.
        self.validate_static_inputs_func = validate_static_inputs_func

        # Initialize handlers
        self.output_handler = BaseOutputHandler()

        # Check for preprocessing
        self._uses_preprocessing = getattr(self.torch_fn, "preprocessing", None) is not None
        self._postprocess_fn = None
        if not self._uses_preprocessing:
            self.kernel = TorchMlirKernel(
                torch_fn=torch_fn,
                op_name=aten_op_name,
                output_params=output_params,
                static_argnums=static_argnums,
                static_argnames=static_argnames,
            )
        else:
            self.kernel = None

    def execute(self, *args, **kwargs) -> ExecutionResult:
        """Execute the operation.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            ExecutionResult with output tensor(s)
        """
        # Run validator on raw args before preprocessing. Mirrors what the C++
        # dispatch path does in kernel.lower
        if self.validate_static_inputs_func is not None:
            self.validate_static_inputs_func(*args, **kwargs)

        if self._uses_preprocessing:
            preprocess_result = self.torch_fn(*args, **kwargs)

            # check if postprocess function is provided
            if len(preprocess_result) == 4:
                actual_torch_fn, args, kwargs, self._postprocess_fn = preprocess_result
            else:
                actual_torch_fn, args, kwargs = preprocess_result
                self._postprocess_fn = None

            if self.kernel is None:
                self.kernel = TorchMlirKernel(
                    torch_fn=actual_torch_fn,
                    op_name=self.aten_op_name,
                    output_params=self.output_params,
                    static_argnums=self.static_argnums,
                    static_argnames=self.static_argnames,
                )

        exec_context = ExecutionContext(original_inputs=args, original_kwargs=kwargs.copy())

        # CPU fallback for int64 when policy demands it.
        # Covers UNSUPPORTED ops (matmul/conv) plus native_strict, which
        # routes any non-native int64 op away from Neuron. See
        # torch_neuronx.int64_policy.Int64Mode.
        #
        # Routes through PyTorch's at::native::cpu_fallback via the C++
        # _cpu_fallback helper, mirroring what the C++ dispatcher's
        # neuron_cpu_fallback does. This preserves Tensor(a!) write
        # aliasing (out= and inplace), multi-return device placement, and
        # schema-based kwarg filtering — none of which can be done
        # correctly from a hand-rolled tensor.cpu() / .to("neuron") loop.
        from torch_neuronx.int64_policy import Int64Policy

        policy = Int64Policy.get()
        if policy.should_cpu_fallback(self.aten_op_name):
            has_int64 = any(isinstance(a, torch.Tensor) and a.dtype == torch.int64 for a in args)
            if has_int64:
                # auto_registration.py wraps inplace ops by injecting a
                # synthetic ``out=self`` kwarg so downstream Neuron-side
                # bookkeeping has a uniform handle for the write target.
                # Schema-strict ATen ops (e.g. ``aten::zero_(Tensor(a!) self)``)
                # don't declare ``out``; ``createStackForSchema`` would
                # reject the call. Strip the synthetic kwarg before the
                # fallback — ``self`` is already the first positional arg
                # and the Tensor(a!) alias is enough for cpu_fallback to
                # mutate it in place. Mirrors the equivalent strip in
                # python_ops/base.py for the non-int64 CPU fallback.
                fallback_kwargs = kwargs
                # Inplace detection mirrors Operation.is_inplace in base.py:
                # an op is inplace if its name contains ``_.`` (overloaded
                # inplace, e.g. ``aten::add_.Tensor``) or ends in ``_`` with
                # no overload (e.g. ``aten::zero_``).
                op_name = self.aten_op_name
                is_inplace = "_." in op_name or (op_name.endswith("_") and "." not in op_name)
                if (
                    is_inplace
                    and "out" in fallback_kwargs
                    and len(args) > 0
                    and fallback_kwargs["out"] is args[0]
                ):
                    fallback_kwargs = {k: v for k, v in fallback_kwargs.items() if k != "out"}

                from torch_neuronx import _C

                result = _C._cpu_fallback(self.aten_op_name, *args, **fallback_kwargs)
                return ExecutionResult(success=True, output=result)

        # Apply dtype autocast only if not disabled
        if not self.DISABLE_DTYPE_AUTOCAST:
            # When native int64 is enabled, don't cast int64 — S64 passes through to compiler.
            # When disabled, cast int64→int32 (old behavior).
            with autocast_neuron(
                cast_dtype_params=True,
                cast_int64=not policy.native_int64_enabled,
            ) as context:
                # Process arguments, excluding output parameters from casting
                original_kwargs = kwargs
                args, kwargs, _ = context.process_args(
                    args, kwargs, output_params=self.output_params
                )

                # Validate output dtypes
                context.validate_output_dtypes(original_kwargs)

        # Execute kernel
        try:
            result = self.kernel(*args, context=exec_context, **kwargs)

            # Apply postprocessing if available
            if self._postprocess_fn is not None:
                result = self._postprocess_fn(result)

            return ExecutionResult(success=True, output=result)
        except Exception as e:
            # Do not fall back for user/input contract errors; match PyTorch behavior
            error_traceback = traceback.format_exc()
            if isinstance(e, IndexError | ValueError | TypeError):
                raise
            error_message = f"{e!s}\n{error_traceback}"
            logger.debug(f"Execution failed: {error_message}")
            return ExecutionResult(success=False, error_msg=error_message)

    def _execute_impl(self):
        pass

    def _should_handle_empty(self, args) -> bool:
        """Check if any input tensors are empty."""
        return any(isinstance(arg, torch.Tensor) and arg.numel() == 0 for arg in args)

    def _get_device(self, args: tuple) -> torch.device:
        """Get device from arguments.

        Args:
            args: Input arguments

        Returns:
            Device from first tensor, or CPU
        """
        for arg in args:
            if isinstance(arg, torch.Tensor):
                return arg.device
        return torch.device("cpu")

    def _handle_empty_tensors(self, args, kwargs, context):
        """Handle empty tensor cases."""
        output_specs, _, _ = self.kernel._get_cached_output_specs(args, kwargs, context=context)
        device = self._get_device(args)

        outputs = []
        for output_spec in output_specs:
            outputs.append(torch.zeros(output_spec[0], dtype=output_spec[1], device=device))

        return tuple(outputs)

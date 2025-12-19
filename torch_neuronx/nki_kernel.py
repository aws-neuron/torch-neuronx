"""TorchNeuronNKIKernel - NKI kernel execution for torch-neuronx."""

import base64
import hashlib
import json
import logging
import os
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch
from neuronxcc.nki.compiler.backends.neuron import sema
from neuronxcc.nki.compiler.backends.neuron.dimensions import serialize_dims
from neuronxcc.nki.compiler.backends.neuron.FrameworkKernel import _base_name
from neuronxcc.nki.compiler.backends.neuron.metaclasses import modified_tensor_type
from neuronxcc.nki.compiler.backends.neuron.NumpyKernel import BaremetalKernel
from neuronxcc.nki.compiler.backends.neuron.tensor import tensor as nki_tensor
from neuronxcc.nki.compiler.backends.neuron.tensors import (
    KernelTensor,
    TensorRef,
)
from neuronxcc.starfish.penguin.ir.Function import AliasType
from neuronxcc.starfish.penguin.ir.NativeKernel import KERNEL_VERSION
from neuronxcc.starfish.support import dtype as neuron_dtype

from . import _C
from .utils import (
    get_logical_neuron_cores,
    get_platform_target,
    is_neff_cache_disabled,
    log_neff_cache_hit,
    log_neff_cache_miss,
    log_neff_cache_store,
    map_external_dtype_to_torch,
)

logger = logging.getLogger(__name__)
# Sub-logger for return-tensor allocation/override details.
# Users can enable just this category via standard logging config:
#   logging.getLogger("torch_neuronx.nki_kernel.ret").setLevel(logging.DEBUG)
ret_logger = logger.getChild("ret")


@dataclass(frozen=True)
class NKIKernelKey:
    """Immutable cache key for NKI kernel IR caching.

    Captures all input parameters that affect kernel tracing and IR generation:
    - Function identity and compilation options
    - Input arguments (tensors, scalars, tuples, custom objects, etc.)

    Supported argument types:
    - torch.Tensor: captured with shape and dtype
    - tuple: captured with full value
    - int, float, bool, str, None: captured with type and value
    - Custom objects: must have working __str__() method for string representation

    Raises TypeError for unsupported types or objects with broken __str__() methods.
    """

    func_name: str
    compile_opts: str
    input_tensors: tuple[tuple[str, tuple[int, ...], str], ...]  # (name, shape, dtype_str)
    input_scalars: tuple[tuple[str, str, str], ...]  # (name, type_name, value_str)
    input_tuples: tuple[tuple[str, tuple], ...]  # (name, tuple_value)

    @classmethod
    def from_boundargs(cls, func_name: str, boundargs, compile_opts: str = "") -> "NKIKernelKey":
        """Create cache key from bound arguments for IR caching."""
        input_tensors = []
        input_scalars = []
        input_tuples = []

        for name, arg in boundargs.arguments.items():
            if torch.is_tensor(arg):
                input_tensors.append((name, tuple(arg.shape), str(arg.dtype)))
            elif isinstance(arg, tuple):
                input_tuples.append((name, arg))
            elif isinstance(arg, int | float | bool | str) or arg is None:
                input_scalars.append((name, type(arg).__name__, str(arg)))
            else:
                # For custom objects, use type name and string representation
                try:
                    obj_str = str(arg)
                    input_scalars.append((name, type(arg).__name__, obj_str))
                except Exception as e:
                    raise TypeError(
                        f"Cannot create cache key for argument '{name}' of type {type(arg)}. "
                        f"Object must have a working __str__() method for caching. "
                        f"str() failed with: {e}"
                    ) from e

        return cls(
            func_name=func_name,
            compile_opts=compile_opts,
            input_tensors=tuple(sorted(input_tensors)),
            input_scalars=tuple(sorted(input_scalars)),
            input_tuples=tuple(sorted(input_tuples)),
        )


@dataclass(frozen=True)
class TraceResult:
    dumped_config: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    return_types: tuple[tuple[Any, tuple[int, ...]], ...]
    operand_output_aliases: dict[int, int]
    result_is_sequence: bool
    constant_values: tuple
    cache_key: int
    has_collectives: bool


class TorchNeuronNKIKernel(BaremetalKernel):
    """NKI kernel with torch-neuronx execution.

    This class extends BaremetalKernel to execute compiled NKI kernels
    using torch-neuronx's NRT bindings. BaremetalKernel handles compilation
    while this class handles caching and execution.
    """

    # Class-level cache for loaded models
    _loaded_models: ClassVar[dict[str, int]] = {}  # neff_hash -> model_handle
    # Class-level cache for compiled NEFFs by shape/dtype signature
    _neff_cache: ClassVar[dict[str, tuple[str, bytes]]] = {}  # cache_key -> (neff_path, neff_bytes)
    # Class-level cache for traced IR to skip tracing on subsequent calls
    _ir_cache: ClassVar[
        dict[NKIKernelKey, tuple[Any, Any, tuple]]
    ] = {}  # cache_key -> (ir, ret_val, constants)

    def __init__(self, **kwargs):
        self._return_tensor_overrides = kwargs.pop("return_tensor_overrides", None)
        lnc = get_logical_neuron_cores()
        default_compile_opts_auto = " ".join(
            ["-O1", "--model-type", "transformer", "--auto-cast=none", "--lnc", lnc]
        )

        # Check for default compile options from environment variable, otherwise default to
        # default_opts_parts.
        default_compile_opts = os.environ.get("NEURON_NKI_COMPILE_FLAGS", default_compile_opts_auto)

        # Get any additional compile options passed to this kernel
        additional_opts = kwargs.get("additional_compile_opt", "")

        # If no additional options were provided, use the defaults
        if not additional_opts:
            kwargs["additional_compile_opt"] = default_compile_opts
        # Otherwise, merge defaults with provided options (provided options take precedence)
        else:
            kwargs["additional_compile_opt"] = f"{default_compile_opts} {additional_opts}".strip()
        # NKI uses the kernel_return calling convention by default
        kwargs.setdefault("kernel_return", True)
        kwargs["platform_target"] = get_platform_target()
        super().__init__(**kwargs)

    def translate_to_neuron_dtype(self, _dtype):
        """Convert PyTorch dtype to Neuron dtype."""

        torch_to_neuron_dtype = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.float16: np.float16,
            torch.int64: np.int64,
            torch.int32: np.int32,
            torch.int16: np.int16,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
            torch.uint16: np.uint16,
            torch.uint32: np.uint32,
            torch.uint64: np.uint64,
            torch.bool: np.uint8,
            torch.complex64: np.complex64,
            torch.complex128: np.complex128,
            torch.bfloat16: neuron_dtype.bfloat16,
        }

        if _dtype == getattr(torch, "float8_e4m3fn", None):
            raise RuntimeError("float8_e4m3fn is not supported in neuronxcc. ")

        if hasattr(torch, "float8_e5m2"):
            torch_to_neuron_dtype[torch.float8_e5m2] = neuron_dtype.float8_e5m2

        if hasattr(torch, "float8_e4m3fn"):
            torch_to_neuron_dtype[torch.float8_e4m3fn] = neuron_dtype.float8_e4m3fn

        if hasattr(torch, "float8_e4m3fnuz"):
            torch_to_neuron_dtype[torch.float8_e4m3fnuz] = neuron_dtype.float8_e4m3fn

        if hasattr(torch, "float8_e5m2fnuz"):
            torch_to_neuron_dtype[torch.float8_e5m2fnuz] = neuron_dtype.float8_e5m2

        if _dtype in torch_to_neuron_dtype:
            return torch_to_neuron_dtype[_dtype]

        # For other dtype that is common with numpy, use builtin pytorch to do the translation
        return torch.empty(1, dtype=_dtype, device="cpu").numpy().dtype

    def is_framework_tensor(self, t):
        """Check if this is a PyTorch tensor."""
        return torch.is_tensor(t)

    def map_framework_tensor(self, t):
        """Extract shape and dtype from PyTorch tensor."""
        return t.shape, t.dtype

    def _translate_param(self, ctx, o, name, annotation=None):
        """Override parameter translation to handle PyTorch tensors properly.

        This prevents the base TraceKernel from catching PyTorch tensors
        with data_ptr/dtype attributes and allows them to flow through
        to the baremetal execution path.
        """
        # Check if this is a PyTorch tensor BEFORE calling super()
        # This ensures we handle it before TraceKernel's data_ptr check
        if torch.is_tensor(o):
            # PyTorch tensors should be treated like numpy arrays:
            # create a parameter during trace phase
            numpy_dtype = self.translate_to_neuron_dtype(o.dtype)
            return self.create_parameter(
                ctx=ctx, name=name, shape=tuple(o.shape), dtype=numpy_dtype, annotation=annotation
            )

        # For non-tensors, use the default translation
        return super()._translate_param(ctx, o, name, annotation)

    def _generate_ir_cache_key(self, boundargs) -> NKIKernelKey:
        """Generate cache key from boundargs for IR caching."""
        compile_opts = getattr(self, "additional_compile_opt", "")
        return NKIKernelKey.from_boundargs(
            func_name=self.func.__name__, boundargs=boundargs, compile_opts=compile_opts
        )

    def _get_original_tensor_name(self, ir_name):
        """Extract original parameter name from IR tensor name.

        For mutable tensors, NKI compiler transforms 'input' -> 'input.must_alias_input'
        This method extracts 'input' from 'input.must_alias_input'
        """
        if "must_alias" in ir_name:
            return ir_name.split(".")[0]
        return ir_name

    def _generate_neff_cache_key(self, ir, boundargs) -> str:
        """Generate cache key from IR and boundargs for NEFF caching."""
        key_parts = []

        # Add function name
        key_parts.append(self.func.__name__)

        # Add input tensor info
        for intensor in ir.inputs:
            lookup_name = self._get_original_tensor_name(intensor.name)
            tensor = self.load_tensor_from_arguments(boundargs.arguments, lookup_name)
            if torch.is_tensor(tensor):
                key_parts.extend([intensor.name, str(tensor.shape), str(tensor.dtype)])

        # Add output tensor info
        for outtensor in ir.outputs:
            lookup_name = self._get_original_tensor_name(outtensor.name)
            tensor = self.load_tensor_from_arguments(boundargs.arguments, lookup_name)
            if torch.is_tensor(tensor):
                key_parts.extend([outtensor.name, str(tensor.shape), str(tensor.dtype)])

        # Add scalar parameters (non-tensor arguments)
        # This ensures different scalar values create different NEFF cache entries
        for name, arg in boundargs.arguments.items():
            if not torch.is_tensor(arg):
                key_parts.extend([name, str(type(arg).__name__), str(arg)])

        return "_".join(key_parts)

    def call_impl(self, boundargs):
        """Override to implement IR caching and skip tracing on subsequent calls."""
        # Check if caching is disabled
        cache_disabled = is_neff_cache_disabled()

        # Generate cache key from boundargs (before tracing)
        cache_key = self._generate_ir_cache_key(boundargs)

        # Check IR cache first (unless caching is disabled)
        if not cache_disabled and cache_key in self._ir_cache:
            ir, ret_val, constants = self._ir_cache[cache_key]
            self._code = ir

            # Return cached ret_val for parent's __call__ to pass to post_process_call
            return ret_val

        # No cache hit - proceed with normal tracing via parent's specialize_and_call
        ret_val, ir, constants = self.specialize_and_call(boundargs)

        # Cache the IR, ret_val, and constants (unless caching is disabled)
        if not cache_disabled:
            self._ir_cache[cache_key] = (ir, ret_val, constants)

        # Return the result (post_process_call will be called by parent)
        return ret_val

    def _ensure_output_tensors(self, ir, boundargs):
        """Ensure boundargs has torch tensors allocated for each IR output."""

        def _pick_device():
            for v in boundargs.arguments.values():
                if torch.is_tensor(v):
                    return v.device
            return torch.device("neuron", 0)

        def _to_torch_dtype(nki_dtype):
            return map_external_dtype_to_torch(nki_dtype)

        device = _pick_device()
        for outtensor in ir.outputs:
            name = outtensor.name
            if name in boundargs.arguments and torch.is_tensor(boundargs.arguments[name]):
                continue
            shape = tuple(int(s) for s in outtensor.shape)
            dtype = _to_torch_dtype(outtensor.dtype)
            boundargs.arguments[name] = torch.empty(shape, dtype=dtype, device=device)

    def post_process_call(self, ir, ret, boundargs):
        """Override to implement NEFF caching before compilation."""
        # Make sure output placeholders exist before using them (for cache key)
        self._ensure_output_tensors(ir, boundargs)
        # Generate cache key based on shapes/dtypes
        cache_key = self._generate_neff_cache_key(ir, boundargs)

        # Check if caching is disabled
        cache_disabled = is_neff_cache_disabled()

        # Check if we have a cached NEFF for this configuration (unless caching is disabled)
        if not cache_disabled and cache_key in self._neff_cache:
            # Log cache hit
            log_neff_cache_hit(cache_key)

            # Use cached NEFF
            neff_path, neff_bytes = self._neff_cache[cache_key]

            # Execute the cached NEFF
            ret = self.create_return_tensors(ret, boundargs=boundargs)
            self._ensure_output_tensors(ir, boundargs)
            ntff_name = self.execute_neff(neff_bytes, ir=ir, boundargs=boundargs)
            return ret

        # Log cache miss
        log_neff_cache_miss(cache_key)

        # No cache hit - compile and cache
        ret = self.create_return_tensors(ret, boundargs=boundargs)
        self._ensure_output_tensors(ir, boundargs)
        cwd = os.getcwd()

        try:
            with self.create_tmp_dir() as tempdir:
                os.chdir(tempdir)

                # Compile
                neff = self._compile(ir)

                # Read the compiled NEFF for caching
                with open(neff, "rb") as f:
                    neff_bytes = f.read()

                # Cache it (unless caching is disabled)
                if not cache_disabled:
                    self._neff_cache[cache_key] = (neff, neff_bytes)
                    log_neff_cache_store(cache_key)

                # Execute
                ntff_name = self.execute_neff(neff_bytes, ir=ir, boundargs=boundargs)
                self.post_process_artifacts(cwd, neff=neff, ntff_name=ntff_name)
                return ret
        finally:
            os.chdir(cwd)

    # Allocate return tensors as torch tensors on Neuron device when
    # NKI kernels return outputs instead of taking them as arguments.
    def create_return_tensors(self, ret, boundargs):  # type: ignore[override]
        def _pick_device():
            for v in boundargs.arguments.values():
                if torch.is_tensor(v):
                    return v.device
            return torch.device("neuron", 0)

        def _to_torch_dtype(nki_dtype):
            return map_external_dtype_to_torch(nki_dtype)

        def _maybe_override(name, shape, tdtype, index=None):
            ov = self._return_tensor_overrides
            override = None
            if isinstance(ov, dict) and name in ov:
                override = ov[name]
            elif isinstance(ov, list | tuple) and index is not None and 0 <= index < len(ov):
                override = ov[index]
            elif torch.is_tensor(ov) and index in (None, 0):
                override = ov

            if override is None:
                return None

            if not torch.is_tensor(override):
                ret_logger.debug("Ignoring non-tensor override for %s: %s", name, type(override))
                return None

            if tuple(override.shape) != tuple(shape) or override.dtype != tdtype:
                ret_logger.debug(
                    "Override for %s has mismatched shape/dtype. "
                    "expected shape=%s, dtype=%s; got shape=%s, dtype=%s",
                    name,
                    shape,
                    tdtype,
                    tuple(override.shape),
                    override.dtype,
                )
                return None
            if override.device.type != "neuron":
                ret_logger.debug(
                    "Override for %s must be on Neuron device; got %s",
                    name,
                    override.device,
                )
                return None
            if not override.is_contiguous():
                ret_logger.debug(
                    "Override for %s is not contiguous; making contiguous copy",
                    name,
                )
                override = override.contiguous()

            return override

        # Single tensor
        if isinstance(ret, TensorRef | KernelTensor):
            t = ret.tensor
            shape = getattr(t, "sourceShape", None) or getattr(t, "shape", None)
            tdtype = _to_torch_dtype(getattr(t, "dtype", None))
            device = _pick_device()
            out_tensor = _maybe_override(getattr(t, "name", ""), shape, tdtype, index=0)
            if out_tensor is None:
                out_tensor = torch.empty(shape, dtype=tdtype, device=device)
            boundargs.arguments[t.name] = out_tensor
            ret_logger.debug(
                "Allocated return tensor name=%s, shape=%s, dtype=%s, device=%s",
                t.name,
                shape,
                tdtype,
                device,
            )
            return out_tensor

        # Tuple/list of tensors
        if isinstance(ret, list | tuple):
            new_rets = []
            for idx, item in enumerate(ret):
                if isinstance(item, TensorRef | KernelTensor):
                    t = item.tensor
                    shape = getattr(t, "sourceShape", None) or getattr(t, "shape", None)
                    tdtype = _to_torch_dtype(getattr(t, "dtype", None))
                    device = _pick_device()
                    out_tensor = _maybe_override(getattr(t, "name", ""), shape, tdtype, index=idx)
                    if out_tensor is None:
                        out_tensor = torch.empty(shape, dtype=tdtype, device=device)
                    boundargs.arguments[t.name] = out_tensor
                    ret_logger.debug(
                        "Allocated return tensor name=%s, shape=%s, dtype=%s, device=%s",
                        t.name,
                        shape,
                        tdtype,
                        device,
                    )
                    new_rets.append(out_tensor)
                else:
                    new_rets.append(item)
            return type(ret)(new_rets)

        # Fallback for non-NKI return types (e.g., None/scalars/unsupported containers).
        # This path does not honor return_tensor_overrides and may allocate fresh tensors,
        # incurring extra copies instead of writing directly into user-provided outputs.
        return super().create_return_tensors(ret, boundargs)

    def _validate_tensor(self, tensor, name):
        """Validate that tensor is a PyTorch tensor on Neuron device."""
        if not torch.is_tensor(tensor):
            raise TypeError(
                f"TorchNeuronNKIKernel expects PyTorch tensors, but {name} is {type(tensor)}."
            )

        if tensor.device.type != "neuron":
            raise ValueError(
                f"TorchNeuronNKIKernel expects tensors on Neuron device, but {name} is on "
                f"{tensor.device}. Move tensor to Neuron device first with tensor.to('neuron')."
            )

    def execute_neff(self, neff_bytes: bytes, ir: Any, boundargs: Any) -> str:
        """Execute the compiled NEFF using torch-neuronx's NRT.

        Args:
            neff: Path to the compiled NEFF file
            ir: Intermediate representation with input/output info
            boundargs: Bound arguments containing tensor data

        Returns:
            Trace filename for compatibility
        """
        # Create tensor sets
        input_set = _C._nrt_allocate_tensor_set()
        output_set = _C._nrt_allocate_tensor_set()

        # Ensure that all operations prior to the NKI execution are completed, in case
        # prior operations are dispatched asynchronously.
        # TODO: Remove once NKI is extended to go through the C++ async pipeline.
        torch.neuron.synchronize()

        try:
            tensor_devices = []
            # Add inputs to tensor set
            for intensor in ir.inputs:
                lookup_name = self._get_original_tensor_name(intensor.name)
                tensor = self.load_tensor_from_arguments(boundargs.arguments, lookup_name)
                self._validate_tensor(tensor, intensor.name)
                # Ensure tensor is contiguous for NKI kernel execution
                # NKI kernels expect contiguous memory layout
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                    # Update the boundargs with the contiguous tensor
                    boundargs.arguments[intensor.name] = tensor
                tensor_devices.append(tensor.device)
                # Ensure that all operations prior to the NKI execution are completed, in case
                # prior operations are dispatched asynchronously.
                # TODO: Remove once NKI is extended to go through the C++ async pipeline.
                torch.neuron.synchronize()
                _C._nrt_add_tensor_to_tensor_set(input_set, tensor, intensor.name)

            # Add outputs to tensor set
            for outtensor in ir.outputs:
                lookup_name = self._get_original_tensor_name(outtensor.name)
                tensor = self.load_tensor_from_arguments(boundargs.arguments, lookup_name)
                self._validate_tensor(tensor, outtensor.name)
                tensor_devices.append(tensor.device)
                # Ensure that all operations prior to the NKI execution are completed, in case
                # prior operations are dispatched asynchronously.
                # TODO: Remove once NKI is extended to go through the C++ async pipeline.
                torch.neuron.synchronize()
                _C._nrt_add_tensor_to_tensor_set(output_set, tensor, outtensor.name)

            assert all(tensor_devices[0] == device for device in tensor_devices), (
                f"All input and output tensors should be on the same device,"
                f"received {tensor_devices}"
            )

            # Load NEFF if not already loaded (using content hash)
            neff_hash = hashlib.sha256(neff_bytes).hexdigest()
            if neff_hash not in self._loaded_models:
                model_handle = _C._nrt_load(neff_bytes, tensor_devices[0].index)
                self._loaded_models[neff_hash] = model_handle
            else:
                model_handle = self._loaded_models[neff_hash]

            # Execute
            _C._nrt_execute(model_handle, input_set, output_set)

        finally:
            # Cleanup tensor sets
            _C._nrt_destroy_tensor_set(input_set)
            _C._nrt_destroy_tensor_set(output_set)

        # Return trace filename (for compatibility)
        return self.save_trace_name or "profile.ntff"

    def dump_config(self, *args, **kwargs):
        r"""Returns the `backend_config`, the list of input names and the list of the output name,
        based on given arguments.

        If `self.enable_cache` is True, `dump_config` will try to retrieve the results
        from the cache using `args`, `kwargs` and the spmd launch grid and other
        kernel attributes as key to identify the identical backend_config.

        Otherwise, `dump_config` will always generate new backend_config.

        # NOTE: THis is still used by legacy framework code, dont change the signature
        """
        with self.bind_arguments(*args, **kwargs) as boundargs:
            return self.dump_config_with_boundargs(boundargs)

    def dump_config_with_boundargs(self, boundargs):
        self._map_args(boundargs)
        hash_key = self._generate_hash_key(boundargs.arguments)

        cache = getattr(self.func, "__neuron_kernel_interface_kernel_cache__", {})
        existing_config = cache.get(hash_key, None)
        if existing_config is not None:
            return existing_config

        ret_val, code, constants = self.specialize_and_call(boundargs)
        dumped_config = self.encode_backend_config(code)
        has_collectives = self.has_collectives

        assert (
            constants is not None
        ), "enable_const_rewrite should be turned on for framework kernels."

        return_names = set()
        result_is_sequence = False
        if isinstance(ret_val, Sequence):
            return_names = {_base_name(t) for t in ret_val}
            result_is_sequence = True
        elif ret_val:
            return_names = {_base_name(ret_val)}

        input_names = tuple(t.name for t in code.inputs)
        input_names = input_names[: len(input_names) - len(constants)]
        output_names = tuple(t.name for t in code.outputs if t.name not in return_names)
        return_types = self.translate_return_types(ret_val)

        operand_output_aliases = self.generate_operand_output_aliases(code)

        result = self.assemble_result(
            dumped_config,
            input_names=input_names,
            output_names=output_names,
            return_types=return_types,
            operand_output_aliases=operand_output_aliases,
            result_is_sequence=result_is_sequence,
            constant_values=constants,
            cache_key=hash_key,
            has_collectives=has_collectives,
        )

        if self.enable_cache:
            cache[hash_key] = result

        return result

    def _map_to_decltensor_or_passthrough(self, t):
        if isinstance(t, tuple) and hasattr(t, "_fields"):
            return t
        if isinstance(t, tuple | list):
            res = [self._map_to_decltensor_or_passthrough(e) for e in t]
            if isinstance(t, tuple):
                res = tuple(res)
            return res
        if isinstance(t, dict):
            return {k: self._map_to_decltensor_or_passthrough(v) for k, v in t.items()}
        if not self.is_framework_tensor(t):
            return t

        try:
            shape, dtype = self.map_framework_tensor(t)
        except ValueError:
            raise ValueError("map_framework_tensor must return a tuple of shape and type") from None

        return nki_tensor[shape, self.translate_to_neuron_dtype(dtype)]

    def _return_shape_dtype_or_hashable(self, t, name):
        if isinstance(t, modified_tensor_type):
            return (t.tensor_shape, t.tensor_dtype)

        if isinstance(t, str):
            return t

        if isinstance(t, Sequence):
            return tuple(
                self._return_shape_dtype_or_hashable(e, name=f"{name}[{i}]")
                for i, e in enumerate(t)
            )
        elif isinstance(t, dict):
            return tuple(
                (k, self._return_shape_dtype_or_hashable(v, name=f"{name}[{k}]"))
                for k, v in sorted(t.items())
            )

        if not isinstance(t, Hashable):
            sema.err_nki_param_not_hashable(name=name, ty=type(t))
        return t

    def _map_args(self, boundargs):
        for name, value in boundargs.arguments.items():
            boundargs.arguments[name] = self._map_to_decltensor_or_passthrough(value)

    def _generate_hash_key(self, args):
        hash_key = [
            self._return_shape_dtype_or_hashable(arg, name=name) for name, arg in args.items()
        ]

        hash_key.append(tuple(self.grid))
        hash_key.append(self.opts)
        hash_key.append(self._platform_target_is_fallback)

        return hash(tuple(hash_key))

    def generate_operand_output_aliases(self, code):
        outputs = code.outputs

        def get_alias_output_index(aliased_tensor, alias_type):
            assert alias_type == AliasType.must_alias, f"Unexpected alias type {alias_type}!"
            return outputs.index(aliased_tensor)

        operand_output_aliases = {
            i: get_alias_output_index(aliased_tensor, alias_type)
            for i, in_tensor in enumerate(code.inputs)
            for aliased_tensor, alias_type in code.findAliasTensor(in_tensor)
        }
        return operand_output_aliases

    def encode_backend_config(self, code):
        config = {}
        config["kernel_version"] = KERNEL_VERSION
        config["func_literal"] = code.serialize_ir_string(f"{self.func.__name__}_ir")
        config["grid"] = serialize_dims(self.grid)
        config["func_name"] = self.func.__name__
        config["has_collectives"] = self.has_collectives
        config["mac_count"] = self.matmul_mac_count
        config["tiled"] = code.has_attr("tiled")
        if not self._platform_target_is_fallback:
            config["platform_target"] = self.opts.platform_target
        dumped_config = base64.b64encode(json.dumps(config).encode("utf-8")).decode("utf-8")
        return dumped_config

    def assemble_result(
        self,
        dumped_config,
        input_names,
        output_names,
        return_types,
        operand_output_aliases,
        result_is_sequence,
        constant_values,
        cache_key,
        has_collectives,
    ):
        result = TraceResult(
            dumped_config=dumped_config,
            input_names=input_names,
            output_names=output_names,
            return_types=return_types,
            operand_output_aliases=operand_output_aliases,
            result_is_sequence=result_is_sequence,
            constant_values=constant_values,
            cache_key=cache_key,
            has_collectives=has_collectives,
        )

        return result

    def translate_return_types(self, ret_val):
        if ret_val is None:
            return []

        if isinstance(ret_val, Sequence):
            return [self.translate_return_type(t) for t in ret_val]

        assert isinstance(ret_val, nki_tensor), "Unexpected return type!"
        return [self.translate_return_type(ret_val)]

    def translate_return_type(self, ret_val):
        assert isinstance(ret_val, nki_tensor), "Unexpected return type!"
        return ret_val.dtype, ret_val.shape

    @property
    def enable_cache(self):
        return True

    @classmethod
    def clear_model_cache(cls):
        """Clear the loaded model cache and unload all models."""
        for neff_hash, model_handle in cls._loaded_models.items():
            try:
                _C._nrt_unload(model_handle)
            except Exception as e:
                # Log but don't fail if unload fails
                logger.warning("Failed to unload model %s: %s", neff_hash, e)
        cls._loaded_models.clear()

    @classmethod
    def clear_neff_cache(cls):
        """Clear the compiled NEFF cache."""
        cls._neff_cache.clear()

    @classmethod
    def clear_ir_cache(cls):
        """Clear the traced IR cache."""
        cls._ir_cache.clear()

    @classmethod
    def clear_all_caches(cls):
        """Clear model, NEFF, and IR caches."""
        cls.clear_model_cache()
        cls.clear_neff_cache()
        cls.clear_ir_cache()

    @classmethod
    def get_ir_cache_size(cls):
        """Get the current size of the IR cache."""
        return len(cls._ir_cache)

    @classmethod
    def get_neff_cache_size(cls):
        """Get the current size of the NEFF cache."""
        return len(cls._neff_cache)

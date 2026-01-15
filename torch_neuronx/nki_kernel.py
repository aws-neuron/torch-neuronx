"""TorchNeuronNKIKernel - NKI kernel execution for torch-neuronx."""

from collections.abc import Hashable, Sequence

import numpy as np
import torch
from neuronxcc.nki.compiler.backends.neuron.FrameworkKernel import UnifiedKernel as UnifiedKernelV1

try:
    from nki.compiler.backends.neuron.FrameworkKernel import UnifiedKernel as UnifiedKernelV2

    IS_NKI_V2_AVAILABLE = True
except ImportError:
    IS_NKI_V2_AVAILABLE = False

    class UnifiedKernelV2:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Placeholder class - cannot instantiate")


from neuronxcc.starfish.support import dtype as neuron_dtype

from .utils import get_platform_target


class TorchNeuronNKIKernel:
    """NKI kernel with torch-neuronx execution.

    This class extends FrameworkKernel to execute NKI kernels.
    """

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

    def dump_config_with_boundargs(self, boundargs):
        hash_key = self._generate_hash_key(boundargs.arguments)
        cache = getattr(self.func, "__neuron_kernel_interface_kernel_cache__", {})
        existing_config = cache.get(hash_key, None)
        if existing_config is not None:
            return existing_config
        self._map_args(boundargs)
        result = super().dump_config_with_boundargs(boundargs)
        cache[hash_key] = result
        return result


class TorchNeuronNKIKernelV1(TorchNeuronNKIKernel, UnifiedKernelV1):
    pass


class TorchNeuronNKIKernelV2(TorchNeuronNKIKernel, UnifiedKernelV2):
    def _map_to_decltensor_or_passthrough(self, t):
        if isinstance(t, (tuple | list)):
            res = [self._map_to_decltensor_or_passthrough(e) for e in t]
            if isinstance(t, tuple):
                res = tuple(res)
            return res
        if isinstance(t, dict):
            return {k: self._map_to_decltensor_or_passthrough(v) for k, v in t.items()}
        if not self.is_framework_tensor(t):
            return t

        return torch.empty_like(t, device="meta")

    def _return_shape_dtype_or_hashable(self, t, name):
        if isinstance(t, torch.Tensor):
            return (t.shape, t.dtype)

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
            raise RuntimeError("sema.err_nki_param_not_hashable(name=name, ty=type(t))")
        return t

    def _platform_target(self, opts):
        return get_platform_target()

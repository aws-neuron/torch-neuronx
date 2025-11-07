import functools
from functools import partial, wraps
from unittest.mock import patch

import torch
from torch.testing._internal.distributed._shard import sharded_tensor

try:
    import torch_neuron

    HAS_NEURON = True
except ImportError:
    HAS_NEURON = False


def _create_noop_decorator():
    def decorator(func):
        return func

    return decorator


NEURON_DIST_TIMEOUT_DEFAULT = 7200


def patch_skip_decorator_early():
    """
    Patches the skip_if_lt_x_gpu decorator in PyTorch's distributed testing module
    to allow tests to run regardless of GPU count.

    Returns:
        function: A dummy decorator that doesn't skip any tests
    """
    import torch.testing._internal.common_distributed

    def neuron_skip_if_lt_x_gpu(x):
        return _create_noop_decorator()

    def patch_requires_nccl():
        return _create_noop_decorator()

    # Patch with_comms to use neuron backend by default
    def patch_with_comms(func=None, init_rpc=True, backend="neuron"):
        if func is None:
            return partial(patch_with_comms, init_rpc=init_rpc, backend=backend)

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Skip the device count check for neuron backend
            self.init_comms(init_rpc=init_rpc, backend=backend)
            func(self, *args, **kwargs)
            self.destroy_comms(destroy_rpc=init_rpc)

        return wrapper

    torch.testing._internal.common_distributed.skip_if_lt_x_gpu = neuron_skip_if_lt_x_gpu
    torch.testing._internal.common_distributed.requires_nccl = patch_requires_nccl
    sharded_tensor.with_comms = patch_with_comms


def patch_requires_accelerator_backend_early():
    def neuron_requires_accelerator_dist_backend(backends):
        def decorator(func):
            return func

        return decorator

    torch.testing._internal.common_distributed.requires_accelerator_dist_backend = (
        neuron_requires_accelerator_dist_backend
    )


def patch_distributed_tests_timeout():
    import torch.testing._internal.common_distributed

    torch.testing._internal.common_distributed.TIMEOUT_DEFAULT = NEURON_DIST_TIMEOUT_DEFAULT


def patch_instantiate_device_type_tests_neuron():
    """
    Patches instantiate_device_type_tests to parametrize for neuron.
    """
    from torch.testing._internal.common_device_type import instantiate_device_type_tests

    original_instantiate = instantiate_device_type_tests

    def patched_instantiate(test_class, global_dict, only_for=None, allow_xpu=None):
        return original_instantiate(test_class, global_dict, only_for=("neuron",))

    torch.testing._internal.common_device_type.instantiate_device_type_tests = patched_instantiate


def patch_common_fsdp():
    from torch.testing._internal import common_fsdp

    common_fsdp.DEVICE_TYPE = "neuron"
    common_fsdp.DISTRIBUTED_BACKEND = "neuron"


def patch_distributed_backend_early():
    """
    Patches torch.distributed.init_process_group to automatically replace
    nccl/gloo backends with neuron backend. Uses inspect.signature to handle
    backend parameter regardless of whether it's passed positionally or as keyword.
    """
    import inspect

    import torch.distributed as dist

    _original_init_process_group = dist.init_process_group
    _signature = inspect.signature(_original_init_process_group)

    def neuron_init_process_group(*args, **kwargs):
        # Map arguments to parameter names using function signature
        bound = _signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        # Replace nccl/gloo with neuron backend
        if "backend" in bound.arguments and bound.arguments["backend"] in ["nccl", "gloo"]:
            bound.arguments["backend"] = "neuron"

        # Call original function with modified arguments
        return _original_init_process_group(*bound.args, **bound.kwargs)

    dist.init_process_group = neuron_init_process_group


def pre_run_patches():
    # Running Pre-run Patches.
    patch_skip_decorator_early()
    patch_requires_accelerator_backend_early()
    patch_distributed_tests_timeout()
    patch_distributed_backend_early()
    patch_instantiate_device_type_tests_neuron()


class NeuronPatcher:
    @staticmethod
    def patch_current_device():
        """
        Creates a patch for the current device function that returns current device index.

        Returns:
            function: A function that returns current device index
        """
        return lambda: torch_neuron.current_device()

    @staticmethod
    def patch_device_count():
        """
        Creates a patch for the device count function that returns the number of
        available Neuron devices or 0 if Neuron is not available.

        Returns:
            function: A function that returns the Neuron device count
        """

        def get_device_count():
            if HAS_NEURON:
                count = torch_neuron.device_count()
                return count
            return 0

        return get_device_count

    @staticmethod
    def patch_set_device():
        """
        Creates a patch for the set_device function that sets the Neuron device to 0
        if Neuron is available.

        Returns:
            function: A function that sets the Neuron device
        """

        def patch_setdevice(device):
            if HAS_NEURON:
                if isinstance(device, int):
                    torch_neuron.set_device(device)
                elif hasattr(device, "index") and device.index is not None:
                    torch_neuron.set_device(device.index)
                else:
                    torch_neuron.set_device(0)

        return patch_setdevice

    @staticmethod
    def patch_tensor_cuda(tensor_self, device=None, *args, **kwargs):
        """
        Patches the tensor.cuda() method to move tensors to Neuron device instead of CUDA.

        Args:
            tensor_self: The tensor instance
            device: The target device
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            torch.Tensor: The tensor moved to Neuron device
        """

        return tensor_self.to("neuron")

    @staticmethod
    def patch_is_cuda_property():
        """
        Creates a patch for the is_cuda property to check if a tensor is on a Neuron device.

        Returns:
            function: A getter function that checks if tensor is on Neuron device
        """

        def getter(tensor_self):
            try:
                return tensor_self.device.type in ("neuron")
            except AttributeError:
                return False

        return getter

    @staticmethod
    def create_tensor_function_wrapper(original_func):
        """
        Creates a wrapper for tensor creation functions that redirects CUDA device
        specifications to Neuron.

        Args:
            original_func: The original tensor creation function

        Returns:
            function: Wrapped function that redirects CUDA to Neuron
        """

        def wrapper(*args, **kwargs):
            if "device" in kwargs:
                device = kwargs["device"]
                if (
                    (isinstance(device, str)) and (device == "cuda" or device.startswith("cuda:"))
                ) or ((isinstance(device, torch.device)) and (device.type == "cuda")):
                    kwargs["device"] = "neuron"
            return original_func(*args, **kwargs)

        return wrapper

    @classmethod
    def patch_module_to(cls):
        """
        Patches the torch.nn.Module.to() method to redirect CUDA device specifications
        to Neuron.

        Returns:
            function: Patch to() method that handles device conversion to Neuron
        """
        original_module_to = torch.nn.Module.to

        def patch_to(self, device=None, *args, **kwargs):
            if device is not None:
                if isinstance(device, int):
                    return original_module_to(self, f"neuron:{device}", *args, **kwargs)
                elif isinstance(device, str):
                    if device.startswith("cuda:") or device == "cuda":
                        return original_module_to(self, "neuron", *args, **kwargs)
                elif hasattr(device, "type") and device.type == "cuda":
                    return original_module_to(self, torch.device("neuron"), *args, **kwargs)
            return original_module_to(self, device, *args, **kwargs)

        return patch_to

    @staticmethod
    def patch_tensor_to_method():
        """
        Patches the tensor.to() method to handle device conversion to Neuron.

        Returns:
            function: Patches to() method that redirects CUDA device specifications to Neuron
        """
        original_to = torch.Tensor.to

        def patch_to(tensor_self, device=None, *args, **kwargs):
            if device is not None:
                if isinstance(device, int):
                    return original_to(tensor_self, "neuron", *args, **kwargs)
                elif isinstance(device, str):
                    if device.startswith("cuda:") or device == "cuda":
                        return original_to(tensor_self, "neuron", *args, **kwargs)
                elif hasattr(device, "type") and device.type == "cuda":
                    return original_to(tensor_self, torch.device("neuron"), *args, **kwargs)
            return original_to(tensor_self, device, *args, **kwargs)

        return patch_to

    @classmethod
    def patch_module_cuda(cls):
        """
        Patches torch.nn.Module.cuda() method to use Neuron instead of CUDA.
        Lookout for changes after neuron device update.

        Returns:
            function: Patched cuda() method that moves module to Neuron device
        """

        def patch_cuda(self, device=None, *args, **kwargs):
            return self.to("neuron", *args, **kwargs)

        return patch_cuda

    @classmethod
    def patch_device_class(cls):
        """
        Creates a patched device class that redirects CUDA device specifications to Neuron.
        """
        original_torch_device = torch.device

        class PatchDevice:
            def __new__(cls, device_type=None, index=None, *args, **kwargs):
                if device_type is None:
                    return original_torch_device(*args, **kwargs)

                if isinstance(device_type, torch.device):
                    device_str = str(device_type)
                else:
                    device_str = str(device_type)
                    # Handle case where device_type is just a number (device index)
                    if device_str.isdigit():
                        device_str = f"neuron:{device_str}"
                    elif index is not None:
                        device_str = f"{device_str}:{index}"

                if device_str.startswith("cuda"):
                    device_str = device_str.replace("cuda", "neuron")
                return original_torch_device(device_str, *args, **kwargs)

        return PatchDevice

    @staticmethod
    def patch_requires_cuda():
        """
        Patches the requires_cuda decorator to handle Neuron availability instead of CUDA.

        Returns:
            unittest.mock.patch: A patch object that modifies the requires_cuda decorator
        """
        try:
            from torch.testing._internal.common_utils import requires_cuda

            def neuron_decorator(func):
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    if not HAS_NEURON:
                        import pytest

                        pytest.skip("Neuron not available")
                    return func(*args, **kwargs)

                return wrapper

            return patch("torch.testing._internal.common_utils.requires_cuda", neuron_decorator)
        except ImportError:
            return patch("builtins.id", side_effect=lambda x: x)

    @staticmethod
    def patch_cuda_peer_access():
        """
        Patches torch._C._cuda_canDeviceAccessPeer to always return True for Neuron devices.

        Returns:
            unittest.mock.patch: A patch object that modifies the peer access function
        """

        # TODO: ADD neuron check peer device access in next revisions
        def neuron_can_device_access_peer(device1, device2):
            return True

        if hasattr(torch._C, "_cuda_canDeviceAccessPeer"):
            return patch.object(
                torch._C, "_cuda_canDeviceAccessPeer", neuron_can_device_access_peer
            )
        else:
            return patch("builtins.id", side_effect=lambda x: x)  # No-op patch

    @staticmethod
    def patch_sm80_or_later():
        """
        Patches SM80OrLater check from torch.testing._internal.common_cuda to True for Neuron.

        Returns:
            unittest.mock.patch: A patch object that modifies SM80OrLater
        """
        try:
            from torch.testing._internal import common_cuda

            return patch.object(common_cuda, "SM80OrLater", True)
        except ImportError:
            return patch("builtins.id", side_effect=lambda x: x)  # No-op patch

    @classmethod
    def collect_patches(cls):
        """
        Collects all patches needed to redirect CUDA operations to Neuron.

        Returns:
            list: A list of patch objects that modify PyTorch's CUDA functionality to use Neuron
        """
        patches = []

        # Patch torch.cuda functions
        patches.append(
            patch.multiple(
                torch.cuda,
                current_device=cls.patch_current_device(),
                device_count=cls.patch_device_count(),
                set_device=cls.patch_set_device(),
            )
        )

        patches.append(patch.object(torch.Tensor, "to", cls.patch_tensor_to_method()))

        # Patch tensor.cuda() method
        patches.append(patch.object(torch.Tensor, "cuda", cls.patch_tensor_cuda))

        # Patch Module.to and Module.cuda
        patches.append(patch.object(torch.nn.Module, "to", cls.patch_module_to()))
        patches.append(patch.object(torch.nn.Module, "cuda", cls.patch_module_cuda()))

        patches.append(patch("torch.device", cls.patch_device_class()))

        # Patch tensor creation functions
        tensor_patches = {}
        for func_name in [
            "ones",
            "zeros",
            "empty",
            "ones_like",
            "zeros_like",
            "empty_like",
            "arange",
            "tensor",
            "rand",
            "randn",
            "full",
        ]:
            if hasattr(torch, func_name):
                orig_func = getattr(torch, func_name)
                tensor_patches[func_name] = cls.create_tensor_function_wrapper(orig_func)

        patches.append(patch.multiple(torch, **tensor_patches))

        # Patch requires_cuda
        patches.append(cls.patch_requires_cuda())

        # Patch CUDA peer access and SM80 checks
        patches.append(cls.patch_cuda_peer_access())
        patches.append(cls.patch_sm80_or_later())

        return patches

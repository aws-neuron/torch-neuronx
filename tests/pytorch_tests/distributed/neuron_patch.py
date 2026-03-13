import functools
import inspect
from functools import partial, wraps
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist

try:
    import torch_neuronx

    HAS_NEURON = True
except ImportError:
    HAS_NEURON = False

from enum import Enum


def _create_noop_decorator():
    def decorator(func):
        return func

    return decorator


NEURON_DIST_TIMEOUT_DEFAULT = 7200


def override_world_size(
    globals_dict: dict[str, Any],
    world_size: int = 2,
    include_test_classes: list[str] | None = None,
    exclude_test_classes: list[str] | None = None,
) -> None:
    """
    Override world_size property for test classes based on include/exclude lists.

    Args:
        globals_dict: Dictionary containing global variables
        world_size: The world_size value to set
        include_test_classes: List of class names to include (if specified, exclude is ignored)
        exclude_test_classes: List of class names to exclude (ignored if include is specified)
    """

    def should_override(name: str) -> bool:
        if include_test_classes is not None:
            target_classes = {"TestNeuron" + class_name for class_name in include_test_classes}
            return name in target_classes
        elif exclude_test_classes is not None:
            excluded_classes = {"TestNeuron" + class_name for class_name in exclude_test_classes}
            return name not in excluded_classes
        else:
            return True

    for name, obj in list(globals_dict.items()):
        # __bases__ is to ensure that we only overide when this is a TestClass
        if hasattr(obj, "world_size") and hasattr(obj, "__bases__") and should_override(name):
            obj.world_size = property(lambda self, ws=world_size: ws)


def patch_module_function_from_cuda_to_neuron(module_path, func_name):
    """
    Template to patch PyTorch functions that redirect CUDA device requests to Neuron.

    Args:
        module_path: Module containing the function (e.g., torch, torch.distributed)
        func_name: Function name to patch (e.g., "get_device_module")
    """

    def patched_func(device_type=None):
        if device_type is None:
            return torch.neuron
        if device_type == "cuda" or (
            isinstance(device_type, torch.device) and device_type.type == "cuda"
        ):
            return torch.neuron
        return torch.neuron

    # Preserve __wrapped__ for dynamo compatibility
    original_func = getattr(module_path, func_name)
    patched_func.__wrapped__ = getattr(original_func, "__wrapped__", original_func)

    setattr(module_path, func_name, patched_func)


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

    def patch_requires_nccl_version(version, msg):
        return _create_noop_decorator()

    def patch_skip_but_pass_in_sandcastle_if(condition, reason):
        return _create_noop_decorator()

    # Patch with_comms for sharded_tensor (func, init_rpc, backend)
    def patch_with_comms_sharded(func=None, init_rpc=False, backend="neuron"):
        if func is None:
            return partial(patch_with_comms_sharded, init_rpc=init_rpc, backend=backend)

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self.init_comms(init_rpc=init_rpc, backend=backend)
            func(self, *args, **kwargs)
            self.destroy_comms(destroy_rpc=init_rpc)

        return wrapper

    # Patch with_comms for dtensor (eager_init, backend)
    def patch_with_comms_dtensor(eager_init=False, backend="neuron"):
        def decorator(func, eager_init=False, backend="neuron"):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                self.init_pg(eager_init, backend)

                try:
                    func(self, *args, **kwargs)
                except Exception as e:
                    dist.destroy_process_group()
                    raise e

                self.destroy_pg()

            return wrapper

        return (
            decorator(func=eager_init)
            if callable(eager_init)
            else partial(decorator, eager_init=eager_init, backend=backend)
        )

    torch.testing._internal.common_distributed.skip_if_lt_x_gpu = neuron_skip_if_lt_x_gpu
    torch.testing._internal.common_distributed.requires_nccl = patch_requires_nccl
    torch.testing._internal.common_distributed.requires_nccl_version = patch_requires_nccl_version
    torch.testing._internal.common_distributed.skip_but_pass_in_sandcastle_if = (
        patch_skip_but_pass_in_sandcastle_if
    )
    torch.testing._internal.common_utils.skip_but_pass_in_sandcastle_if = (
        patch_skip_but_pass_in_sandcastle_if
    )

    from torch.testing._internal.distributed._shard import sharded_tensor

    sharded_tensor.with_comms = patch_with_comms_sharded

    from torch.testing._internal.distributed._tensor import common_dtensor

    common_dtensor.with_comms = patch_with_comms_dtensor
    common_dtensor.DEVICE_TYPE = "neuron"
    common_dtensor.PG_BACKEND = "neuron"


def patch_test_cuda_flags():
    """
    Enable TEST_CUDA and TEST_MULTIGPU flags for Neuron.
    """
    import torch.testing._internal.common_cuda as common_cuda
    import torch.testing._internal.common_utils as common_utils

    if HAS_NEURON:
        common_cuda.TEST_CUDA = True
        common_utils.TEST_CUDA = True
        common_cuda.TEST_MULTIGPU = torch_neuronx.device_count() >= 2


def patch_requires_cuda_early():
    """
    Patches the @requires_cuda decorator to work with Neuron devices.
    """
    import torch.testing._internal.common_utils as common_utils

    def requires_neuron(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not HAS_NEURON:
                import unittest

                raise unittest.SkipTest("PrivateUse1Neuron not available")
            return func(*args, **kwargs)

        return wrapper

    common_utils.requires_cuda = requires_neuron


def patch_requires_accelerator_backend_early():
    def neuron_requires_accelerator_dist_backend(backends=None):
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
    signature = inspect.signature(original_instantiate)

    def patched_instantiate(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        bound.arguments["only_for"] = "neuron"
        return original_instantiate(*bound.args, **bound.kwargs)

    torch.testing._internal.common_device_type.instantiate_device_type_tests = patched_instantiate


def patch_get_desired_device_type_neuron():
    from torch.testing._internal.common_device_type import PrivateUse1TestBase

    # Create a new class that inherits from PrivateUse1TestBase
    class NeuronTestBase(PrivateUse1TestBase):
        device_type = "neuron"
        device_mod = torch_neuronx.neuron
        primary_device = "neuron:0"

    def patched_get_desired_device_type_test_bases(*args, **kwargs):
        return [NeuronTestBase]

    torch.testing._internal.common_device_type.get_desired_device_type_test_bases = (
        patched_get_desired_device_type_test_bases
    )


def patch_common_fsdp():
    from torch.testing._internal import common_fsdp

    common_fsdp.DEVICE_TYPE = "neuron"
    common_fsdp.DISTRIBUTED_BACKEND = "neuron"
    common_fsdp.DEVICE_COUNT = 2


def patch_amp_not_available():
    from torch.cuda.amp.common import amp_definitely_not_available

    original_amp_not_available = amp_definitely_not_available

    def patched_amp_not_available():
        return (not HAS_NEURON) and original_amp_not_available()

    torch.cuda.amp.common.amp_definitely_not_available = patched_amp_not_available


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


def patch_get_cycles_per_ms():
    """Patch get_cycles_per_ms to return 1 for neuron"""
    try:
        import torch.testing._internal.common_utils as common_utils

        common_utils.get_cycles_per_ms = lambda: 1
    except (ImportError, AttributeError):
        pass


def patch_decorator_to_return_obj_noop():
    """
    Patches decorator to return its original object, e.g. fn, class etc.
    """

    def patch_skip_if_hpu_to_return_obj(obj):
        return obj

    import torch.testing._internal.common_utils as common_utils

    common_utils.skipIfHpu = patch_skip_if_hpu_to_return_obj


def patch_is_nccl_available():
    """Patch is_nccl_available to return True for Neuron tests."""
    dist.is_nccl_available = lambda: True


def patch_nccl_version():
    """Patch torch.cuda.nccl.version to return a dummy version for Neuron.

    This is needed because test_c10d_nccl.py has class-level decorators that call
    torch.cuda.nccl.version() during module import, which fails on non-NCCL PyTorch builds.
    """

    def patched_nccl_version():
        return (2, 19, 0)  # Return a reasonable NCCL version tuple

    if hasattr(torch.cuda, "nccl"):
        torch.cuda.nccl.version = patched_nccl_version


def patch_multiline_equal_whitespace_tolerance():
    """
    Patch unittest.TestCase.assertMultiLineEqual to be tolerant of
    whitespace-only differences (extra blank lines).

    This is needed because some patches (e.g., @torch.jit.ignore on tensor
    creation wrappers) cause minor formatting differences in FX graph output.
    """
    import re
    import unittest

    # Store original to avoid re-patching
    if hasattr(unittest.TestCase.assertMultiLineEqual, "_neuron_patched"):
        return

    _original = unittest.TestCase.assertMultiLineEqual

    def _patched_assert_multi_line_equal(self, first, second, msg=None):
        # Fast path: exact match
        if first == second:
            return

        def normalize(s):
            if not isinstance(s, str):
                return s
            lines = s.split("\n")
            lines = [line.rstrip() for line in lines]
            s = "\n".join(lines)
            # Collapse multiple consecutive blank lines to single blank line
            s = re.sub(r"\n\n+", "\n", s)
            return s.strip()

        # If normalized versions match, pass the test
        if normalize(first) == normalize(second):
            return

        # Otherwise, call original to get proper error message
        _original(self, first, second, msg)

    _patched_assert_multi_line_equal._neuron_patched = True
    unittest.TestCase.assertMultiLineEqual = _patched_assert_multi_line_equal


def pre_run_patches():
    # Must be first - before any test imports that check NCCL availability
    patch_is_nccl_available()
    patch_nccl_version()
    # Running Pre-run Patches.
    patch_test_cuda_flags()
    patch_skip_decorator_early()
    patch_module_function_from_cuda_to_neuron(torch, "get_device_module")
    patch_module_function_from_cuda_to_neuron(torch.distributed, "get_default_backend_for_device")
    patch_requires_accelerator_backend_early()
    patch_distributed_tests_timeout()
    patch_distributed_backend_early()
    patch_instantiate_device_type_tests_neuron()
    patch_get_desired_device_type_neuron()
    patch_amp_not_available()
    patch_requires_cuda_early()
    patch_decorator_to_return_obj_noop()
    patch_get_cycles_per_ms()
    patch_multiline_equal_whitespace_tolerance()

    # Patch torch.cuda.device_count for test class creation
    torch.cuda.device_count = lambda: torch_neuronx.device_count() if HAS_NEURON else 0


# Store reference to original torch.device for use in PatchDevice
_original_torch_device = torch.device

# allow isinstance to be called on our patched device properly
_original_torch_device_class = type(torch.device("cpu"))


class PatchDeviceMeta(type):
    """
    Metaclass for PatchDevice that enables isinstance() checks to work correctly.

    When torch.device is patched to PatchDevice, isinstance(obj, torch.device) would
    normally return False for real torch.device objects. This metaclass overrides
    __instancecheck__ to return True for both PatchDevice instances AND real
    torch.device objects, making isinstance() checks work correctly in both
    production code and tests.
    """

    def __instancecheck__(cls, instance):
        # Return True if instance is a real torch.device OR a PatchDevice
        return isinstance(instance, _original_torch_device) or type.__instancecheck__(cls, instance)


class PatchDevice(metaclass=PatchDeviceMeta):
    """
    Patched device class that redirects CUDA device specifications to Neuron.
    Defined at module level to support pickling.

    Uses PatchDeviceMeta metaclass to ensure isinstance(obj, torch.device) returns
    True for real torch.device objects even after patching.
    """

    def __new__(cls, device_type=None, index=None, *args, **kwargs):
        if device_type is None:
            return _original_torch_device(*args, **kwargs)

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
        return _original_torch_device(device_str, *args, **kwargs)


class NeuronPatcher:
    @staticmethod
    def patch_current_device():
        """
        Creates a patch for the current device function that returns current device index.

        Returns:
            function: A function that returns current device index
        """
        return lambda: torch_neuronx.current_device()

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
                count = torch_neuronx.device_count()
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
                    torch_neuronx.set_device(device)
                elif hasattr(device, "index") and device.index is not None:
                    torch_neuronx.set_device(device.index)
                else:
                    torch_neuronx.set_device(0)

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

        @torch.jit.ignore
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

        @torch.compiler.allow_in_graph
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
        Returns the module-level PatchDevice class that redirects CUDA device
        specifications to Neuron. Using a module-level class allows proper pickling.
        Also updates fx.graph custom builtins to use the patched device class.
        """
        import torch.fx.graph as fx_graph

        fx_graph._register_custom_builtin("device", "from torch import device", PatchDevice)
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
            # No-op patch using a dummy target that won't affect anything
            return patch.object(torch, "__version__", torch.__version__)

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
            # No-op patch using a dummy target that won't affect anything
            return patch.object(torch, "__version__", torch.__version__)

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
            # No-op patch using a dummy target that won't affect anything
            return patch.object(torch, "__version__", torch.__version__)

    @classmethod
    def collect_patches(cls):
        """
        Collects all patches needed to redirect CUDA operations to Neuron.

        Returns:
            list: A list of patch objects that modify PyTorch's CUDA functionality to use Neuron
        """
        patches = {}

        # Patch torch.cuda functions
        patches[PatchName.DEVICE_PATCHES.name] = patch.multiple(
            torch.cuda,
            current_device=cls.patch_current_device(),
            device_count=cls.patch_device_count(),
            set_device=cls.patch_set_device(),
        )

        patches[PatchName.TENSOR_TO_METHOD.name] = patch.object(
            torch.Tensor, "to", cls.patch_tensor_to_method()
        )

        # Patch tensor.cuda() method
        patches[PatchName.TENSOR_CUDA.name] = patch.object(
            torch.Tensor, "cuda", cls.patch_tensor_cuda
        )

        # Patch Module.to and Module.cuda
        patches[PatchName.MODULE_TO.name] = patch.object(
            torch.nn.Module, "to", cls.patch_module_to()
        )
        patches[PatchName.MODULE_CUDA.name] = patch.object(
            torch.nn.Module, "cuda", cls.patch_module_cuda()
        )

        patches[PatchName.DEVICE_CLASS.name] = patch("torch.device", cls.patch_device_class())

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
            "randint",
            "full",
        ]:
            if hasattr(torch, func_name):
                orig_func = getattr(torch, func_name)
                tensor_patches[func_name] = cls.create_tensor_function_wrapper(orig_func)

        patches[PatchName.TENSOR_PATCHES.name] = patch.multiple(torch, **tensor_patches)

        # Patch requires_cuda
        patches[PatchName.REQUIRES_CUDA.name] = cls.patch_requires_cuda()

        # Patch CUDA peer access and SM80 checks
        patches[PatchName.CUDA_PEER_ACCESS.name] = cls.patch_cuda_peer_access()
        patches[PatchName.SM80_OR_LATER.name] = cls.patch_sm80_or_later()

        return patches


class PatchName(Enum):
    DEVICE_PATCHES = "device_patches"
    TENSOR_TO_METHOD = "patch_tensor_to_method"
    TENSOR_CUDA = "patch_tensor_cuda"
    MODULE_TO = "patch_module_to"
    MODULE_CUDA = "patch_module_cuda"
    DEVICE_CLASS = "patch_device_class"
    TENSOR_PATCHES = "tensor_patches"
    REQUIRES_CUDA = "patch_requires_cuda"
    CUDA_PEER_ACCESS = "patch_cuda_peer_access"
    SM80_OR_LATER = "patch_sm80_or_later"

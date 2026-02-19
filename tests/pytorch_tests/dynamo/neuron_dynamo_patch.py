"""Neuron-specific patches for PyTorch dynamo tests."""

import functools
import os
from contextlib import nullcontext

import torch._dynamo

import torch_neuronx
from tests.pytorch_tests.distributed.neuron_patch import NeuronPatcher, pre_run_patches

HAS_NEURON = True


def patch_requires_cuda_and_triton_early():
    """
    Patches the @requires_cuda_and_triton decorator to work with Neuron devices.
    Triton is not required for Neuron backend.
    """
    try:
        from torch.testing._internal.triton_utils import requires_cuda_and_triton

        def requires_neuron_decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not HAS_NEURON:
                    import unittest

                    raise unittest.SkipTest("PrivateUse1Neuron not available")
                return func(*args, **kwargs)

            return wrapper

        import torch.testing._internal.triton_utils

        torch.testing._internal.triton_utils.requires_cuda_and_triton = requires_neuron_decorator
    except ImportError:
        # triton_utils may not exist in all PyTorch versions
        pass


def patch_torch_compile_backend():
    """Patch torch.compile to redirect real backends to neuron."""
    original_compile = torch.compile

    # Real backends that should be redirected to neuron
    redirect_backends = {
        "eager",
        "aot_eager",
        "inductor",
        "aot_inductor",
        "aot_eager_decomp_partition",
    }

    def patched_compile(model=None, *, backend="neuron", **kwargs):
        # Redirect real backends to neuron
        if backend in redirect_backends:
            backend = "neuron"

        # Preserve test/debug backends and callable backends
        kwargs["dynamic"] = False
        return original_compile(model, backend=backend, **kwargs)

    torch.compile = patched_compile


def patch_instantiate_device_type_tests_add_neuron():
    """Patch instantiate_device_type_tests to add neuron to device lists."""
    import inspect

    from torch.testing._internal.common_device_type import instantiate_device_type_tests

    original_instantiate = instantiate_device_type_tests
    signature = inspect.signature(original_instantiate)

    def patched_instantiate(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        # Add neuron to only_for list if it exists
        if bound.arguments.get("only_for"):
            only_for = bound.arguments["only_for"]
            if isinstance(only_for, list | tuple) and "neuron" not in only_for:
                bound.arguments["only_for"] = [*list(only_for), "neuron"]

        return original_instantiate(*bound.args, **bound.kwargs)

    import torch.testing._internal.common_device_type

    torch.testing._internal.common_device_type.instantiate_device_type_tests = patched_instantiate


def patch_torch_cuda_to_neuron():
    """Patch torch.cuda references to use torch.neuron for device guard tests."""
    import torch

    # Store original torch.cuda
    if not hasattr(torch, "_original_cuda"):
        torch._original_cuda = torch.cuda

    # Replace torch.cuda with torch.neuron for tests
    cuda_module = torch.cuda
    torch.cuda = torch.neuron
    torch_neuronx.neuron.amp = torch.amp
    torch.cuda.amp.common = cuda_module.amp.common
    torch.cuda.streams = torch_neuronx.streams
    # Set the cuda version to ensure no rocm
    torch.version.cuda = "neuron"


def patch_torch_stream_context_to_neuron():
    torch.neuron.StreamContext = torch_neuronx.StreamContext


def patch_device_capability():
    def get_device_capability():
        return (0, 0)

    torch.neuron.get_device_capability = get_device_capability


def patch_tensor_types():
    torch.neuron.FloatTensor = torch.Tensor
    torch.neuron.DoubleTensor = torch.Tensor
    torch.neuron.HalfTensor = torch.Tensor
    torch.neuron.BFloat16Tensor = torch.Tensor
    torch.neuron.ByteTensor = torch.Tensor
    torch.neuron.CharTensor = torch.Tensor
    torch.neuron.IntTensor = torch.Tensor
    torch.neuron.ShortTensor = torch.Tensor
    torch.neuron.LongTensor = torch.Tensor


def patch_pin_memory_utils():
    torch.neuron._pin_memory_utils = None


def get_context_for_class(class_name):
    """
    Some test classes require dynamo config patches to be on that
    seem to get ignored when we import the module, add them here
    """
    if class_name == "TestAOTCompile":
        return torch._dynamo.config.patch(enable_aot_compile=True)
    return nullcontext()


def patch_device_for_fx_namespace():
    """
    PatchDevice doesn't get recogized within fx graphs, patch it into python builtins
    so it gets recognized
    """
    import builtins

    builtins.device = torch.device


def patch_minifier_gen_test_code():
    """Fix _gen_test_code to handle empty codegen_config()."""
    import torch._dynamo.test_minifier_common as tmc

    original_gen = tmc.MinifierTestBase._gen_test_code

    def patched_gen(self, run_code, repro_after, repro_level):
        code = original_gen(self, run_code, repro_after, repro_level)
        # Remove lines that are just "." from empty codegen_config()
        lines = [line for line in code.split("\n") if line.strip() != "."]
        return "\n".join(lines)

    tmc.MinifierTestBase._gen_test_code = patched_gen


def apply_dynamo_patches():
    """Apply all patches needed for dynamo tests."""
    pre_run_patches()

    patch_requires_cuda_and_triton_early()

    patch_instantiate_device_type_tests_add_neuron()

    patches = NeuronPatcher.collect_patches()
    for patch in patches.values():
        patch.start()

    patch_torch_compile_backend()

    patch_torch_cuda_to_neuron()

    patch_torch_stream_context_to_neuron()

    patch_device_capability()

    patch_tensor_types()

    patch_pin_memory_utils()

    patch_device_for_fx_namespace()

    patch_minifier_gen_test_code()

    # Ensure retain device is set
    os.environ["TORCH_NEURONX_RETAIN_DEVICE_MODE"] = "1"

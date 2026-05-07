"""Neuron-specific patches for PyTorch dynamo tests."""

import functools
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import torch._dynamo

import torch_neuronx
from tests.pytorch_tests.distributed.neuron_patch import NeuronPatcher, pre_run_patches

HAS_NEURON = True


def patch_jiterator_utils():
    """Mock CUDA-dependent test utilities that fail to import on Neuron."""
    import torch

    if not hasattr(torch.cuda, "jiterator"):
        torch.cuda.jiterator = mock.MagicMock()
    # Mock missing modules that don't exist in this PyTorch version
    if "torch._higher_order_ops.inline_asm_elementwise" not in sys.modules:
        sys.modules["torch._higher_order_ops.inline_asm_elementwise"] = mock.MagicMock()
    sys.modules["torch.testing._internal.jit_utils"] = mock.MagicMock()


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


def patch_memory_stats_to_synchronize():
    """Wrap peak-memory stat APIs on torch.neuron to synchronize first.

    Neuron's allocator is asynchronous. Upstream PyTorch tests that sample peak memory
    assume these reads are synchronous with allocator activity (as they are on
    CUDA). To align with upstream PyTorch memory stats test expectations, we synchronize.
    """
    import torch

    # Readers: sync so pending alloc/free ops are reflected in the stats.
    reader_names = (
        "memory_stats",
        "memory_allocated",
        "max_memory_allocated",
        "memory_reserved",
        "max_memory_reserved",
    )
    # Resetters: sync so pending ops land in stats before peak is reset.
    resetter_names = (
        "reset_peak_memory_stats",
        "reset_accumulated_memory_stats",
    )

    targets = [torch.neuron]
    memory_mod = getattr(torch.neuron, "memory", None)
    if memory_mod is not None and memory_mod is not torch.neuron:
        targets.append(memory_mod)

    def _wrap_with_sync(target, fn_name, *, pass_through_return):
        if not hasattr(target, fn_name):
            return
        original = getattr(target, fn_name)
        if getattr(original, "_neuron_sync_patched", False):
            return

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            torch_neuronx.synchronize()
            result = original(*args, **kwargs)
            return result if pass_through_return else None

        wrapper._neuron_sync_patched = True
        setattr(target, fn_name, wrapper)

    for target in targets:
        for name in reader_names:
            _wrap_with_sync(target, name, pass_through_return=True)
        for name in resetter_names:
            _wrap_with_sync(target, name, pass_through_return=False)

        # Legacy alias used by a few upstream tests.
        # CUDA keeps this as a deprecated pointer to reset_peak_memory_stats.
        if not hasattr(target, "reset_max_memory_allocated") and hasattr(
            target, "reset_peak_memory_stats"
        ):
            target.reset_max_memory_allocated = target.reset_peak_memory_stats


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
    if class_name == "TestForwardLossBackward":
        return torch._dynamo.config.patch(trace_autograd_ops=True)
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

    # Added sys insert path below bcs of relative import needed
    # in test_dynamic_shapes.py pt upstream file.
    dynamo_test_dir = str(Path(__file__).resolve().parent.parent / "test" / "dynamo")
    if dynamo_test_dir not in sys.path:
        sys.path.insert(0, dynamo_test_dir)

    pre_run_patches()

    patch_requires_cuda_and_triton_early()

    patch_instantiate_device_type_tests_add_neuron()

    patches = NeuronPatcher.collect_patches()
    for patch in patches.values():
        patch.start()

    patch_torch_compile_backend()

    patch_torch_cuda_to_neuron()

    patch_memory_stats_to_synchronize()

    patch_torch_stream_context_to_neuron()

    patch_device_capability()

    patch_tensor_types()

    patch_pin_memory_utils()

    patch_device_for_fx_namespace()

    patch_minifier_gen_test_code()

    patch_jiterator_utils()

    # Ensure retain device is set
    os.environ["TORCH_NEURONX_RETAIN_DEVICE_MODE"] = "1"

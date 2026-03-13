"""Utility functions for testing Neuron operations."""

import os
import tempfile
from contextlib import contextmanager
from functools import wraps
from typing import Any

import pytest
import torch

import torch_neuronx
from torch_neuronx.utils import is_sync_mode_enabled


@contextmanager
def track_neuron_ops():
    """Context manager that tracks operations executed within its scope.

    This clears the op tracking before entering the context and provides
    access to executed and fallback ops after operations are performed.

    Yields:
        None

    Example:
        with track_neuron_ops():
            output = model(input)
            assert_op_runs_on_neuron("aten::linear")
    """
    # Clear any existing tracking state
    torch_neuronx.clear_op_tracking()

    try:
        yield
    finally:
        # Op tracking persists after context exits
        # Users can call assert_op_runs_on_neuron or get_executed_op_list
        pass


def check_op_runs_on_neuron(op_name: str) -> tuple[bool, list, list]:
    """Check if an operation ran on Neuron (not CPU fallback).

    Uses the runtime op tracking APIs instead of file-based checking.
    Should be called within or after a track_neuron_ops() context.

    Args:
        op_name: Name of the operation to check (e.g., "aten::scaled_dot_product_attention")

    Returns:
        Tuple of (ran_on_neuron, executed_ops, fallback_ops)
    """
    # need to synchronize when using async flow, to make sure all ops execution is completed
    torch.neuron.synchronize()
    executed_ops = torch_neuronx.get_executed_ops()
    fallback_ops = torch_neuronx.get_fallback_ops()

    if op_name in fallback_ops:
        print(f"❌ {op_name} found in fallback ops (running on CPU!)")
        return False, executed_ops, fallback_ops

    if op_name in executed_ops:
        print(f"✓ {op_name} found in executed ops (running on Neuron)")
        return True, executed_ops, fallback_ops

    nki_variants = [
        f"nki_{op_name}",
        op_name.replace("aten::", "nki_"),
        op_name.replace("aten::", ""),
    ]

    for variant in nki_variants:
        if any(variant in op for op in executed_ops):
            print(f"✓ NKI kernel variant '{variant}' detected (running on Neuron)")
            return True, executed_ops, fallback_ops

    print(f"⚠ {op_name} not found in executed or fallback ops")
    print(f"  Executed ops: {executed_ops}")
    print(f"  Fallback ops: {fallback_ops}")
    return False, executed_ops, fallback_ops


def assert_op_did_not_run_on_neuron(op_name: str):
    """Assert that an operation did not run on Neuron.

    Uses the runtime op tracking APIs instead of file-based checking.
    Should be called within or after a track_neuron_ops() context.

    Args:
        op_name: Name of the operation to check (e.g., "aten::scaled_dot_product_attention")

    Raises:
        AssertionError: If the operation ran on Neuron
    """
    ran_on_neuron, _, _ = check_op_runs_on_neuron(op_name)
    if ran_on_neuron:
        raise AssertionError(f"Operation {op_name} ran on Neuron when it was not expected to run")


def assert_op_runs_on_neuron(op_name: str):
    """Assert that an operation ran on Neuron (not CPU fallback).

    Uses the runtime op tracking APIs instead of file-based checking.
    Should be called within or after a track_neuron_ops() context.

    Args:
        op_name: Name of the operation to check (e.g., "aten::scaled_dot_product_attention")

    Raises:
        AssertionError: If the operation didn't run on Neuron
    """
    ran_on_neuron, executed_ops, fallback_ops = check_op_runs_on_neuron(op_name)
    if not ran_on_neuron:
        raise AssertionError(
            f"Operation {op_name} did not run on Neuron.\n"
            f"Executed: {executed_ops}\nFallback: {fallback_ops}"
        )


def get_executed_op_list() -> dict:
    """Get a detailed report of operations executed on Neuron vs CPU.

    Returns:
        Dictionary with 'executed' and 'fallback' lists of operation names
    """
    return {
        "executed": torch_neuronx.get_executed_ops(),
        "fallback": torch_neuronx.get_fallback_ops(),
    }


def assert_op_falls_back_on_cpu(op_name: str):
    """Assert that an operation falls back to CPU (not running on Neuron).

    Uses the runtime op tracking APIs instead of file-based checking.
    Should be called within or after a track_neuron_ops() context.

    Args:
        op_name: Name of the operation to check (e.g., "aten::scaled_dot_product_attention")

    Raises:
        AssertionError: If the operation didn't fall back to CPU
    """
    ran_on_neuron, _, fallback_ops = check_op_runs_on_neuron(op_name)

    if ran_on_neuron:
        raise AssertionError(f"Operation {op_name} ran on Neuron instead of falling back to CPU")

    if op_name in fallback_ops:
        return

    raise AssertionError(f"Operation {op_name} was not tracked in execution")


def assert_op_does_not_run(op_name: str):
    """Assert that an operation does not run at all (neither on Neuron nor CPU).

    This verifies that the operation is not in the executed ops list AND not in the
    fallback ops list. This is useful for verifying that lazy materialization
    optimizations completely eliminate unnecessary operations like contiguous().

    Uses the runtime op tracking APIs instead of file-based checking.
    Should be called within or after a track_neuron_ops() context.

    Args:
        op_name: Name of the operation to check (e.g., "aten::contiguous")

    Raises:
        AssertionError: If the operation ran on Neuron or fell back to CPU
    """
    executed_ops = torch_neuronx.get_executed_ops()
    fallback_ops = torch_neuronx.get_fallback_ops()

    # Check if op ran on Neuron
    if op_name in executed_ops:
        print(f"❌ {op_name} found in executed ops (running on Neuron!)")
        raise AssertionError(
            f"Operation {op_name} ran on Neuron when it should not have run at all"
        )

    # Check if op fell back to CPU
    if op_name in fallback_ops:
        print(f"❌ {op_name} found in fallback ops (offloaded to CPU!)")
        raise AssertionError(
            f"Operation {op_name} fell back to CPU when it should not have run at all"
        )

    # Check for NKI kernel execution patterns
    nki_variants = [
        f"nki_{op_name}",
        op_name.replace("aten::", "nki_"),
        op_name.replace("aten::", ""),
    ]

    for variant in nki_variants:
        if any(variant in op for op in executed_ops):
            print(f"❌ NKI kernel variant '{variant}' detected (running on Neuron!)")
            raise AssertionError(
                f"NKI kernel variant of {op_name} ran on Neuron when it should not have run at all"
            )

    print(f"✓ {op_name} did not run (neither on Neuron nor CPU) - operation was eliminated")


def count_cpu_bounce_calls(fn) -> dict[str, int]:
    """Count CPU bounce calls for a function by wrapping NRT read/write.

    Tracks calls to:
      - _nrt_copy_neuron_to_cpu_tensor (reads)
      - _nrt_copy_cpu_to_neuron_tensor (writes)

    Args:
        fn: Callable to execute while counting

    Returns:
        dict with keys 'reads' and 'writes'
    """
    import torch_neuronx._C as _C

    reads = 0
    writes = 0

    orig_read = _C._nrt_copy_neuron_to_cpu_tensor
    orig_write = _C._nrt_copy_cpu_to_neuron_tensor

    def wrap_read(*args, **kwargs):
        nonlocal reads
        reads += 1
        return orig_read(*args, **kwargs)

    def wrap_write(*args, **kwargs):
        nonlocal writes
        writes += 1
        return orig_write(*args, **kwargs)

    _C._nrt_copy_neuron_to_cpu_tensor = wrap_read
    _C._nrt_copy_cpu_to_neuron_tensor = wrap_write

    try:
        fn()
    finally:
        _C._nrt_copy_neuron_to_cpu_tensor = orig_read
        _C._nrt_copy_cpu_to_neuron_tensor = orig_write

    return {"reads": reads, "writes": writes}


def get_pytest_worker_count():
    return int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))


def get_pytest_worker_index():
    return int(os.environ.get("PYTEST_XDIST_WORKER", "gw0")[2:])


def get_total_core_count():
    return int(os.environ.get("TOTAL_DEVICE_COUNT", "64"))


def get_core_start_index():
    return get_pytest_worker_index() * get_total_core_count() // get_pytest_worker_count()


def get_cache_size(kernel):
    if is_sync_mode_enabled():
        return kernel.cache.size()

    torch.neuron.synchronize()
    return torch_neuronx._C._get_compilation_cache_stats()["total_entries"]


def requires_nrt_streams(test_func):
    """Decorator to enable multi-stream mode for tests that require unique streams.

    Sets TORCH_NEURONX_ENABLE_HOST_CC=1 for the duration of the test.
    """

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        old_host_cc = os.environ.get("TORCH_NEURONX_ENABLE_HOST_CC")
        os.environ["TORCH_NEURONX_ENABLE_HOST_CC"] = "1"
        try:
            return test_func(*args, **kwargs)
        finally:
            if old_host_cc is None:
                del os.environ["TORCH_NEURONX_ENABLE_HOST_CC"]
            else:
                os.environ["TORCH_NEURONX_ENABLE_HOST_CC"] = old_host_cc

    return wrapper


def assert_raises(
    expected_exception: type[BaseException] | tuple[type[BaseException], ...],
    *args: Any,
    **kwargs: Any,
):
    r"""Decorator for negative tests that handles async mode synchronization.

    This decorator uses pytest.raises to handle the asynchronous nature of Neuron
    operations. It ensures that torch.neuron.synchronize() is called after the
    operation to properly catch and assert on the expected exception message.

    Args:
        expected_exception: Exception type or tuple of exception types to catch.
        match: The regex pattern to match against the error message when running on Neuron.
               Can be a simple string (which will be treated as a literal regex)
               or a regex pattern with special characters.
               If None, no message matching is performed.
        match_cpu: The regex pattern to match against the error message when running on CPU
                   (when TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS is not set).
                   If provided, this will be used instead of 'match' when in CPU fallback mode.
                   If None, 'match' will be used regardless of the environment variable.

    Usage:
        @assert_raises(RuntimeError, match="Cannot add non-contiguous tensor")
        def test_some_negative_case():
            # test code that should raise an exception
            tensor.some_operation()

        @assert_raises(ValueError)
        def test_value_error():
            # test code that should raise ValueError
            some_invalid_operation()

        @assert_raises(
            IndexError,
            match=r"duplicate value in the list of dims",
            match_cpu=r"dim \d+ appears multiple times in the list of dims"
        )
        def test_with_different_cpu_message():
            # test code that has different error messages on Neuron vs CPU
            some_operation_with_different_messages()
    """

    def decorator(test_func):
        @wraps(test_func)
        def wrapper(*func_args, **func_kwargs):
            # Extract match_cpu parameter if present
            modified_kwargs = kwargs.copy()
            match_cpu = modified_kwargs.pop("match_cpu", None)

            # Determine which match pattern to use based on environment variable
            if (
                match_cpu is not None
                and os.environ.get("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS") != "1"
            ):
                # Environment variable is not set - use 'match_cpu' pattern (CPU fallback mode)
                modified_kwargs["match"] = match_cpu

            with pytest.raises(expected_exception, *args, **modified_kwargs):
                test_func(*func_args, **func_kwargs)
                # Synchronize ops execution to propagate raised error
                torch.neuron.synchronize()

        return wrapper

    return decorator

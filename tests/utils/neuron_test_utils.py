"""Utility functions for testing Neuron operations."""

import os
import tempfile
from contextlib import contextmanager
from functools import wraps
from typing import Any, Optional, TextIO

import numpy as np
import pytest
import torch
from neuronxcc.starfish.support.dtype import is_float_type

import torch_neuronx


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


def get_cache_size():
    torch.neuron.synchronize()
    return torch_neuronx._C._get_compilation_cache_stats()["total_entries"]


def _requires_stream_config(test_func, host_cc_enabled):
    expected_host_cc = "1" if host_cc_enabled else "0"

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        if not getattr(wrapper, "_nrt_stream_config_seeded", False):
            raise RuntimeError(
                "Pytest did not apply the NRT stream configuration. "
                "Apply this decorator directly to a collected test."
            )

        if (
            os.environ.get("TORCH_NEURONX_ENABLE_STREAMS") != "1"
            or os.environ.get("TORCH_NEURONX_ENABLE_HOST_CC") != expected_host_cc
        ):
            raise RuntimeError(
                "The NRT stream environment does not match this decorator. "
                "Check the pytest setup hook."
            )
        if host_cc_enabled and os.environ.get("NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE") != "true":
            raise RuntimeError(
                "The HostCC test environment does not enable DMA transpose. "
                "Check the pytest setup hook."
            )
        return test_func(*args, **kwargs)

    wrapper._nrt_stream_config_seeded = False
    return pytest.mark.nrt_stream_config(host_cc_enabled=host_cc_enabled)(wrapper)


def requires_nrt_streams(test_func):
    """Enable distinct streams and keep HostCC off.

    Use this for a test that only needs a stream separate from the default
    stream. HostCC costs kernel performance, so do not enable it here.

    Apply this decorator directly to a collected test. Mark the test with
    needs_process_isolation. The C++ stream configuration keeps both flags for
    the process lifetime. See "Trap 2" in docs/AI-testing.md.
    """
    return _requires_stream_config(test_func, host_cc_enabled=False)


def requires_nrt_streams_with_host_cc(test_func):
    """Enable distinct streams and HostCC.

    Use this only for a test that asserts HostCC behavior, such as which copy
    path runs on a non-default stream. HostCC also enables the DMA transpose
    setting that the hang guard needs.

    Apply this decorator directly to a collected test. Mark the test with
    needs_process_isolation. See "Trap 2" in docs/AI-testing.md.
    """
    return _requires_stream_config(test_func, host_cc_enabled=True)


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


def max_all_close(
    a,
    b,
    rtol=1e-05,
    atol=1e-08,
    equal_nan_inf=False,
    verbose=0,
    mode="max",
    logfile: TextIO | None = None,
    min_pass_rate: float = 1.0,
) -> bool:
    """
    Compare two arrays for approximate equality with configurable tolerance modes.

    This function provides a custom comparison utility with lower sensitivity by considering
    the overall value range. Unlike numpy.allclose, it offers detailed mismatch reporting and
    uses the overall value range for relative difference calculations, making it suitable for
    validating numerical computation results where global tolerance is more appropriate than
    per-element tolerance.

    Reference: Spec Issue NCC-181

    Args:
        a (np.ndarray): Test array to compare (typically the computed result).
        b (np.ndarray): Reference/gold array to compare against (ground truth).
        rtol (float, optional): Relative tolerance multiplier. Elements are considered equal if
            abs(a - b) <= atol + rtol * brange. Defaults to 1e-05.
        atol (float, optional): Absolute tolerance threshold. Provides a minimum tolerance
            independent of magnitude. Defaults to 1e-08.
        equal_nan_inf (bool, optional): If True, matching NaN values and matching infinity
            values (same sign) are treated as equal. Defaults to False.
        verbose (int, optional): Verbosity level controlling output detail. Defaults to 0.
            - 0: Silent mode, returns boolean only
            - 1: Prints summary with largest absolute and relative differences
            - 3: Prints detailed information for every mismatched element
        mode (str, optional): Comparison mode determining how relative differences are
            calculated. Defaults to "max".
            - "py": Python-style allclose with per-element range. Computes
              relDiff = absDiff / abs(b[i]) for each element. Reports the largest
              relative difference, matching np.testing.assert_allclose behavior.
            - "max": Uses global maximum absolute value in b for all comparisons.
              Computes relErr = (absDiff - atol) / max(abs(b)) as percentage.
              Provides looser, more uniform match criteria across all elements.
              Reports relErr for the element with largest absDiff.
        logfile (str, optional): File path for logging verbose output. When specified,
            all verbose output is appended to this file in addition to stdout.
            Defaults to None (no file logging).
        min_pass_rate (float, optional): Minimum fraction of elements that must pass
            tolerance check (0 to 1). Defaults to 1.0 (all elements must pass).
            When < 1.0, comparison passes if at least this fraction of elements
            are within tolerance.

    Returns:
        bool: True if all elements are approximately equal within specified tolerances.
            False if any element exceeds tolerance or if array shapes don't match.

    Raises:
        No exceptions are raised. Shape mismatches return False with optional error message.

    Notes:
        - Shape validation: Arrays must have identical shapes or comparison fails immediately.
        - Tolerance formula: abs(a - b) <= atol + rtol * brange
        - In "max" mode: brange = max(abs(b)), excluding NaN values
        - In "py" mode: brange = abs(b) computed per-element
        - Special value handling for floating point types:
          * inf - inf = 0.0 (matching infinities considered equal)
          * -inf - (-inf) = 0.0 (matching negative infinities considered equal)
          * NaN comparisons handled via equal_nan_inf parameter
        - Invalid operations (like inf - inf) are suppressed to prevent numpy warnings.

    Examples:
        Basic usage with default settings:
        >>> a = np.array([1.0, 2.0, 3.0])
        >>> b = np.array([1.0, 2.0, 3.0001])
        >>> max_all_close(a, b, rtol=1e-3)
        True

        Verbose output with summary:
        >>> maxAllClose(a, b, rtol=1e-3, verbose=1)
        INFO: allclose comparison summary: largest abs diff = 0.0001, ...
        True

        Per-element detailed output:
        >>> maxAllClose(a, b, rtol=1e-5, verbose=3, mode="py")
        INFO: allclose comparison failed at (2,): abs(3 - 3.0001) = 0.0001 ...
        False

        Handling special values:
        >>> a = np.array([1.0, np.nan, np.inf])
        >>> b = np.array([1.0, np.nan, np.inf])
        >>> maxAllClose(a, b, equal_nan_inf=True)
        True

        Logging to file:
        >>> max_all_close(a, b, rtol=1e-5, verbose=1, logfile="comparison.log")
        False
    """
    assert 0 <= rtol <= 1, f"Relative tolerance has to be between 0 and 1 but got {rtol=}"

    # b is refernce, a is new
    if a.shape != b.shape:
        if verbose > 0:
            print(f"ERROR: allclose mismatching shapes {a.shape} != {b.shape}")
        return False

    brange = abs(b)
    if mode == "max":
        brange = np.amax(brange[np.isfinite(brange)])

    # We ignore invalid errors here because inf-inf will cause numpy to produce nan
    # and throw an invalid error
    with np.errstate(invalid="ignore"):
        abs_diff = abs(a.astype(float) - b.astype(float))
    # Calling np.sign() on boolean types will crash, so we do this only for floats
    if is_float_type(a.dtype):
        # Replace matching inf or matching -inf with 0.0
        abs_diff = np.where((np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))), 0.0, abs_diff)
    close = abs_diff <= atol + rtol * brange
    if equal_nan_inf:
        close |= (a == b) | (np.isnan(a) & np.isnan(b))

    finiteness_mismatch_tensor = np.isfinite(a) != np.isfinite(b)
    is_finiteness_matching = not finiteness_mismatch_tensor.any()

    def print_with_log(s: str):
        print(s)
        if logfile is not None:
            print(s, file=logfile)

    sf = ""

    if not is_finiteness_matching and verbose >= 1:
        sf += "ERROR: There are indices with finite-infinite mismatches\n"

        if verbose >= 3:
            indices = zip(*np.where(finiteness_mismatch_tensor), strict=False)
            for index in indices:
                sf += f"{index}: {a[index]} vs {b[index]}\n"

        print_with_log(sf)

    if verbose >= 3:
        indices_array = np.argwhere(~close)
        s = (
            "INFO: allclose comparison failed at {indices}: "
            "abs({a:.4g} - {b:.4g}) = {absDiff:.4g} is greater than "
            "({atol:.4g} + {rtol:.4g} * {brange:.4g}) = {val:.4g}, "
        )
        s_py = "relative difference {relDiff:4g} "
        s_max = "relative error {relErr:4g} % "
        for indices in indices_array:
            indices = tuple(indices)
            brangep = brange[indices] if mode == "py" else brange
            val = atol + rtol * brangep
            abs_d = abs_diff[indices]
            if mode == "py":
                ss = s + s_py
                sf = ss.format(
                    a=a[indices],
                    b=b[indices],
                    atol=atol,
                    rtol=rtol,
                    brange=brangep,
                    indices=indices,
                    val=val,
                    absDiff=abs_d,
                    relDiff=abs_d / brangep,
                )
            else:
                ss = s + s_max
                rel_err = (abs_d - atol) / brange
                sf = ss.format(
                    a=a[indices],
                    b=b[indices],
                    atol=atol,
                    rtol=rtol,
                    brange=brangep,
                    indices=indices,
                    val=val,
                    absDiff=abs_d,
                    relErr=rel_err * 100,
                )
            print_with_log(sf)

    if verbose >= 1:
        largest_abs_diff = np.amax(abs_diff)
        s = "INFO: allclose comparison summary: largest abs diff = {:g}, "
        s_py = "largest relative difference = {:g} \n"
        s_max = "rel diff = {:g} % (check against current rel tolerance of {:g} %)\n"
        if mode == "py":
            with np.errstate(invalid="ignore", divide="ignore"):
                largest_relative_diff = np.amax(abs_diff / brange)
            sf = (s + s_py).format(largest_abs_diff, largest_relative_diff)
        else:
            max_abs_diff_rel = (
                (largest_abs_diff - atol) / brange if largest_abs_diff >= atol else 0.0
            )
            sf = (s + s_max).format(largest_abs_diff, max_abs_diff_rel * 100.0, rtol * 100.0)
        print_with_log(sf)

    all_close = np.all(close) and is_finiteness_matching
    if all_close or min_pass_rate >= 1.0:
        return all_close

    # Check if pass rate meets minimum threshold
    pass_rate = np.mean(close)
    passed = pass_rate >= min_pass_rate and is_finiteness_matching
    if verbose >= 1:
        status = "PASSED" if passed else "FAILED"
        print_with_log(
            f"INFO: pass rate = {pass_rate:.2%} (required: {min_pass_rate:.2%}) - {status}\n"
        )
    return passed

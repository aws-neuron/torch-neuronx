"""Utility functions for testing Neuron operations."""

import os
import tempfile
from contextlib import contextmanager
from functools import wraps

import torch

import torch_neuron


@contextmanager
def temporary_dump_directory():
    """Create a temporary directory for torch_neuron dumps and set the environment variable.

    This ensures each test run has its own dump directory and doesn't collide with
    other processes.

    Yields:
        str: Path to the temporary dump directory
    """
    # Save original env var if it exists
    original_dump_dir = os.environ.get("TORCH_NEURON_DUMP")

    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="torch_neuron_dump_test_")

    try:
        # Set the environment variable
        os.environ["TORCH_NEURON_DUMP"] = temp_dir
        yield temp_dir
    finally:
        # Restore original env var
        if original_dump_dir is not None:
            os.environ["TORCH_NEURON_DUMP"] = original_dump_dir
        else:
            os.environ.pop("TORCH_NEURON_DUMP", None)

        # Clean up temp directory
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def check_neuron_execution(dump_dir: str, op_name: str) -> tuple[bool, str]:
    """Check if an operation executed on Neuron device vs CPU fallback.

    Args:
        dump_dir: Path to the torch_neuron dump directory
        op_name: Name of the operation to check (e.g., "scaled_dot_product_attention")

    Returns:
        Tuple of (success: bool, message: str)
        - success is True if the op ran on Neuron (not offloaded)
        - message describes what was found
    """
    import glob

    # Find subdirectories in dump_dir (there should be one per process/run)
    dump_subdirs = glob.glob(os.path.join(dump_dir, "*"))

    if not dump_subdirs:
        return False, f"No dump directories found in {dump_dir}"

    # Use the most recent dump directory
    latest_dump = max(dump_subdirs, key=os.path.getctime)

    # Check used_ops.txt (not used_neuron_ops.txt)
    used_ops_file = os.path.join(latest_dump, "used_ops.txt")
    used_ops_content = ""
    if os.path.exists(used_ops_file):
        with open(used_ops_file) as f:
            used_ops_content = f.read()

    # Check offloaded_ops.txt
    offloaded_ops_file = os.path.join(latest_dump, "offloaded_ops.txt")
    offloaded_ops_content = ""
    if os.path.exists(offloaded_ops_file):
        with open(offloaded_ops_file) as f:
            offloaded_ops_content = f.read()

    # Check if op is in used_neuron_ops
    op_in_used = op_name in used_ops_content
    # Check if op is in offloaded_ops
    op_in_offloaded = op_name in offloaded_ops_content

    # Determine success and message
    if op_in_offloaded:
        return False, f"❌ {op_name} found in offloaded_ops.txt (running on CPU!)"
    elif op_in_used:
        return True, f"✓ {op_name} found in used_ops.txt (running on Neuron)"
    elif not os.path.exists(used_ops_file) and not os.path.exists(offloaded_ops_file):
        return False, f"⚠ No ops tracking files found in {latest_dump}"
    else:
        # Op not found in either file
        # Check if any related ops are present
        if "nki_" in used_ops_content:
            # NKI kernels are logged with nki_ prefix
            return True, "✓ NKI kernel execution detected (op may be part of NKI kernel)"
        return False, f"⚠ {op_name} not found in either used_ops.txt or offloaded_ops.txt"


def assert_runs_on_neuron(dump_dir: str, op_name: str):
    """Assert that an operation runs on Neuron device (not CPU fallback).

    Args:
        dump_dir: Path to the torch_neuron dump directory
        op_name: Name of the operation to check

    Raises:
        AssertionError: If the operation doesn't run on Neuron
    """
    success, message = check_neuron_execution(dump_dir, op_name)
    print(message)
    assert success, f"Operation {op_name} did not run on Neuron: {message}"


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
    torch_neuron.clear_op_tracking()

    try:
        yield
    finally:
        # Op tracking persists after context exits
        # Users can call assert_op_runs_on_neuron or get_executed_op_list
        pass


def assert_op_runs_on_neuron(op_name: str):
    """Assert that an operation ran on Neuron (not CPU fallback).

    Uses the runtime op tracking APIs instead of file-based checking.
    Should be called within or after a track_neuron_ops() context.

    Args:
        op_name: Name of the operation to check (e.g., "aten::scaled_dot_product_attention")

    Raises:
        AssertionError: If the operation didn't run on Neuron
    """
    # need to synchronize when using async flow, to make sure all ops execution is completed
    # before any asserts can be done
    torch.neuron.synchronize()
    executed_ops = torch_neuron.get_executed_ops()
    fallback_ops = torch_neuron.get_fallback_ops()

    # Check if op ran as fallback (on CPU)
    if op_name in fallback_ops:
        print(f"❌ {op_name} found in fallback ops (running on CPU!)")
        raise AssertionError(f"Operation {op_name} fell back to CPU execution")

    # Check if op ran on Neuron
    if op_name in executed_ops:
        print(f"✓ {op_name} found in executed ops (running on Neuron)")
        return

    # Op not found in either list
    # Check for NKI kernel execution patterns
    nki_variants = [
        f"nki_{op_name}",
        op_name.replace("aten::", "nki_"),
        op_name.replace("aten::", ""),
    ]

    for variant in nki_variants:
        if any(variant in op for op in executed_ops):
            print(f"✓ NKI kernel variant '{variant}' detected (running on Neuron)")
            return

    # If we get here, op wasn't found
    print(f"⚠ {op_name} not found in executed or fallback ops")
    print(f"  Executed ops: {executed_ops}")
    print(f"  Fallback ops: {fallback_ops}")
    raise AssertionError(f"Operation {op_name} was not tracked in execution")


def get_executed_op_list() -> dict:
    """Get a detailed report of operations executed on Neuron vs CPU.

    Returns:
        Dictionary with 'executed' and 'fallback' lists of operation names
    """
    return {
        "executed": torch_neuron.get_executed_ops(),
        "fallback": torch_neuron.get_fallback_ops(),
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
    executed_ops = torch_neuron.get_executed_ops()
    fallback_ops = torch_neuron.get_fallback_ops()

    # Check if op ran on Neuron
    if op_name in executed_ops:
        print(f"❌ {op_name} found in executed ops (running on Neuron!)")
        raise AssertionError(f"Operation {op_name} ran on Neuron instead of falling back to CPU")

    # Check if op ran as fallback (on CPU)
    if op_name in fallback_ops:
        print(f"✓ {op_name} found in fallback ops (running on CPU)")
        return

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
                f"NKI kernel variant of {op_name} ran on Neuron instead of falling back to CPU"
            )

    # If we get here, op wasn't found in either list
    print(f"⚠ {op_name} not found in executed or fallback ops")
    print(f"  Executed ops: {executed_ops}")
    print(f"  Fallback ops: {fallback_ops}")
    raise AssertionError(f"Operation {op_name} was not tracked in execution")


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
    import torch_neuron._C as _C

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


def cpu_fallback_test(func):
    """Decorator that removes TORCH_NEURON_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS for CPU fallback"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        original_value = os.environ.pop("TORCH_NEURON_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", None)
        try:
            return func(*args, **kwargs)
        finally:
            if original_value is not None:
                os.environ["TORCH_NEURON_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS"] = original_value

    return wrapper

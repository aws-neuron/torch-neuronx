"""Pytest configuration for running upstream PyTorch tests natively on Neuron."""

import os
import shutil
from pathlib import Path


def _overlay_torch_internal() -> None:
    """Overlay ``torch/testing/_internal/`` with our forked version.

    Must run before ``import torch`` (see call site below) so the new files
    are on disk before Python caches any ``_internal`` submodules. The
    xdist controller does the swap once; workers skip to avoid races on
    the shared ``_internal_tmp`` staging directory.
    """
    # Only the xdist controller (master) runs the overlay.
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return

    overlay = Path(__file__).parent / "torch" / "testing" / "_internal"
    if not overlay.exists():
        return  # Overlay source not present; leave install untouched.

    import importlib.util

    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.origin:
        return  # torch isn't installed; nothing to overlay.

    installed = Path(spec.origin).parent / "testing" / "_internal"

    # Atomic swap: copy overlay → temp → rename over the installed dir.
    # If the process crashes mid-operation, site-packages is left either
    # fully old or fully new but never half-overlaid.
    temp = installed.with_name("_internal_tmp")
    if temp.exists():
        shutil.rmtree(temp)
    shutil.copytree(str(overlay), str(temp))
    if installed.exists():
        shutil.rmtree(installed)
    temp.rename(installed)


# Run the overlay BEFORE any torch import. Order matters.
_overlay_torch_internal()

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.testing._comparison as _torch_cmp  # noqa: E402
from skip_registry.neuron_blocklist import BLOCKLIST  # noqa: E402

os.environ["DISTRIBUTED_TESTS_DEFAULT_TIMEOUT"] = "600"  # 10 minutes
os.environ.setdefault("PYTORCH_TESTING_DEVICE_ONLY_FOR", "neuron")
# Prefer multi-process over multi-threaded tests (NRT is not thread-safe)
os.environ.setdefault("PYTORCH_TESTING_PREFER_MULTIPROCESS", "1")
# Prevent torch.manual_seed() from triggering CUDA lazy init (segfaults without GPU driver)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Neuron downcasts float64→float32 and int64→int32 internally.
# Relax default test tolerances for float64 to float32-equivalent levels
# so that expected precision loss from the downcast doesn't cause test failures.
_torch_cmp._DTYPE_PRECISIONS[torch.float64] = (1e-3, 1e-3)


# Collection-time ignores — processed before any hooks or imports.
# collect_ignore_glob is a global variable you can define in a conftest.py file
# to exclude specific test files or directories from collection using shell-style wildcard patterns
def _to_glob(e):
    """Convert a blocklist entry to a collect_ignore_glob pattern.

    Every entry is a path relative to the ``test/`` root.
    """
    test_dir = str(Path(__file__).parent / "test")
    if e.endswith(".py"):
        return f"{test_dir}/{e}"
    return f"{test_dir}/{e}/**"


collect_ignore_glob = [_to_glob(entry) for group in BLOCKLIST for entry in group.tests]


def _setup_neuron_cores_for_xdist(config):
    """Partition visible cores across xdist workers.

    Distributed tests get WORLD_SIZE cores per worker (default 8).
    All other tests get 1 core per worker.
    Without xdist, no core partitioning is applied.
    """
    test_paths = [str(a) for a in getattr(config, "args", [])]
    if not any("pytorch_tests_native" in p for p in test_paths):
        return

    from tests.utils.neuron_test_utils import (
        get_core_start_index,
        get_pytest_worker_count,
        get_total_core_count,
    )

    if get_pytest_worker_count() <= 1:
        return

    is_distributed = any("distributed" in p for p in test_paths)
    cores_per_worker = int(os.environ.get("WORLD_SIZE", "8")) if is_distributed else 1
    _core_start = get_core_start_index()
    _core_end = min(_core_start + cores_per_worker - 1, get_total_core_count() - 1)
    if _core_start > _core_end:
        return
    os.environ["NEURON_RT_VISIBLE_CORES"] = f"{_core_start}-{_core_end}"


@pytest.fixture(autouse=True)
def _reset_fx_tracing_flag():
    yield
    torch.fx._symbolic_trace._is_fx_tracing_flag = False


def pytest_configure(config):
    """Apply Neuron-specific patches after the torch._internal overlay.

    The overlay of ``torch/testing/_internal/`` is performed at conftest
    module load time (see :func:`_overlay_torch_internal`) — before any
    ``import torch`` so that subsequent imports pick up the new files.
    This hook only applies runtime Neuron patches (compile redirect,
    cross-device comparison, OpInfo sample filter) and xdist setup.
    """
    # Apply Neuron test patches (compile redirect, cross-device comparison)
    from tests.pytorch_tests_native.neuron_overrides import apply_patches

    apply_patches()

    # Set up neuron core allocation for xdist workers
    _setup_neuron_cores_for_xdist(config)


# ---------------------------------------------------------------------------
# Skip/xfail markers
# ---------------------------------------------------------------------------
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Apply Neuron skip/xfail markers from the registry."""
    from tests.pytorch_tests_native.neuron_test_registry import apply_markers

    for item in items:
        apply_markers(item)


@pytest.hookimpl(hookwrapper=True)
def pytest_report_teststatus(report, config):
    """Add test duration to verbose output."""
    pluggy_result = yield
    if not hasattr(report, "duration"):
        return
    outcome, letter, verbose = pluggy_result.get_result()
    if verbose:
        pluggy_result.force_result((outcome, letter, f"{verbose} [{report.duration:.4f}s]"))

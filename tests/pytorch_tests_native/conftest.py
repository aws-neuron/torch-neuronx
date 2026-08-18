"""Pytest configuration for running upstream PyTorch tests natively on Neuron."""

import os
import shutil
import sys
from pathlib import Path

# Env var the xdist controller uses to hand the per-worker core count to its
# workers. The controller knows the test selection (via ``sys.argv``) but has no
# worker index; each worker knows its index but not the selection (an xdist
# worker's ``sys.argv`` is just ``['-c']``). The controller resolves the count
# and exports it here at conftest module load — before workers are spawned, so
# they inherit it.
_CORES_PER_WORKER_ENV = "NEURON_CORES_PER_WORKER"


def _setup_neuron_cores_for_xdist() -> None:
    """Partition visible cores across xdist workers, BEFORE ``import torch``.

    ``torch.get_device_module(...).device_count()`` (and
    ``torch.accelerator.device_count()``) memoize the Neuron device count on
    their first call. If that first call happens before ``NEURON_RT_VISIBLE_CORES``
    is set, every later reader sees the full physical count (e.g. 64) instead of
    this worker's core allocation (e.g. 8). MultiProcContinuousTest classes whose
    ``world_size`` is unset then resolve world_size from that stale count and
    spawn 64 rank processes onto 8 usable cores — every rank dies in
    ``_init_pg`` and the dispatcher hangs draining the dead queues. Setting
    NEURON_RT_VISIBLE_CORES at module load, before the first torch import,
    guarantees it precedes all the workers' device_count call.

    Distributed tests get WORLD_SIZE cores per worker (default 8); other tests
    get 1. The controller resolves the per-worker count (NEURON_CORES_PER_WORKER);
    Each worker use it plus its own index to configure NEURON_RT_VISIBLE_CORES.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER")

    if worker is None:
        # Controller: it has the test selection in sys.argv, worker index is None.
        # Resolve cores-per-worker and export it for the workers to inherit.
        if not any("pytorch_tests_native" in a for a in sys.argv):
            return
        # PYTEST_XDIST_WORKER_COUNT is mostly available on worker, not controller.
        # This falls back to "_argv_worker_count()" parsing "-n" in pytest cmd
        worker_count = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "0")) or _argv_worker_count()
        if worker_count <= 1:
            return
        is_distributed = any("distributed" in a for a in sys.argv)
        cores_per_worker = int(os.environ.get("WORLD_SIZE", "8")) if is_distributed else 1
        os.environ[_CORES_PER_WORKER_ENV] = str(cores_per_worker)
        return

    # Worker: sees _CORES_PER_WORKER_ENV since controller already set it in os.environ
    cores_per_worker = int(os.environ.get(_CORES_PER_WORKER_ENV, "0"))
    if cores_per_worker <= 0:
        return

    from tests.utils.neuron_test_utils import (
        get_core_start_index,
        get_pytest_worker_count,
        get_total_core_count,
    )

    if get_pytest_worker_count() <= 1:
        return
    _core_start = get_core_start_index()
    _core_end = min(_core_start + cores_per_worker - 1, get_total_core_count() - 1)
    if _core_start > _core_end:
        return
    os.environ["NEURON_RT_VISIBLE_CORES"] = f"{_core_start}-{_core_end}"


def _argv_worker_count() -> int:
    """Parse ``-n <N>`` / ``-nN`` from the controller's argv (xdist worker count)."""
    for i, a in enumerate(sys.argv):
        if a == "-n" and i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                return 0
        if a.startswith("-n") and len(a) > 2:
            try:
                return int(a[2:])
            except ValueError:
                return 0
    return 0


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

# Partition Neuron cores per xdist worker BEFORE importing torch, so the device
# count is memoized against this worker's window (see the function docstring).
_setup_neuron_cores_for_xdist()

# Sibling-import shims for upstream test directories
_TEST_ROOT = Path(__file__).parent / "test"
for _sibling_import_dir in ("distributed/pipelining",):
    _p = _TEST_ROOT / _sibling_import_dir
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.testing._comparison as _torch_cmp  # noqa: E402
from skip_registry.neuron_blocklist import BLOCKLIST  # noqa: E402

os.environ["DISTRIBUTED_TESTS_DEFAULT_TIMEOUT"] = "600"  # 10 minutes
os.environ.setdefault("PYTORCH_TESTING_DEVICE_ONLY_FOR", "neuron")
# Some upstream files hardcode only_for=["cpu","cuda","hpu","xpu"] in their
# instantiate_device_type_tests() call, which drops the PrivateUse1 (neuron)
# base before PYTORCH_TESTING_DEVICE_ONLY_FOR=neuron is even applied. This re-adds the
# PrivateUse1 base from the full list, bypassing the file's only_for, and runs
# before the env only_for filter so "privateuse1" survives.
os.environ.setdefault("PYTORCH_TESTING_DEVICE_FOR_CUSTOM", "privateuse1")
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
    cross-device comparison, OpInfo sample filter).
    """
    # Apply Neuron test patches (compile redirect, cross-device comparison)
    from tests.pytorch_tests_native.neuron_overrides import apply_patches

    apply_patches()


# ---------------------------------------------------------------------------
# Skip/xfail markers
# ---------------------------------------------------------------------------
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Apply Neuron skip/xfail markers from the registry."""
    from tests.pytorch_tests_native.neuron_test_registry import apply_markers

    for item in items:
        # Guard: only apply skip/xfail markers to pytorch_tests_native tests.
        # pytest_collection_modifyitems receives ALL collected items, and
        # run-test-parallel collects the entire tests/ tree in one invocation.
        # Without this guard, patterns like *complex* in the skip registry
        # would incorrectly skip tests in tests/operations/ and tests/views/.
        if "pytorch_tests_native" in str(item.fspath):
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

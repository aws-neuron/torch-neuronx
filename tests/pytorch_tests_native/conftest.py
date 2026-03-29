"""Pytest configuration for running upstream PyTorch tests natively on Neuron."""

import os
import shutil
from pathlib import Path

import pytest

os.environ["DISTRIBUTED_TESTS_DEFAULT_TIMEOUT"] = "1800"  # 30 minutes
os.environ.setdefault("PYTORCH_TESTING_DEVICE_ONLY_FOR", "neuron")


def pytest_configure(config):
    """Override installed torch.testing._internal with our forked version.

    Runs only in the master process (before xdist workers spawn), ensuring
    all processes see the same replaced files without race conditions.
    """
    # Skip in xdist workers — they have workerinput, master doesn't
    if hasattr(config, "workerinput"):
        return

    import torch.testing._internal as _internal

    installed = Path(os.path.dirname(_internal.__file__))
    overlay = Path(__file__).parent / "torch" / "testing" / "_internal"
    if overlay.exists():
        shutil.rmtree(installed)
        shutil.copytree(str(overlay), str(installed))


import torch_neuronx  # noqa: E402


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

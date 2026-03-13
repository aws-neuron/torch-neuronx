"""Pytest configuration for running upstream PyTorch tests natively on Neuron."""

import os
from pathlib import Path

import pytest

os.environ["DISTRIBUTED_TESTS_DEFAULT_TIMEOUT"] = "1800"  # 30 minutes
os.environ.setdefault("PYTORCH_TESTING_DEVICE_ONLY_FOR", "neuron")

# Setup overlay for torch.testing._internal
try:
    import torch

    root = Path(__file__).parent
    testing_internal = root / "torch" / "testing" / "_internal"
    if testing_internal.exists():
        os.environ["TORCH_TESTING_INTERNAL_PATH"] = str(testing_internal)
        import torch.testing._internal as _internal

        _internal.__path__.insert(0, str(testing_internal))
except ModuleNotFoundError:
    pass


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

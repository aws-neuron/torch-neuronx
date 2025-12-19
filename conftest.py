"""
pytest configuration for torch-neuronx tests
"""

import os

import pytest

SANITY_PT_DISTRIBUTED_TESTS = None


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "multi_device: marks tests as distributed tests")

    global SANITY_PT_DISTRIBUTED_TESTS
    # Read the option from the distributed configuration object
    SANITY_PT_DISTRIBUTED_TESTS = config.getoption("--sanity_pt_distributed_tests")


def is_distributed_test(item):
    """Check if test is in distributed folder or has distributed marker."""
    return (
        "distributed" in str(item.fspath)  # Check path
        or bool(list(item.iter_markers(name="multi_device")))  # Check marker
    )


def is_launcher_env(session):
    return not os.path.exists("/dev/neuron0")


def pytest_addoption(parser):
    """Register custom command line options."""
    parser.addoption(
        "--test_suite_path", action="store", default="tests/", help="Path to test suite to run"
    )
    parser.addoption(
        "--num_workers",
        action="store",
        help="Num workers to run the pytest command with",
    )
    # custom ignore arg is parsed to --ignore on the command run by compute node;
    # Can be passed multiple times
    # standard ignore conflicts with reserved keyword and does not allow manual parsing
    parser.addoption(
        "--custom-ignore",
        action="append",
        dest="ignore_paths",
        help="Ignore paths during test collection (can be used multiple times)",
    )
    parser.addoption(
        "--skip_pt_test_distributed",
        action="store_true",
        default="",
        help="Skip PT Distributed tests.",
    )
    parser.addoption(
        "--sanity_pt_distributed_tests",
        action="store_true",
        default=False,
        help="Enable sanity mode for distributed PyTorch tests.",
    )

    parser.addoption(
        "--json_report",
        action="store_true",
        default="",
        help="Collect distributed tests in upstream pytorch tests",
    )

    parser.addoption(
        "--skip-distributed", action="store_true", default=False, help="Skip distributed tests"
    )

    parser.addoption(
        "--cpp_unit_tests",
        action="store_true",
        default=False,
        help="Run C++ unit tests",
    )

    parser.addoption(
        "--async",
        action="store_true",
        default=False,
        help="Run tests in async mode",
    )


@pytest.fixture(scope="session", autouse=True)
def ensure_neuron_initialized(request):
    """Ensure torch_neuronx is fully initialized before any test runs.
    Skip for distributed tests."""

    # torch import inside this function, as TOD worker doesn't have torch
    # exit early if in tod environment or bb test
    if is_launcher_env(request.session):
        return

    import torch
    from packaging.version import Version

    torch_neuronx = None
    if Version("2.9.0") > Version(torch.__version__):
        import torch_neuronx

    # Skip if any test in the session has distributed marker
    # # Check if there are any NON-distributed tests
    has_distributed_tests = all(is_distributed_test(item) for item in request.session.items)

    # Only skip if ALL tests are distributed (i.e., no non-distributed tests)
    if has_distributed_tests:
        return

    # Use only one core for non-distributed tests (when not using xdist)
    os.environ["NEURON_RT_NUM_CORES"] = "1"
    # NEURON_RT_VIRTUAL_CORE_SIZE can be set externally to control LNC
    # If not set, defaults to 2 (LNC=2)

    # Force initialization
    if torch_neuronx is not None:
        torch_neuronx._lazy_init()

    # Force a dummy tensor operation to complete initialization
    # This ensures all Python ops are registered before tests run
    t = torch.empty(1, device="neuron")
    del t


@pytest.fixture(scope="class")
def device():
    """Get the neuron device."""
    import torch

    return torch.device("neuron")


if hasattr(pytest, "hookimpl"):
    # Only use xdist hooks if xdist is available
    try:
        import xdist  # noqa: F401

        @pytest.hookimpl(tryfirst=True)
        def pytest_configure_node(node):
            """Configure each xdist worker node with a unique neuron core."""

            import json
            import subprocess

            from torch_neuronx.utils import get_worker_multiplier

            result = subprocess.run(
                ["/opt/aws/neuron/bin/neuron-ls", "-j"], capture_output=True, text=True, check=True
            )
            devices = json.loads(result.stdout)
            total_cores = sum(device["nc_count"] for device in devices)

            # Determine worker multiplier based on NEURON_RT_VIRTUAL_CORE_SIZE
            multiplier = get_worker_multiplier()

            device_count = total_cores * multiplier
            # Need to set this so that the distributed test can start from correct core count
            node.workerinput["TOTAL_DEVICE_COUNT"] = str(device_count)

            # Check if this is a distributed test by looking at the test file paths
            test_paths = getattr(node.config, "args", [])
            is_distributed_test = any("distributed" in str(path) for path in test_paths)

            if is_distributed_test:
                # Distributed tests: divide total cores by number of workers
                worker_count = getattr(node.config.option, "numprocesses", 1)
                cores_per_worker = device_count // worker_count
                os.environ["NEURON_RT_NUM_CORES"] = str(cores_per_worker)
            else:
                # Non-distributed tests: 1 core per process
                os.environ["NEURON_RT_NUM_CORES"] = "1"
            # NEURON_RT_VIRTUAL_CORE_SIZE can be set externally to control LNC
            # If not set, defaults to 2 (LNC=2)
    except ImportError:
        # xdist not available, skip parallel configuration
        pass


def pytest_runtest_setup(item):
    """Set environment variables from worker input before each test."""
    # Check if we have worker input (running under xdist)
    if hasattr(item.config, "workerinput"):
        worker_input = item.config.workerinput
        if "TOTAL_DEVICE_COUNT" in worker_input:
            os.environ["TOTAL_DEVICE_COUNT"] = worker_input["TOTAL_DEVICE_COUNT"]

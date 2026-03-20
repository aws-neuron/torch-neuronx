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


def is_lnc1_test(item):
    """Check if test file has lnc1 in name (needs LNC=1 configuration)."""
    return "_lnc1" in str(item.fspath)


def is_all_cores_test(item):
    """Check if test needs all cores visible (no NEURON_RT_NUM_CORES limit) -
    Tests with "_all_cores" in filename need all cores visible.
    """
    return "_all_cores" in str(item.fspath)


def is_launcher_env(session):
    return not os.path.exists("/dev/neuron0")


def pytest_addoption(parser):
    """Register custom command line options."""
    parser.addoption(
        "--test_suite_path",
        action="store",
        default="tests/",
        help=(
            "Path to test suite to run. Supports comma-separated paths "
            "e.g. 'tests/a.py,tests/b.py'."
        ),
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

    parser.addoption(
        "--instance_type",
        action="store",
        default="trn2.48xlarge",
        help="Instance type for remote workload tests",
    )

    # Block performance test options
    parser.addoption(
        "--block",
        action="store",
        default="qwen3_torchtitan",
        help="Block module name",
    )
    parser.addoption(
        "--preset",
        action="store",
        default="qwen3-8b-tp",
        help="Block preset config",
    )
    parser.addoption(
        "--tp_size",
        action="store",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.addoption(
        "--nproc",
        action="store",
        type=int,
        default=1,
        help="Number of processes",
    )
    parser.addoption(
        "--warmup_runs",
        action="store",
        type=int,
        default=2,
        help="Number of warmup runs",
    )
    parser.addoption(
        "--benchmark_runs",
        action="store",
        type=int,
        default=3,
        help="Number of benchmark runs",
    )
    parser.addoption(
        "--singlecore",
        action="store_true",
        default=False,
        help="Run singlecore block perf (no distributed)",
    )
    parser.addoption(
        "--n_layers",
        action="store",
        type=int,
        default=4,
        help="Number of transformer layers for block perf benchmarks "
        "(default: 4, mixtral_moe uses 2)",
    )

    # Collective performance test options
    parser.addoption(
        "--collective_op",
        action="store",
        default="all",
        help="Collective op (all_reduce, all_gather_into_tensor, reduce_scatter_tensor, all)",
    )
    parser.addoption(
        "--collective_sizes",
        action="store",
        default="2M",
        help="Comma-separated tensor sizes for collective benchmark",
    )
    parser.addoption(
        "--collective_dtype",
        action="store",
        default="float32",
        help="Data type for collective benchmark (float32, float16, bfloat16)",
    )

    parser.addoption(
        "--remote_workload_timeout",
        action="store",
        default=None,
        help="Timeout in seconds for remote workload execution",
    )
    parser.addoption(
        "--torch_version",
        action="store",
        default="stable",
        choices=["stable", "nightly"],
        help="PyTorch version: 'stable' or 'nightly'",
    )

    # Performance regression gate options
    parser.addoption(
        "--mfu_threshold",
        action="store",
        type=float,
        default=None,
        help="Minimum MFU percentage threshold for perf regression gate",
    )
    parser.addoption(
        "--peak_memory_gb",
        action="store",
        type=float,
        default=None,
        help="Maximum peak memory in GB for perf regression gate",
    )
    parser.addoption(
        "--mfu_regression_tolerance",
        action="store",
        type=float,
        default=0.10,
        help="MFU regression tolerance (default: 0.10 = 10%)",
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

    # Check if ONLY lnc1 tests are being run
    all_lnc1 = all(is_lnc1_test(item) for item in request.session.items)
    # Check if all tests need all cores visible
    needs_all_cores = all(is_all_cores_test(item) for item in request.session.items)

    # Set LNC=1 if all tests require it
    if all_lnc1:
        os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"

    # If test needs all cores be visible, don't limit NEURON_RT_NUM_CORES
    if needs_all_cores:
        pass
    else:
        os.environ["NEURON_RT_NUM_CORES"] = "1"

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

            # Determine worker multiplier based on NEURON_LOGICAL_NC_CONFIG
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
            # NEURON_LOGICAL_NC_CONFIG can be set externally to control LNC
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


@pytest.fixture(scope="class")
def enable_metrics_for_class():
    """Enable TORCH_NEURONX_METRICS_ENABLED for classes using this fixture."""
    original_value = os.environ.get("TORCH_NEURONX_METRICS_ENABLED")
    os.environ["TORCH_NEURONX_METRICS_ENABLED"] = "1"
    yield
    if original_value is None:
        os.environ.pop("TORCH_NEURONX_METRICS_ENABLED", None)
    else:
        os.environ["TORCH_NEURONX_METRICS_ENABLED"] = original_value


@pytest.fixture(scope="function", autouse=True)
def reset_dynamo(request):
    """Reset torch dynamo state before each test to ensure isolation."""
    # Skip if torch is not available (TOD worker, C++ tests, etc.)
    if is_launcher_env(request.session):
        return
    import torch._dynamo

    torch._dynamo.reset()


def pytest_sessionfinish(session, exitstatus):
    """Print S3 results URL after all tests complete (outside pytest capture)."""
    s3_url = getattr(session.config, "_aten_upstream_results_s3_url", None)
    if s3_url:
        print(f"\nINFO: [ATEN_UPSTREAM_RESULTS] S3 results folder: {s3_url}")

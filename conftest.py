"""
pytest configuration for torch-neuronx tests
"""

import contextlib
import os
import sys
from pathlib import Path

import pytest

# tests/test_infra and tests/testing check the CI tooling and read files under
# .github/. A runtime image that ships the test tree without .github (the DLC
# prunes it) cannot run them. Skip them there. Normal CI keeps .github, so this
# does nothing in CI.
_REPO_ROOT = Path(__file__).resolve().parent
_HAS_GITHUB = (_REPO_ROOT / ".github").is_dir()
_CI_TOOLING_DIRS = (("tests", "test_infra"), ("tests", "testing"))


@contextlib.contextmanager
def _module_auto_compile_patch_scope():
    previous = {
        name: os.environ.get(name)
        for name in (
            "TORCH_NEURONX_MODULE_AUTO_COMPILE",
            "TORCH_NEURONX_MODULE_AUTO_COMPILE_DEBUG",
        )
    }
    existing_module = sys.modules.get("torch_neuronx")
    patches_were_active = bool(
        existing_module
        and getattr(
            existing_module,
            "_module_auto_compile_patches_are_active",
            lambda: False,
        )()
    )
    os.environ["TORCH_NEURONX_MODULE_AUTO_COMPILE"] = "1"
    os.environ["TORCH_NEURONX_MODULE_AUTO_COMPILE_DEBUG"] = "1"

    installed_by_fixture = False
    try:
        import torch_neuronx

        torch_neuronx._install_module_auto_compile_patches()
        installed_by_fixture = (
            not patches_were_active and torch_neuronx._module_auto_compile_patches_are_active()
        )
        yield
    finally:
        try:
            if installed_by_fixture:
                torch_neuronx._uninstall_module_auto_compile_patches()
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


@pytest.fixture(scope="session")
def _module_auto_compile_patch_scope_factory():
    return _module_auto_compile_patch_scope


def pytest_configure(config):
    """Register the custom pytest markers.

    Args:
        config (pytest.Config): Pytest configuration supplied by pytest for this hook.
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "multi_device: marks tests as distributed tests")
    config.addinivalue_line(
        "markers",
        "neuron_cores(n): the distributed test uses n NeuronCores. The default is 4.",
    )
    config.addinivalue_line(
        "markers",
        "needs_process_isolation(reason): test cannot share a process with other tests. "
        "A run with isolated_pass deselects these from the shared run. The runner "
        "runs them in a second, forked pass. See docs/AI-testing.md.",
    )
    config.addinivalue_line(
        "markers",
        "nrt_stream_config(host_cc_enabled, streams_enabled=True): "
        "seed the stream flags before Neuron initialization.",
    )
    if not _HAS_GITHUB:
        print(
            "conftest: .github/ is absent. Skipping the CI-tooling tests in "
            "tests/test_infra and tests/testing; they read .github/.",
            file=sys.stderr,
        )


def pytest_ignore_collect(collection_path, config):
    """Skip the CI-tooling test directories when .github/ is absent.

    Args:
        collection_path (pathlib.Path): Candidate path pytest wants to collect.
        config (pytest.Config): Pytest configuration for this run.

    Returns:
        bool | None: True to skip tests/test_infra and tests/testing when
        .github/ is absent. None otherwise, so pytest collects as usual.
    """
    if _HAS_GITHUB:
        return None
    try:
        parts = Path(collection_path).resolve().relative_to(_REPO_ROOT).parts
    except ValueError:
        return None
    return parts[:2] in _CI_TOOLING_DIRS or None


def is_distributed_test(item):
    """Return whether a collected test is distributed.

    Args:
        item (pytest.Item): Collected test whose path and markers are inspected.

    Returns:
        bool: ``True`` for tests under a distributed path or marked ``multi_device``.
    """
    return (
        "distributed" in str(item.fspath)  # Check path
        or bool(list(item.iter_markers(name="multi_device")))  # Check marker
    )


def is_lnc1_test(item):
    """Return whether a test requires the ``LNC=1`` configuration.

    Args:
        item (pytest.Item): Collected test whose file path is inspected.

    Returns:
        bool: ``True`` when the test file path contains ``_lnc1``.
    """
    return "_lnc1" in str(item.fspath)


def is_all_cores_test(item):
    """Return whether a test requires all Neuron cores to remain visible.

    Args:
        item (pytest.Item): Collected test whose file path is inspected.

    Returns:
        bool: ``True`` when the test file path contains ``_all_cores``.
    """
    return "_all_cores" in str(item.fspath)


def is_launcher_env(session):
    """Return whether ``/dev/neuron0`` is absent.

    Args:
        session (pytest.Session): Test session accepted but not inspected.

    Returns:
        bool: ``True`` when ``/dev/neuron0`` does not exist.
    """
    return not os.path.exists("/dev/neuron0")


def pytest_addoption(parser):
    """Register the repository-specific pytest command-line options.

    Args:
        parser (pytest.Parser): Command-line parser supplied by pytest for this hook.
    """
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
    """Initialize the Neuron device before eligible tests execute.

    Args:
        request (pytest.FixtureRequest): Session fixture request supplied by pytest.

    Returns:
        None: This setup-only fixture does not provide a value.

    Note:
        This setup-only autouse fixture skips when ``/dev/neuron0`` is absent or every
        collected test is distributed, configures core visibility, calls
        ``torch_neuronx._lazy_init()`` before PyTorch 2.9, allocates one device tensor,
        and performs no teardown.
    """
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
    """Return the PyTorch Neuron device.

    Returns:
        torch.device: Device object for the ``neuron`` device.
    """
    import torch

    return torch.device("neuron")


if hasattr(pytest, "hookimpl"):
    # Only use xdist hooks if xdist is available
    try:
        import xdist  # noqa: F401

        @pytest.hookimpl(tryfirst=True)
        def pytest_configure_node(node):
            """Configure Neuron core visibility for an xdist worker.

            Args:
                node (object): Worker controller supplied by pytest-xdist for this hook.

            Raises:
                subprocess.CalledProcessError: If ``neuron-ls`` cannot enumerate devices.
                json.JSONDecodeError: If ``neuron-ls`` returns invalid JSON.
            """
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


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Configure per-test stream flags and copy the xdist device count.

    Args:
        item (pytest.Item): Test item supplied by pytest before its setup phase.

    Note:
        The ``nrt_stream_config`` marker saves the current stream-related environment,
        seeds its requested streams and HostCC settings, and marks the test callable as seeded.
    """
    stream_config = item.get_closest_marker("nrt_stream_config")
    if stream_config is not None:
        host_cc_enabled = stream_config.kwargs["host_cc_enabled"]
        test_func = getattr(item.obj, "__func__", item.obj)
        item._nrt_stream_config_test_func = test_func
        item._original_nrt_stream_config = {
            "TORCH_NEURONX_ENABLE_STREAMS": os.environ.get("TORCH_NEURONX_ENABLE_STREAMS"),
            "TORCH_NEURONX_ENABLE_HOST_CC": os.environ.get("TORCH_NEURONX_ENABLE_HOST_CC"),
            "NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE": os.environ.get(
                "NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE"
            ),
            "NEURON_RT_EXEC_TIMEOUT": os.environ.get("NEURON_RT_EXEC_TIMEOUT"),
        }
        os.environ["TORCH_NEURONX_ENABLE_STREAMS"] = (
            "1" if stream_config.kwargs.get("streams_enabled", True) else "0"
        )
        os.environ["TORCH_NEURONX_ENABLE_HOST_CC"] = "1" if host_cc_enabled else "0"
        if host_cc_enabled:
            # Package initialization applies these when HostCC is on, but it already
            # ran with the launch flags, so seed what a HostCC process would have.
            os.environ["NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE"] = "true"
            os.environ.setdefault("NEURON_RT_EXEC_TIMEOUT", "300")
        else:
            # A true value with HostCC off is rejected at import.
            os.environ.pop("NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE", None)
        test_func._nrt_stream_config_seeded = True

    # Check if we have worker input (running under xdist)
    if hasattr(item.config, "workerinput"):
        worker_input = item.config.workerinput
        if "TOTAL_DEVICE_COUNT" in worker_input:
            os.environ["TOTAL_DEVICE_COUNT"] = worker_input["TOTAL_DEVICE_COUNT"]


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    """Restore process-level stream flags changed before fixture initialization.

    Args:
        item (pytest.Item): Test item whose saved stream configuration is restored.
        nextitem (pytest.Item | None): Next test item, accepted for the pytest hook protocol.

    Returns:
        None: This teardown hook does not return a value.
    """
    original = getattr(item, "_original_nrt_stream_config", None)
    if original is None:
        return

    item._nrt_stream_config_test_func._nrt_stream_config_seeded = False
    for name, value in original.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture(scope="class")
def enable_metrics_for_class():
    """Enable torch-neuronx metrics for a test class.

    Yields:
        None: Control to the test class while metrics are enabled.

    Note:
        Teardown restores the original environment value or removes it if previously unset.
    """
    original_value = os.environ.get("TORCH_NEURONX_METRICS_ENABLED")
    os.environ["TORCH_NEURONX_METRICS_ENABLED"] = "1"
    yield
    if original_value is None:
        os.environ.pop("TORCH_NEURONX_METRICS_ENABLED", None)
    else:
        os.environ["TORCH_NEURONX_METRICS_ENABLED"] = original_value


@pytest.fixture(scope="function", autouse=True)
def reset_dynamo(request):
    """Reset TorchDynamo state before each eligible test.

    Args:
        request (pytest.FixtureRequest): Function fixture request supplied by pytest.

    Returns:
        None: This setup-only fixture does not provide a value.

    Note:
        This setup-only autouse fixture skips launcher environments and performs no teardown.
    """
    # Skip if torch is not available (TOD worker, C++ tests, etc.)
    if is_launcher_env(request.session):
        return
    import torch._dynamo

    torch._dynamo.reset()


def pytest_sessionfinish(session, exitstatus):
    """Print the upstream-results S3 URL at session finish when cached.

    Args:
        session (pytest.Session): Completed test session supplied by pytest for this hook.
        exitstatus (int or pytest.ExitCode): Session exit status supplied by pytest but
            ignored by this hook.
    """
    s3_url = getattr(session.config, "_aten_upstream_results_s3_url", None)
    if s3_url:
        print(f"\nINFO: [ATEN_UPSTREAM_RESULTS] S3 results folder: {s3_url}")

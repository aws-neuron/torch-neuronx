"""Conftest for distributed tests - automatically marks all tests in this folder."""

import os

import pytest

# A test without the neuron_cores mark can use this many cores.
DEFAULT_NEURON_CORES = 4
# Run only the tests of this core window. A CI node sets it in the test config.
CORE_WINDOW = os.environ.get("NEURON_TEST_CORE_WINDOW")


def pytest_configure(config):
    # Pin each xdist worker to its own core window, the same window that
    # get_core_start_index() gives to the ranks. Without it, a runtime with no
    # pinned core (a subprocess or a one-rank test) takes free cores from other
    # windows. gw0 also has no NEURON_RT_NUM_CORES, so it takes all free cores.
    if not hasattr(config, "workerinput"):
        return
    from tests.utils.neuron_test_utils import get_core_start_index, get_pytest_worker_count

    total = int(config.workerinput["TOTAL_DEVICE_COUNT"])
    os.environ["TOTAL_DEVICE_COUNT"] = str(total)
    start = get_core_start_index()
    os.environ["NEURON_RT_VISIBLE_CORES"] = (
        f"{start}-{start + total // get_pytest_worker_count() - 1}"
    )
    os.environ.pop("NEURON_RT_NUM_CORES", None)


def neuron_cores(item):
    mark = item.get_closest_marker("neuron_cores")
    return mark.args[0] if mark else DEFAULT_NEURON_CORES


def core_window(cores):
    """Return the smallest core window that has ``cores`` cores.

    The window of a worker is the core count divided by the worker count, so it is a power of 2.
    """
    window = DEFAULT_NEURON_CORES
    while window < cores:
        window *= 2
    return window


def pytest_collection_modifyitems(config, items):
    """Add 'distributed' marker to all tests in the distributed folder."""
    distributed_marker = pytest.mark.multi_device

    for item in items:
        # Check if this test is in the distributed folder
        if "distributed" in str(item.fspath):
            item.add_marker(distributed_marker)

    # A core-window run uses 64 / window workers, and keeps only the tests of its window.
    if CORE_WINDOW is None:
        return
    window = int(CORE_WINDOW)
    deselected = [item for item in items if core_window(neuron_cores(item)) != window]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if core_window(neuron_cores(item)) == window]


@pytest.fixture(autouse=True)
def override_main_fixtures():
    # Override any fixtures from main conftest
    os.environ["NEURON_RT_PORT"] = "2025"


@pytest.fixture(autouse=True)
def neuron_core_budget(request):
    # get_rank_core() reads the budget in each rank. In a core-window run the budget comes
    # from the mark, not from the worker count, so a test with too many ranks also fails in the
    # retry pass. Other runs, for example the host-cc run, keep the core window of the worker.
    if CORE_WINDOW is None:
        yield
        return
    os.environ["NEURON_TEST_CORE_BUDGET"] = str(neuron_cores(request.node))
    yield
    os.environ.pop("NEURON_TEST_CORE_BUDGET", None)

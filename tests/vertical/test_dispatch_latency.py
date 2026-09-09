"""Per-op dispatch-latency benchmark.

Each case times the op's dispatch layer against its end-to-end call and writes a
JSON row; ``conftest.py`` aggregates those into ``dispatch_latency_<ts>.csv`` at
session end, which feeds the Grafana panel. A pytest case per op gives per-op
reporting, isolation so one bad op cannot lose the sweep, and ``-k`` selection for
free. Measurement lives in ``internal/dispatch_timing.py``.

Run with ``-n 1``. This is a host-side measurement at microsecond scale, so xdist
workers would add scheduler noise, and ``met.clear_all()`` would wipe
process-global metrics for anything sharing the interpreter.

Export ``TORCH_NEURONX_DISPATCH_TIMING=1`` and ``NEURON_LAUNCH_BLOCKING=0`` before
starting python. The fixture below checks both.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

import torch_neuronx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from internal.common import build_params, classify_op, get_neuron_op_db
from internal.dispatch_timing import measure_one_op

# conftest.py reads this directory to build the CSV.
METRICS_DIR = (
    Path(os.environ.get("NEURON_COVERAGE_METRICS_DIR", tempfile.gettempdir()))
    / "neuron_dispatch_metrics"
)


@pytest.fixture(scope="module", autouse=True)
def _dispatch_metrics_enabled():
    """Turn the C++ ``Dispatch.Total`` metric on, and reject a blocking launch.

    At ``NEURON_LAUNCH_BLOCKING=1`` the timed section would include the device wait.
    The flag latches on the first submit, so it has to be set before python starts.
    """
    import torch_neuronx.metrics as met

    if os.environ.get("NEURON_LAUNCH_BLOCKING") == "1":
        pytest.fail(
            "NEURON_LAUNCH_BLOCKING=1 puts device execution inside the Dispatch.Total "
            "span. Export NEURON_LAUNCH_BLOCKING=0 before starting python."
        )

    was_enabled = met.is_enabled()
    met.set_enabled(True)
    yield
    met.set_enabled(was_enabled)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestDispatchLatency:
    def test_op(self, op_info):
        iters = max(1, int(os.environ.get("NEURON_DISPATCH_ITERS", "5")))
        warmup = max(0, int(os.environ.get("NEURON_DISPATCH_WARMUP", "2")))

        row = measure_one_op(op_info, iters=iters, warmup=warmup)

        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        name = row["op_name"].replace("::", "_").replace(".", "_")
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        (METRICS_DIR / f"{name}_{worker}.json").write_text(json.dumps(row))

        # No latency bound: a threshold on shared CI hardware would flake, and one
        # op's limitation is data rather than a failure. Losing the row is the failure.
        assert row["op_name"] == op_info.op_name
        assert row["last_updated"], "row cannot be partitioned without a timestamp"


def pytest_generate_tests(metafunc):
    if "op_info" in metafunc.fixturenames:
        ops = [
            op
            for op in get_neuron_op_db()
            if op.op_resolved and classify_op(op.op_name) == "registered"
        ]
        metafunc.parametrize("op_info", build_params(ops))

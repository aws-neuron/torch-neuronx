"""Per-op dispatch-latency measurement, driven by ``test_dispatch_latency.py``.

For one ATen op, measures the time inside torch-neuronx's dispatch layer against
the end-to-end call time. On the C++ path that is the ``Dispatch.Total`` metric.
Every op dispatches in C++, so every row comes from that metric and
``timing_source`` reports ``cpp_metric``.

Each op lands in one bucket: ``timed``, ``no_dispatch`` (ran but never entered an
instrumented path), ``no_sample`` (no valid input), or ``errored``.
"""

from __future__ import annotations

import csv
import gc
import os
import subprocess
import time
from datetime import datetime, timezone
from functools import cache

import torch

import torch_neuronx
import torch_neuronx.metrics as met

# Also the parquet/Grafana schema. Append columns, never insert, so positions hold.
DISPATCH_FIELDS = [
    "op_name",
    "op_group",
    "status",
    "dispatch_us",
    "e2e_us",
    "dispatch_us_min",
    "dispatch_us_max",
    "dispatch_samples",
    "timing_source",
    "input_shapes",
    "input_dtype",
    "n_iters",
    "error",
    "last_updated",
    "torch_neuronx_version",
    "neuronx_cc_version",
    "aws_neuronx_runtime_lib_version",
]


@cache
def _versions():
    """Component versions for this host, or "" for any that cannot be read.

    Probed the same way as the upstream test runner
    (``tests/pytorch_tests_native/runner/job_metadata.py``): Python packages report
    their ``__version__``, and runtime-lib comes from dpkg.
    """
    try:
        runtime_lib = subprocess.run(
            ["dpkg-query", "-W", "-f=${Version}", "aws-neuronx-runtime-lib"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        runtime_lib = ""
    return {
        "torch_neuronx_version": getattr(torch_neuronx, "__version__", ""),
        # Set from neuronxcc at import; GetCompilerVersion() reads the same value.
        "neuronx_cc_version": os.environ.get("NEURONXCC_VERSION", ""),
        "aws_neuronx_runtime_lib_version": runtime_lib,
    }


def write_csv(rows, csv_path):
    """Write the CSV, stripping newlines so a multi-line error cannot break a row."""
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=DISPATCH_FIELDS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: str(row.get(k, "")).replace("\n", " ").replace("\r", "")
                    for k in DISPATCH_FIELDS
                }
            )


def _sanitize_sample(op_name, sample):
    """Clamp integer ``remainder`` divisors to 1, in place.

        remainder(x, tensor([0, 0]))  ->  remainder(x, tensor([1, 1]))
        remainder(x, 0)              ->  remainder(x, 1)

    Dividing an integer by zero hangs the compiled NEFF rather than raising, and a
    hang takes the whole sweep down with it. Only ops actually named ``remainder``
    are touched.
    """
    if op_name.split("::")[-1].split(".")[0] != "remainder":
        return
    fixed = []
    for arg in sample.args:
        if isinstance(arg, torch.Tensor) and not arg.is_floating_point():
            arg.clamp_(min=1)
        elif isinstance(arg, int) and not isinstance(arg, bool) and arg == 0:
            arg = 1
        fixed.append(arg)
    sample.args = tuple(fixed)


def _describe_sample(sample):
    """Return ``(shapes, dtype)`` for the sample measured.

        add(x[2048, 12288], y[2048, 12288])  ->  ("2048x12288,2048x12288", "float32")
        _foreach_add([15 tensors of 256x4096], 1.0)  ->  ("[15]256x4096", "float32")

    Kwargs are included because ``sample_inputs`` passes ``out=`` buffers that way.
    """
    shapes = []
    dtype = ""
    for arg in (sample.input, *sample.args, *sample.kwargs.values()):
        if isinstance(arg, torch.Tensor):
            shapes.append("x".join(str(d) for d in arg.shape) or "scalar")
            if not dtype:
                dtype = str(arg.dtype).replace("torch.", "")
        elif isinstance(arg, list | tuple) and arg and isinstance(arg[0], torch.Tensor):
            shapes.append(f"[{len(arg)}]" + "x".join(str(d) for d in arg[0].shape))
            if not dtype:
                dtype = str(arg[0].dtype).replace("torch.", "")
    return ",".join(shapes), dtype


def _move_to_neuron(value):
    """Copy tensors onto the device, recursing into containers."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to("neuron")
    if isinstance(value, torch.Size):
        return value  # holds ints, nothing to move
    if isinstance(value, list | tuple):
        moved = [_move_to_neuron(v) for v in value]
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return type(value)(*moved)  # namedtuple takes positional fields
        return type(value)(moved)
    if isinstance(value, dict):
        return {k: _move_to_neuron(v) for k, v in value.items()}
    return value


def _union_duration_ns(samples):
    """Total time the dispatcher was busy, given when each of its runs started and
    finished. Overlapping runs are counted once.

    ``TimedSection`` records the end time and duration of each run, so a run is an
    interval. There are two mechanisms for this:

    Nested, when the op's own dispatch triggers a sub-op's dispatch inside it::

        run A  |------------------|   0 to 100 us
        run B        |------|         20 to 50 us, inside A
        answer 100 us. Adding them would count B twice.

    Sequential, when the op has no dispatch of its own and its sub-ops each run
    separately::

        run A  |-----|                 0 to 30 us
        run B            |-----|      40 to 70 us
        answer 60 us. Taking the longest run alone would report only 30.

    Adding the durations up is right for the second case and too high for the first.
    Simply measures the clock time actually covered instead.
    """
    intervals = sorted((s["timestamp_ns"] - s["value"], s["timestamp_ns"]) for s in samples)
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start > end:  # a gap, so close this interval and open the next
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + (end - start)


def _read_dispatch_time_us():
    """Dispatch time of the op just called.

    Returns ``(microseconds, dispatcher runs, which clock)``, or ``(None, 0, "")`` if
    the metric recorded nothing.

    A count above 1024 means the metric's ring overwrote older timings and the time
    is an underestimate; the most any op reached in a full sweep is 6.

    Four of the six dispatch entry points write this metric, so ``copy_`` and
    ``_to_copy`` come back as ``no_dispatch``.
    """
    data = met.get_metric_data("Dispatch.Total")
    if data and data["total_samples"] > 0:
        return _union_duration_ns(data["samples"]) / 1000.0, data["total_samples"], "cpp_metric"
    return None, 0, ""


def measure_op(op, sample, iters, warmup):
    """Run ``op`` and return ``(per-iteration samples, error)``.

    Each sample is ``(e2e_us, dispatch_us|None, n_runs, source)``, read as a pair
    from the same call so the two timings always describe one dispatch.
    """
    # Outside the loop: a .to("neuron") inside it would be timed as dispatch too.
    sample.kwargs.pop("requires_grad", None)
    try:
        args = tuple(_move_to_neuron(a) for a in (sample.input, *sample.args))
        kwargs = {k: _move_to_neuron(v) for k, v in sample.kwargs.items()}
    except Exception as e:
        return None, "alloc: " + " ".join(str(e).split())

    try:
        for _ in range(warmup):  # also absorbs the cold NEFF compile
            with torch.no_grad():
                op(*args, **kwargs)
        torch_neuronx.synchronize()
    except Exception as e:
        return None, " ".join(str(e).split())

    samples = []
    for _ in range(iters):
        met.clear_all()
        torch_neuronx.synchronize()

        start_ns = time.perf_counter_ns()
        with torch.no_grad():
            op(*args, **kwargs)
        torch_neuronx.synchronize()
        end_ns = time.perf_counter_ns()

        dispatch_us, n_runs, source = _read_dispatch_time_us()
        samples.append(((end_ns - start_ns) / 1000.0, dispatch_us, n_runs, source))
    return samples, None


def _reduce(samples):
    """Extract the fastest dispatch time to minimise device noise.

    We measure each op several times. This picks the fastest end-to-end run and
    reports the corresponding dispatch time; the fastest run is the one least
    affected by the machine being busy elsewhere. Runs where the dispatcher recorded
    nothing are only considered if no run recorded anything.
    """
    with_dispatch = [s for s in samples if s[1] is not None]
    e2e, dispatch, n_runs, source = min(with_dispatch or samples, key=lambda s: s[0])
    values = [s[1] for s in samples if s[1] is not None]
    # Reachable because the metric arena is process-global: another thread's dispatch
    # can land inside this window and clear_all() cannot fence it.
    anomaly = ""
    if dispatch is not None and dispatch > e2e:
        anomaly = (
            f"dispatch_us ({dispatch:.0f}) exceeds e2e_us ({e2e:.0f}); {n_runs} dispatcher runs"
        )
    return {
        "dispatch_us": dispatch,
        "e2e_us": e2e,
        "dispatch_us_min": min(values) if values else None,
        "dispatch_us_max": max(values) if values else None,
        "dispatch_samples": n_runs or "",
        "timing_source": source,
        "n_iters": len(samples),
    }, anomaly


def measure_one_op(op_info, iters=5, warmup=2, last_updated=None):
    """Measure one op and return its finished CSV row.

    Never raises for an op-specific problem. A missing sample, a compiler rejection
    or an unexpected exception each come back as a row with the matching status,
    because for this benchmark those are data rather than test failures.
    """
    if last_updated is None:
        last_updated = os.environ.get("NEURON_DISPATCH_TIMESTAMP") or datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")
    row = dict.fromkeys(DISPATCH_FIELDS, "")
    row.update(
        op_name=op_info.op_name,
        op_group=op_info.op_group,
        status="errored",
        n_iters=0,
        last_updated=last_updated,
        **_versions(),
    )
    try:
        try:
            samples_iter = op_info.sample_inputs("cpu", torch.float32, requires_grad=False)
            sample = next(iter(samples_iter), None)
        except Exception as e:
            row.update(status="no_sample", error="sample_inputs: " + " ".join(str(e).split()))
            return row
        if sample is None:
            row.update(status="no_sample", error="sample_inputs yielded no samples")
            return row

        _sanitize_sample(op_info.op_name, sample)
        row["input_shapes"], row["input_dtype"] = _describe_sample(sample)

        samples, error = measure_op(op_info._resolved_op, sample, iters, warmup)
        if error is not None:
            row["error"] = error
            return row
        if not samples:  # iters <= 0
            row["error"] = f"no timed iterations (iters={iters})"
            return row

        timings, anomaly = _reduce(samples)
        for field in ("dispatch_us", "e2e_us", "dispatch_us_min", "dispatch_us_max"):
            value = timings[field]
            # "" rather than 0, so a missing timing cannot be read as a real zero.
            row[field] = "" if value is None else f"{value:.0f}"
        row.update(
            dispatch_samples=timings["dispatch_samples"],
            timing_source=timings["timing_source"],
            n_iters=timings["n_iters"],
            error=anomaly,
            status="timed" if timings["dispatch_us"] is not None else "no_dispatch",
        )
        return row
    except Exception as e:  # one op blowing up is a row, not a lost sweep
        row["error"] = " ".join(str(e).split())
        return row
    finally:
        # Keep one op's buffers resident at a time; the whole sweep will not fit.
        gc.collect()
        torch_neuronx.synchronize()
        torch_neuronx.empty_cache()

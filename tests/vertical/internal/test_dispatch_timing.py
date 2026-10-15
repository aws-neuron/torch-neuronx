"""Device-free tests for ``dispatch_timing``: the reductions, the sample fixups,
and the CSV contract the parquet consumer depends on. The on-device measurement is
exercised by ``tests/vertical/test_dispatch_latency.py``.
"""

import csv
from types import SimpleNamespace

import pytest
import torch

from . import dispatch_timing as dt


def _iter(e2e, dispatch, n_runs=1, source="cpp_metric"):
    """One per-iteration sample as ``measure_op`` collects them."""
    return (e2e, dispatch, n_runs, source)


def _metric(*spans, total_samples=None):
    """Metric data for spans given as ``(start_ns, end_ns)``. ``TimedSection``
    records the end timestamp and the duration, so that is the shape the binding
    returns. ``total_samples`` overrides the count to model a full sample ring."""
    return {
        "total_samples": len(spans) if total_samples is None else total_samples,
        "accumulator": sum(end - start for start, end in spans),
        "samples": [{"value": end - start, "timestamp_ns": end} for start, end in spans],
    }


class TestReduce:
    def test_reports_one_iteration_not_two_reduced_series(self):
        # Reducing separately would pair the lowest e2e (10) with the lowest
        # dispatch (1), which came from a different call.
        timings, _ = dt._reduce([_iter(30, 9), _iter(10, 3), _iter(20, 1)])
        assert (timings["e2e_us"], timings["dispatch_us"]) == (10, 3)

    def test_reports_spread_across_all_iterations(self):
        timings, _ = dt._reduce([_iter(100, 10), _iter(50, 4), _iter(70, 7)])
        assert timings["dispatch_us_min"] == 4 and timings["dispatch_us_max"] == 10
        assert timings["n_iters"] == 3 and timings["timing_source"] == "cpp_metric"

    def test_prefers_iterations_that_timed_dispatch(self):
        # The lowest-e2e iteration recorded nothing, so it must not mask an op the
        # other iterations did time.
        timings, _ = dt._reduce([_iter(10, None), _iter(30, 3), _iter(40, 4)])
        assert (timings["e2e_us"], timings["dispatch_us"]) == (30, 3)

    def test_falls_back_when_no_iteration_timed_dispatch(self):
        timings, _ = dt._reduce([_iter(30, None), _iter(10, None)])
        assert timings["dispatch_us"] is None and timings["e2e_us"] == 10
        assert timings["dispatch_us_min"] is None and timings["dispatch_us_max"] is None

    def test_flags_dispatch_above_e2e(self):
        _, anomaly = dt._reduce([_iter(10, 40, n_runs=3)])
        assert "exceeds e2e_us" in anomaly

    def test_no_anomaly_on_a_normal_row(self):
        _, anomaly = dt._reduce([_iter(300, 20)])
        assert anomaly == ""


class TestUnionDuration:
    def test_nested_spans_count_the_outer_span_once(self):
        assert dt._union_duration_ns(_metric((0, 100_000), (20_000, 50_000))["samples"]) == 100_000

    def test_sibling_spans_add_up(self):
        assert dt._union_duration_ns(_metric((0, 30_000), (40_000, 70_000))["samples"]) == 60_000

    def test_partial_overlap_counts_shared_time_once(self):
        assert dt._union_duration_ns(_metric((0, 50_000), (30_000, 80_000))["samples"]) == 80_000

    def test_touching_spans_merge(self):
        assert dt._union_duration_ns(_metric((0, 20_000), (20_000, 35_000))["samples"]) == 35_000

    def test_single_span_is_its_own_duration(self):
        assert dt._union_duration_ns(_metric((10_000, 52_000))["samples"]) == 42_000


class TestReadDispatchTimeUs:
    def test_prefers_cpp_metric(self, monkeypatch):
        monkeypatch.setattr(dt.met, "get_metric_data", lambda name: _metric((0, 42_000)))
        assert dt._read_dispatch_time_us() == (42.0, 1, "cpp_metric")

    def test_reports_span_count_so_reentrancy_is_visible(self, monkeypatch):
        monkeypatch.setattr(
            dt.met, "get_metric_data", lambda name: _metric((0, 100_000), (20_000, 50_000))
        )
        assert dt._read_dispatch_time_us() == (100.0, 2, "cpp_metric")

    def test_span_count_is_uncapped(self, monkeypatch):
        # The metric keeps a bounded ring but counts every span. Reporting the ring
        # size would hide that the reduction saw an incomplete set.
        monkeypatch.setattr(
            dt.met, "get_metric_data", lambda name: _metric((0, 5_000), total_samples=2000)
        )
        assert dt._read_dispatch_time_us()[1] == 2000

    def test_none_when_the_metric_recorded_nothing(self, monkeypatch):
        monkeypatch.setattr(dt.met, "get_metric_data", lambda name: {"total_samples": 0})
        assert dt._read_dispatch_time_us() == (None, 0, "")


class TestSanitizeSample:
    """An integer ``remainder`` divisor of 0 hangs the NEFF, taking the sweep with
    it, so it is clamped."""

    @staticmethod
    def _sample():
        return SimpleNamespace(args=(torch.zeros(3, dtype=torch.int32), 0))

    def test_clamps_tensor_and_scalar_divisors(self):
        sample = self._sample()
        dt._sanitize_sample("aten::remainder.Scalar", sample)
        assert sample.args[0].tolist() == [1, 1, 1] and sample.args[1] == 1

    def test_leaves_unrelated_op_untouched(self):
        # Matched on the exact base name, not a substring.
        sample = self._sample()
        dt._sanitize_sample("aten::my_remainder_helper", sample)
        assert sample.args[0].tolist() == [0, 0, 0] and sample.args[1] == 0

    def test_leaves_float_divisors_alone(self):
        # Float division by zero gives inf on the device rather than hanging.
        sample = SimpleNamespace(args=(torch.zeros(2), 0.0))
        dt._sanitize_sample("aten::remainder.Tensor", sample)
        assert sample.args[0].tolist() == [0.0, 0.0]


class TestDescribeSample:
    def test_records_shapes_and_first_dtype(self):
        sample = SimpleNamespace(
            input=torch.zeros(2048, 12288), args=(torch.zeros(2048, 12288),), kwargs={}
        )
        assert dt._describe_sample(sample) == ("2048x12288,2048x12288", "float32")

    def test_summarizes_a_tensor_list_by_length_and_element_shape(self):
        sample = SimpleNamespace(input=[torch.zeros(256, 4096)] * 15, args=(1.0,), kwargs={})
        assert dt._describe_sample(sample) == ("[15]256x4096", "float32")

    def test_includes_kwarg_tensors(self):
        # sample_inputs passes out= buffers as kwargs.
        sample = SimpleNamespace(
            input=torch.zeros(4, 4), args=(), kwargs={"out": torch.zeros(4, 4)}
        )
        assert dt._describe_sample(sample) == ("4x4,4x4", "float32")


class TestWriteCsv:
    """``write_csv`` is the parquet consumer's contract: exact column set, exact
    order, and one line per row."""

    def test_header_order_matches_the_schema(self, tmp_path):
        out = tmp_path / "d.csv"
        dt.write_csv([{"op_name": "aten::add.Tensor", "dispatch_us": "23"}], str(out))
        with open(out, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert list(rows[0]) == dt.DISPATCH_FIELDS
        assert rows[0]["op_name"] == "aten::add.Tensor" and rows[0]["dispatch_us"] == "23"

    def test_missing_timings_stay_blank_not_zero(self, tmp_path):
        # A no_dispatch or errored op has no dispatch time; 0 would read as a real
        # measurement downstream.
        out = tmp_path / "d.csv"
        dt.write_csv([{"op_name": "aten::x", "status": "errored", "error": "boom"}], str(out))
        with open(out, newline="") as fh:
            row = next(csv.DictReader(fh))
        assert row["dispatch_us"] == "" and row["e2e_us"] == ""
        assert row["error"] == "boom"

    def test_newlines_are_stripped(self, tmp_path):
        # A multi-line error must not break the row structure.
        out = tmp_path / "d.csv"
        dt.write_csv([{"op_name": "aten::x", "error": "line one\nline two\r"}], str(out))
        with open(out, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert "\n" not in rows[0]["error"] and "\r" not in rows[0]["error"]


class TestVersions:
    def test_reports_the_columns_the_schema_declares(self):
        # A compiler or runtime bump moves dispatch_us for reasons outside this
        # repo, so every row carries what produced it.
        versions = dt._versions()
        assert set(versions) == {
            "torch_neuronx_version",
            "neuronx_cc_version",
            "aws_neuronx_runtime_lib_version",
        }
        assert set(versions) <= set(dt.DISPATCH_FIELDS)
        assert all(isinstance(v, str) for v in versions.values())


class TestBucketing:
    """The status cascade, which decides what a row means. Driven with a fake
    op_info so all four buckets are reachable without a device."""

    @pytest.fixture(autouse=True)
    def _no_device(self, monkeypatch):
        """measure_one_op always frees buffers in a finally; stub that out."""
        monkeypatch.setattr(dt.torch_neuronx, "synchronize", lambda: None)
        monkeypatch.setattr(dt.torch_neuronx, "empty_cache", lambda: None)

    @staticmethod
    def _op_info(samples=(), resolved_op=None, sample_error=None):
        def sample_inputs(*args, **kwargs):
            if sample_error:
                raise RuntimeError(sample_error)
            return iter(samples)

        return SimpleNamespace(
            op_name="aten::fake",
            op_group="elementwise",
            sample_inputs=sample_inputs,
            _resolved_op=resolved_op,
        )

    @staticmethod
    def _sample():
        return SimpleNamespace(input=torch.zeros(4, 4), args=(), kwargs={})

    def test_no_sample_when_the_generator_yields_nothing(self, monkeypatch):
        monkeypatch.setattr(dt, "measure_op", lambda *a: (None, "unused"))
        row = dt.measure_one_op(self._op_info(samples=()), last_updated="T")
        assert row["status"] == "no_sample"
        assert "yielded no samples" in row["error"]
        assert row["dispatch_us"] == "" and row["e2e_us"] == ""

    def test_no_sample_carries_the_generator_error(self, monkeypatch):
        monkeypatch.setattr(dt, "measure_op", lambda *a: (None, "unused"))
        row = dt.measure_one_op(self._op_info(sample_error="bad shape"), last_updated="T")
        assert row["status"] == "no_sample" and "bad shape" in row["error"]

    def test_errored_when_the_op_fails_to_run(self, monkeypatch):
        monkeypatch.setattr(dt, "measure_op", lambda *a: (None, "offs must be divisible by 128"))
        row = dt.measure_one_op(self._op_info(samples=[self._sample()]), last_updated="T")
        assert row["status"] == "errored"
        assert row["error"] == "offs must be divisible by 128"
        assert row["n_iters"] == 0

    def test_no_dispatch_when_the_op_ran_but_recorded_no_span(self, monkeypatch):
        # What a CPU fallback looks like: real e2e, no dispatch span.
        monkeypatch.setattr(dt, "measure_op", lambda *a: ([_iter(8, None, 0, "")], None))
        row = dt.measure_one_op(self._op_info(samples=[self._sample()]), last_updated="T")
        assert row["status"] == "no_dispatch"
        assert row["dispatch_us"] == "" and row["e2e_us"] == "8"

    def test_timed_when_a_span_was_recorded(self, monkeypatch):
        monkeypatch.setattr(dt, "measure_op", lambda *a: ([_iter(317.6, 23.4)], None))
        row = dt.measure_one_op(self._op_info(samples=[self._sample()]), last_updated="T")
        assert row["status"] == "timed"
        assert row["dispatch_us"] == "23" and row["e2e_us"] == "318"  # whole us
        assert row["timing_source"] == "cpp_metric" and row["n_iters"] == 1

    def test_zero_iters_does_not_silently_report_a_timed_row(self, monkeypatch):
        # An empty sweep must not look like a successful one.
        monkeypatch.setattr(dt, "measure_op", lambda *a: ([], None))
        row = dt.measure_one_op(self._op_info(samples=[self._sample()]), iters=0, last_updated="T")
        assert row["status"] == "errored" and "no timed iterations" in row["error"]

    def test_every_row_is_writable_whatever_the_bucket(self, monkeypatch, tmp_path):
        # The CSV writer must accept any bucket; a missing key would raise here.
        monkeypatch.setattr(dt, "measure_op", lambda *a: ([_iter(8, None, 0, "")], None))
        row = dt.measure_one_op(self._op_info(samples=[self._sample()]), last_updated="T")
        out = tmp_path / "d.csv"
        dt.write_csv([row], str(out))
        with open(out, newline="") as fh:
            assert list(next(csv.DictReader(fh))) == dt.DISPATCH_FIELDS

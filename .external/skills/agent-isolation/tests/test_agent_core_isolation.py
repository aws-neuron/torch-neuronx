"""Hardware-free tests for the agent_core_isolation pytest plugin.

These tests prove the plugin is a NO-OP everywhere it must not act, so it
cannot break another skill's run (the native tree, distributed, a non-sliced
window, a worker index past the slice), and that it pins one distinct core per
worker only in the sliced single-core lane. No Neuron device is needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import agent_core_isolation as m

# --- _slice_bounds --------------------------------------------------------


def test_slice_bounds_range(monkeypatch):
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "8-15")
    assert m._slice_bounds() == (8, 8)


def test_slice_bounds_single_core_is_none(monkeypatch):
    # A single index (no dash) is not a range; the plugin must not act.
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "4")
    assert m._slice_bounds() is None


def test_slice_bounds_comma_list_is_none(monkeypatch):
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "0,1,2,3")
    assert m._slice_bounds() is None


def test_slice_bounds_unset_is_none(monkeypatch):
    monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
    assert m._slice_bounds() is None


def test_slice_bounds_reversed_is_none(monkeypatch):
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "15-8")
    assert m._slice_bounds() is None


# --- pytest_configure -----------------------------------------------------


class _Config:
    """Minimal stand-in for a pytest Config. A worker carries workerinput."""

    def __init__(self, workerinput=None):
        if workerinput is not None:
            self.workerinput = workerinput


def test_configure_no_worker_is_noop(monkeypatch):
    # The xdist controller / -n 0 carry no workerinput.
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "8-15")
    monkeypatch.setattr(m.sys, "argv", ["pytest"])
    m.pytest_configure(_Config(workerinput=None))
    assert m.os.environ["NEURON_RT_VISIBLE_CORES"] == "8-15"


def test_configure_native_tree_is_noop(monkeypatch):
    # The pytorch_tests_native tree pins cores in its own conftest.
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "8-15")
    monkeypatch.setattr(m.sys, "argv", ["pytest", "tests/pytorch_tests_native/test"])
    m.pytest_configure(_Config(workerinput={"workerid": "gw3"}))
    assert m.os.environ["NEURON_RT_VISIBLE_CORES"] == "8-15"


def test_configure_distributed_is_noop(monkeypatch):
    # WORLD_SIZE > 1 needs a block of cores per worker, not one.
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "8-15")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(m.sys, "argv", ["pytest"])
    m.pytest_configure(_Config(workerinput={"workerid": "gw3"}))
    assert m.os.environ["NEURON_RT_VISIBLE_CORES"] == "8-15"


def test_configure_non_range_is_noop(monkeypatch):
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "0,1,2,3")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(m.sys, "argv", ["pytest"])
    m.pytest_configure(_Config(workerinput={"workerid": "gw1"}))
    assert m.os.environ["NEURON_RT_VISIBLE_CORES"] == "0,1,2,3"


def test_configure_worker_past_slice_is_noop(monkeypatch):
    # gw8 in a 4-core slice would collide; keep the full window instead of
    # pinning to a core outside the slice.
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "0-3")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(m.sys, "argv", ["pytest"])
    m.pytest_configure(_Config(workerinput={"workerid": "gw8"}))
    assert m.os.environ["NEURON_RT_VISIBLE_CORES"] == "0-3"


def test_configure_pins_one_core(monkeypatch):
    # gw3 of slice 8-15 gets logical core 8 + 3 = 11.
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "8-15")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(m.sys, "argv", ["pytest"])
    m.pytest_configure(_Config(workerinput={"workerid": "gw3"}))
    assert m.os.environ["NEURON_RT_VISIBLE_CORES"] == "11-11"


def test_configure_pins_first_worker(monkeypatch):
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "8-15")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(m.sys, "argv", ["pytest"])
    m.pytest_configure(_Config(workerinput={"workerid": "gw0"}))
    assert m.os.environ["NEURON_RT_VISIBLE_CORES"] == "8-8"

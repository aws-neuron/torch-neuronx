"""Per-xdist-worker NeuronCore pinning for the bare-pytest lanes.

Activated inside ``agent-run`` (it sets ``NEURON_RT_VISIBLE_CORES`` to the
agent's slice and loads this plugin). Under ``pytest -n``, xdist spawns N
workers that all inherit the SAME ``NEURON_RT_VISIBLE_CORES`` window. Without
this plugin every worker asks the runtime for the WHOLE slice, so the workers
fight over the same cores and all but the first fail with
``Failed to initialize Neuron Runtime`` / ``Logical Neuron Core(s) not
available``. This plugin gives each worker ONE distinct core in the slice.

Scope: the single-core lanes only (``tests/python_ops``, e2e, vertical). It is
a no-op when:
  * there is no xdist worker (controller or ``-n 0``);
  * ``NEURON_RT_VISIBLE_CORES`` is not a simple ``start-end`` range;
  * the run is the ``pytorch_tests_native`` tree (its own conftest pins there);
  * the run is distributed (``WORLD_SIZE`` > 1 needs many cores per worker).

Run ``-n`` at most the slice size. A worker index beyond the slice keeps the
full window (it would collide), so cap the worker count to the number of cores.

``NEURON_RT_VISIBLE_CORES`` indexes LOGICAL cores. On an LNC=2 host a slice
``8-15`` is 8 logical cores, so ``-n 8`` gets one logical core each.
"""

import os
import sys

import pytest


def _slice_bounds():
    """Return (base, size) from NEURON_RT_VISIBLE_CORES, or None if not a range."""
    val = os.environ.get("NEURON_RT_VISIBLE_CORES", "")
    if "-" not in val:
        return None
    start, _, end = val.partition("-")
    try:
        base, last = int(start), int(end)
    except ValueError:
        return None
    if last < base:
        return None
    return base, last - base + 1


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    # Let the native tree's own conftest handle its pinning.
    if any("pytorch_tests_native" in a for a in sys.argv):
        return
    # Distributed tests need a block of cores per worker, not one.
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return
    # Only real xdist workers carry workerinput; controller / -n 0 do not.
    workerinput = getattr(config, "workerinput", None)
    if not workerinput:
        return
    bounds = _slice_bounds()
    if bounds is None:
        return
    base, size = bounds
    # workerid is "gw<N>", local to this run (gw0, gw1, ...).
    idx = int(workerinput["workerid"][2:])
    if idx >= size:
        return
    core = base + idx
    os.environ["NEURON_RT_VISIBLE_CORES"] = f"{core}-{core}"

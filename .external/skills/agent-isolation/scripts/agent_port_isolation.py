"""Per-agent port-range isolation pytest plugin.

Activated by setting ``AGENT_PORT_OFFSET`` in the agent's environment (the
``agent-run`` wrapper does this). Without that env var the plugin is a no-op,
so it's safe to leave in PYTEST_PLUGINS unconditionally.

Two surfaces are handled:

1. The repo's ``tests/distributed/utils.py`` uses a module constant
   ``_BASE_MASTER_PORT = 29500``. We bump it by the agent's offset so each
   agent's ``DistributedTester`` scans a disjoint port window.

2. PyTorch upstream multiprocess tests typically read ``MASTER_PORT`` from
   env via the env-style rendezvous. We set it (offset from 29500) when the
   caller hasn't already set it.

Tests that hardcode a port literal in their source (e.g.
``init_method="tcp://localhost:29501"``) bypass both surfaces  -  those need
per-test fixes, not per-agent isolation.
"""

import importlib
import os


def pytest_configure(config):
    offset = int(os.environ.get("AGENT_PORT_OFFSET", "0"))
    if not offset:
        return

    # Surface 1: repo's DistributedTester
    try:
        u = importlib.import_module("tests.distributed.utils")
        u._BASE_MASTER_PORT = u._BASE_MASTER_PORT + offset
    except ImportError:
        # Repo's DistributedTester isn't on the path (e.g. running only
        # upstream tests); Surface 2 below still applies the offset.
        pass

    # Surface 2: upstream tests that read MASTER_PORT
    os.environ.setdefault("MASTER_PORT", str(29500 + offset))

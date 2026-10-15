"""Hardware-free tests for the agent_port_isolation pytest plugin."""

import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import agent_port_isolation as m


def test_no_offset_is_a_no_op(monkeypatch):
    monkeypatch.delenv("AGENT_PORT_OFFSET", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)
    assert m.pytest_configure(None) is None
    assert "MASTER_PORT" not in os.environ


def test_offset_sets_master_port_and_bumps_base(monkeypatch):
    utils = types.ModuleType("tests.distributed.utils")
    utils._BASE_MASTER_PORT = 29500
    monkeypatch.setitem(sys.modules, "tests.distributed.utils", utils)
    monkeypatch.setenv("AGENT_PORT_OFFSET", "1128")
    monkeypatch.delenv("MASTER_PORT", raising=False)
    assert m.pytest_configure(None) is None
    assert utils._BASE_MASTER_PORT == 29500 + 1128
    assert os.environ["MASTER_PORT"] == str(29500 + 1128)


def test_existing_master_port_is_kept(monkeypatch):
    utils = types.ModuleType("tests.distributed.utils")
    utils._BASE_MASTER_PORT = 29500
    monkeypatch.setitem(sys.modules, "tests.distributed.utils", utils)
    monkeypatch.setenv("AGENT_PORT_OFFSET", "1000")
    monkeypatch.setenv("MASTER_PORT", "12345")
    assert m.pytest_configure(None) is None
    assert os.environ["MASTER_PORT"] == "12345"

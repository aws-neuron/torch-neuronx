import os

import pytest

from tests.utils.bootstrap_ports import configure_worker_bootstrap_port


def test_worker_port_uses_neuron_rt_port_base(monkeypatch):
    monkeypatch.setenv("NEURON_RT_PORT", "23000")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")

    configure_worker_bootstrap_port(2)

    assert os.environ["NEURON_RT_ROOT_COMM_ID"] == "127.0.0.1:23002"


def test_worker_port_defaults_addr_to_localhost(monkeypatch):
    monkeypatch.setenv("NEURON_RT_PORT", "20000")
    monkeypatch.delenv("MASTER_ADDR", raising=False)

    configure_worker_bootstrap_port(0)

    assert os.environ["NEURON_RT_ROOT_COMM_ID"] == "localhost:20000"


def test_workers_get_disjoint_ports(monkeypatch):
    monkeypatch.setenv("NEURON_RT_PORT", "20000")
    monkeypatch.setenv("MASTER_ADDR", "localhost")

    configure_worker_bootstrap_port(0)
    first = os.environ["NEURON_RT_ROOT_COMM_ID"]
    configure_worker_bootstrap_port(8)
    replacement = os.environ["NEURON_RT_ROOT_COMM_ID"]

    assert first != replacement
    assert first == "localhost:20000"
    assert replacement == "localhost:20008"


def test_worker_port_rejects_out_of_range(monkeypatch):
    monkeypatch.setenv("NEURON_RT_PORT", "65535")

    with pytest.raises(RuntimeError, match="No valid CCOM bootstrap port"):
        configure_worker_bootstrap_port(2)

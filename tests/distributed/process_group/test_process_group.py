"""Unit tests for runtime-setup helpers in torch_neuronx.distributed.backend.

Covers the helpers still used by the C++ NeuronBackend factory
(`_create_neuron_process_group`):
  - `_neuron_runtime_setup`
  - `_set_root_comm_id`
  - `_set_rt_visible_cores`

The Python `ProcessGroupNeuron` class and its tests were removed when the
backend was migrated to C++ (`NeuronBackend`). End-to-end coverage of the
new backend lives in `test_process_group_neuron.py` (requires WORLD_SIZE>=2).
"""

import os
from unittest.mock import MagicMock, patch

import pytest

import torch_neuronx
from torch_neuronx.distributed.backend import (
    _neuron_runtime_setup,
    _set_root_comm_id,
    _set_rt_visible_cores,
)


@patch("torch_neuronx.is_neuron_runtime_initialized", return_value=False)
@patch("torch_neuronx._C._nrt_barrier")
def test_set_root_comm_id(mock_barrier, mock_is_neuron_runtime_initialized):
    store = MagicMock()

    _set_root_comm_id(0, 0, 2, store)
    assert store.set.call_count == 1
    assert store.set.call_args[0][0] == "NEURON_RT_PORT"
    actual_port = store.set.call_args[0][1]

    store.get.return_value = actual_port.encode()

    _set_root_comm_id(1, 1, 2, store)
    assert os.environ.get("NEURON_RT_ROOT_COMM_ID") == f"localhost:{actual_port}"


@patch("socket.gethostbyname", return_value="192.168.1.100")
@patch("socket.gethostname", return_value="worker1")
def test_set_rt_visible_cores_tcp_method(mock_hostname, mock_gethostbyname):
    store = MagicMock()

    def mock_store_get(key):
        worker_map = {
            "worker_0": b"192.168.1.100:0",
            "worker_1": b"192.168.1.100:1",
            "worker_2": b"192.168.1.101:2",
            "worker_3": b"192.168.1.101:3",
        }
        if key in worker_map:
            return worker_map[key]
        raise Exception("Key not found")

    store.get.side_effect = mock_store_get

    local_rank, local_world_size = _set_rt_visible_cores(1, 4, store)

    assert local_rank == 1
    assert local_world_size == 2
    assert os.environ.get("NEURON_RT_VISIBLE_CORES") == "1"


def test_set_rt_visible_cores_with_env_vars():
    os.environ["LOCAL_RANK"] = "2"
    os.environ["LOCAL_WORLD_SIZE"] = "4"

    store = MagicMock()
    local_rank, local_world_size = _set_rt_visible_cores(0, 8, store)

    assert local_rank == 2
    assert local_world_size == 4
    assert os.environ.get("NEURON_RT_VISIBLE_CORES") == "2"

    del os.environ["LOCAL_RANK"]
    del os.environ["LOCAL_WORLD_SIZE"]


@patch("torch_neuronx.is_neuron_runtime_initialized", return_value=False)
@patch("torch_neuronx._lazy_init")
@patch("socket.gethostbyname", return_value="127.0.0.1")
@patch("socket.gethostname", return_value="localhost")
def test_neuron_runtime_setup_barrier_call(
    mock_hostname, mock_gethostbyname, mock_lazy_init, mock_is_neuron_runtime_initialized
):
    """Test that _neuron_runtime_setup calls barrier with correct parameters"""
    store = MagicMock()

    def mock_store_get(key):
        worker_map = {
            "worker_0": b"127.0.0.1:0",
            "worker_1": b"127.0.0.1:1",
            "worker_2": b"127.0.0.1:2",
            "worker_3": b"127.0.0.1:3",
            "NEURON_RT_PORT": b"12345",
        }
        return worker_map.get(key, b"")

    store.get.side_effect = mock_store_get
    store.set = MagicMock()

    with (
        patch.object(torch_neuronx._C, "_nrt_barrier") as mock_barrier,
        patch.object(torch_neuronx._C, "_set_world_size") as mock_set_world_size,
        patch.object(torch_neuronx._C, "_set_rank") as mock_set_rank,
        patch.object(torch_neuronx._C, "_set_local_world_size") as mock_set_local_world_size,
        patch.object(torch_neuronx._C, "_set_local_device_start_index") as mock_set_start,
        patch.object(torch_neuronx._C, "_vnc_count", return_value=1),
    ):
        _neuron_runtime_setup(rank=1, size=4, store=store)
        mock_set_world_size.assert_called_once_with(4)
        mock_set_rank.assert_called_once_with(1)
        mock_barrier.assert_called_once_with(0, 1, 4)
        mock_set_local_world_size.assert_called_once()
        mock_set_start.assert_called_once()


@patch("torch_neuronx.is_neuron_runtime_initialized", return_value=False)
@patch("torch_neuronx._lazy_init")
def test_neuron_runtime_setup_with_env_vars(mock_lazy_init, mock_is_neuron_runtime_initialized):
    """Test _neuron_runtime_setup with LOCAL_RANK/LOCAL_WORLD_SIZE set"""
    os.environ["LOCAL_RANK"] = "2"
    os.environ["LOCAL_WORLD_SIZE"] = "3"

    store = MagicMock()

    with (
        patch.object(torch_neuronx._C, "_nrt_barrier") as mock_barrier,
        patch.object(torch_neuronx._C, "_set_world_size") as mock_set_world_size,
        patch.object(torch_neuronx._C, "_set_rank") as mock_set_rank,
        patch.object(torch_neuronx._C, "_set_local_world_size") as mock_set_local_world_size,
        patch.object(torch_neuronx._C, "_set_local_device_start_index") as mock_set_start,
        patch.object(torch_neuronx._C, "_vnc_count", return_value=1),
    ):
        _neuron_runtime_setup(rank=5, size=6, store=store)

        mock_set_world_size.assert_called_once_with(6)
        mock_set_rank.assert_called_once_with(5)
        mock_set_local_world_size.assert_called_once_with(3)
        mock_set_start.assert_called_once_with(2)
        mock_barrier.assert_called_once_with(0, 5, 6)

    del os.environ["LOCAL_RANK"]
    del os.environ["LOCAL_WORLD_SIZE"]


@pytest.fixture(autouse=True)
def reset_global_state():
    import torch_neuronx.distributed.backend as backend_module

    backend_module._WORLD_SIZE = None

    env_vars_to_clean = [
        "NEURON_RT_ROOT_COMM_ID",
        "NEURON_RT_VISIBLE_CORES",
        "NEURON_RT_PORT",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
    ]
    original_env = {}
    for var in env_vars_to_clean:
        original_env[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    yield

    backend_module._WORLD_SIZE = None
    for var in env_vars_to_clean:
        if var in os.environ:
            del os.environ[var]
        if original_env[var] is not None:
            os.environ[var] = original_env[var]

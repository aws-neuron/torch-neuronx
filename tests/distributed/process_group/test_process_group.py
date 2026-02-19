import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises
from torch_neuronx.distributed.backend import (
    ProcessGroupNeuron,
    _neuron_runtime_setup,
    _set_root_comm_id,
    _set_rt_visible_cores,
)


@contextmanager
def distributed_context(world_size, rank):
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    yield
    del os.environ["WORLD_SIZE"]
    del os.environ["RANK"]


@pytest.fixture
def mock_store():
    store = MagicMock()

    def mock_store_get(key):
        worker_map = {
            "worker_0": b"127.0.0.1:0",
            "worker_1": b"127.0.0.1:1",
            "NEURON_RT_PORT": b"12345",
        }
        return worker_map.get(key, b"")

    store.get.side_effect = mock_store_get
    store.set = MagicMock()
    return store


@pytest.fixture(autouse=True)
def mock_torch_neuronx():
    with patch("torch_neuronx._C") as mock_c:
        mock_c._vnc_count.return_value = 1
        mock_c._set_local_world_size = MagicMock()
        mock_c._set_local_device_start_index = MagicMock()
        mock_c._set_world_size = MagicMock()
        mock_c._set_rank = MagicMock()
        mock_c._nrt_barrier = MagicMock()
        yield mock_c


@pytest.fixture
def process_group(mock_store):
    with (
        distributed_context(2, 0),
        patch("torch_neuronx._lazy_init"),
        patch("socket.gethostbyname", return_value="127.0.0.1"),
        patch("socket.gethostname", return_value="localhost"),
        patch("torch_neuronx.is_neuron_runtime_initialized", return_value=False),
    ):
        return ProcessGroupNeuron(mock_store, rank=0, size=2, timeout=None)


@patch("torch_neuronx.is_neuron_runtime_initialized", return_value=False)
@patch("torch_neuronx._lazy_init")
@patch("socket.gethostbyname", return_value="127.0.0.1")
@patch("socket.gethostname", return_value="localhost")
def test_init_default_group(
    mock_hostname,
    mock_gethostbyname,
    mock_lazy_init,
    mock_is_neuron_runtime_initialized,
    mock_store,
):
    with distributed_context(2, 0):
        process_group = ProcessGroupNeuron(mock_store, rank=0, size=2, timeout=None)

        assert process_group.rank() == 0
        assert process_group.size() == 2
        torch_neuronx._C._set_world_size.assert_called_once_with(2)
        torch_neuronx._C._set_rank.assert_called_once_with(0)
        torch_neuronx._C._set_local_world_size.assert_called_once()
        torch_neuronx._C._set_local_device_start_index.assert_called_once()
        torch_neuronx._C._nrt_barrier.assert_called_with(0, 0, 2)


@patch("torch_neuronx.is_neuron_runtime_initialized", return_value=False)
@patch("torch_neuronx._lazy_init")
def test_init_with_local_rank_env(mock_lazy_init, mock_is_neuron_runtime_initialized, mock_store):
    with distributed_context(4, 0):
        os.environ["LOCAL_RANK"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "2"

        process_group = ProcessGroupNeuron(mock_store, rank=0, size=4, timeout=None)

        assert process_group.rank() == 0
        assert process_group.size() == 4
        assert os.environ.get("NEURON_RT_VISIBLE_CORES") == "1"

        del os.environ["LOCAL_RANK"]
        del os.environ["LOCAL_WORLD_SIZE"]


@patch("torch_neuronx.is_neuron_runtime_initialized", return_value=True)
@assert_raises(
    AssertionError, match="Neuron runtime should not be initialized before the init_process_group."
)
def test_init_with_runtime_initialized(mock_is_neuron_runtime_initialized, mock_store):
    torch_neuronx.distributed.backend._WORLD_SIZE = None
    with distributed_context(4, 0):
        _ = ProcessGroupNeuron(mock_store, rank=0, size=4, timeout=None)


def test_get_backend_name(process_group):
    assert process_group.getBackendName() == "neuron"


def test_group_name(process_group):
    test_name = "test_group"
    process_group._set_group_name(test_name)
    assert process_group.group_name == test_name


def test_barrier(process_group):
    mock_stream = MagicMock()
    mock_stream.wait_stream = MagicMock()
    mock_work = MagicMock(spec=torch._C._distributed_c10d.Work)
    with (
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch_neuronx.current_device", return_value=0),
        patch("torch_neuronx.current_stream", return_value=mock_stream),
        patch("torch_neuronx.stream"),
        patch("torch_neuronx._C._nrt_barrier"),
        patch.object(process_group, "_get_neuron_stream", return_value=mock_stream),
        patch("torch_neuronx.distributed.backend._ret_work", return_value=mock_work),
    ):
        process_group.barrier(None)


@assert_raises(NotImplementedError)
def test_barrier_error(process_group):
    with patch("torch.distributed.get_world_size", return_value=1):
        process_group.barrier(None)


def test_allreduce(process_group):
    mock_work = MagicMock(spec=torch._C._distributed_c10d.Work)

    def mock_ret_work(pg, tensors, op_type, opts, collective_fn=None, device=None):
        # Execute the collective_fn to verify all_reduce_op is called
        if collective_fn is not None:
            collective_fn()
        return mock_work

    with (
        patch.object(torch_neuronx.distributed.backend, "all_reduce_op") as mock_all_reduce_op,
        patch("torch.distributed.get_process_group_ranks", return_value=[0, 1]),
        patch.object(torch_neuronx.distributed.backend, "_ret_work", side_effect=mock_ret_work),
    ):
        tensors = [torch.tensor([1.0, 2.0])]
        options = MagicMock()
        options.reduceOp = dist.ReduceOp.SUM

        result = process_group.allreduce(tensors, options)

        mock_all_reduce_op.assert_called_once()
        assert result == mock_work


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
        patch("torch_neuronx._C._nrt_barrier") as mock_barrier,
        patch("torch_neuronx._C._set_world_size") as mock_set_world_size,
        patch("torch_neuronx._C._set_rank") as mock_set_rank,
        patch("torch_neuronx._C._set_local_world_size") as mock_set_local_world_size,
        patch("torch_neuronx._C._set_local_device_start_index") as mock_set_start,
        patch("torch_neuronx._C._vnc_count", return_value=1),
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
        patch("torch_neuronx._C._nrt_barrier") as mock_barrier,
        patch("torch_neuronx._C._set_world_size") as mock_set_world_size,
        patch("torch_neuronx._C._set_rank") as mock_set_rank,
        patch("torch_neuronx._C._set_local_world_size") as mock_set_local_world_size,
        patch("torch_neuronx._C._set_local_device_start_index") as mock_set_start,
        patch("torch_neuronx._C._vnc_count", return_value=1),
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


def test_set_enable_nan_check(process_group):
    """Test _set_enable_nan_check method functionality."""
    # Test default state
    assert process_group._enable_nan_check is False

    # Test enabling
    process_group._set_enable_nan_check(True)
    assert process_group._enable_nan_check is True

    # Test disabling
    process_group._set_enable_nan_check(False)
    assert process_group._enable_nan_check is False

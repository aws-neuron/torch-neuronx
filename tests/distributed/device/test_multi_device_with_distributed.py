import os
import socket
from collections.abc import Callable
from contextlib import contextmanager

import pytest
import torch

from tests.utils.neuron_test_utils import assert_raises, get_core_start_index

from ..utils import DistributedTester

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 2))


@contextmanager
def distributed_context(master_addr, master_port, rank: int, world_size: int):
    """Context manager for distributed setup"""
    try:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        yield
    finally:
        for key in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK"]:
            if key in os.environ:
                del os.environ[key]


class DistributedTesterForMultiDevice(DistributedTester):
    def _worker(
        self, rank: int, test_fn: Callable, func_args, master_addr, master_port, error_queue
    ):
        """Worker function that runs on each process"""
        try:
            with distributed_context(master_addr, master_port, rank, self.world_size):
                test_fn(rank, self.world_size, func_args)
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            error_queue.put((rank, e, tb))
            raise


def _test_multi_process_single_device(rank, world_size, args):
    """Test each process uses single device"""
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(get_core_start_index() + rank)

    import torch.distributed as dist

    import torch_neuronx

    dist.init_process_group(backend="neuron")

    # Test device count is 1 per process
    device_count = torch_neuronx.device_count()
    assert device_count != 1, f"device_count returned {device_count}, expected {world_size}"

    # Test device creation
    device = torch.device(f"neuron:{rank}")
    assert device.type == "neuron"
    assert device.index == rank

    # Test tensor operations
    tensor = torch.tensor([rank + 1.0]).to(device)
    result = tensor + 1.0
    expected = torch.tensor([rank + 2.0])
    assert torch.allclose(result.cpu(), expected)

    dist.destroy_process_group()


def _test_multi_process_multi_device(rank, world_size, args):
    """Test each process uses multiple devices"""
    devices_per_process = args.get("devices_per_process", 2)
    start_core = get_core_start_index() + rank * devices_per_process
    visible_cores = ",".join(str(start_core + i) for i in range(devices_per_process))
    os.environ["NEURON_RT_VISIBLE_CORES"] = visible_cores

    import torch.distributed as dist

    import torch_neuronx

    dist.init_process_group(backend="neuron")


def _test_cross_device_access_failure(rank, world_size, args):
    """Test that tries to access device from other process"""
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(get_core_start_index() + rank)

    import torch.distributed as dist

    import torch_neuronx

    dist.init_process_group(backend="neuron")

    _ = torch.tensor([rank + 1.0]).to("neuron:2")


def _test_vnc_id_mapping(rank, world_size, args):
    """Test local_rank to vnc_id mapping"""
    devices_per_process = args.get("devices_per_process", 1)
    start_core = get_core_start_index() + rank * devices_per_process

    if devices_per_process > 1:
        visible_cores = ",".join(str(start_core + i) for i in range(devices_per_process))
        os.environ["NEURON_RT_VISIBLE_CORES"] = visible_cores
    else:
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(start_core)

    import torch.distributed as dist

    import torch_neuronx

    dist.init_process_group(backend="neuron")

    # Test vnc_id mapping
    for i in range(devices_per_process):
        local_rank = rank + i
        expected_vnc_id = i
        vnc_id = torch_neuronx._C._get_vnc_id(local_rank)
        assert (
            vnc_id == expected_vnc_id
        ), f"local_rank {local_rank} -> vnc_id {vnc_id}, expected {expected_vnc_id}"

    dist.destroy_process_group()


def _test_device_properties(rank, world_size, args):
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(get_core_start_index() + rank)

    import torch.distributed as dist

    import torch_neuronx

    dist.init_process_group(backend="neuron")

    # Test device properties
    props = torch_neuronx.get_device_properties(0)
    assert hasattr(props, "name")
    assert hasattr(props, "total_memory")
    assert isinstance(props.name, str)
    assert isinstance(props.total_memory, int)

    dist.destroy_process_group()


class TestMultiDevicesWithDistributed:
    @property
    def world_size(self):
        return _WORLD_SIZE

    @pytest.fixture(autouse=True)
    def process_index(self, worker_id):
        """Setup process index and return it"""
        process_index = int(worker_id[2:]) if worker_id != "master" else 0
        return process_index

    @pytest.fixture(autouse=True)
    def distributed_tester(self, process_index):
        """Setup distributed tester using process_index"""
        rt_port = int(os.environ.get("NEURON_RT_PORT", "2025"))
        os.environ["NEURON_RT_COMM_ID"] = f"localhost:{rt_port + process_index}"
        self.distributed_tester = DistributedTesterForMultiDevice(
            world_size=self.world_size, process_index=process_index
        )
        return self.distributed_tester

    def test_multi_process_single_device(self):
        """Test multi-process with single device per process (use case 2)"""
        self.distributed_tester.run_test(_test_multi_process_single_device)

    @assert_raises(
        (RuntimeError, ValueError),
        match=r"(Attempted to have more than one device per process.*"
        r"|Device index.*out of range.*)",
    )
    def test_multi_process_multi_device(self):
        """Test multi-process with multiple devices per process (use case 4)"""
        self.distributed_tester.run_test(_test_multi_process_multi_device, devices_per_process=2)

    @assert_raises(RuntimeError, match=r"Device index 2 is out of range.*")
    def test_cross_device_access_failure(self):
        """Test multi-process with single device per process (use case 2)"""
        self.distributed_tester.run_test(_test_cross_device_access_failure)

    def test_vnc_id_mapping_single_device(self):
        """Test vnc_id mapping for single device per process"""
        self.distributed_tester.run_test(_test_vnc_id_mapping, devices_per_process=1)

    def test_device_properties_distributed(self):
        """Test device properties in distributed setting"""

        self.distributed_tester.run_test(_test_device_properties)

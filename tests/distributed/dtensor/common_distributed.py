import os

import torch
import torch.distributed as dist

from tests.utils.neuron_test_utils import get_core_start_index, get_pytest_worker_index


class NeuronDistributedBase:
    _world_size = 8

    @property
    def world_size(self):
        return self._world_size

    @world_size.setter
    def world_size(self, value):
        self._world_size = value

    @property
    def backend(self):
        return "neuron"

    @property
    def device(self):
        return torch.device("neuron", self.rank)

    @classmethod
    def _setup_neuron_environment(cls, rank, set_rt_visible_cores=False):
        process_index = get_pytest_worker_index()
        rt_port = int(os.environ.get("NEURON_RT_PORT", "2025"))
        os.environ["NEURON_RT_ROOT_COMM_ID"] = f"localhost:{rt_port + process_index}"
        if set_rt_visible_cores:
            core_start_index = get_core_start_index()
            os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_start_index + rank)
        os.environ.setdefault("FI_EFA_ENABLE_SHM_TRANSFER", "0")

    def _setup_process_group(self, register_default=False):
        if not dist.is_initialized():
            store = dist.FileStore(self.file_name, self.world_size)
            core_start_index = get_core_start_index()
            os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_start_index + self.rank)
            dist.init_process_group(
                backend=self.backend,
                world_size=self.world_size,
                rank=self.rank,
                store=store,
            )

        if register_default:
            torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

        return dist.distributed_c10d._get_default_group()

    def init_pg(self, *args, **kwargs) -> None:
        self._setup_process_group()

    def _get_process_group(self):
        return self._setup_process_group()

    def create_pg(self, device=None):
        return self._setup_process_group()

    def _init_process_group(self, *args, **kwargs):
        self._setup_process_group(register_default=True)

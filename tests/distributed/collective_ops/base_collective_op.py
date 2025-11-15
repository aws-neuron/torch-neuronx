import os

import pytest

from ..utils import DistributedTester

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 2))


class BaseCollectiveOpTest:
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
        # Presetting the port and adjusting based on process_index, so that
        # we don't have a clash when there are 64 tests running in parallel
        rt_port = int(os.environ.get("NEURON_RT_PORT", "2025"))
        os.environ["NEURON_RT_ROOT_COMM_ID"] = f"localhost:{rt_port + process_index}"
        self.distributed_tester = DistributedTester(
            world_size=self.world_size, process_index=process_index
        )
        return self.distributed_tester

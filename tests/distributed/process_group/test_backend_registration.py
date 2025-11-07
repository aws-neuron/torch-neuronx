import os

import pytest

from ..collective_ops.base_collective_op import BaseCollectiveOpTest


def _test_registration_and_rt_comm_id_preset(rank, world_size, kwargs):
    import torch.distributed as dist

    assert dist.get_backend() == "neuron"
    assert os.environ.get("NEURON_RT_ROOT_COMM_ID", None) == kwargs["neuron_rt_comm_id"]
    dist.destroy_process_group()


class TestBackendRegistration(BaseCollectiveOpTest):
    def _test_registration_and_rt_comm_id_preset(self):
        os.environ["NEURON_RT_ROOT_COMM_ID"] = "localhost:2024"
        self.distributed_tester.run_test(
            _test_registration_and_rt_comm_id_preset, neuron_rt_comm_id="localhost:2024"
        )
        del os.environ["NEURON_RT_ROOT_COMM_ID"]

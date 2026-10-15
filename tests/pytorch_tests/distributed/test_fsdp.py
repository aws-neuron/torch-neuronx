import os
import sys

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_fsdp import FSDPTest

import torch_neuronx
from tests.pytorch_tests.distributed.common_distributed import (
    NeuronDistributedBase,
    create_test_classes,
)
from tests.pytorch_tests.distributed.neuron_patch import patch_common_fsdp


class NeuronFSDPTest(NeuronDistributedBase, FSDPTest):
    @classmethod
    def _run(cls, rank, test_name, file_name, parent_pipe, **kwargs):
        from torch.testing._internal.common_distributed import TEST_SKIPS

        super()._setup_neuron_environment(rank)
        cls._apply_patches()

        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        try:
            self._setup_process_group()
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)
            raise

        # Same as FSDPTest
        torch._dynamo.reset()
        device_ids = [dist.get_rank()]
        self.run_test(test_name, parent_pipe)
        dist.barrier(device_ids=device_ids)
        dist.destroy_process_group()

    @property
    def world_size(self):
        return 2


patch_common_fsdp()
create_test_classes(NeuronFSDPTest, globals(), "fsdp_spec.json")


if __name__ == "__main__":
    import pytest

    pytest.main(["-vs", __file__])

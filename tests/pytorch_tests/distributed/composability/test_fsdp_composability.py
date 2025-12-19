import torch
from torch.testing._internal.common_fsdp import get_devtype

import torch_neuronx
from tests.pytorch_tests.distributed.common_distributed import create_test_classes
from tests.pytorch_tests.distributed.neuron_patch import override_world_size, patch_common_fsdp
from tests.pytorch_tests.distributed.test_fsdp import NeuronFSDPTest

device_type = torch.device(get_devtype())
device_module = torch.get_device_module(device_type)

patch_common_fsdp()
create_test_classes(NeuronFSDPTest, globals(), "fsdp_composability_spec.json")
# Use centralized override_world_size with dynamic calculation matching upstream behavior
override_world_size(globals(), world_size=min(4, device_module.device_count()))

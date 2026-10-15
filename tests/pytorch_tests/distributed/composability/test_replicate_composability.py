import pytest

import torch_neuronx
from tests.pytorch_tests.distributed.common_distributed import NeuronCommonTest, create_test_classes
from tests.pytorch_tests.distributed.neuron_patch import override_world_size

create_test_classes(NeuronCommonTest, globals(), "replicate_composability_spec.json")

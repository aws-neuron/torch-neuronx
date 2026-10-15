import pytest

import torch_neuronx
from tests.pytorch_tests.distributed.common_distributed import NeuronCommonTest, create_test_classes

create_test_classes(NeuronCommonTest, globals(), "dtensor_composability_spec.json")

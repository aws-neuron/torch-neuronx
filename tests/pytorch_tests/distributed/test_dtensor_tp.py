import pytest
from common_distributed import NeuronCommonTest, create_test_classes

import torch_neuronx
from tests.pytorch_tests.distributed.neuron_patch import override_world_size

create_test_classes(NeuronCommonTest, globals(), "dtensor_tp_spec.json")

# World size is being set to 64 in the code, setting this to 4
# for TensorParallelAPITests in test_parallelize_api.py
override_world_size(globals(), world_size=4)


if __name__ == "__main__":
    import pytest

    pytest.main(["-vs", __file__])

import pytest
from common_distributed import NeuronCommonTest, create_test_classes

import torch_neuronx

create_test_classes(NeuronCommonTest, globals(), "dtensor_spec.json")


if __name__ == "__main__":
    import pytest

    pytest.main(["-vs", __file__])

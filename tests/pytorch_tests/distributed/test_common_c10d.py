import pytest
from common_distributed import NeuronCommonTest, create_test_classes

import torch_neuronx

create_test_classes(NeuronCommonTest, globals(), "common_c10d_spec.json")


if __name__ == "__main__":
    import pytest

    pytest.main(["-vs", __file__])

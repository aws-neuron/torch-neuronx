import pytest
from common_distributed import NeuronMultiProcessTest, create_test_classes

import torch_neuronx

create_test_classes(NeuronMultiProcessTest, globals(), "checkpoint_spec.json")


if __name__ == "__main__":
    import pytest

    pytest.main(["-vs", __file__])

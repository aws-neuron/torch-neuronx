from typing import Any

import pytest

import torch_neuronx
from tests.pytorch_tests.distributed.common_distributed import (
    NeuronMultiProcessTestDDP,
    create_test_classes,
)

create_test_classes(NeuronMultiProcessTestDDP, globals(), "ddp_spec.json")


if __name__ == "__main__":
    import pytest

    pytest.main(["-vs", __file__])

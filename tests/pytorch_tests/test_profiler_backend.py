import logging
import os

os.environ["PYTORCH_TESTING_DEVICE_FOR_CUSTOM"] = "privateuse1"

# ---------------------------------------------------------------------------
# Import-time patches: prevent upstream skip decorators from firing.
#
# These patches must run AFTER torch is loaded (via torch_neuronx) but BEFORE
# setup_pytorch_tests() imports the upstream test modules, because skip
# decorators like @skipIf(not TEST_CUDA) evaluate their condition at class-
# definition time (import).
# ---------------------------------------------------------------------------
import torch.testing._internal.common_cuda as _common_cuda
import torch.testing._internal.common_utils as _common_utils

import torch_neuronx

_common_cuda.TEST_CUDA = True
_common_utils.TEST_CUDA = True

_common_cuda.TEST_MULTIGPU = True
_common_utils.TEST_MULTIGPU = True

import pytest  # noqa: E402

from tests.pytorch_tests.utils import (  # noqa: E402
    create_pytorch_test_wrappers,
    setup_pytorch_tests,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SPEC_FILE = os.environ.get("PROFILER_BACKEND_SPEC_FILE", "profiler_backend_spec.json")

PYTORCH_TEST_CONFIGS = setup_pytorch_tests(spec_file=SPEC_FILE)

_test_wrappers = create_pytorch_test_wrappers(PYTORCH_TEST_CONFIGS)
for _class_name, _test_cls in _test_wrappers.items():
    globals()[_class_name] = _test_cls

if __name__ == "__main__":
    pytest.main(["-vs", __file__])

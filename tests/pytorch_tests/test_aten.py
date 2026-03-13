import logging
import os

import pytest

import torch_neuronx
from tests.pytorch_tests.utils import (
    _apply_opinfo_patches,
    create_pytorch_test_wrappers,
    setup_pytorch_tests,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Apply OpInfo patches once before any test module imports
_apply_opinfo_patches()

SPEC_FILE = os.environ.get("ATEN_SPEC_FILE", "aten_spec.json")
CLASS_NAME_FILTER = os.environ.get("ATEN_CLASS_NAME", None)

PYTORCH_TEST_CONFIGS = setup_pytorch_tests(spec_file=SPEC_FILE, class_name_filter=CLASS_NAME_FILTER)

_test_wrappers = create_pytorch_test_wrappers(PYTORCH_TEST_CONFIGS)
for _class_name, _test_cls in _test_wrappers.items():
    globals()[_class_name] = _test_cls

if __name__ == "__main__":
    pytest.main(["-vs", __file__])

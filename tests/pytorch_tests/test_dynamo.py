import logging
import re
import unittest

import pytest

import torch_neuronx
from tests.pytorch_tests.dynamo.neuron_dynamo_patch import (
    apply_dynamo_patches,
    get_context_for_class,
)
from tests.pytorch_tests.utils import setup_pytorch_tests

# Apply patches before test discovery
apply_dynamo_patches()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PYTORCH_TEST_CONFIGS = setup_pytorch_tests(spec_file="dynamo_spec.json", collect_skipped_tests=True)


def get_class_key(config):
    file_name = config.test_module.__file__.split("/")[-1].split(".")[0]
    return f"{file_name}__{config.test_class_name}"


def get_method_name(config, test):
    config_name = config.test_module.__file__.split("/")[-1]
    config_name = config_name.split(".")[0]
    return f"{config_name}__{test}"


class TestPytorchDynamoWrapper:
    """Dynamically generated pytest methods for PyTorch dynamo tests."""

    @classmethod
    def setup_class(cls):
        cls.pytorch_test_classes = {}
        for config in PYTORCH_TEST_CONFIGS:
            test_class = getattr(config.test_module, config.test_class_name)
            cls.pytorch_test_classes[get_class_key(config)] = test_class

    def _run_pytorch_test(self, class_key, method_name):
        class_name = class_key.split("__")[-1]
        with get_context_for_class(class_name):
            test_instance = self.pytorch_test_classes[class_key]()
            if hasattr(test_instance, "setUp"):
                test_instance.setUp()
            try:
                method = getattr(test_instance, method_name)
                expecting_failure = getattr(method, "__unittest_expecting_failure__", False)
                try:
                    method()
                except unittest.SkipTest as e:
                    pytest.skip(str(e))
                except Exception:
                    if expecting_failure:
                        pytest.xfail("unittest.expectedFailure")
                    raise
                else:
                    if expecting_failure:
                        pytest.fail("Expected test to fail but it passed")
            finally:
                if hasattr(test_instance, "tearDown"):
                    test_instance.tearDown()


for config in PYTORCH_TEST_CONFIGS:
    for method_name in config.test_methods:

        def create_test_method(config, method_name):
            class_key = get_class_key(config)
            test_attribute_name = get_method_name(config, method_name)

            def test_method(self):
                self._run_pytorch_test(class_key, method_name)

            test_method.__name__ = f"{test_attribute_name}"
            test_method.__doc__ = f"PyTorch Dynamo {method_name} test"
            return test_method

        test_attribute_name = get_method_name(config, method_name)
        setattr(
            TestPytorchDynamoWrapper,
            test_attribute_name,
            create_test_method(config, method_name),
        )

    for xfail in config.xfail:
        xfail_method_kw = next(iter(xfail.keys()))
        xfail_method_name_list = [
            method_name for method_name in config.test_methods if xfail_method_kw in method_name
        ]
        for xfail_method_name in xfail_method_name_list:
            xfail_test_attribute_name = get_method_name(config, xfail_method_name)
            method = getattr(TestPytorchDynamoWrapper, xfail_test_attribute_name)
            method = pytest.mark.xfail(reason=xfail[xfail_method_kw]["reason"])(method)
            setattr(TestPytorchDynamoWrapper, xfail_test_attribute_name, method)

    for skip in config.skip_tests:
        skip_method_kw = next(iter(skip.keys()))
        skip_method_name_list = [
            method_name for method_name in config.test_methods if skip_method_kw in method_name
        ]
        for skip_method_name in skip_method_name_list:
            skip_test_attribute_name = get_method_name(config, skip_method_name)
            method = getattr(TestPytorchDynamoWrapper, skip_test_attribute_name)
            method = pytest.mark.skip(reason=skip[skip_method_kw]["reason"])(method)
            setattr(TestPytorchDynamoWrapper, skip_test_attribute_name, method)

if __name__ == "__main__":
    pytest.main(["-vs", __file__])

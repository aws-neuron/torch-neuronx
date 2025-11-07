import logging
import os

import pytest

import torch_neuron
from tests.pytorch_tests.utils import setup_pytorch_tests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Setup at module level
PYTORCH_TEST_CONFIGS = setup_pytorch_tests(spec_file="aten_spec.json")


class TestPytorchWrapper:
    """Dynamically generated pytest methods for each PyTorch test."""

    @classmethod
    def setup_class(cls):
        """Store the test class."""
        cls.pytorch_test_classes = {}
        for config in PYTORCH_TEST_CONFIGS:
            test_class = getattr(config.test_module, config.test_class_name)
            cls.pytorch_test_classes[config.test_class_name] = test_class

    def _run_pytorch_test(self, class_name, method_name):
        """Helper to run individual PyTorch test."""
        test_instance = self.pytorch_test_classes[class_name]()
        if hasattr(test_instance, "setUp"):
            test_instance.setUp()

        try:
            method = getattr(test_instance, method_name)
            method()
        except Exception as e:
            logger.error(f"{method_name} failed: {e}")
            raise
        finally:
            if hasattr(test_instance, "tearDown"):
                test_instance.tearDown()


# Create the test methods
for config in PYTORCH_TEST_CONFIGS:
    for method_name in config.test_methods:

        def create_test_method(test_class_name, method_name):
            def test_method(self):
                """Dynamically created test method."""
                self._run_pytorch_test(test_class_name, method_name)

            test_method.__name__ = f"{method_name}"
            test_method.__doc__ = f"PyTorch {method_name} test"
            return test_method

        # Add the method to the class
        setattr(
            TestPytorchWrapper,
            f"{method_name}",
            create_test_method(config.test_class_name, method_name),
        )

    for xfail in config.xfail:
        xfail_method_kw = next(iter(xfail.keys()))
        xfail_method_name_list = [
            method_name for method_name in config.test_methods if xfail_method_kw in method_name
        ]

        for xfail_method_name in xfail_method_name_list:
            method = getattr(TestPytorchWrapper, xfail_method_name)
            method = pytest.mark.xfail(reason=xfail[xfail_method_kw]["reason"])(method)
            setattr(TestPytorchWrapper, xfail_method_name, method)

    for skip in config.skip_tests:
        skip_method_kw = next(iter(skip.keys()))
        skip_method_name_list = [
            method_name for method_name in config.test_methods if skip_method_kw in method_name
        ]
        for skip_method_name in skip_method_name_list:
            method = getattr(TestPytorchWrapper, skip_method_name)
            method = pytest.mark.skip(reason=skip[skip_method_kw]["reason"])(method)
            setattr(TestPytorchWrapper, skip_method_name, method)

if __name__ == "__main__":
    # This allows running the tests directly with python
    pytest.main(["-vs", __file__])

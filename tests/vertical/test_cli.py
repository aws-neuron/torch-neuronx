import argparse
import logging

import pytest

from tests.vertical.op_spec_base import OpSpecBase
from tests.vertical.test_registry import get_test_registry, get_xfail_tests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_test_configuration():
    """Parse command line arguments for test configuration."""
    parser = argparse.ArgumentParser(description="Run vertical test for PyTorch ops")
    parser.add_argument(
        "--test-name",
        type=str,
        default=None,
        help="Comma-delimited names of the tests. If not specified, all registered tests will run",
    )
    parser.add_argument(
        "--op-spec",
        type=str,
        default=None,
        help="Comma-delimited names of the op specs. If not specified, all op specs will run.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args, _ = parser.parse_known_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    test_registry = get_test_registry()
    input_test_name_list = args.test_name.split(",") if args.test_name else None
    test_name_list = []
    if input_test_name_list:
        for test_name in input_test_name_list:
            if test_name in test_registry:
                test_name_list.append(test_name)
            else:
                raise KeyError(f"Test '{test_name}' not found in the registry: {test_registry}")
    else:
        test_name_list = list(test_registry.keys())

    op_spec_list = args.op_spec.split(",") if args.op_spec else None
    test_configs = {}
    for test_name in test_name_list:
        test_entry = test_registry[test_name]
        ops_spec_lu = test_entry.op_specs
        if op_spec_list:
            # Filter for specific op specs
            spec_list = []
            for op_spec_name in op_spec_list:
                if op_spec_name in ops_spec_lu:
                    spec_list.append(ops_spec_lu[op_spec_name])
                else:
                    logger.warning(f"Op spec '{op_spec_name}' not found for test '{test_name}'")
                    raise KeyError(f"Op spec '{op_spec_name}' not found for test '{test_name}'")
        else:
            spec_list = list(ops_spec_lu.values())

        test_configs[test_name] = spec_list

    logger.info(f"Running test config '{test_configs}'")

    return test_configs


TEST_CONFIG = get_test_configuration()


class TestVerticalWrapper:
    """Dynamically generated pytest methods for each Vertical test."""

    @classmethod
    def setup_class(cls):
        """Setup test registry."""
        cls.test_registry = get_test_registry()

    def _run_vertical_test(self, class_name: str, op_spec: type[OpSpecBase]):
        """Helper to run individual PyTorch test."""
        try:
            test_instance = self.test_registry[class_name].test_class()
            test_name = f"{class_name}.{op_spec.__name__}"

            results = test_instance.execute(op_spec)
            failed_tests = []
            total_tests = 0

            for _result in results:
                total_tests += 1
                print(_result)

                if not _result.passed:
                    failed_tests.append(f"{class_name}.{op_spec.__name__}: {_result}")

            logger.info(f"{test_name} total cases run: {total_tests}")
            logger.info(f"{test_name} failed cases: {len(failed_tests)}")
            logger.info(f"{test_name} passed cases: {total_tests - len(failed_tests)}")

            # Fail if any tests failed
            if failed_tests:
                failure_message = f"\n{len(failed_tests)} test(s) failed:\n" + "\n".join(
                    failed_tests
                )
                pytest.fail(failure_message)

        except Exception as e:
            logger.error(f"{op_spec.__name__} failed: {e}")
            raise


# Create the test methods
xfailed_tests = get_xfail_tests()
for class_name, op_spec_list in TEST_CONFIG.items():
    for op_spec in op_spec_list:

        def create_test_method(class_name: str, op_spec: type[OpSpecBase]):
            def test_method(self):
                """Dynamically created test method."""
                self._run_vertical_test(class_name, op_spec)

            method_name = f"test_{class_name}_{op_spec.__name__}"
            test_method.__name__ = method_name
            test_method.__doc__ = f"{class_name} {op_spec.__name__} test"

            mark_xfail_reason = xfailed_tests[class_name][op_spec.__name__]
            if mark_xfail_reason:
                test_method = pytest.mark.xfail(reason=mark_xfail_reason)(test_method)

            return test_method

        # Add the method to the class
        method_name = f"test_{class_name}_{op_spec.__name__}"
        setattr(
            TestVerticalWrapper,
            method_name,
            create_test_method(class_name, op_spec),
        )


if __name__ == "__main__":
    # This allows running the tests directly with python
    pytest.main(["-vs", __file__])

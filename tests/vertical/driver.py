import logging
import os
from typing import Any, Optional

import torch

import torch_neuronx
from tests.vertical.op_test_result import OpTestResult
from tests.vertical.test_registry import get_test_registry

logger = logging.getLogger(__name__)


class VerticalTestDriver:
    """Main test driver that orchestrates the execution of vertical test against set of op specs.

    This class manages the initialization of the Neuron runtime environment
    and coordinates the execution of various vertical tests across different
    op specs.
    """

    def run_tests(
        self,
        test_name: str,
        op_spec_list: list[str] | None = None,
    ) -> dict[str, list[OpTestResult]]:
        """Run all test combinations for specified operations and tests.

        Args:
            test_name: Name of the vertical test to run.
            op_spec_list: Optional list of op sepc names to test.
             If None, all available op specs will be tested.

        Returns:
            Dictionary mapping test names to their execution results

        Raises:
            KeyError: If specified test names or op specs are not found in registry
            RuntimeError: If test execution fails
        """

        test_registry = get_test_registry()
        logger.debug(f"test registry: {test_registry}")
        results: dict[str, Any] = {}

        try:
            logger.info(f"Running test: {test_name}")
            test_entry = test_registry[test_name]
            ops_spec_lu = test_entry.op_specs
            test_instance = test_entry.test_class()

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

            logger.info(
                f"Running '{test_name}' with op_specs: {[spec.__name__ for spec in spec_list]}"
            )

            test_results = test_instance.execute(spec_list)
            results[test_name] = test_results

        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            raise RuntimeError(f"Test execution failed: {e}") from e

        return results

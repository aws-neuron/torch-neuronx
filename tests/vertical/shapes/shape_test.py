import logging
from typing import Any, Union

import torch

from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

from ..op_spec_base import OpSpecBase
from ..op_test_result import OpTestResult
from ..test_registry import register_test
from ..vertical_test_base import VerticalTestBase

logger = logging.getLogger(__name__)


@register_test()
class ShapeTest(VerticalTestBase):
    """Test different input shapes for ops across CPU and Neuron devices.

    This test validates that ops produce consistent results when executed
    on CPU versus Neuron devices with various input shapes.
    """

    def execute(self, op_spec_class: type[OpSpecBase]) -> list[OpTestResult]:
        """Execute shape tests for all provided op specs.

        Args:
            op_spec_class: op spec class to test

        Returns:
            List of OpTestResult objects containing test results for all ops
        """
        all_results = []

        op_spec = op_spec_class()
        test_op = self._get_op(op_spec.op_name)
        for _args, _kwargs in op_spec.generate_combinations():
            torch.manual_seed(self.random_seed)
            test_args = self._allocate(_args, device="cpu")
            test_kwargs = self._allocate(_kwargs, device="cpu")
            out_cpu = test_op(*test_args, **test_kwargs)

            torch.manual_seed(self.random_seed)
            test_args = self._allocate(_args, device="neuron")
            test_kwargs = self._allocate(_kwargs, device="neuron")
            with track_neuron_ops():
                out_neuron = test_op(*test_args, **test_kwargs)
                expected_neuron_op_name = op_spec.expected_neuron_op(_args, _kwargs)
                if expected_neuron_op_name is not None:
                    assert_op_runs_on_neuron(expected_neuron_op_name)
                else:
                    logger.debug("Skip valdiating op runs on Neuron")

            all_results.append(self.validate_result(out_cpu, out_neuron, op_spec, _args, _kwargs))

        return all_results

    def validate_result(
        self,
        out_cpu: torch.Tensor | tuple[torch.Tensor, ...],
        out_neuron: torch.Tensor | tuple[torch.Tensor, ...],
        op_spec: Any,
        args: tuple[Any, ...],
        kwargs: dict,
    ) -> OpTestResult:
        """Validate that CPU and Neuron outputs match for the given op spec.

        Args:
            out_cpu: Output from CPU execution
            out_neuron: Output from Neuron execution
            op_spec: Op spec object
            args: Input arguments used for the op
            kwargs: Keyword arguments used for the op

        Returns:
            OpTestResult object containt test metadata and result
        """
        result = OpTestResult(
            test_name=self.__class__.__name__,
            op_spec_name=op_spec.__class__.__name__,
            op_name=op_spec.op_name,
            args_shapes=[_arg.shape for _arg in args],
            args_dtypes=[_arg.dtype for _arg in args],
            kwargs_spec=kwargs,
        )
        try:
            if op_spec.output_indices:
                for _index in op_spec.output_indices:
                    if not isinstance(out_cpu, tuple):
                        raise ValueError(f"expected tuple output but received {type(out_cpu)}")

                    torch.testing.assert_close(out_cpu[_index], out_neuron[_index].cpu())
            else:
                torch.testing.assert_close(out_cpu, out_neuron.cpu())
        except AssertionError as e:
            logger.error(f"{result} failed tensor comparison with with msg: {e}\n")
            result.passed = False
            return result

        result.passed = True
        return result

import logging
from typing import Any, Union

import torch
from neuronxcc.starfish.support.util import allclose

from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

from ..op_spec_base import OpSpecBase
from ..op_test_result import OpTestResult
from ..test_registry import register_test
from ..vertical_test_base import VerticalTestBase

logger = logging.getLogger(__name__)


@register_test()
class CheckBackwardImplTest(VerticalTestBase):
    """Check backward implementation for ops across CPU and Neuron devices.

    This test validates checks ops backward implementation produce consistent
    results when executed on CPU versus Neuron devices.
    """

    rtol: float = 1e-02
    atol: float = 1e-05

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
            cpu_test_args = self._allocate(_args, device="cpu")
            cpu_test_kwargs = self._allocate(_kwargs, device="cpu")
            out_cpu = test_op(*cpu_test_args, **cpu_test_kwargs)

            if op_spec.output_loss_index is not None:
                loss_cpu = out_cpu[op_spec.output_loss_index].sum()
            else:
                loss_cpu = out_cpu.sum()

            loss_cpu.backward()

            torch.manual_seed(self.random_seed)
            neuron_test_args = self._allocate(_args, device="neuron")
            neuron_test_kwargs = self._allocate(_kwargs, device="neuron")
            with track_neuron_ops():
                out_neuron = test_op(*neuron_test_args, **neuron_test_kwargs)
                if op_spec.output_loss_index is not None:
                    loss_neuron = out_neuron[op_spec.output_loss_index].sum()
                else:
                    loss_neuron = out_neuron.sum()

                loss_neuron.backward()
                expected_neuron_op_name_list = op_spec.expected_neuron_op(_args, _kwargs)
                for expected_neuron_op_name in expected_neuron_op_name_list:
                    assert_op_runs_on_neuron(expected_neuron_op_name)
                else:
                    logger.debug("Skip valdiating op runs on Neuron")

            all_results.append(
                self.validate_result(
                    out_cpu,
                    out_neuron,
                    op_spec,
                    cpu_test_args,
                    cpu_test_kwargs,
                    neuron_test_args,
                    neuron_test_kwargs,
                )
            )

        return all_results

    def validate_result(
        self,
        out_cpu: torch.Tensor | tuple[torch.Tensor, ...],
        out_neuron: torch.Tensor | tuple[torch.Tensor, ...],
        op_spec: Any,
        input_cpu_args: tuple[Any, ...],
        input_cpu_kwargs: dict,
        input_neuron_args: tuple[Any, ...],
        input_neuron_kwargs: dict,
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
            args_shapes=[
                _arg.shape for _arg in input_neuron_args if isinstance(_arg, torch.Tensor)
            ],
            args_dtypes=[
                _arg.dtype for _arg in input_neuron_args if isinstance(_arg, torch.Tensor)
            ],
            kwargs_spec=input_neuron_kwargs,
        )
        try:
            if op_spec.output_indices:
                for _index in op_spec.output_indices:
                    if not isinstance(out_cpu, tuple):
                        raise ValueError(f"expected tuple output but received {type(out_cpu)}")

                    assert allclose(
                        out_cpu[_index].detach().float().numpy().flatten(),
                        out_neuron[_index].detach().float().cpu().numpy().flatten(),
                        rtol=self.rtol,
                        atol=self.atol,
                        equal_nan=True,
                        equal_inf=True,
                        verbose=1,
                        mode="max",
                    ), f"{op_spec.op_name}: output mismatch between CPU and Neuron"
            else:
                assert allclose(
                    out_cpu.detach().float().numpy().flatten(),
                    out_neuron.detach().float().cpu().numpy().flatten(),
                    rtol=self.rtol,
                    atol=self.atol,
                    equal_nan=True,
                    equal_inf=True,
                    verbose=1,
                    mode="max",
                ), f"{op_spec.op_name}: output mismatch between CPU and Neuron"

            for index in op_spec.input_indices:
                input_neuron = input_neuron_args[index]
                input_cpu = input_cpu_args[index]
                assert (
                    input_neuron.requires_grad == input_cpu.requires_grad
                ), f"{op_spec.op_name}: requires grad mismatch between CPU and Neuron"
                if not input_neuron.requires_grad:
                    continue

                assert input_neuron.grad is not None, f"{op_spec.op_name}: No gradients computed"
                assert (
                    input_neuron.grad.shape == input_neuron.shape
                ), f"{op_spec.op_name}: Gradient shape mismatch"
                assert (
                    input_neuron.grad.device.type == "neuron"
                ), f"{op_spec.op_name}: Gradients not on neuron device"

                assert input_cpu.grad is not None, "CPU gradients should exist"
                assert allclose(
                    input_cpu.grad.detach().float().numpy().flatten(),
                    input_neuron.grad.detach().float().cpu().numpy().flatten(),
                    rtol=self.rtol,
                    atol=self.atol,
                    equal_nan=True,
                    equal_inf=True,
                    verbose=1,
                    mode="max",
                ), f"{op_spec.op_name}: Gradient mismatch between CPU and Neuron"

        except AssertionError as e:
            logger.error(f"{result} failed tensor comparison with with msg: {e}\n")
            result.passed = False
            return result

        result.passed = True
        return result

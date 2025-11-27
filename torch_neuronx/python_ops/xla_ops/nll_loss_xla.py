"""XLA implementation of NLL loss forward and backward."""

import jax.nn as jnn
import jax.numpy as jnp
import torch

from torch_neuronx.kernels import TorchNeuronXLAKernel
from torch_neuronx.python_ops.auto_registration import neuron_op
from torch_neuronx.python_ops.base import ExecutionResult, OperationImplementation


@neuron_op("aten::nll_loss_forward.output")
class NLLLossForwardXLAImpl(OperationImplementation):
    """XLA implementation of nll loss forward."""

    def __init__(self):
        self._kernels = {}

    @classmethod
    def _compute_target_probs(cls, log_probs, target, ignore_index):
        # Create a mask for ignored indices
        mask = target != ignore_index

        # Gather the log probabilities corresponding to the targets
        # target_log_probs = jnp.take_along_axis(
        #     log_probs,
        #     jnp.expand_dims(target * mask, axis=1),
        #     axis=1
        # ).squeeze(1)
        # using a scatter based implementatio because compiler has a bug in the gather based
        # implementation: P283302344
        zeros = jnp.zeros_like(log_probs)
        target_log_probs = jnp.sum(
            zeros.at[jnp.arange(log_probs.shape[0]), target * mask].add(1) * log_probs, axis=1
        )

        # Apply the mask
        target_log_probs = target_log_probs * mask

        return target_log_probs, mask

    @classmethod
    def _compute_loss(cls, target_log_probs, weight, reduction):
        # Calculate total_weight
        total_weight = jnp.sum(weight)

        # Calculate negative log likelihood
        loss = -target_log_probs * weight

        # Apply reduction
        if reduction == 0:  # none
            return loss, total_weight
        elif reduction == 2:  # sum
            return jnp.sum(loss), total_weight
        elif reduction == 1:  # mean
            # Prevent division by zero
            total_weight = jnp.maximum(total_weight, 1e-8)
            return jnp.sum(loss) / total_weight, total_weight
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def _get_kernel(self, weight, reduction):
        if weight is None:
            if f"weight_none_{reduction}" not in self._kernels:

                def _nll_loss_forward_weight_none(input, target, ignore_index):
                    target_log_probs, mask = NLLLossForwardXLAImpl._compute_target_probs(
                        input, target, ignore_index
                    )
                    weight = mask.astype(input.dtype)
                    return NLLLossForwardXLAImpl._compute_loss(target_log_probs, weight, reduction)

                if reduction == 2:
                    kernel = TorchNeuronXLAKernel(
                        _nll_loss_forward_weight_none, "nll_loss_forward_weight_none_reduction_sum"
                    )
                elif reduction == 1:
                    kernel = TorchNeuronXLAKernel(
                        _nll_loss_forward_weight_none, "nll_loss_forward_weight_none_reduction_mean"
                    )
                elif reduction == 0:
                    kernel = TorchNeuronXLAKernel(
                        _nll_loss_forward_weight_none, "nll_loss_forward_weight_none_reduction_none"
                    )
                else:
                    raise ValueError(f"Unknown reduction: {reduction}")

                self._kernels[f"weight_none_{reduction}"] = kernel
            return self._kernels[f"weight_none_{reduction}"]
        else:
            if reduction not in self._kernels:

                def _nll_loss_forward(input, target, ignore_index, weight):
                    target_log_probs, mask = NLLLossForwardXLAImpl._compute_target_probs(
                        input, target, ignore_index
                    )
                    weight = weight.astype(input.dtype)
                    weight = jnp.take(weight, target * mask) * mask
                    return NLLLossForwardXLAImpl._compute_loss(target_log_probs, weight, reduction)

                if reduction == 2:
                    kernel = TorchNeuronXLAKernel(
                        _nll_loss_forward, "nll_loss_forward_reduction_sum"
                    )
                elif reduction == 1:
                    kernel = TorchNeuronXLAKernel(
                        _nll_loss_forward, "nll_loss_forward_reduction_mean"
                    )
                elif reduction == 0:
                    kernel = TorchNeuronXLAKernel(
                        _nll_loss_forward, "nll_loss_forward_reduction_none"
                    )
                else:
                    raise ValueError(f"Unknown reduction: {reduction}")
                self._kernels[reduction] = kernel
            return self._kernels[reduction]

    def _execute_impl(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weight=None,
        reduction: int = 1,
        ignore_index=None,
        output=None,
        total_weight=None,
    ) -> ExecutionResult:
        """Execute nll forward using XLA."""
        try:
            # Use provided output tensor or create a new one
            if output is None:
                output = (
                    torch.empty(
                        target.shape,
                        dtype=input.dtype,
                        requires_grad=input.requires_grad,
                        device="neuron",
                    )
                    if reduction == 0
                    else torch.empty(
                        (0,), dtype=input.dtype, requires_grad=input.requires_grad, device="neuron"
                    )
                )
                output.requires_grad = True

            total_weight = torch.empty(0, device="neuron") if total_weight is None else total_weight

            inputs = (
                [input, target, ignore_index]
                if weight is None
                else [input, target, ignore_index, weight]
            )

            # Execute kernel
            self._get_kernel(weight, reduction)(*inputs, output=(output, total_weight))

            return ExecutionResult(success=True, output=(output, total_weight))
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


@neuron_op("aten::nll_loss_backward.grad_input")
class NLLLossBackwardXLAImpl(OperationImplementation):
    """XLA implementation of nll loss backward."""

    def __init__(self):
        self._kernels = {}

    @classmethod
    def _generate_mask(cls, target, ignore_index):
        return target != ignore_index

    @classmethod
    def _compute_grad_input(cls, grad_output, target, weight, reduction, mask, num_elements):
        # Calculate total_weight
        total_weight = jnp.sum(weight)
        # Scale gradient based on reduction
        grad_scale = (
            grad_output / jnp.maximum(total_weight, 1e-8) if reduction == 1 else grad_output
        )  # 1 == mean
        target_one_hot = jnn.one_hot(target * mask, num_elements, dtype=grad_output.dtype)
        return -target_one_hot * weight[..., None] * grad_scale

    def _get_kernel(self, weight, reduction, input):
        if weight is None:
            if f"weight_none_{reduction}_{input.shape[1]}" not in self._kernels:

                def _nll_loss_backward_weight_none(grad_output, target, ignore_index):
                    mask = NLLLossBackwardXLAImpl._generate_mask(target, ignore_index)
                    weight = mask.astype(grad_output.dtype)
                    return NLLLossBackwardXLAImpl._compute_grad_input(
                        grad_output, target, weight, reduction, mask, input.shape[1]
                    )

                if reduction == 1:
                    kernel = TorchNeuronXLAKernel(
                        _nll_loss_backward_weight_none,
                        "nll_loss_backward_weight_none_reduction_mean",
                    )
                elif reduction == 2 or reduction == 0:
                    kernel = TorchNeuronXLAKernel(
                        _nll_loss_backward_weight_none,
                        "nll_loss_backward_weight_none_reduction_sum",
                    )
                else:
                    raise ValueError(f"Unknown reduction: {reduction}")

                self._kernels[f"weight_none_{reduction}"] = kernel
            return self._kernels[f"weight_none_{reduction}"]
        else:
            if reduction not in self._kernels:

                def _nll_loss_backward(grad_output, target, ignore_index, weight):
                    mask = NLLLossBackwardXLAImpl._generate_mask(target, ignore_index)
                    weight = weight.astype(grad_output.dtype)
                    weight = jnp.take(weight, target * mask) * mask
                    return NLLLossBackwardXLAImpl._compute_grad_input(
                        grad_output, target, weight, reduction, mask, input.shape[1]
                    )

                if reduction == 1:
                    kernel = TorchNeuronXLAKernel(
                        _nll_loss_backward, "nll_loss_backward_reduction_mean"
                    )
                elif reduction == 2 or reduction == 0:
                    kernel = TorchNeuronXLAKernel(
                        _nll_loss_backward, "nll_loss_backward_reduction_sum"
                    )
                else:
                    raise ValueError(f"Unknown reduction: {reduction}")
                self._kernels[reduction] = kernel
            return self._kernels[reduction]

    def _execute_impl(
        self,
        grad_output: torch.Tensor,
        input: torch.Tensor,
        target: torch.Tensor,
        weight=None,
        reduction: int = 1,
        ignore_index=-100,
        total_weight=None,
        grad_input: torch.Tensor = None,
    ) -> ExecutionResult:
        """Execute nll backward using XLA."""
        try:
            # Use provided output tensor or create a new one
            if grad_input is None:
                grad_input = torch.empty(input.shape, dtype=input.dtype, device=input.device)

            inputs = (
                [grad_output, target, ignore_index]
                if weight is None
                else [grad_output, target, ignore_index, weight]
            )
            # Execute kernel
            self._get_kernel(weight, reduction, input)(*inputs, output=grad_input)

            return ExecutionResult(success=True, output=grad_input)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))

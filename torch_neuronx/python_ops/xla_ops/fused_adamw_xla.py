"""XLA implementation of fused AdamW optimizer."""

import jax.numpy as jnp
import torch

from torch_neuronx.kernels import TorchNeuronXLAKernel
from torch_neuronx.python_ops.auto_registration import neuron_op, set_wrapper_override
from torch_neuronx.python_ops.base import ExecutionResult, OperationImplementation


@neuron_op("aten::_fused_adamw_")
@neuron_op("aten::_fused_adamw_.tensor_lr")
class FusedAdamWXLAImpl(OperationImplementation):
    """XLA implementation of fused AdamW optimizer"""

    def __init__(self):
        self.num_static_args = 7
        self.num_tensor_args = 6

        def adamw_computation(*inputs):
            """Computation for AdamW optimizer.

            Args:
                *inputs: flattened list
                    [param1, grad1, exp_avg1, exp_avg_sq1, max_exp_avg_sq1, step1, param2, ...,
                amsgrad, lr, beta1, beta2, weight_decay, eps, maximize]

            Returns:
                Tuple (
                    updated_params,
                    updated_exp_avgs,
                    updated_exp_avg_sqs,
                    updated_max_exp_avg_sqs
                )
            """
            # get tensor and static args
            amsgrad, lr, beta1, beta2, weight_decay, eps, maximize = inputs[-self.num_static_args :]
            param_data = inputs[: -self.num_static_args]
            num_params = len(param_data) // self.num_tensor_args

            updated_params, updated_exp_avgs, updated_exp_avg_sqs, updated_max_exp_avg_sqs = (
                [],
                [],
                [],
                [],
            )

            for i in range(num_params):
                base_idx = i * self.num_tensor_args
                param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, step = param_data[
                    base_idx : base_idx + self.num_tensor_args
                ]

                param_dtype = param.dtype
                exp_avg_dtype = exp_avg.dtype
                exp_avg_sq_dtype = exp_avg_sq.dtype
                max_exp_avg_sq_dtype = max_exp_avg_sq.dtype

                # negate gradient to maximize
                grad_to_use = jnp.where(maximize, -grad, grad)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                exp_avg_new = beta1 * exp_avg + (1.0 - beta1) * grad_to_use
                exp_avg_sq_new = beta2 * exp_avg_sq + (1.0 - beta2) * grad_to_use * grad_to_use

                # AMSGrad: use max
                max_exp_avg_sq_new = jnp.where(
                    amsgrad, jnp.maximum(max_exp_avg_sq, exp_avg_sq_new), max_exp_avg_sq
                )

                denom = jnp.where(
                    amsgrad,
                    (jnp.sqrt(max_exp_avg_sq_new) / jnp.sqrt(bias_correction2)) + eps,
                    (jnp.sqrt(exp_avg_sq_new) / jnp.sqrt(bias_correction2)) + eps,
                )
                step_size = lr / bias_correction1

                param_new = param * (1.0 - lr * weight_decay) - step_size * exp_avg_new / denom

                updated_params.append(param_new.astype(param_dtype))
                updated_exp_avgs.append(exp_avg_new.astype(exp_avg_dtype))
                updated_exp_avg_sqs.append(exp_avg_sq_new.astype(exp_avg_sq_dtype))
                # If amsgrad is False, we alias both max_exp_avg_sq_new
                # and updated_max_exp_avg_sqs to be exp_avg_sq_new
                max_exp_avg_sq_new = jnp.where(amsgrad, max_exp_avg_sq_new, exp_avg_sq_new)
                updated_max_exp_avg_sqs.append(max_exp_avg_sq_new.astype(max_exp_avg_sq_dtype))

            return tuple(
                updated_params + updated_exp_avgs + updated_exp_avg_sqs + updated_max_exp_avg_sqs
            )

        self.kernel = TorchNeuronXLAKernel(adamw_computation, "_fused_adamw_")

    def can_handle(self, *args, **kwargs) -> bool:
        params = args[0]
        if not all(p.device.type == "neuron" for p in params):
            raise RuntimeError("All parameters must be on Neuron device.")
        return True

    def _execute_impl(
        self,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale=None,
        found_inf=None,
        out=None,
        **kwargs,
    ) -> ExecutionResult:
        try:
            assert grad_scale is None and found_inf is None
            if isinstance(lr, torch.Tensor):
                lr = lr.item()
            static_args = [amsgrad, lr, beta1, beta2, weight_decay, eps, maximize]
            # case when amdgrad=False
            if not max_exp_avg_sqs:
                # If amsgrad is False, we alias both max_exp_avg_sq_new
                # and updated_max_exp_avg_sqs to be exp_avg_sq_new
                assert not amsgrad, "max_exp_avg_sqs must not be None when amsgrad=False"
                max_exp_avg_sqs = exp_avg_sqs

            # flatten tensors and static args
            inputs_flattened = []
            donate_argnums = []
            donate_baselist = [0, 2, 3, 4]
            for i in range(len(params)):
                inputs_flattened.extend(
                    [
                        params[i],
                        grads[i],
                        exp_avgs[i],
                        exp_avg_sqs[i],
                        max_exp_avg_sqs[i],
                        state_steps[i],
                    ]
                )
                donate_argnums_cur = [x + self.num_tensor_args * i for x in donate_baselist]
                donate_argnums.extend(donate_argnums_cur)
            donate_argnums = tuple(donate_argnums)
            inputs_flattened.extend(static_args)

            # Pre-allocate output tuple pointing to original tensors
            # to in-place update from within the kernel
            output_tensors = tuple(params + exp_avgs + exp_avg_sqs + max_exp_avg_sqs)
            self.kernel(*inputs_flattened, output=output_tensors, donate_argnums=donate_argnums)

            return ExecutionResult(success=True, output=None)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


def _fused_adamw_wrapper(aten_name, impl_class):
    """Wrapper for fused AdamW that returns None"""
    cached_op = None

    def wrapper(*args, **kwargs):
        nonlocal cached_op

        if cached_op is None:
            from torch_neuronx.python_ops.auto_registration import create_auto_operation

            cached_op = create_auto_operation(aten_name, [impl_class])

        kwargs["out"] = args[0]
        cached_op(*args, **kwargs)
        return None

    return wrapper


set_wrapper_override(
    "aten::_fused_adamw_", _fused_adamw_wrapper("aten::_fused_adamw_", FusedAdamWXLAImpl)
)
set_wrapper_override(
    "aten::_fused_adamw_.tensor_lr",
    _fused_adamw_wrapper("aten::_fused_adamw_.tensor_lr", FusedAdamWXLAImpl),
)

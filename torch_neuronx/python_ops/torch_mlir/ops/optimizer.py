"""Optimizer operations."""

import torch

from ..operation_registry import register_aten


@register_aten(
    ["aten::_fused_adamw_", "aten::_fused_adamw_.tensor_lr"],
    uses_preprocessing=True,
)
def torch_fused_adamw(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.01,
    eps=1e-8,
    amsgrad=False,
    maximize=False,
    grad_scale=None,
    found_inf=None,
    **kwargs,
):
    if isinstance(lr, torch.Tensor):
        lr = lr.item()

    use_amsgrad = amsgrad
    use_maximize = maximize

    def compute_fn(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
    ):
        updated_params = []
        updated_exp_avgs = []
        updated_exp_avg_sqs = []
        updated_max_exp_avg_sqs = []

        for i in range(len(params)):
            grad = grads[i]
            if use_maximize:
                grad = -grad

            step = state_steps[i]

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            bias_correction2_sqrt = torch.sqrt(bias_correction2)

            exp_avg_new = exp_avgs[i] * beta1 + grad * (1.0 - beta1)

            exp_avg_sq_new = exp_avg_sqs[i] * beta2 + grad * grad * (1.0 - beta2)

            if use_amsgrad:
                max_exp_avg_sq_new = torch.maximum(max_exp_avg_sqs[i], exp_avg_sq_new)
                denom = (torch.sqrt(max_exp_avg_sq_new) / bias_correction2_sqrt) + eps
                updated_max_exp_avg_sqs.append(max_exp_avg_sq_new)
            else:
                denom = (torch.sqrt(exp_avg_sq_new) / bias_correction2_sqrt) + eps

            step_size = lr / bias_correction1
            param_new = params[i] * (1.0 - lr * weight_decay) - step_size * exp_avg_new / denom

            updated_params.append(param_new)
            updated_exp_avgs.append(exp_avg_new)
            updated_exp_avg_sqs.append(exp_avg_sq_new)

        if use_amsgrad:
            return tuple(
                updated_params + updated_exp_avgs + updated_exp_avg_sqs + updated_max_exp_avg_sqs
            )

        # Note that we can't return None here since the fx graph requires an output node
        return tuple(updated_params + updated_exp_avgs + updated_exp_avg_sqs)

    # in-place variant returns None
    def postprocess_fn(results):
        return None

    if use_amsgrad:
        out_tensors = tuple(params + exp_avgs + exp_avg_sqs + max_exp_avg_sqs)
    else:
        out_tensors = tuple(params + exp_avgs + exp_avg_sqs)

    return (
        compute_fn,
        (
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
        ),
        {"out": out_tensors},  # write directly to original tensors
        postprocess_fn,
    )

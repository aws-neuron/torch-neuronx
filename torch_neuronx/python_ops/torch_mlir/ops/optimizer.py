"""Optimizer operations."""

from ..operation_registry import register_aten


@register_aten(["aten::_fused_adamw_", "aten::_fused_adamw_.tensor_lr"])
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
):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        if maximize:
            grad = -grad

        step += 1
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        denom = exp_avg_sq.sqrt().add_(eps)
        param.mul_(1 - lr * weight_decay)
        param.addcdiv_(exp_avg, denom, value=-lr)

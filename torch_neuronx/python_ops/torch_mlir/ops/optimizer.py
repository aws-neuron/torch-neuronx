"""Optimizer operations."""

import torch

from torch_neuronx.python_ops.nki_kernels.adamw import _has_nki_adam_kernels

from ..operation_registry import Preprocessing, register_aten


def _adam_step_mlir(
    param,
    grad,
    exp_avg,
    exp_avg_sq,
    max_exp_avg_sq,
    step,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    maximize,
    decoupled_wd,
    amsgrad,
):
    """Single-param Adam update using elementwise ops (no NKI kernel)."""
    if maximize:
        grad = -grad

    if not decoupled_wd:
        grad = grad + weight_decay * param

    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    bias_correction2_sqrt = torch.sqrt(bias_correction2)

    exp_avg_new = exp_avg * beta1 + grad * (1.0 - beta1)
    exp_avg_sq_new = exp_avg_sq * beta2 + grad * grad * (1.0 - beta2)

    if amsgrad:
        max_exp_avg_sq_new = torch.maximum(max_exp_avg_sq, exp_avg_sq_new)
        denom = (torch.sqrt(max_exp_avg_sq_new) / bias_correction2_sqrt) + eps
    else:
        max_exp_avg_sq_new = None
        denom = (torch.sqrt(exp_avg_sq_new) / bias_correction2_sqrt) + eps

    step_size = lr / bias_correction1
    if decoupled_wd:
        param_new = param * (1.0 - lr * weight_decay) - step_size * exp_avg_new / denom
    else:
        param_new = param - step_size * exp_avg_new / denom

    return param_new, exp_avg_new, exp_avg_sq_new, max_exp_avg_sq_new


def _make_adam_compute_fn(amsgrad, maximize, decoupled_wd):
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
        from torch_neuronx.python_ops.nki_kernels.adamw import adam_nki, adamw_nki

        wd_val = (1.0 - lr * weight_decay) if decoupled_wd else weight_decay
        # retain step-dependent params as tensors to avoid recompilation
        step = state_steps[0]
        bc1 = 1.0 - torch.pow(beta1, step)
        bc2_sqrt = torch.sqrt(1.0 - torch.pow(beta2, step))
        step_size = lr / bc1
        inv_bc2_sqrt = 1.0 / bc2_sqrt

        results_p, results_ea, results_eas, results_meas = [], [], [], []

        for i in range(len(params)):
            grad = -grads[i] if maximize else grads[i]
            meas = max_exp_avg_sqs[i] if amsgrad else params[i]

            # NKI kernel requires at least 1-dimensional tensors; fall back to
            # elementwise math for scalar (0-dim) params (e.g. CLIP logit_scale).
            if params[i].ndim == 0:
                p_new, ea_new, eas_new, meas_new = _adam_step_mlir(
                    params[i],
                    grads[i],
                    exp_avgs[i],
                    exp_avg_sqs[i],
                    meas if amsgrad else None,
                    state_steps[i],
                    lr,
                    beta1,
                    beta2,
                    weight_decay,
                    eps,
                    maximize,
                    decoupled_wd,
                    amsgrad,
                )
            else:
                results = (adamw_nki if decoupled_wd else adam_nki)(
                    params[i],
                    grad,
                    exp_avgs[i],
                    exp_avg_sqs[i],
                    meas,
                    step_size.expand(128, 1),
                    inv_bc2_sqrt.expand(128, 1),
                    wd_val * torch.ones((128, 1), dtype=torch.float32, device=params[i].device),
                    beta1,
                    beta2,
                    eps,
                    amsgrad,
                )
                if amsgrad:
                    p_new, ea_new, eas_new, meas_new = results
                else:
                    p_new, ea_new, eas_new = results
                    meas_new = None

            results_p.append(p_new)
            results_ea.append(ea_new)
            results_eas.append(eas_new)
            if amsgrad:
                results_meas.append(meas_new)

        if amsgrad:
            return tuple(results_p + results_ea + results_eas + results_meas)
        return tuple(results_p + results_ea + results_eas)

    # TODO remove MLIR fallback once fused_adam NKI kernel is available
    def compute_fn_mlir(
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
            param_new, exp_avg_new, exp_avg_sq_new, max_exp_avg_sq_new = _adam_step_mlir(
                params[i],
                grads[i],
                exp_avgs[i],
                exp_avg_sqs[i],
                max_exp_avg_sqs[i] if amsgrad else None,
                state_steps[i],
                lr,
                beta1,
                beta2,
                weight_decay,
                eps,
                maximize,
                decoupled_wd,
                amsgrad,
            )
            updated_params.append(param_new)
            updated_exp_avgs.append(exp_avg_new)
            updated_exp_avg_sqs.append(exp_avg_sq_new)
            if amsgrad:
                updated_max_exp_avg_sqs.append(max_exp_avg_sq_new)

        if amsgrad:
            return tuple(
                updated_params + updated_exp_avgs + updated_exp_avg_sqs + updated_max_exp_avg_sqs
            )

        # Note that we can't return None here since the fx graph requires an output node
        return tuple(updated_params + updated_exp_avgs + updated_exp_avg_sqs)

    if _has_nki_adam_kernels:
        return compute_fn
    else:
        return compute_fn_mlir


def _register_fused_adam(op_names, decoupled_wd):
    static_args = (7, 8, 9, 10) if _has_nki_adam_kernels else ()

    @register_aten(
        op_names, preprocessing=Preprocessing.VALUE_DEPENDENT, static_argnums=static_args
    )
    def fused_adam_impl(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.01 if decoupled_wd else 0.0,
        eps=1e-8,
        amsgrad=False,
        maximize=False,
        grad_scale=None,
        found_inf=None,
        **kwargs,
    ):
        compute_fn = _make_adam_compute_fn(amsgrad, maximize, decoupled_wd)

        # Filter out zero-numel params
        non_empty_idx = [i for i, p in enumerate(params) if p.numel() > 0]
        ne_params = [params[i] for i in non_empty_idx]
        ne_grads = [grads[i] for i in non_empty_idx]
        ne_exp_avgs = [exp_avgs[i] for i in non_empty_idx]
        ne_exp_avg_sqs = [exp_avg_sqs[i] for i in non_empty_idx]
        ne_max_exp_avg_sqs = [max_exp_avg_sqs[i] for i in non_empty_idx] if amsgrad else []
        ne_state_steps = [state_steps[i] for i in non_empty_idx]

        if not ne_params:
            return compute_fn, (), {}, lambda results: None

        if amsgrad:
            out_tensors = tuple(ne_params + ne_exp_avgs + ne_exp_avg_sqs + ne_max_exp_avg_sqs)
        else:
            out_tensors = tuple(ne_params + ne_exp_avgs + ne_exp_avg_sqs)

        return (
            compute_fn,
            (
                ne_params,
                ne_grads,
                ne_exp_avgs,
                ne_exp_avg_sqs,
                ne_max_exp_avg_sqs,
                ne_state_steps,
                lr,
                beta1,
                beta2,
                weight_decay,
                eps,
            ),
            {"out": out_tensors},
            lambda results: None,
        )

    return fused_adam_impl


torch_fused_adamw = _register_fused_adam(
    ["aten::_fused_adamw_", "aten::_fused_adamw_.tensor_lr"], decoupled_wd=True
)
torch_fused_adam = _register_fused_adam(
    ["aten::_fused_adam_", "aten::_fused_adam_.tensor_lr"], decoupled_wd=False
)

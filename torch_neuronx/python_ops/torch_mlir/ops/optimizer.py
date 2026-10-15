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


def _make_adam_compute_fn(decoupled_wd):
    """Build the traced compute function.

    amsgrad and maximize are parameters, not captured values: they select which
    graph is emitted and they vary from call to call, and the kernel cache keys on
    compute_fn's code object, which every closure from this def shares. Captured,
    they would be invisible to it and all four combinations would collide.

    decoupled_wd stays captured: it is not an ATen argument, it is bound per
    registration (_fused_adam_ False, _fused_adamw_ True), and each op name has its
    own kernel cache, so one cache never sees both values.
    """

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
        amsgrad,
        maximize,
    ):
        """Compute fused Adam updates with NKI kernels and scalar fallback.

        Args:
            params (list[torch.Tensor]): Parameter tensors to update.
            grads (list[torch.Tensor]): Gradients for the parameters.
            exp_avgs (list[torch.Tensor]): First-moment state tensors.
            exp_avg_sqs (list[torch.Tensor]): Second-moment state tensors.
            max_exp_avg_sqs (list[torch.Tensor]): Maximum second-moment state tensors.
            state_steps (list[torch.Tensor]): Per-parameter optimizer step tensors.
            lr (torch.Tensor): Learning rate.
            beta1 (torch.Tensor): First-moment decay coefficient.
            beta2 (torch.Tensor): Second-moment decay coefficient.
            weight_decay (torch.Tensor): Weight-decay coefficient.
            eps (torch.Tensor): Denominator stability term.
            amsgrad (bool): Whether to maintain maximum second moments.
            maximize (bool): Whether to maximize the objective.

        Returns:
            tuple[torch.Tensor, ...]: Parameters, moments, and optional AMSGrad maxima in list
                order; zero-element outputs alias their corresponding inputs.
        """
        from torch_neuronx.python_ops.nki_kernels.adamw import adam_nki, adamw_nki

        wd_val = (1.0 - lr * weight_decay) if decoupled_wd else weight_decay

        results_p, results_ea, results_eas, results_meas = [], [], [], []

        for i in range(len(params)):
            # Counter is a runtime tensor; no host scalar or per-step recompilation.
            step = state_steps[i]
            bc1 = 1.0 - torch.pow(beta1, step)
            bc2_sqrt = torch.sqrt(1.0 - torch.pow(beta2, step))
            step_size = lr / bc1
            inv_bc2_sqrt = 1.0 / bc2_sqrt
            grad = -grads[i] if maximize else grads[i]
            meas = max_exp_avg_sqs[i] if amsgrad else params[i]

            # Zero-numel param shard (e.g. FSDPv2 last-rank shard with shape
            # [0, N]): the Adam update has no elements to compute, and the NKI
            # kernel cannot allocate a zero-size tile ("shape must be a tuple of
            # positive integers"). Pass the buffers through unchanged so the
            # output count still lines up with the out= tuple for aliasing.
            if params[i].numel() == 0:
                results_p.append(params[i])
                results_ea.append(exp_avgs[i])
                results_eas.append(exp_avg_sqs[i])
                if amsgrad:
                    results_meas.append(max_exp_avg_sqs[i])
                continue

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
        amsgrad,
        maximize,
    ):
        """Compute fused Adam updates with elementwise MLIR operations.

        Args:
            params (list[torch.Tensor]): Parameter tensors to update.
            grads (list[torch.Tensor]): Gradients for the parameters.
            exp_avgs (list[torch.Tensor]): First-moment state tensors.
            exp_avg_sqs (list[torch.Tensor]): Second-moment state tensors.
            max_exp_avg_sqs (list[torch.Tensor]): Maximum second-moment state tensors.
            state_steps (list[torch.Tensor]): Optimizer step tensors.
            lr (torch.Tensor): Learning rate.
            beta1 (torch.Tensor): First-moment decay coefficient.
            beta2 (torch.Tensor): Second-moment decay coefficient.
            weight_decay (torch.Tensor): Weight-decay coefficient.
            eps (torch.Tensor): Denominator stability term.
            amsgrad (bool): Whether to maintain maximum second moments.
            maximize (bool): Whether to maximize the objective.

        Returns:
            tuple[torch.Tensor, ...]: Parameters, moments, and optional AMSGrad maxima in list
                order; zero-element outputs alias their corresponding inputs.
        """
        updated_params = []
        updated_exp_avgs = []
        updated_exp_avg_sqs = []
        updated_max_exp_avg_sqs = []

        for i in range(len(params)):
            # Zero-numel param shard (e.g. FSDPv2 last-rank [0, N] shard): no
            # elements to update. Pass the buffers through unchanged, matching
            # the NKI compute_fn, so the output count lines up with the out=
            # tuple for aliasing.
            if params[i].numel() == 0:
                updated_params.append(params[i])
                updated_exp_avgs.append(exp_avgs[i])
                updated_exp_avg_sqs.append(exp_avg_sqs[i])
                if amsgrad:
                    updated_max_exp_avg_sqs.append(max_exp_avg_sqs[i])
                continue

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
    # 11, 12 are amsgrad and maximize. Static so they reach the trace as Python
    # bools: amsgrad sets the number of outputs (4N vs 3N) and indexes
    # max_exp_avg_sqs, which is [] when it is False, so neither can be deferred to
    # a tensor. bool subclasses int, so a non-static one would be tensorized.
    static_args = ((7, 8, 9, 10) if _has_nki_adam_kernels else ()) + (11, 12)

    @register_aten(
        op_names,
        preprocessing=Preprocessing.SHAPE_ONLY,
        static_argnums=static_args,
        # C++ cached path. The lowering depends only on shapes, dtypes, and the
        # bool flags, all of which are in the cache key, so it replays on a hit.
        # lr is a non-static float kwarg, so PreprocessArgs makes it a runtime
        # NEFF input: one NEFF serves every learning rate (no recompile per LR
        # step). Mutated state is donated back in place via the out= tuple below.
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
        """Return fused Adam's compute function, inputs, donated buffers, and no-op postprocessor.

        Args:
            params (list[torch.Tensor]): Parameter tensors to update.
            grads (list[torch.Tensor]): Gradients for the parameters.
            exp_avgs (list[torch.Tensor]): First-moment state tensors.
            exp_avg_sqs (list[torch.Tensor]): Second-moment state tensors.
            max_exp_avg_sqs (list[torch.Tensor]): Maximum second-moment state tensors.
            state_steps (list[torch.Tensor]): Optimizer step tensors.
            lr (float or torch.Tensor, optional): Learning rate. Defaults to ``0.001``.
            beta1 (float, optional): First-moment decay. Defaults to ``0.9``.
            beta2 (float, optional): Second-moment decay. Defaults to ``0.999``.
            weight_decay (float, optional): Defaults to ``0.01`` for AdamW, else ``0.0``.
            eps (float, optional): Denominator stability term. Defaults to ``1e-8``.
            amsgrad (bool, optional): Maintain maximum second moments. Defaults to ``False``.
            maximize (bool, optional): Maximize the objective. Defaults to ``False``.
            grad_scale (torch.Tensor | None, optional): Ignored metadata. Defaults to ``None``.
            found_inf (torch.Tensor | None, optional): Ignored metadata. Defaults to ``None``.
            **kwargs (Any): Additional dispatcher keyword arguments; currently ignored.

        Returns:
            tuple: Compute function, inputs, donated state buffers, and no-op postprocessor.
        """
        compute_fn = _make_adam_compute_fn(decoupled_wd)

        if not params:
            return compute_fn, (), {}, lambda results: None

        # Keep lists full-length (do not filter zero-numel shards): the cached
        # path relies on kept-full-length inputs so cache-hit indices stay valid,
        # and the out= tuple below must line up 1:1 with the compute_fn outputs.
        if amsgrad:
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
                max_exp_avg_sqs if amsgrad else [],
                state_steps,
                lr,
                beta1,
                beta2,
                weight_decay,
                eps,
                amsgrad,
                maximize,
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

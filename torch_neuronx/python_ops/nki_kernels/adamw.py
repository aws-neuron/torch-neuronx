"""NKI op wrappers for fused Adam/AdamW optimizer."""

import torch

# TODO remove try/except once fused_adam NKI kernel is available
try:
    from nkilib.experimental.optimizer import adam_kernel, adamw_kernel

    from torch_neuronx.nki_hop import nki_op, wrap_nki

    _wrapped_adam = wrap_nki(adam_kernel)
    _wrapped_adamw = wrap_nki(adamw_kernel)

    @nki_op("nki_kernels::adam", mutates_args={})
    def adam_nki(
        param: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        max_exp_avg_sq: torch.Tensor,
        ss: torch.Tensor,
        ibc: torch.Tensor,
        wd: torch.Tensor,
        beta1: float,
        beta2: float,
        eps: float,
        amsgrad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _wrapped_adam[(2,)](
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            ss,
            ibc,
            wd,
            param.numel(),
            beta1,
            beta2,
            eps,
            amsgrad,
        )

    @nki_op("nki_kernels::adamw", mutates_args={})
    def adamw_nki(
        param: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        max_exp_avg_sq: torch.Tensor,
        ss: torch.Tensor,
        ibc: torch.Tensor,
        wd: torch.Tensor,
        beta1: float,
        beta2: float,
        eps: float,
        amsgrad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _wrapped_adamw[(2,)](
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            ss,
            ibc,
            wd,
            param.numel(),
            beta1,
            beta2,
            eps,
            amsgrad,
        )

    _has_nki_adam_kernels = True
except (ImportError, ModuleNotFoundError):
    _has_nki_adam_kernels = False

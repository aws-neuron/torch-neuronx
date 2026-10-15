"""NKI op wrappers for fused Adam/AdamW optimizer."""

import torch

# TODO remove try/except once fused_adam NKI kernel is available
try:
    from nkilib.experimental.optimizer import adam_kernel, adamw_kernel

    from torch_neuronx.nki_hop import nki_op, wrap_nki
    from torch_neuronx.utils import get_logical_neuron_cores

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
        """Apply a fused Adam optimizer update with coupled weight decay.

        Args:
            param (torch.Tensor): Parameter values to update.
            grad (torch.Tensor): Gradient values.
            exp_avg (torch.Tensor): First-moment state.
            exp_avg_sq (torch.Tensor): Second-moment state.
            max_exp_avg_sq (torch.Tensor): Maximum second-moment state or placeholder.
            ss (torch.Tensor): Step-size values expanded for the NKI kernel.
            ibc (torch.Tensor): Inverse square-root bias-correction values.
            wd (torch.Tensor): Coupled weight-decay values.
            beta1 (float): First-moment decay coefficient.
            beta2 (float): Second-moment decay coefficient.
            eps (float): Denominator stability term.
            amsgrad (bool): Whether to update the maximum second moment.

        Returns:
            tuple[torch.Tensor, ...]: Updated parameter and optimizer states, including
                maximum second moment only when ``amsgrad`` is enabled.
        """
        return _wrapped_adam[(int(get_logical_neuron_cores()),)](
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
        """Apply a fused AdamW optimizer update with decoupled weight decay.

        Args:
            param (torch.Tensor): Parameter values to update.
            grad (torch.Tensor): Gradient values.
            exp_avg (torch.Tensor): First-moment state.
            exp_avg_sq (torch.Tensor): Second-moment state.
            max_exp_avg_sq (torch.Tensor): Maximum second-moment state or placeholder.
            ss (torch.Tensor): Step-size values expanded for the NKI kernel.
            ibc (torch.Tensor): Inverse square-root bias-correction values.
            wd (torch.Tensor): Decoupled parameter-decay factors.
            beta1 (float): First-moment decay coefficient.
            beta2 (float): Second-moment decay coefficient.
            eps (float): Denominator stability term.
            amsgrad (bool): Whether to update the maximum second moment.

        Returns:
            tuple[torch.Tensor, ...]: Updated parameter and optimizer states, including
                maximum second moment only when ``amsgrad`` is enabled.
        """
        return _wrapped_adamw[(int(get_logical_neuron_cores()),)](
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

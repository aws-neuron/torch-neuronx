from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from ..op_spec_base import OpSpecBase
from ..test_registry import register_spec


@register_spec(vertical_test="CheckBackwardImplTest")
@dataclass
class NativeDropoutOpSpec(OpSpecBase):
    """torch native dropout op spec for CheckBackwardImplTest"""

    op_name: str = "native_dropout"
    args_specs: list[list[Any]] = field(
        default_factory=lambda: [
            [torch.randn(2, 5, requires_grad=True), torch.randn(2, 3, 32, 32, requires_grad=True)]
        ]
    )
    kwargs_specs: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "p": [0.1],
            "train": [True],
        }
    )
    output_indices: list[int] = field(default_factory=lambda: [0, 1])
    output_loss_index: int = 0

    def expected_neuron_op(self, *args, **kwargs) -> str | None:
        """get name of the op expected to run on neuron device"""
        return ["aten::native_dropout", "aten::native_dropout_backward"]


@register_spec(vertical_test="CheckBackwardImplTest")
@dataclass
class MMOpSpec(OpSpecBase):
    """MM op spec for CheckBackwardImplTest"""

    op_name: str = "mm"
    args_specs: list[list[Any]] = field(
        default_factory=lambda: [
            [torch.randn(3, 4, requires_grad=True, dtype=torch.bfloat16)],
            [torch.randn(4, 6, requires_grad=False, dtype=torch.bfloat16)],
        ]
    )

    def expected_neuron_op(self, *args, **kwargs) -> str | None:
        """get name of the op expected to run on neuron device"""
        return ["aten::mm"]


@register_spec(vertical_test="CheckBackwardImplTest")
@dataclass
class SiluOpSpec(OpSpecBase):
    """silu op spec for CheckBackwardImplTest"""

    op_name: str = "F.silu"
    args_specs: list[list[Any]] = field(
        default_factory=lambda: [[torch.randn(2, 3, requires_grad=True)]]
    )

    def expected_neuron_op(self, *args, **kwargs) -> str | None:
        return ["aten::silu", "aten::silu_backward"]


@register_spec(vertical_test="CheckBackwardImplTest")
@dataclass
class ScaledDotProductFusedAttentionOverrideableOpSpec(OpSpecBase):
    """scaled_dot_product_attention op spec for CheckBackwardImplTest"""

    op_name: str = "F.scaled_dot_product_attention"
    args_specs: list[list[Any]] = field(
        default_factory=lambda: [
            [torch.randn(1, 4, 512, 64, dtype=torch.bfloat16, requires_grad=True)],  # query
            [torch.randn(1, 4, 512, 64, dtype=torch.bfloat16, requires_grad=True)],  # key
            [torch.randn(1, 4, 512, 64, dtype=torch.bfloat16, requires_grad=True)],  # value
        ]
    )
    kwargs_specs: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "attn_mask": [None],
            "dropout_p": [0.0],
            "is_causal": [False],
            "scale": [None],
        }
    )

    def expected_neuron_op(self, *args, **kwargs) -> str | None:
        return ["aten::_scaled_dot_product_fused_attention_overrideable"]


@register_spec(vertical_test="CheckBackwardImplTest")
@dataclass
class NllLossForwardOpSpec(OpSpecBase):
    """nll_loss_forward op spec for CheckBackwardImplTest"""

    op_name: str = "nn.NLLLoss"
    args_specs: list[list[Any]] = field(
        default_factory=lambda: [
            [torch.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)],
            [torch.tensor([1, 0, 4])],
        ]
    )

    def expected_neuron_op(self, *args, **kwargs) -> str | None:
        return ["aten::nll_loss_forward", "aten::nll_loss_backward"]


@register_spec(vertical_test="CheckBackwardImplTest")
@dataclass
class IndexSelectOpSpec(OpSpecBase):
    """index_select op spec for CheckBackwardImplTest"""

    op_name: str = "index_select"
    args_specs: list[list[Any]] = field(
        default_factory=lambda: [
            [torch.randn(3, 4, requires_grad=True)],
            [0],
            [torch.tensor([0, 2])],
        ]
    )

    def expected_neuron_op(self, *args, **kwargs) -> str | None:
        return ["aten::index_select"]  # index_select_backward get deomposed


@register_spec(vertical_test="CheckBackwardImplTest")
@dataclass
class LogSoftmaxOpSpec(OpSpecBase):
    """_log_softmax op spec for CheckBackwardImplTest"""

    op_name: str = "_log_softmax"
    args_specs: list[list[Any]] = field(
        default_factory=lambda: [[torch.randn(2, 3, requires_grad=True)]]
    )
    kwargs_specs: dict[str, list[Any]] = field(
        default_factory=lambda: {"dim": [1], "half_to_float": [False]}
    )

    def expected_neuron_op(self, *args, **kwargs) -> str | None:
        return ["aten::_log_softmax", "aten::_log_softmax_backward_data"]


@register_spec(vertical_test="CheckBackwardImplTest")
@dataclass
class GatherOpSpec(OpSpecBase):
    """gather op spec for CheckBackwardImplTest"""

    op_name: str = "gather"
    args_specs: list[list[Any]] = field(
        default_factory=lambda: [
            [torch.randn(3, 4, requires_grad=True)],
            [1],
            [torch.tensor([[0, 1, 2, 0], [2, 0, 0, 1], [1, 2, 1, 0]], dtype=torch.int64)],
        ]
    )

    def expected_neuron_op(self, *args, **kwargs) -> str | None:
        return ["aten::gather", "aten::scatter_add"]  # gather_backward lowers to aten::scatter_add


@register_spec(vertical_test="CheckBackwardImplTest")
@dataclass
class GeluOpSpec(OpSpecBase):
    """gelu op spec for CheckBackwardImplTest"""

    op_name: str = "F.gelu"
    args_specs: list[list[Any]] = field(
        default_factory=lambda: [[torch.randn(2, 3, requires_grad=True)]]
    )
    kwargs_specs: dict[str, list[Any]] = field(default_factory=lambda: {"approximate": ["none"]})

    def expected_neuron_op(self, *args, **kwargs) -> str | None:
        return ["aten::gelu"]  #  "aten::gelu_backward" not implemented, gets decomposed


@register_spec(vertical_test="CheckBackwardImplTest")
@dataclass
class SigmoidOpSpec(OpSpecBase):
    """sigmoid op spec for CheckBackwardImplTest"""

    op_name: str = "sigmoid"
    args_specs: list[list[Any]] = field(
        default_factory=lambda: [[torch.randn(2, 3, requires_grad=True)]]
    )

    def expected_neuron_op(self, *args, **kwargs) -> str | None:
        return ["aten::sigmoid"]  # "aten::sigmoid_backward"


@register_spec(vertical_test="CheckBackwardImplTest")
@dataclass
class SoftmaxOpSpec(OpSpecBase):
    """softmax op spec for CheckBackwardImplTest"""

    op_name: str = "F.softmax"
    args_specs: list[list[Any]] = field(
        default_factory=lambda: [[torch.randn(2, 3, requires_grad=True)]]
    )
    kwargs_specs: dict[str, list[Any]] = field(default_factory=lambda: {"dim": [1]})

    def expected_neuron_op(self, *args, **kwargs) -> str | None:
        return ["aten::_softmax", "aten::_softmax_backward_data"]

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from ..op_spec_base import OpSpecBase
from ..test_registry import register_spec


@register_spec(vertical_test="ShapeTest")
@dataclass
class NativeDropoutOpSpec(OpSpecBase):
    """torch native dropout op spec for vertical shape test"""

    op_name: str = "native_dropout"
    args_specs: list[list[Any]] = field(
        default_factory=lambda: [[torch.randn(2, 5), torch.randn(2, 3, 32, 32)]]
    )
    kwargs_specs: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "p": [0.1, 0.3, 0.5, 0.7],
            "train": [True, False],
        }
    )
    output_indices: list[int] = field(default_factory=lambda: [0, 1])

    def expected_neuron_op(self, *args, **kwargs) -> str | None:
        """get name of the op expected to run on neuron device"""
        return "aten::native_dropout"


@register_spec(vertical_test="ShapeTest")
@dataclass
class FunctionalDropoutOpSpec(NativeDropoutOpSpec):
    """Functional dropout op spec for vertical shape test"""

    op_name: str = "F.dropout"
    args_specs: list[list[Any]] = field(
        default_factory=lambda: [[torch.randn(2, 5), torch.randn(2, 3, 32, 32)]]
    )
    kwargs_specs: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "p": [0.1, 0.3, 0.5, 0.7],
            "training": [True, False],
        }
    )
    output_indices: list[int] = None

    def expected_neuron_op(self, *args, **kwargs) -> str | None:
        """get name of the op expected to run on neuron device"""
        p = kwargs.get("p")
        train = kwargs.get("train")
        if p not in [0.0, 1.0] and train:
            return "aten::native_dropout"

        return None

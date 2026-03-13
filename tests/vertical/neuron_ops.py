"""Decorator for parametrizing tests across PyTorch ops for Neuron device."""

import functools
from collections.abc import Callable, Sequence
from typing import Any

import pytest
import torch

from .skip_xfail_ops import get_skip_reason, get_xfail_reason


class NeuronOps:
    """Decorator that parametrizes a test method across ops and dtypes.

    Creates individual test methods for each op x dtype combination.
    Uses PyTorch's OpInfo.sample_inputs() to generate test inputs.
    Skip/xfail is determined by the test class name from skip_xfail_ops.py.

    Example:
        @NeuronOps(neuron_op_db, dtypes=[torch.float32, torch.bfloat16])
        def test_correctness(self, op, dtype):
            for sample in op.sample_inputs("cpu", dtype):
                ...
    """

    def __init__(
        self,
        op_list: Sequence,
        dtypes: Sequence[torch.dtype] = (torch.float32,),
    ):
        self.op_list = list(op_list)
        self.default_dtypes = tuple(dtypes)
        self._test_class_name: str | None = None

    def _get_marks(self, op_name: str) -> list:
        """Get pytest marks for this op based on skip/xfail config."""
        if not self._test_class_name:
            return []
        marks = []
        skip_reason = get_skip_reason(op_name, self._test_class_name)
        if skip_reason:
            marks.append(pytest.mark.skip(reason=skip_reason))
        else:
            xfail_reason = get_xfail_reason(op_name, self._test_class_name)
            if xfail_reason:
                marks.append(pytest.mark.xfail(reason=xfail_reason))
        return marks

    def _get_dtypes(self, op) -> tuple[torch.dtype, ...]:
        """Get dtypes to test for this op."""
        supported = set(op.dtypes) if op.dtypes else set()
        return tuple(d for d in self.default_dtypes if d in supported)

    def __call__(self, test_fn: Callable) -> Callable:
        """Generate parametrized test methods."""
        # Extract test class name from the qualified name (e.g., "TestNeuronOps.test_correctness")
        qualname = test_fn.__qualname__
        if "." in qualname:
            self._test_class_name = qualname.split(".")[0]

        # Build list of (op, dtype) combinations with marks
        params = []
        for op in self.op_list:
            marks = self._get_marks(op.name)
            for dtype in self._get_dtypes(op):
                param = pytest.param(
                    op,
                    dtype,
                    id=f"{op.name}_{dtype}".replace("torch.", ""),
                    marks=marks,
                )
                params.append(param)

        @pytest.mark.parametrize("op,dtype", params)
        @functools.wraps(test_fn)
        def wrapper(self, op, dtype):
            return test_fn(self, op, dtype)

        return wrapper


def allocate_to_device(data: Any, device: str) -> Any:
    """Recursively move tensors to device."""
    if isinstance(data, torch.Tensor):
        return data.detach().clone().to(device)
    elif isinstance(data, list | tuple):
        return type(data)(allocate_to_device(item, device) for item in data)
    elif isinstance(data, dict):
        return {k: allocate_to_device(v, device) for k, v in data.items()}
    return data


def has_zero_dim_tensor(data: Any) -> bool:
    """Check if data contains any 0-dimensional tensors."""
    if isinstance(data, torch.Tensor):
        return data.dim() == 0
    elif isinstance(data, list | tuple):
        return any(has_zero_dim_tensor(item) for item in data)
    elif isinstance(data, dict):
        return any(has_zero_dim_tensor(v) for v in data.values())
    return False


def filter_zero_dim_samples(samples: list) -> list:
    """Filter out samples that contain 0-dimensional tensors.

    Neuron doesn't support 0-d tensors for many ops, so we skip these edge cases.
    """
    return [
        s
        for s in samples
        if not has_zero_dim_tensor(s.input)
        and not has_zero_dim_tensor(s.args)
        and not has_zero_dim_tensor(s.kwargs)
    ]


def has_empty_tensor(data: Any) -> bool:
    """Check if data contains any empty tensors (numel == 0)."""
    if isinstance(data, torch.Tensor):
        return data.numel() == 0
    elif isinstance(data, list | tuple):
        return any(has_empty_tensor(item) for item in data)
    elif isinstance(data, dict):
        return any(has_empty_tensor(v) for v in data.values())
    return False


def filter_empty_samples(samples: list) -> list:
    """Filter out samples that contain empty tensors (numel == 0)."""
    return [
        s
        for s in samples
        if not has_empty_tensor(s.input)
        and not has_empty_tensor(s.args)
        and not has_empty_tensor(s.kwargs)
    ]

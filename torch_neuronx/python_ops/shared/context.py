"""Execution context passed alongside inputs during op execution.

Provides a typed container for auxiliary information that should not be
treated as program inputs (i.e., not part of the compiled signature or
cache key). This avoids smuggling hidden kwargs and keeps interfaces clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionContext:
    """Context for a single op execution.

    - original_inputs: The original ATen inputs before preprocessing. Used for
      tasks like meta dtype inference where the ATen schema matters.
    - original_kwargs: The original kwargs before dtype casting. Used for
      inferring original output dtypes when dtype kwarg is present.
    - expected_dtypes: Optionally, precomputed expected output dtypes.
    """

    original_inputs: tuple[Any, ...] | None = None
    original_kwargs: dict[str, Any] | None = None
    expected_dtypes: list[Any] | None = None

    def has_original_inputs(self) -> bool:
        return self.original_inputs is not None and len(self.original_inputs) > 0

    def has_original_kwargs(self) -> bool:
        return self.original_kwargs is not None

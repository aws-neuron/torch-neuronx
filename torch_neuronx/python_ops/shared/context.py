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
    """Carry auxiliary state for a single operation execution.

    Args:
        original_inputs (tuple[Any, ...], optional): Original ATen inputs before
            preprocessing. Defaults to ``None``.
        original_kwargs (dict[str, Any], optional): Original keyword arguments before
            dtype casting. Defaults to ``None``.
        expected_dtypes (list[Any], optional): Reserved output dtypes that are stored
            but not consumed. Defaults to ``None``.

    Note:
        The context remains separate from program inputs and the compiled signature.
    """

    original_inputs: tuple[Any, ...] | None = None
    original_kwargs: dict[str, Any] | None = None
    expected_dtypes: list[Any] | None = None

    def has_original_inputs(self) -> bool:
        """Determine whether nonempty original inputs are available.

        Returns:
            bool: ``True`` when ``original_inputs`` contains at least one item.
        """
        return self.original_inputs is not None and len(self.original_inputs) > 0

    def has_original_kwargs(self) -> bool:
        """Determine whether original keyword arguments are available.

        Returns:
            bool: ``True`` when ``original_kwargs`` is not ``None``.
        """
        return self.original_kwargs is not None

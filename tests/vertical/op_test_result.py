from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class OpTestResult:
    """Represents the result of a single op spec instance execution.

    Attributes:
        test_name: Name of the test class that executed this test
        op_spec_name: Name of the op spec class used
        op_name: Op name
        args_shapes: Shapes of the input arguments to the op
        args_dtypes: Data types of the input arguments to the op
        kwargs_spec: Input keyword arguments to the op
        passed: Whether the test passed (True), failed (False), or is undetermined (None)
    """

    test_name: str
    op_spec_name: str
    op_name: str
    args_shapes: list[list[Any]] | None = None
    args_dtypes: list[list[Any]] | None = None
    kwargs_spec: dict[str, list[Any]] | None = None
    passed: bool | None = None

    def __str__(self):
        green = "\033[92m"
        red = "\033[91m"
        status = f"{green}✓" if self.passed else f"{red}✗"
        return (
            f"{status} {self.test_name}::{self.op_spec_name}::{self.op_name} | "
            f"Shapes: {self.args_shapes} | Types: {self.args_dtypes} | "
            f"Kwargs: {self.kwargs_spec}"
        )

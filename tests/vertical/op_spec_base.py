import itertools
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class OpSpecBase:
    """Base class for op spec.

    This class defines the structure for defininf op spec to be tested for a
    particular vertical test. The spec provides op arguments and keyword arguments,
    and logic on how to generate possible combinations for testing.

    Attributes:
        op_name: Name of the op to be tested
        args_specs: Optional list of lists, where each inner list contains
                   possible values for a positional argument
        kwargs_specs: Optional dictionary mapping keyword argument names
                     to lists of possible values
        input_indices: list of indices to specify which inputs
                       to validate for operations
        output_indices: Optional list of indices to specify which outputs
                       to validate for operations that return multiple values
        output_loss_index: Optional index to specify which of the outputs
                       to use for calcualting the loss.
        use_zip: Enable generating zip combinations instead of cartesian.

    """

    op_name: str
    args_specs: list[list[Any]] | None = None
    kwargs_specs: dict[str, list[Any]] | None = None
    input_indices: list[int] = field(default_factory=lambda: [0])
    output_indices: list[int] | None = None
    output_loss_index: list[int] | None = None
    use_zip: bool = False

    def __post_init__(self) -> None:
        """Validate the spec after initialization.

        Raises:
            ValueError: If the configuration is invalid
        """
        if not self.op_name:
            raise ValueError("op_name cannot be empty")

        if self.args_specs is not None:
            if not isinstance(self.args_specs, list):
                raise ValueError("args_specs must be a list or None")
            if not all(isinstance(spec, list) for spec in self.args_specs):
                raise ValueError("Each element in args_specs must be a list")

        if self.kwargs_specs is not None:
            if not isinstance(self.kwargs_specs, dict):
                raise ValueError("kwargs_specs must be a dict or None")
            if not all(isinstance(v, list) for v in self.kwargs_specs.values()):
                raise ValueError("Each value in kwargs_specs must be a list")

        if self.output_indices is not None:
            if not isinstance(self.output_indices, list):
                raise ValueError("output_indices must be a list or None")
            if not all(isinstance(idx, int) and idx >= 0 for idx in self.output_indices):
                raise ValueError("All output_indices must be non-negative integers")

    def generate_combinations(self) -> Iterator[tuple[tuple[Any, ...], dict[str, Any]]]:
        """Generate all combinations of positional and keyword arguments.

        Creates the cartesian product of all possible argument values specified
        in args_specs and kwargs_specs to generate comprehensive test cases.

        Returns:
            Iterator yielding tuples of (args, kwargs) where:
            - args: Tuple of positional arguments
            - kwargs: Dictionary of keyword arguments

        Yields:
            Tuple[Tuple[Any, ...], Dict[str, Any]]: Each combination of
            arguments as (args_tuple, kwargs_dict)

        Example:
            >>> spec = OpSpecBase(
            ...     op_name="add",
            ...     args_specs=[[1, 2], [3, 4]],
            ...     kwargs_specs={"dtype": [int, float]}
            ... )
            >>> list(spec.generate_combinations())
            [((1, 3), {'dtype': int}), ((1, 3), {'dtype': float}),
             ((1, 4), {'dtype': int}), ((1, 4), {'dtype': float}),
             ((2, 3), {'dtype': int}), ((2, 3), {'dtype': float}),
             ((2, 4), {'dtype': int}), ((2, 4), {'dtype': float})]
        """

        if self.use_zip:
            args_combinations = list(zip(*self.args_specs, strict=False))
        else:
            args_combinations = itertools.product(*self.args_specs) if self.args_specs else [()]

        if self.kwargs_specs:
            kwarg_keys = list(self.kwargs_specs.keys())
            kwarg_values = [self.kwargs_specs[key] for key in kwarg_keys]
            if self.use_zip:
                kwarg_combinations = list(zip(*kwarg_values, strict=False))
            else:
                kwarg_combinations = list(itertools.product(*kwarg_values))
        else:
            kwarg_keys = []
            kwarg_combinations = [()]  # Single empty tuple for no kwargs

        if self.use_zip:
            if not self.kwargs_specs:
                kwarg_combinations = [()] * len(args_combinations)
                kwarg_keys = [] * len(args_combinations)

            for args, kwarg_vals in zip(args_combinations, kwarg_combinations, strict=False):
                kwargs = dict(zip(kwarg_keys, kwarg_vals, strict=False))
                yield args, kwargs
        else:
            for args in args_combinations:
                for kwarg_vals in kwarg_combinations:
                    kwargs = dict(zip(kwarg_keys, kwarg_vals, strict=False))
                    yield args, kwargs

    def expected_neuron_op(self, *args, **kwargs) -> str | None:
        """get name of the op expected to run on neuron device"""
        return self.op_name

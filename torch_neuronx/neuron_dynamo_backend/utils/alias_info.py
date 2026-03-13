"""
Aliasing information data structures for StableHLO input-output buffer optimization.

This module defines data structures that track relationships between function inputs
and outputs for buffer aliasing optimization. When a function output is derived from
mutating an input tensor, the compiler can reuse the input buffer for the output,
avoiding unnecessary memory allocation.

These structures are used to set the `mhlo.input_output_alias` attribute on StableHLO
modules, enabling the Neuron compiler to optimize memory usage.
"""

from dataclasses import dataclass, field


@dataclass
class AliasInfo:
    """Single input-output aliasing relationship for StableHLO buffer optimization.

    Represents a mapping where a function output aliases (shares memory with) a
    function input. Used to set the mhlo.input_output_alias attribute.

    Attributes:
        parameter_number (int): Index of the input parameter in the function signature
            (0-indexed). This is the input whose buffer will be reused.
        parameter_index (list[int] | None): Index path into nested tuple parameters.
            Empty list [] for flat parameters, None if not applicable.
        output_index (int): Index of the output in the function's return tuple
            (0-indexed) that aliases the specified input parameter.
    """

    parameter_number: int
    parameter_index: list[int] | None
    output_index: int


@dataclass
class AliasingInfo:
    aliases: list[AliasInfo] = field(default_factory=list)
    output_to_input: dict[int, int] = field(default_factory=dict, repr=False)

    def get_input_index(self, output_index: int) -> int | None:
        """Get input index for an aliased output, or None if not aliased."""
        return self.output_to_input.get(output_index)

    def add(
        self, parameter_number: int, parameter_index: list[int] | None, output_index: int
    ) -> None:
        """Add an aliasing relationship to the collection.

        Args:
            parameter_number (int): Index of the input parameter.
            parameter_index (list[int] | None): Index path for nested parameters.
            output_index (int): Index of the aliased output.
        """
        if output_index in self.output_to_input:
            existing_input = self.output_to_input[output_index]
            if existing_input != parameter_number:
                raise ValueError(
                    f"Output {output_index} already aliases input {existing_input}, "
                    f"cannot also alias input {parameter_number}."
                )

        self.aliases.append(AliasInfo(parameter_number, parameter_index, output_index))
        self.output_to_input[output_index] = parameter_number

    def __iter__(self):
        """Iterate over aliasing relationships.

        Returns:
            Iterator[AliasInfo]: Iterator over alias entries.
        """
        return iter(self.aliases)

    def __len__(self):
        """Return the number of aliasing relationships.

        Returns:
            int: Number of alias entries.
        """
        return len(self.aliases)

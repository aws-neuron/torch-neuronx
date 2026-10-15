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
    """Represent one StableHLO input-output buffer alias.

    Args:
        parameter_number (int): Zero-based input whose buffer is reused.
        parameter_index (list[int] | None): Index path into nested tuple parameters.
            Use an empty list for a flat parameter or ``None`` when inapplicable.
        output_index (int): Zero-based output that aliases the input.

    Note:
        This record supplies the ``mhlo.input_output_alias`` attribute used for
        StableHLO buffer optimization.
    """

    parameter_number: int
    parameter_index: list[int] | None
    output_index: int


@dataclass
class AliasingInfo:
    """Collect StableHLO input-output buffer aliases.

    Args:
        aliases (list[AliasInfo]): Alias records in insertion order. Defaults to
            an empty list.
        output_to_input (dict[int, int]): Output-to-input lookup table. Defaults
            to an empty dictionary.
    """

    aliases: list[AliasInfo] = field(default_factory=list)
    output_to_input: dict[int, int] = field(default_factory=dict, repr=False)

    def get_input_index(self, output_index: int) -> int | None:
        """Get the input index aliased by an output.

        Args:
            output_index (int): Output index to query.

        Returns:
            int | None: Aliased input index, or ``None`` if no alias exists.
        """
        return self.output_to_input.get(output_index)

    def add(
        self, parameter_number: int, parameter_index: list[int] | None, output_index: int
    ) -> None:
        """Add an input-output aliasing relationship.

        Args:
            parameter_number (int): Input parameter index.
            parameter_index (list[int] | None): Index path for a nested parameter.
            output_index (int): Aliased output index.

        Raises:
            ValueError: If the output already aliases a different input.

        Note:
            Re-adding an output mapped to the same parameter number has no effect; a
            different ``parameter_index`` for that mapping is ignored.
        """
        if output_index in self.output_to_input:
            existing_input = self.output_to_input[output_index]
            if existing_input != parameter_number:
                raise ValueError(
                    f"Output {output_index} already aliases input {existing_input}, "
                    f"cannot also alias input {parameter_number}."
                )
        else:
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

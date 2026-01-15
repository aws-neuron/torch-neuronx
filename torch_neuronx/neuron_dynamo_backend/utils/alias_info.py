from dataclasses import dataclass, field


@dataclass
class AliasInfo:
    parameter_number: int
    parameter_index: list[int] | None
    output_index: int


@dataclass
class AliasingInfo:
    aliases: list[AliasInfo] = field(default_factory=list)

    def add(
        self, parameter_number: int, parameter_index: list[int] | None, output_index: int
    ) -> None:
        self.aliases.append(AliasInfo(parameter_number, parameter_index, output_index))

    def __iter__(self):
        return iter(self.aliases)

    def __len__(self):
        return len(self.aliases)

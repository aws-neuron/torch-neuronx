"""Build ordered shards for one target."""

from __future__ import annotations

from typing import NamedTuple


class Shard(NamedTuple):
    index: int
    members: tuple[str, ...]

    @property
    def label(self) -> str:
        return shard_label(self.members)


def shard_label(names: tuple[str, ...]) -> str:
    """Return the job label for a shard."""
    first_name = names[0]
    return first_name if len(names) == 1 else f"{first_name} +{len(names) - 1}"


def build_shards(
    groups: tuple[tuple[str, ...], ...],
) -> tuple[Shard, ...]:
    """Return the configured member groups as ordered shards."""
    return tuple(
        Shard(index=shard_index, members=group) for shard_index, group in enumerate(groups)
    )

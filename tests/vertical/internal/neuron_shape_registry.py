"""Op-specific shape registry backed by a consolidated CSV.

CSV format:
    op_name,arg_shapes,source_model,count
    aten::linear,"{""input"":[1,2048,4096],""weight"":[11008,4096],""bias"":null}",llama-7b,47

Deduplicates on (op_name, arg_shapes). Merges counts across models.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OpShapeEntry:
    """A unique (op_name, arg_shapes) pair with metadata."""

    op_name: str
    arg_shapes: dict[str, tuple[int, ...] | None]
    sources: dict[str, int] = field(default_factory=dict)  # model → count

    @property
    def total_count(self) -> int:
        return sum(self.sources.values())

    @property
    def shapes_key(self) -> str:
        """Stable JSON key for dedup."""
        return json.dumps(self.arg_shapes, sort_keys=True)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.arg_shapes.items())
        return f"OpShapeEntry({self.op_name}, {args}, count={self.total_count})"


class NeuronShapeRegistry:
    """Deduplicated registry of op shape combinations.

    Usage:
        # Register from model trace
        SHAPE_REGISTRY.register("aten::linear",
            arg_shapes={"input": (1, 2048, 4096), "weight": (11008, 4096), "bias": None},
            source="llama-7b", count=47)

        # Query
        for entry in SHAPE_REGISTRY.get("aten::linear"):
            sample = op.make_sample(entry.arg_shapes, device, dtype)

    """

    def __init__(self) -> None:
        # {op_name: {shapes_key: OpShapeEntry}}
        self._entries: dict[str, dict[str, OpShapeEntry]] = defaultdict(dict)

    def register(
        self,
        op_name: str,
        arg_shapes: dict[str, tuple[int, ...] | list[int] | None],
        source: str = "manual",
        count: int = 1,
    ) -> None:
        """Register a shape combination. Deduplicates and merges counts."""
        normalized = {
            k: tuple(v) if isinstance(v, list | tuple) else v for k, v in arg_shapes.items()
        }
        entry = OpShapeEntry(op_name=op_name, arg_shapes=normalized)
        key = entry.shapes_key

        if key in self._entries[op_name]:
            existing = self._entries[op_name][key]
            existing.sources[source] = existing.sources.get(source, 0) + count
        else:
            entry.sources[source] = count
            self._entries[op_name][key] = entry

    def get(self, op_name: str) -> list[OpShapeEntry]:
        """Get all shape entries for an op, sorted by total count (most common first)."""
        return sorted(self._entries.get(op_name, {}).values(), key=lambda e: -e.total_count)

    def has(self, op_name: str) -> bool:
        return bool(self._entries.get(op_name))

    def all_ops(self) -> list[str]:
        return sorted(self._entries.keys())

    def __len__(self) -> int:
        return sum(len(v) for v in self._entries.values())

    def load_json(self, path: str | Path, source: str | None = None) -> int:
        """Load from model trace JSON. Source defaults to filename."""
        if source is None:
            source = Path(path).stem
        n = 0
        with open(path) as f:
            for entry in json.load(f):
                self.register(
                    op_name=entry["op_name"],
                    arg_shapes=entry["arg_shapes"],
                    source=source,
                    count=entry.get("count", 1),
                )
                n += 1
        return n

    def load_csv(self, path: str | Path) -> int:
        """Load from consolidated CSV. Returns number of entries loaded."""
        n = 0
        with open(path) as f:
            for row in csv.DictReader(f):
                self.register(
                    op_name=row["op_name"],
                    arg_shapes=json.loads(row["arg_shapes"]),
                    source=row.get("source_model", "csv"),
                    count=int(row.get("count", 1)),
                )
                n += 1
        return n

    def save_csv(self, path: str | Path) -> int:
        """Save to consolidated CSV. Returns number of rows written."""
        n = 0
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["op_name", "arg_shapes", "source_model", "count"])
            for op_name in sorted(self._entries):
                for entry in sorted(self._entries[op_name].values(), key=lambda e: -e.total_count):
                    for source, count in sorted(entry.sources.items()):
                        writer.writerow([op_name, entry.shapes_key, source, count])
                        n += 1
        return n

    def stats(self) -> dict:
        """Summary stats."""
        return {
            "total_ops": len(self._entries),
            "total_shapes": len(self),
            "top_ops": sorted(
                ((op, len(entries)) for op, entries in self._entries.items()),
                key=lambda x: -x[1],
            )[:10],
        }


SHAPE_REGISTRY = NeuronShapeRegistry()


# Load all shape defaults (hand-crafted + auto-generated)
def _load_defaults():
    from .neuron_shape_defaults_generated import register_all_defaults

    register_all_defaults(SHAPE_REGISTRY.register)


_load_defaults()

# Load model-traced shapes from env var (set by --shape-registry pytest option).
# This ensures xdist workers also load the shapes.
_shape_registry_path = os.environ.get("NEURON_SHAPE_REGISTRY")
if _shape_registry_path and os.path.exists(_shape_registry_path):
    SHAPE_REGISTRY.load_json(_shape_registry_path, source="model_trace")

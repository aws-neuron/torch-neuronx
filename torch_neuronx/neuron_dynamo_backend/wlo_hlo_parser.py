"""
Parse wrapped_neff.hlo to extract WLO weight transforms.

This runs as a subprocess to avoid protobuf descriptor conflicts
with the main torch process.

Usage:
    python wlo_hlo_parser.py /path/to/wrapped_neff.hlo

Output (stdout):
    JSON list of transforms:
    [{"input_index": 0, "reshape": [128, 2, 128], "transpose": [2, 1, 0]}, ...]
"""

import json
import sys

from neuronxcc.thirdparty_libs.xla.service import hlo_pb2

if __package__:
    from torch_neuronx.neuron_dynamo_backend.wlo_contract import validate_transform
else:
    from wlo_contract import validate_transform


def parse_transforms(hlo_path: str) -> list[dict]:
    hlo = hlo_pb2.HloModuleProto()
    with open(hlo_path, "rb") as f:
        hlo.ParseFromString(f.read())

    entry = next(
        (c for c in hlo.computations if c.name == hlo.entry_computation_name),
        None,
    )
    if not entry:
        return []

    transforms = []
    for comp in hlo.computations:
        if not comp.name.endswith("_transpose"):
            continue

        weight_idx = None
        for inst in entry.instructions:
            if inst.opcode != "call":
                continue
            if not inst.frontend_attributes or not inst.frontend_attributes.map:
                continue
            fa = dict(inst.frontend_attributes.map)
            if "transposable_weight_idx" not in fa:
                continue
            for cid in inst.called_computation_ids:
                if cid == comp.id:
                    weight_idx = int(fa["transposable_weight_idx"])
                    break
            if weight_idx is not None:
                break

        if weight_idx is None:
            continue

        reshapes = [i for i in comp.instructions if i.opcode == "reshape"]
        transposes = [i for i in comp.instructions if i.opcode == "transpose"]
        if len(transposes) != 1 or len(reshapes) < 1:
            raise RuntimeError(
                f"WLO: unexpected transform structure for weight {weight_idx} in "
                f"'{comp.name}': expected a leading reshape and exactly one transpose, got "
                f"{len(reshapes)} reshape(s) and {len(transposes)} transpose(s)"
            )

        transform = {
            "input_index": weight_idx,
            "reshape": list(reshapes[0].shape.dimensions),
            "transpose": list(transposes[0].dimensions),
        }
        validate_transform(transform)
        transforms.append(transform)

    return transforms


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: wlo_hlo_parser.py <path_to_wrapped_neff.hlo>", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(parse_transforms(sys.argv[1])))

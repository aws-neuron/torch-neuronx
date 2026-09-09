#!/usr/bin/env python3
"""Refresh the committed cache-key shape registry from the S3 cache-key feed.

The model test suite runs the HuggingFace and torchtitan models in CI. Each job
writes its Neuron compiler cache keys to a daily S3 folder. Every cache key
records the op name, the argument shapes, and the dtypes. This tool downloads a
day's cache keys, maps them to op schema arg names, caps the shapes per op, and
regenerates the ``register_cache_key_shapes`` function in
``tests/vertical/internal/neuron_shape_defaults_generated.py``. The vertical
suite calls that function by default (opt out with ``--no-cache-key-shapes`` in
tests/vertical/conftest.py).

The shapes are Python, not a raw JSON file. The tool rewrites only the block
between the two sentinel comments in the generated file, so the rest of the file
stays intact.

This tool is OFFLINE. Do not fetch S3 at test time. CI has no bucket credentials
and the daily folder is not deterministic. Run this tool by hand (or from a
scheduled job), review the diff, and commit the change.

The S3 location has no default because the bucket moves. Pass it every time with
--bucket / --prefix, or set CACHE_KEY_SHAPES_BUCKET / CACHE_KEY_SHAPES_PREFIX.

Usage:
    # Refresh from the latest available day (needs S3 credentials).
    python tests/vertical/tools/refresh_cache_key_shapes.py \
        --bucket <bucket> --prefix <prefix> --profile <aws-profile>

    # Refresh from a specific day.
    python tests/vertical/tools/refresh_cache_key_shapes.py \
        --bucket <bucket> --prefix <prefix> --date 20260904 --profile <aws-profile>

    # Rebuild from a local folder of *_cache_keys.txt files (no S3).
    python tests/vertical/tools/refresh_cache_key_shapes.py --from-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from math import isfinite, prod
from pathlib import Path

# Make ``tests`` importable when this script runs as a file, not a module.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.vertical.internal.generate_op_shape_registry import (  # noqa: E402
    _annotate,
    _build_arg_shapes,
    _get_schema,
    collect_from_cache_keys_file,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# S3 location of the daily cache-key folders. No hard-coded default: the bucket
# moves, so a baked value goes stale. Pass --bucket/--prefix or set the env vars.
DEFAULT_BUCKET = os.environ.get("CACHE_KEY_SHAPES_BUCKET")
DEFAULT_PREFIX = os.environ.get("CACHE_KEY_SHAPES_PREFIX")

# Generated defaults file that holds register_cache_key_shapes(), relative to the
# repo root. The tool rewrites only the block between the sentinel comments.
DEFAULT_OUTPUT = _REPO_ROOT / "tests/vertical/internal/neuron_shape_defaults_generated.py"

# Sentinels that bracket the generated block in the defaults file. The tool
# replaces the text between them (inclusive). Keep them in sync with the file.
SENTINEL_BEGIN = "# BEGIN GENERATED CACHE-KEY SHAPES"
SENTINEL_END = "# END GENERATED CACHE-KEY SHAPES"

# Source tag every cache-key entry carries. Mirrors the loader in
# neuron_shape_registry.py, which vends model=None (all sources) for coverage.
CACHE_KEY_SOURCE = "cache_keys"

# Cap per op. A model-traced registry can carry hundreds of shapes per op; keep
# the top-N by max tensor element count so the committed file stays reviewable
# and under the repo large-file gate. The pytest run caps again at test time
# (--max-shapes-per-op), so this value only bounds the committed file.
DEFAULT_MAX_SHAPES_PER_OP = 6

# Cap the length of a TensorList arg. Optimizer step ops (_fused_adamw_,
# _foreach_*) record the full model parameter list -- hundreds of tensors. For
# op coverage a representative list is enough, and the full list balloons the
# committed file. Keep the first N tensors of any list arg.
DEFAULT_MAX_TENSORLIST = 6


def _truncate_tensorlists(arg_shapes: dict, limit: int) -> dict:
    """Keep at most ``limit`` tensors in each TensorList arg."""
    if limit <= 0:
        return arg_shapes
    out = {}
    for k, v in arg_shapes.items():
        if isinstance(v, list) and len(v) > limit:
            out[k] = v[:limit]
        else:
            out[k] = v
    return out


def _aws(args: list[str], profile: str | None) -> None:
    """Run an ``aws`` command. Add ``--profile`` when set."""
    cmd = ["aws", *args]
    if profile:
        cmd += ["--profile", profile]
    subprocess.run(cmd, check=True)


def _latest_date(bucket: str, prefix: str, profile: str | None) -> str:
    """Return the most recent YYYYMMDD folder under the prefix."""
    cmd = ["aws", "s3", "ls", f"s3://{bucket}/{prefix}/"]
    if profile:
        cmd += ["--profile", profile]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    days = sorted(
        line.split()[-1].rstrip("/")
        for line in out.splitlines()
        if line.strip().endswith("/") and line.split()[-1].rstrip("/").isdigit()
    )
    if not days:
        raise SystemExit(f"No date folders found under s3://{bucket}/{prefix}/")
    return days[-1]


def _download_day(bucket: str, prefix: str, date: str, dest: Path, profile: str | None) -> Path:
    """Download one day's *_cache_keys.txt files into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    src = f"s3://{bucket}/{prefix}/{date}/"
    logger.info("Download %s -> %s", src, dest)
    _aws(
        [
            "s3",
            "cp",
            src,
            str(dest),
            "--recursive",
            "--exclude",
            "*",
            "--include",
            "*_cache_keys.txt",
        ],
        profile,
    )
    return dest


def _max_elements(arg_shapes: dict) -> int:
    """Largest tensor element count in an arg_shapes dict. Used to rank shapes."""
    best = 0
    for val in arg_shapes.values():
        specs = val if isinstance(val, list) else [val]
        for spec in specs:
            if isinstance(spec, dict) and isinstance(spec.get("shape"), list):
                shape = spec["shape"]
                best = max(best, prod(shape) if shape else 1)
    return best


def build_from_files(paths: list[Path], max_per_op: int, max_tensorlist: int) -> list[dict]:
    """Union the cache-key files into a capped, deduplicated registry.

    Counts how many times each unique (op, arg_shapes) appears and records the
    source job configs, so the registry keeps the "most common first" order.
    """
    # key -> aggregate entry
    agg: dict[tuple[str, str], dict] = {}
    for path in sorted(paths):
        source = path.stem.replace("_cache_keys", "")
        for parsed in collect_from_cache_keys_file(str(path)):
            # Skip ops whose schema does not resolve. Without a schema the parser
            # falls back to positional arg0/arg1 names that never match a target
            # op, so the entry is dead weight. Commit only mappable shapes.
            if _get_schema(parsed.op_name) is None:
                continue
            arg_shapes = _annotate(parsed.op_name, _build_arg_shapes(parsed.op_name, parsed))
            arg_shapes = _truncate_tensorlists(arg_shapes, max_tensorlist)
            key = (parsed.op_name, json.dumps(arg_shapes, sort_keys=True, default=str))
            slot = agg.get(key)
            if slot is None:
                agg[key] = {
                    "op_name": parsed.op_name,
                    "arg_shapes": arg_shapes,
                    "count": 1,
                    "capture_paths": {source},
                }
            else:
                slot["count"] += 1
                slot["capture_paths"].add(source)

    by_op: dict[str, list[dict]] = defaultdict(list)
    for slot in agg.values():
        by_op[slot["op_name"]].append(slot)

    registry: list[dict] = []
    for op_name in sorted(by_op):
        entries = by_op[op_name]
        entries.sort(key=lambda e: (-_max_elements(e["arg_shapes"]), -e["count"]))
        if max_per_op > 0:
            entries = entries[:max_per_op]
        for e in entries:
            registry.append(
                {
                    "op_name": e["op_name"],
                    "arg_shapes": e["arg_shapes"],
                    "count": e["count"],
                    "capture_paths": sorted(e["capture_paths"]),
                }
            )
    return registry


def _py_literal(obj) -> str:
    """Render a JSON-derived value as a deterministic Python literal.

    Sorts dict keys so a regenerated block diffs cleanly. Handles the value
    types the cache keys produce: dict, list, str, int, float, bool, None.
    """
    if isinstance(obj, dict):
        return "{" + ", ".join(f"{k!r}: {_py_literal(v)}" for k, v in sorted(obj.items())) + "}"
    if isinstance(obj, list):
        return "[" + ", ".join(_py_literal(v) for v in obj) + "]"
    if isinstance(obj, tuple):
        return "(" + ", ".join(_py_literal(v) for v in obj) + ")"
    # repr() renders inf/-inf/nan as bare names that do not parse. Emit a call.
    if isinstance(obj, float) and not isfinite(obj):
        return f'float("{obj}")'
    return repr(obj)


def render_block(registry: list[dict]) -> str:
    """Render the sentinel-bracketed register_cache_key_shapes block.

    ruff-format reformats the long literals after the tool writes the file, so
    the tool emits one call per line and lets the formatter wrap them.
    """
    ops = len({e["op_name"] for e in registry})
    lines = [
        SENTINEL_BEGIN,
        "# Do not edit by hand. Regenerate with tests/vertical/tools/refresh_cache_key_shapes.py.",
        f'CACHE_KEY_SOURCE = "{CACHE_KEY_SOURCE}"',
        "",
        "",
        "def register_cache_key_shapes(r):",
        f'    """Register cache-key model shapes ({ops} ops, {len(registry)} shapes).',
        "",
        "    These come from the HuggingFace and torchtitan model runs in CI. Each model job",
        "    writes its Neuron compiler cache keys; each key records an op, its argument shapes,",
        "    and dtypes. refresh_cache_key_shapes.py maps those to op schema arg names, caps the",
        "    shapes per op, and regenerates this function. The vertical suite calls it by default",
        "    (opt out with --no-cache-key-shapes).",
        '    """',
        "",
        "    def t(op, arg_shapes, count=1, paths=()):",
        "        r(op, arg_shapes, source=CACHE_KEY_SOURCE, count=count, paths=paths)",
        "",
    ]
    if not registry:
        lines.append("    pass")
    for e in registry:
        lines.append(
            f"    t({e['op_name']!r}, {_py_literal(e['arg_shapes'])}, "
            f"count={e['count']}, paths={tuple(e['capture_paths'])!r})"
        )
    lines.append(SENTINEL_END)
    return "\n".join(lines) + "\n"


def splice_block(text: str, block: str) -> str:
    """Replace the sentinel-bracketed block in ``text``, or append it."""
    if SENTINEL_BEGIN in text and SENTINEL_END in text:
        head = text[: text.index(SENTINEL_BEGIN)]
        tail = text[text.index(SENTINEL_END) + len(SENTINEL_END) :]
        return head + block.rstrip("\n") + tail
    return text.rstrip("\n") + "\n\n\n" + block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--bucket", default=DEFAULT_BUCKET, help="S3 bucket (or CACHE_KEY_SHAPES_BUCKET)"
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Key prefix of the daily folders (or CACHE_KEY_SHAPES_PREFIX)",
    )
    parser.add_argument("--date", help="YYYYMMDD folder. Default: the most recent folder")
    parser.add_argument("--profile", help="AWS profile with read access to the bucket")
    parser.add_argument(
        "--from-dir",
        help="Skip S3. Build from a local folder of *_cache_keys.txt files",
    )
    parser.add_argument(
        "--max-shapes-per-op",
        type=int,
        default=DEFAULT_MAX_SHAPES_PER_OP,
        help="Cap per op by max tensor element count. Pass 0 to disable.",
    )
    parser.add_argument(
        "--max-tensorlist",
        type=int,
        default=DEFAULT_MAX_TENSORLIST,
        help="Cap the length of a TensorList arg. Pass 0 to disable.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Generated defaults file to rewrite the cache-key block in.",
    )
    args = parser.parse_args()

    if args.from_dir:
        src_dir = Path(args.from_dir)
        paths = sorted(src_dir.glob("*_cache_keys.txt"))
        if not paths:
            raise SystemExit(f"No *_cache_keys.txt files in {src_dir}")
        registry = build_from_files(paths, args.max_shapes_per_op, args.max_tensorlist)
    else:
        if not args.bucket or not args.prefix:
            raise SystemExit(
                "No S3 location. Pass --bucket and --prefix, or set "
                "CACHE_KEY_SHAPES_BUCKET / CACHE_KEY_SHAPES_PREFIX. Use --from-dir for a local run."
            )
        date = args.date or _latest_date(args.bucket, args.prefix, args.profile)
        logger.info("Use cache keys for %s", date)
        with tempfile.TemporaryDirectory() as tmp:
            dest = _download_day(args.bucket, args.prefix, date, Path(tmp), args.profile)
            paths = sorted(dest.glob("*_cache_keys.txt"))
            if not paths:
                raise SystemExit(f"No *_cache_keys.txt files for {date}")
            registry = build_from_files(paths, args.max_shapes_per_op, args.max_tensorlist)

    output = Path(args.output)
    text = output.read_text()
    if SENTINEL_BEGIN not in text:
        logger.warning(
            "Sentinel %r not found in %s; appending a new block at the end.",
            SENTINEL_BEGIN,
            output,
        )
    output.write_text(splice_block(text, render_block(registry)))

    ops = len({e["op_name"] for e in registry})
    logger.info("Wrote %s: %d ops, %d shapes", output, ops, len(registry))


if __name__ == "__main__":
    main()

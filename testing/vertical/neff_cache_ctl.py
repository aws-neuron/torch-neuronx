#!/usr/bin/env python3
"""Hydrate or upload the vertical suite's persistent NEFF cache.

Thin CLI around the upstream ``PersistentNeffCache`` module. The vertical
workflow runs ``hydrate`` before pytest and ``upload`` after, so every vertical
run reads the shared NEFF cache and the nightly seeding run writes it back. This
cuts the cold-compile time the same way the upstream suite does.

The cache is credential-free and fail-open: a missing storage label, a missing
env var, or an S3 error is logged, and the run compiles cold. Caching never
blocks or fails a run.

The compiler reads the hydrated dir automatically. ``conftest`` sets
``TORCH_NEURONX_NEFF_CACHE_DIR`` to the same shared dir and enables
HLO-derived keying, so the pytest workers read and write the hydrated cache.

Usage:
    python testing/vertical/neff_cache_ctl.py hydrate
    python testing/vertical/neff_cache_ctl.py upload   # main branch only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The cache feature lives with the upstream runner. Reuse it verbatim, the same
# way the upstream runner imports it. The runner dir is not a package, so add it
# to sys.path first.
_RUNNER_DIR = Path(__file__).resolve().parents[2] / "tests" / "pytorch_tests_native" / "runner"
sys.path.insert(0, str(_RUNNER_DIR))
from persistent_cache import PersistentNeffCache  # noqa: E402

# One fixed S3 prefix for every vertical run. The nightly seeding run writes it
# and the gate, dep-guard, and perf runs read it. A stable name keeps the read
# and the write aligned. The eager and compile suites share it: a NEFF is keyed
# by its HLO, so the two suites never collide.
_JOB_NAME = "vertical"


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "action",
        choices=("hydrate", "upload"),
        help="hydrate: download the cache before the run. upload: write it back (main only).",
    )
    args = parser.parse_args(argv)

    cache = PersistentNeffCache(_JOB_NAME)
    if args.action == "hydrate":
        cache.hydrate()
    else:
        cache.upload()
    return 0


if __name__ == "__main__":
    sys.exit(main())

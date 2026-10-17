#!/bin/bash
# Thin shim — orchestration moved to runner/run_upstream_tests.py.
# Kept so existing callers (CI workflow, dev scripts) work unchanged.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/runner/run_upstream_tests.py" "$@"

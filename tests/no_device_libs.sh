#!/bin/bash
# Verify _C.so does not have load-time dependencies on device-optional libraries.
# These libraries are only available on specific hardware (e.g. EFA-equipped instances)
# and must be loaded via dlopen at runtime, not linked at build time.
set -euo pipefail

SO_PATH="${1:?Usage: $0 <path-to-_C.so>}"

# Add libraries here as needed. Each entry is checked against DT_NEEDED.
DISALLOWED=(
    "libfabric"
)

FAILED=0
for lib in "${DISALLOWED[@]}"; do
    if readelf -d "$SO_PATH" | grep -q "NEEDED.*$lib"; then
        echo "FAIL: _C.so has disallowed DT_NEEDED entry: $lib"
        readelf -d "$SO_PATH" | grep "NEEDED.*$lib"
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo ""
    echo "Device-optional libraries must be loaded via dlopen, not linked at build time."
    exit 1
fi

echo "PASS: _C.so has no disallowed load-time dependencies."

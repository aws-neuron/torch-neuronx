#!/usr/bin/env python3
"""Check whether any process holds a Neuron device."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from typing import Any


def validate_devices(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("The device query root must be a list.")
    if not value:
        raise ValueError("No Neuron devices found")
    for device_index, device in enumerate(value):
        if not isinstance(device, dict):
            raise ValueError(f"Device {device_index} must be an object.")
        processes = device.get("neuron_processes")
        if not isinstance(processes, list):
            raise ValueError(f"Device {device_index} neuron_processes must be a list.")
        for process_index, process in enumerate(processes):
            if not isinstance(process, dict):
                raise ValueError(
                    f"Device {device_index} process {process_index} must be an object."
                )
    return value


def inspect_devices(command: str, timeout: float) -> tuple[int, str]:
    if not math.isfinite(timeout) or timeout <= 0:
        return 2, "The timeout must be finite and greater than zero."
    try:
        result = subprocess.run(
            [command, "-j"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return 2, f"The device query did not run: {error}"
    if result.returncode != 0:
        detail = result.stderr.strip() or "no error text"
        return 2, f"The device query failed with code {result.returncode}: {detail}"
    try:
        devices = validate_devices(json.loads(result.stdout))
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        return 2, f"The device query returned invalid data: {error}"
    busy = [device for device in devices if device["neuron_processes"]]
    if busy:
        details = []
        for device in busy:
            process_ids = [str(item.get("pid", "unknown")) for item in device["neuron_processes"]]
            details.append(
                f"device {device.get('neuron_device', 'unknown')}: pids={','.join(process_ids)}"
            )
        return 1, "A process holds a Neuron device. " + "; ".join(details)
    return 0, f"All {len(devices)} Neuron devices are free."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", default="neuron-ls", help="The device query executable.")
    parser.add_argument("--timeout", type=float, default=30.0, help="The query timeout in seconds.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code, message = inspect_devices(args.command, args.timeout)
    print(message, file=sys.stderr if code == 2 else sys.stdout)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

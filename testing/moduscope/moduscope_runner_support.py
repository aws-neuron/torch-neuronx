"""Shared support for ModuScope runner adapters."""

from __future__ import annotations

import contextlib
import glob
import importlib
import importlib.util
import os
import shutil
import signal
import time
from dataclasses import dataclass, field

import moduscope_process as process
import yaml

HLO_CACHE_DIR = "/tmp/hlo_cache"
NEFF_CACHE_DIR = "/tmp/neff_cache"
INDUCTOR_CACHE_GLOB = "/tmp/torchinductor_*"
CONFIGS_PACKAGE = "cli_scripts"
PROC_ROOT = "/proc"
NEURON_DEVICE_PREFIX = "/dev/neuron"
DEVICE_GRACE_SECONDS = 30.0
DEVICE_POLL_SECONDS = 0.5


def run_member_process(
    invocation: process.TestInvocation,
    timeout_seconds: float,
    command_prefix: str,
    timeout_message: str = "",
) -> process.InvocationResult:
    """Print and run one shard member process."""
    print(command_prefix + " ".join(invocation.argv))
    result = process.run_test_invocation(invocation, timeout_seconds)
    if (
        timeout_message
        and result.returncode == process.TIMEOUT_EXIT_CODE
        and result.reason.startswith("killed by the runner")
    ):
        print(timeout_message)
    return result


def aggregate_member_rows(member_row_batches, after_batch=None) -> list[dict]:
    """Combine each member's rows and optionally handle every cumulative batch."""
    aggregate_rows = []
    for member_rows in member_row_batches:
        aggregate_rows.extend(member_rows)
        if after_batch:
            after_batch(aggregate_rows)
    return aggregate_rows


def send_sigkill(process_id: int) -> None:
    """Send SIGKILL to one process.

    Ignore a missing process or a process that the runner cannot signal.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(process_id, signal.SIGKILL)


def process_holds_neuron_device(file_descriptor_dir: str) -> bool:
    """Return true when a process has a Neuron device file open."""
    try:
        file_descriptors = os.listdir(file_descriptor_dir)
    except OSError:
        return False
    for file_descriptor in file_descriptors:
        with contextlib.suppress(OSError):
            if os.readlink(os.path.join(file_descriptor_dir, file_descriptor)).startswith(
                NEURON_DEVICE_PREFIX
            ):
                return True
    return False


def neuron_device_holders(process_root: str, runner_process_id: int) -> list[int]:
    """Return process IDs that hold a Neuron device, excluding the runner."""
    if not os.path.isdir(process_root):
        return []
    device_holder_process_ids = []
    for process_name in os.listdir(process_root):
        if not process_name.isdigit() or int(process_name) == runner_process_id:
            continue
        if process_holds_neuron_device(os.path.join(process_root, process_name, "fd")):
            device_holder_process_ids.append(int(process_name))
    return sorted(device_holder_process_ids)


def format_process_ids(process_ids: list[int]) -> str:
    """Return a comma-separated process ID list or ``none``."""
    return ",".join(map(str, process_ids)) or "none"


def device_holder_baseline() -> set[int]:
    """Return the Neuron device holders that existed before the shard started."""
    return set(neuron_device_holders(PROC_ROOT, os.getpid()))


def stop_new_device_holders(
    baseline_process_ids: set[int],
) -> tuple[list[int], list[int], list[int]]:
    """Stop device holders that appeared after the shard started.

    A device holder is a process that still has a Neuron device file open.
    """
    # A reaped process can leave children under process 1.
    # Use the starting process list to protect older device holders.
    runner_process_id = os.getpid()
    device_holder_process_ids = neuron_device_holders(PROC_ROOT, runner_process_id)
    new_process_ids = [
        process_id
        for process_id in device_holder_process_ids
        if process_id not in baseline_process_ids
    ]
    baseline_holder_process_ids = [
        process_id for process_id in device_holder_process_ids if process_id in baseline_process_ids
    ]
    for process_id in new_process_ids:
        send_sigkill(process_id)
    surviving_process_ids = new_process_ids
    grace_end = time.monotonic() + DEVICE_GRACE_SECONDS
    while surviving_process_ids and time.monotonic() < grace_end:
        surviving_process_ids = [
            process_id
            for process_id in neuron_device_holders(PROC_ROOT, runner_process_id)
            if process_id not in baseline_process_ids
        ]
        for process_id in surviving_process_ids:
            if process_id not in new_process_ids:
                new_process_ids.append(process_id)
                send_sigkill(process_id)
        if surviving_process_ids:
            time.sleep(DEVICE_POLL_SECONDS)
    return (
        sorted(new_process_ids),
        baseline_holder_process_ids,
        sorted(surviving_process_ids + baseline_holder_process_ids),
    )


def guard_between_members(
    baseline_process_ids: set[int], member_index: int, member_name: str, *, cleared_cache_count: int
) -> str:
    """Check device holders after one member.

    Return the reason that prevents later members from starting.
    """
    new_process_ids, baseline_holder_process_ids, surviving_process_ids = stop_new_device_holders(
        baseline_process_ids
    )
    process.emit_token_line(
        f"GUARDS entry={member_index} model={member_name} caches_cleared={cleared_cache_count} "
        + f"device_pids={format_process_ids(new_process_ids)} "
        + f"foreign={format_process_ids(baseline_holder_process_ids)}"
    )
    if not surviving_process_ids:
        return ""
    process.emit_token_line(
        f"DEVICE_HELD entry={member_index} pids={format_process_ids(surviving_process_ids)}"
    )
    return f"device held by pids {surviving_process_ids} after {member_name} exited"


def clear_compile_caches() -> tuple[str, ...]:
    """Remove compile caches before one test member starts."""
    removed = []
    for cache_dir in (HLO_CACHE_DIR, NEFF_CACHE_DIR, *sorted(glob.glob(INDUCTOR_CACHE_GLOB))):
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
            removed.append(cache_dir)
    return tuple(removed)


def never_started_reason(started_count: int, member_count: int) -> str:
    return f"never started: shard deadline reached after {started_count} of {member_count} members"


@dataclass
class ShardRunState:
    member_count: int
    baseline_process_ids: set[int] = field(default_factory=device_holder_baseline)
    returncode: int = 0
    started_count: int = 0
    device_failure_reason: str = ""

    def check_member_start(self, deadline: float) -> tuple[float, str]:
        """Return the remaining time and the reason that prevents a member from starting."""
        remaining_seconds = deadline - time.monotonic()
        failure_reason = self.device_failure_reason
        if failure_reason:
            self.returncode = self.returncode or 1
            return remaining_seconds, failure_reason
        if remaining_seconds > 0:
            return remaining_seconds, ""
        self.returncode = self.returncode or process.TIMEOUT_EXIT_CODE
        return remaining_seconds, never_started_reason(self.started_count, self.member_count)

    def finish_member(
        self, member_index: int, member_name: str, returncode: int, cleared_cache_count: int
    ) -> None:
        self.started_count += 1
        self.returncode = self.returncode or returncode
        self.device_failure_reason = guard_between_members(
            self.baseline_process_ids,
            member_index,
            member_name,
            cleared_cache_count=cleared_cache_count,
        )


def installed_config_field(config: str, field_name: str) -> tuple[str, ...]:
    """Return the default of one field from an installed task configuration."""
    package_spec = importlib.util.find_spec(CONFIGS_PACKAGE)
    if package_spec is None or not package_spec.submodule_search_locations:
        raise ModuleNotFoundError(f"{CONFIGS_PACKAGE} is not installed.")
    config_path = os.path.join(
        next(iter(package_spec.submodule_search_locations)), "configs", f"{config}.yaml"
    )
    with open(config_path, encoding="utf-8") as handle:
        target_path = yaml.safe_load(handle)["task_config"]["_target_"]
    module_name, _, class_name = target_path.rpartition(".")
    task_config = getattr(importlib.import_module(module_name), class_name)
    config_fields = getattr(task_config, "model_fields", {})
    if field_name in config_fields:
        field_default = config_fields[field_name].get_default(call_default_factory=True)
    else:
        field_default = getattr(task_config, field_name)
    return tuple(field_default)

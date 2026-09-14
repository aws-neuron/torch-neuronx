"""The pod's entry point: runs the selected target shard and records the target check.

A target check is the JSON report that compares a target with the installed wheel.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import signal
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import moduscope_component_equivalence_shard_runner
import moduscope_hugging_face_shard_runner
import moduscope_titan_shard_runner
from moduscope_artifact_layout import (
    ARTIFACT_ROOT,
    IDENTITY_ENV_VARS,
    OUTPUT_SUBDIR,
    RUN_ID_ENV_VAR,
    STAGED_GRID_RENDERER_RELATIVE_PATH,
    read_run_identity,
    require_env,
    run_id_filename,
    stripped_env,
)
from moduscope_process import emit_token_line, write_token_lines
from moduscope_runner_support import installed_config_field
from moduscope_targets import (
    ADAPTER_NAMES,
    COMPONENT_EQUIVALENCE_ADAPTER,
    HUGGING_FACE_ADAPTER,
    TITAN_ADAPTER,
    ShardRunContext,
    Target,
    resolve_target,
)

# Require MODUSCOPE_CI=github because other values can write to production S3 buckets.
CI_ENV_VAR = "MODUSCOPE_CI"
GITHUB_SIDE = "github"
# The installed wheel reads this value as the suite name for its S3 write guard.
SUITE_ENV_VAR = "MODUSCOPE_SUITE"
TARGET_ENV_VAR = "MODUSCOPE_TARGET"
SHARD_INDEX_ENV_VAR = "MODUSCOPE_SHARD_INDEX"
TIMEOUT_SECONDS_ENV_VAR = "MODUSCOPE_TIMEOUT_SECONDS"
TARGET_CHECK_FILE_NAME = "target_check.json"
RENDERER_MODULE = "modu_scope.utils.grid_report_html"
# Keep this list in sync with workflow exports injected into the pod.
# The installed wheel still reads the forwarded variables.
# TORCH_NEURONX_ENABLE_PROLOGUE matches the launcher and appears in each device result row.
FORWARDED_POD_ENV_VARS = ("MODUSCOPE_REQUESTER_TYPE", "MODUSCOPE_REQUESTER_NAME")
ENV_VARS_READ_ON_POD = frozenset(
    {
        CI_ENV_VAR,
        TARGET_ENV_VAR,
        SHARD_INDEX_ENV_VAR,
        TIMEOUT_SECONDS_ENV_VAR,
        *IDENTITY_ENV_VARS,
        *FORWARDED_POD_ENV_VARS,
        "TORCH_NEURONX_ENABLE_PROLOGUE",
    }
)


@dataclass(frozen=True)
class RunnerAdapter:
    run_shard: Callable[[ShardRunContext], int]
    installed_member_field: str | None = None


EXECUTION_ADAPTERS = {
    HUGGING_FACE_ADAPTER: RunnerAdapter(
        moduscope_hugging_face_shard_runner.run_shard,
        moduscope_hugging_face_shard_runner.HUGGING_FACE_MEMBER_FIELD,
    ),
    TITAN_ADAPTER: RunnerAdapter(
        moduscope_titan_shard_runner.run_shard,
        moduscope_titan_shard_runner.TITAN_CONFIG_FIELD,
    ),
    COMPONENT_EQUIVALENCE_ADAPTER: RunnerAdapter(
        moduscope_component_equivalence_shard_runner.run_shard
    ),
}
assert set(EXECUTION_ADAPTERS) == set(ADAPTER_NAMES)


def stamp_output_files(output_root: str, run_id: str) -> None:
    """Prefix every file basename under the upload root with the run ID."""
    try:
        run_id_prefix = f"{run_id}-"
        source_paths = sorted(
            path
            for path in Path(output_root).rglob("*")
            if path.is_file() and not path.name.startswith(run_id_prefix)
        )
        destination_paths = [
            source_path.with_name(run_id_filename(run_id, source_path.name))
            for source_path in source_paths
        ]
        existing_paths = [
            destination_path for destination_path in destination_paths if destination_path.exists()
        ]
        if existing_paths:
            raise FileExistsError(
                f"The run-ID output file already exists: {existing_paths[0]}. "
                + "Remove it before the runner starts."
            )
        for source_path, destination_path in zip(source_paths, destination_paths, strict=True):
            source_path.rename(destination_path)
    except OSError:
        quarantine_path = f"{output_root}.stamping-error-{uuid.uuid4().hex}"
        os.rename(output_root, quarantine_path)
        raise


def _parse_integer(variable_name: str, raw_value: str) -> int:
    try:
        return int(raw_value)
    except ValueError:
        raise ValueError(f"{variable_name}={raw_value!r} is not an integer.") from None


def read_env() -> tuple[str, str, int, str]:
    """Read and validate the target name, shard index text, timeout, and run ID."""
    environment_values = require_env(
        (CI_ENV_VAR, TARGET_ENV_VAR, SHARD_INDEX_ENV_VAR, TIMEOUT_SECONDS_ENV_VAR, RUN_ID_ENV_VAR)
    )
    if environment_values[CI_ENV_VAR].lower() != GITHUB_SIDE:
        raise ValueError(
            f"{CI_ENV_VAR}={environment_values[CI_ENV_VAR]!r}; only {GITHUB_SIDE!r} keeps "
            + "ModuScope's S3 writes off the production buckets."
        )
    timeout_text = environment_values[TIMEOUT_SECONDS_ENV_VAR]
    timeout_seconds = _parse_integer(TIMEOUT_SECONDS_ENV_VAR, timeout_text)
    if timeout_seconds <= 0:
        raise ValueError(
            f"{TIMEOUT_SECONDS_ENV_VAR}={timeout_text!r} is not positive, so ModuScope would "
            + "be killed before it starts."
        )
    return (
        environment_values[TARGET_ENV_VAR],
        environment_values[SHARD_INDEX_ENV_VAR],
        timeout_seconds,
        environment_values[RUN_ID_ENV_VAR],
    )


def read_shard_index(index_text: str, shard_count: int) -> int:
    """Return a shard index inside the planned shard count."""
    index = _parse_integer(SHARD_INDEX_ENV_VAR, index_text)
    if not 0 <= index < shard_count:
        raise ValueError(
            f"{SHARD_INDEX_ENV_VAR}={index} is outside a target of {shard_count} shard(s)."
        )
    return index


def check_target(target: Target, out_dir: str, installed_member_field: str) -> str:
    """Compare the target members with the installed wheel and write target_check.json.

    A wheel member that the target lists in excluded_models is neither missing nor extra;
    the report lists it under "excluded". A listed member that is not in the wheel is
    reported under "excluded_absent" and does not fail the check.
    Return its TARGET_CHECK token line.
    """
    target_check: dict = {"target": target.name, "config": target.config}
    try:
        target_members = target.members
        wheel_members = installed_config_field(target.config, installed_member_field)
        duplicate_members = [
            member_name for member_name, count in Counter(wheel_members).items() if count > 1
        ]
        if duplicate_members:
            raise ValueError(
                "The installed wheel contains duplicate members: "
                + f"{', '.join(duplicate_members)}. "
                + f"Remove duplicate members from {target.config}."
            )
        missing_members = [
            member_name for member_name in target_members if member_name not in wheel_members
        ]
        excluded_members = [
            member_name for member_name in wheel_members if member_name in target.excluded_models
        ]
        excluded_absent_members = [
            member_name
            for member_name in target.excluded_models
            if member_name not in wheel_members
        ]
        extra_members = [
            member_name
            for member_name in wheel_members
            if member_name not in target_members and member_name not in target.excluded_models
        ]
        target_check.update(
            expected=target_members,
            in_wheel=wheel_members,
            missing=missing_members,
            extra=extra_members,
            excluded=excluded_members,
            excluded_absent=excluded_absent_members,
        )
        token_line = (
            f"TARGET_CHECK target={target.name} missing={len(missing_members)} "
            + f"extra={len(extra_members)} excluded={len(excluded_members)}"
        )
    except Exception as exc:
        target_check["error"] = f"{type(exc).__name__}: {exc}"
        token_line = f"TARGET_CHECK target={target.name} error=1"
    with open(
        os.path.join(out_dir, TARGET_CHECK_FILE_NAME), "w", encoding="utf-8"
    ) as target_check_file:
        json.dump(target_check, target_check_file, indent=2)
    return token_line


def moduscope_wheel_token_line() -> str:
    """Return the installed ModuScope wheel token."""
    try:
        wheel_version = importlib.metadata.version("moduscope")
        distribution_files = importlib.metadata.files("moduscope") or ()
    except importlib.metadata.PackageNotFoundError:
        return "MODUSCOPE_WHEEL version=unavailable sha256=unavailable"
    record_file = next(
        (
            distribution_file
            for distribution_file in distribution_files
            if distribution_file.name == "RECORD"
            and distribution_file.parent.name.endswith(".dist-info")
        ),
        None,
    )
    record_sha256 = "none"
    if record_file is not None:
        with contextlib.suppress(OSError):
            record_sha256 = hashlib.sha256(record_file.locate().read_bytes()).hexdigest()
    return f"MODUSCOPE_WHEEL version={wheel_version} sha256={record_sha256}"


def stage_grid_renderer(output_dir: str) -> None:
    """Copy the installed grid renderer. Record its SHA-256 in a token line."""
    renderer_path = Path(output_dir) / STAGED_GRID_RENDERER_RELATIVE_PATH
    try:
        renderer_path.parent.mkdir(parents=True, exist_ok=True)
        renderer_spec = importlib.util.find_spec(RENDERER_MODULE)
        if renderer_spec is None or not renderer_spec.origin:
            raise ModuleNotFoundError(RENDERER_MODULE)
        renderer_bytes = Path(renderer_spec.origin).read_bytes()
        renderer_sha256 = hashlib.sha256(renderer_bytes).hexdigest()
        renderer_path.write_bytes(renderer_bytes)
    except Exception:
        with contextlib.suppress(OSError):
            renderer_path.unlink(missing_ok=True)
        emit_token_line("RENDERER copied=0 sha256=-")
        return
    emit_token_line(f"RENDERER copied=1 sha256={renderer_sha256}")


def main() -> int:
    """Run one target shard and return its exit code."""
    output_root = os.path.join(os.getcwd(), ARTIFACT_ROOT)
    run_id = stripped_env(RUN_ID_ENV_VAR)
    output_ready = False
    try:
        target_name, raw_index, timeout_seconds, _validated_run_id = read_env()
        target = resolve_target(target_name)
        adapter = EXECUTION_ADAPTERS[target.adapter]
        shard_index = read_shard_index(raw_index, len(target.shards))
        shard = target.shards[shard_index]
        output_dir = os.path.join(os.getcwd(), OUTPUT_SUBDIR)
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_ready = True
        os.environ[SUITE_ENV_VAR] = target.config
        emit_token_line(moduscope_wheel_token_line())
        emit_token_line(f"RUN target={target.name} shard={shard_index} run={run_id}")
        print(f"Output directory (emptied): {output_dir}")
        try:
            context = ShardRunContext(
                target=target,
                shard=shard,
                output_dir=output_dir,
                run_identity=read_run_identity(),
                deadline=time.monotonic() + timeout_seconds,
            )
            return adapter.run_shard(context)
        finally:
            try:
                if adapter.installed_member_field is not None:
                    emit_token_line(
                        check_target(target, output_dir, adapter.installed_member_field)
                    )
            finally:
                stage_grid_renderer(output_dir)
                write_token_lines(output_dir)
    finally:
        if not output_ready:
            if os.path.isdir(output_root):
                quarantine_path = f"{output_root}.validation-error-{uuid.uuid4().hex}"
                os.rename(output_root, quarantine_path)
                shutil.rmtree(quarantine_path, ignore_errors=True)
                os.makedirs(os.path.join(output_root, "test-reports"), exist_ok=True)
        elif run_id:
            stamp_output_files(output_root, run_id)


def exit_on_sigterm(signal_number, _stack_frame):
    """Raise SystemExit so each cleanup block runs after SIGTERM.

    Use 128 plus the signal number as the exit code.
    """
    raise SystemExit(128 + signal_number)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, exit_on_sigterm)
    raise SystemExit(main())

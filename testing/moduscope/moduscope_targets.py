"""Validate a target.

Build the hosted workflow plan for the planner and device runner.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple

import moduscope_shard_planner

TARGETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "targets.json")
DEFAULT_INSTANCE = "trn2.48xlarge"
SAFE_NAME_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
INSTANCE_NAME_PATTERN = re.compile(r"[a-z0-9.]+")
MEMBER_PATTERN = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)?")
TARGET_FIELDS = {
    "instance",
    "timeout_minutes",
    "config",
    "shards",
    "xfail",
    "torch_compile_enabled",
}
# The relay's external upload function uses this exact artifact name as the S3 key.
HOSTED_RESULT_ARTIFACT_NAME = "custom-test-artifacts"


class Target(NamedTuple):
    name: str
    timeout_minutes: int
    instance: str = DEFAULT_INSTANCE
    config: str = ""
    shards: tuple[moduscope_shard_planner.Shard, ...] = ()
    torch_compile_enabled: bool = False
    xfail: Mapping[str, str] = MappingProxyType({})


def _raise_target_error(target_name: str, field_name: str, reason: str) -> None:
    raise ValueError(f"targets.json: target {target_name!r}, field {field_name!r}: {reason}")


def _require_matching_string(target_name: str, field_name: str, value, pattern: re.Pattern) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        _raise_target_error(
            target_name,
            field_name,
            f"must be a string over {pattern.pattern}, got {value!r}",
        )
    return value


def _parse_member_groups(
    target_name: str,
    value,
) -> tuple[tuple[str, ...], ...]:
    field_name = "shards"
    if not isinstance(value, list) or not value:
        _raise_target_error(target_name, field_name, "must be a nonempty list of member lists")
    groups = []
    for shard_index, members in enumerate(value):
        if not isinstance(members, list) or not members:
            _raise_target_error(
                target_name,
                field_name,
                f"shard {shard_index} must be a nonempty list of members",
            )
        if not all(isinstance(member, str) for member in members):
            _raise_target_error(
                target_name,
                field_name,
                f"shard {shard_index} must contain only member strings",
            )
        if invalid_members := [
            member for member in members if not MEMBER_PATTERN.fullmatch(member)
        ]:
            _raise_target_error(
                target_name,
                field_name,
                f"shard {shard_index} has members not over {MEMBER_PATTERN.pattern}: "
                + f"{invalid_members}",
            )
        groups.append(tuple(members))
    configured_members = [member for group in groups for member in group]
    if repeated_members := sorted(
        member for member in set(configured_members) if configured_members.count(member) > 1
    ):
        _raise_target_error(target_name, field_name, f"repeat members: {repeated_members}")
    return tuple(groups)


def _require_nonempty_reasons(target_name: str, field_name: str, value: dict) -> None:
    items_without_reasons = [
        item for item, reason in value.items() if not isinstance(reason, str) or not reason.strip()
    ]
    if items_without_reasons:
        _raise_target_error(
            target_name,
            field_name,
            f"every id needs a reason, none for: {items_without_reasons}",
        )


def _parse_expected_failures(
    target_name: str,
    value,
    members: tuple[str, ...],
) -> dict[str, str]:
    if not isinstance(value, dict):
        _raise_target_error(target_name, "xfail", "must be an object of member to reason")
    if unknown_items := [item for item in value if item not in members]:
        _raise_target_error(
            target_name,
            "xfail",
            f"not in this target's shards: {unknown_items}",
        )
    _require_nonempty_reasons(target_name, "xfail", value)
    return dict(value)


def _parse_target_data(target_name: str, target_data) -> Target:
    if not isinstance(target_data, dict):
        _raise_target_error(target_name, "", "must be an object")
    if unknown_fields := sorted(set(target_data) - TARGET_FIELDS):
        _raise_target_error(
            target_name,
            unknown_fields[0],
            "is not a valid target field. Remove this field.",
        )
    instance = _require_matching_string(
        target_name,
        "instance",
        target_data.get("instance", DEFAULT_INSTANCE),
        INSTANCE_NAME_PATTERN,
    )
    timeout_minutes = target_data.get("timeout_minutes")
    if (
        isinstance(timeout_minutes, bool)
        or not isinstance(timeout_minutes, int)
        or timeout_minutes <= 0
    ):
        _raise_target_error(
            target_name,
            "timeout_minutes",
            f"must be an integer above 0, got {timeout_minutes!r}",
        )
    config = target_data.get("config", "")
    if config:
        config = _require_matching_string(
            target_name,
            "config",
            config,
            SAFE_NAME_PATTERN,
        )
    elif not isinstance(config, str):
        _raise_target_error(target_name, "config", f"must be a string, got {config!r}")
    groups = _parse_member_groups(target_name, target_data.get("shards"))
    members = tuple(member for group in groups for member in group)
    members_by_grid_id: dict[str, list[str]] = {}
    for member in members:
        members_by_grid_id.setdefault(member.replace("/", "_"), []).append(member)
    if colliding_members := {
        stored_member: source_members
        for stored_member, source_members in members_by_grid_id.items()
        if len(source_members) > 1
    }:
        _raise_target_error(
            target_name,
            "shards",
            f"members collide once the slash becomes an underscore: {colliding_members}",
        )
    torch_compile_enabled = target_data.get("torch_compile_enabled", False)
    if not isinstance(torch_compile_enabled, bool):
        _raise_target_error(
            target_name,
            "torch_compile_enabled",
            f"must be a boolean, got {torch_compile_enabled!r}",
        )
    expected_failures = _parse_expected_failures(
        target_name,
        target_data.get("xfail", {}),
        members,
    )
    return Target(
        name=target_name,
        timeout_minutes=timeout_minutes,
        instance=instance,
        config=config,
        shards=moduscope_shard_planner.build_shards(groups),
        torch_compile_enabled=torch_compile_enabled,
        xfail=expected_failures,
    )


def read_targets(path: str = TARGETS_PATH) -> dict[str, Target]:
    """Return each validated target from ``path``.

    Tests can pass a temporary file whose validation errors name the target and field.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("targets"), dict):
        raise ValueError(f"{path} has no 'targets' object at its top level.")
    return {
        target_name: _parse_target_data(target_name, target_data)
        for target_name, target_data in payload["targets"].items()
    }


def resolve_target(name: str, path: str = TARGETS_PATH) -> Target:
    """Return the named target. Raise `ValueError` with the known names when it is absent."""
    targets = read_targets(path)
    if name not in targets:
        raise ValueError(f"no target named {name!r} in {path}; known: {list(targets)}")
    return targets[name]


def shard_plan(target: Target) -> list[moduscope_shard_planner.Shard]:
    """Return the normalized shards."""
    return list(target.shards)


def shard_matrix(target: Target) -> list[dict]:
    """Return one row for each configured pod shard."""
    return [{"index": shard.index, "label": shard.label} for shard in target.shards]


def _build_plan(
    target: Target,
    timeout_minutes: int,
    copy_buffer_minutes: int,
) -> dict:
    if copy_buffer_minutes < 0:
        raise ValueError(f"copy_buffer_minutes={copy_buffer_minutes} must not be negative.")
    effective_timeout_minutes = timeout_minutes or target.timeout_minutes
    # run_test.sh copies outputs/ after the test command ends.
    # End the inner timeout before the hosted job limit to reserve the copy buffer.
    if effective_timeout_minutes <= copy_buffer_minutes:
        raise ValueError(
            f"timeout_minutes={effective_timeout_minutes} must exceed "
            + f"copy_buffer_minutes={copy_buffer_minutes}, "
            + "or the inner bound is not positive."
        )
    return {
        "shards": shard_matrix(target),
        "timeout_minutes": effective_timeout_minutes,
        "inner_timeout_seconds": (effective_timeout_minutes - copy_buffer_minutes) * 60,
        "instance": target.instance,
        "relay_artifact": HOSTED_RESULT_ARTIFACT_NAME,
    }


def plan(
    target: Target,
    timeout_minutes: int,
    copy_buffer_minutes: int,
) -> dict:
    """Return the workflow plan, using the target timeout when ``timeout_minutes`` is zero.

    The runner reports that value in a token line.
    """
    return _build_plan(target, timeout_minutes, copy_buffer_minutes)


def main(argv: list[str] | None = None) -> int:
    """Print plan token lines. Send a refusal as a ``::error::`` annotation to standard error."""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    subcommands = argument_parser.add_subparsers(dest="command", required=True)
    plan_parser = subcommands.add_parser("plan")
    plan_parser.add_argument("--target", required=True)
    plan_parser.add_argument("--timeout-minutes", type=int, required=True)
    plan_parser.add_argument("--copy-buffer-minutes", type=int, required=True)
    arguments = argument_parser.parse_args(argv)
    try:
        target = resolve_target(arguments.target)
        plan_outputs = _build_plan(
            target,
            arguments.timeout_minutes,
            arguments.copy_buffer_minutes,
        )
    except ValueError as error:
        # The workflow sends standard output to $GITHUB_OUTPUT, so annotations use standard error.
        print(f"::error::{error}", file=sys.stderr)
        return 1
    for key, value in plan_outputs.items():
        print(f"{key}={json.dumps(value, separators=(',', ':')) if key == 'shards' else value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

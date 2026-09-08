"""Decide what a finished entry's result rows mean and how they are reported."""

from __future__ import annotations

import json

import moduscope_artifact_layout as artifact_layout

# Both suites write "running" before completion.
NON_TERMINAL_STATUSES = frozenset({None, "", "running"})
PASSING_STATUSES = frozenset({"pass", "xfail", "xpass"})
DEFAULT_XFAIL_REASON = "xfail reason unset; using default reason"
# Keep GitHub annotation previews short. Stored grid rows keep the complete error text.
ANNOTATION_PREVIEW_CHARS = 200
# GitHub reads workflow commands from stdout. A percent sign or line break ends a command.
ANNOTATION_UNSAFE_CHARS = str.maketrans("%\r\n", "   ")


def infra_fail_row(
    index: int, name: str | None, reason: str, module_configs_str: str | None = None
) -> dict:
    """Return a row for ``infra_fail``, a runner or pod failure, not a model-result failure.

    Use it when an entry has no usable grid.
    """
    failure_row = {
        "grid_entry_name": name or f"grid_entry_{index}",
        "grid_entry_index": index,
        "mfu(%)": -1,
        "status": "infra_fail",
        "error": reason,
    }
    if module_configs_str:
        failure_row["module_configs_str"] = module_configs_str
    return failure_row


def status_after_expected_failure(status, expected_to_fail: bool, process_passed: bool):
    """Return the status unchanged unless the model result is expected to fail.

    Map ``fail`` to ``xfail`` or a successful ``pass`` to ``xpass``.
    """
    if not expected_to_fail:
        return status
    if status == "fail":
        return "xfail"
    if status == "pass" and process_passed:
        return "xpass"
    return status


def expected_failure_absorbs_exit(rows: list[dict] | None, listed: bool) -> bool:
    """Return whether expected failures explain a nonzero process exit.

    Require every row to have ``pass``, ``xfail``, or ``xpass`` status.
    """
    if not rows:
        return False
    has_expected_failure = listed or any(row.get("expected_to_fail") is True for row in rows)
    return has_expected_failure and all(row.get("status") in PASSING_STATUSES for row in rows)


def _model_name_from_row(row: dict) -> str:
    config_prefix, separator, unused_suffix = str(row.get("module_configs_str") or "").rpartition(
        "_config"
    )
    return config_prefix if separator else str(row.get("grid_entry_name") or "")


def github_failure_annotations(
    rows: list[dict] | None, xfail_reason: str | None = None
) -> list[str]:
    """Return one GitHub workflow command for each row without ``pass`` status.

    Use a notice for ``xfail`` and warnings for all other statuses.
    """
    annotations = []
    for row in rows or []:
        status = row.get("status")
        if status == "pass":
            continue
        model_name = _model_name_from_row(row)
        entry_name = str(row.get("grid_entry_name") or "")
        annotation_label = (
            model_name if entry_name in ("", model_name) else f"{model_name} {entry_name}"
        ).translate(ANNOTATION_UNSAFE_CHARS)
        if status == "xpass":
            annotations.append(f"::warning::model {annotation_label} xpass: prune its xfail entry")
            continue
        error_text = str(row.get("error") or "")[:ANNOTATION_PREVIEW_CHARS]
        error_text = error_text.translate(ANNOTATION_UNSAFE_CHARS)
        if status == "xfail":
            expected_failure_reason = (xfail_reason or DEFAULT_XFAIL_REASON).translate(
                ANNOTATION_UNSAFE_CHARS
            )
            annotations.append(
                f"::notice::model {annotation_label} status=xfail "
                + f"reason={expected_failure_reason} error={error_text}"
            )
        else:
            safe_status = str(status).translate(ANNOTATION_UNSAFE_CHARS)
            annotations.append(
                f"::warning::model {annotation_label} status={safe_status} error={error_text}"
            )
    return annotations


def _fail_unfinished_rows(
    grid_rows: list[dict] | None,
    entry_index: int,
    entry_name: str | None,
    failure_reason: str,
    module_configs_str: str | None,
) -> tuple[list[dict], int]:
    fallback_entry_name = entry_name or f"grid_entry_{entry_index}"
    if not grid_rows:
        return [
            infra_fail_row(
                entry_index,
                entry_name,
                failure_reason,
                module_configs_str,
            )
        ], 1
    failed_row_count = 0
    for row in grid_rows:
        if not row.get("grid_entry_name"):
            row["grid_entry_name"] = fallback_entry_name
        if row.get("status") in NON_TERMINAL_STATUSES:
            row["status"] = "fail"
            row["error"] = failure_reason
            failed_row_count += 1
    return grid_rows, failed_row_count


def _fail_moduscope_error_rows(grid_rows: list[dict] | None) -> int:
    failed_row_count = 0
    for row in grid_rows or []:
        error_message = row.get("error_message")
        has_moduscope_error = bool(row.get("error_case")) or (
            isinstance(error_message, str) and error_message.startswith("Crash")
        )
        if row.get("status") not in NON_TERMINAL_STATUSES or not has_moduscope_error:
            continue
        row["status"] = "fail"
        if "error" not in row:
            row["error"] = error_message
        failed_row_count += 1
    return failed_row_count


def finalize_entry_grid(
    path: str,
    *,
    index: int,
    name: str | None,
    returncode: int,
    reason: str,
    identity: dict[str, str] | None,
    module_configs_str: str | None = None,
    expected_to_fail: bool = False,
) -> str:
    """Finalize the grid after one entry process exits and apply expected-failure rules.

    Return the exact ``ENTRY_GRID`` token line.
    """
    rows = artifact_layout.load_grid_rows(path)
    original_rows_json = json.dumps(rows, sort_keys=True)
    has_unfinished_rows = not rows or any(
        row.get("status") in NON_TERMINAL_STATUSES for row in rows
    )
    failed_row_count = _fail_moduscope_error_rows(rows)
    if not rows or (returncode != 0 and has_unfinished_rows):
        failure_reason = (
            reason
            if returncode != 0
            else "ModuScope produced no usable grid rows and exited with code 0."
        )
        rows, unfinished_row_count = _fail_unfinished_rows(
            rows,
            index,
            name,
            failure_reason,
            module_configs_str,
        )
        failed_row_count += unfinished_row_count
    for row in rows:
        final_status = status_after_expected_failure(
            row.get("status"),
            expected_to_fail or row.get("expected_to_fail") is True,
            returncode == 0,
        )
        # Keep a missing status key missing. Adding "status": null changes the row.
        if final_status != row.get("status"):
            row["status"] = final_status
        artifact_layout.add_run_identity(row, identity)
    # Write only changed rows because callers compare untouched grid bytes.
    if json.dumps(rows, sort_keys=True) != original_rows_json:
        try:
            artifact_layout.write_grid_rows(path, rows)
        except OSError as exc:
            return f"ENTRY_GRID entry={index} write_failed reason={type(exc).__name__}: {exc}"
    return (
        f"ENTRY_GRID entry={index} rows={len(rows)} marked_failed={failed_row_count} "
        + "profile_s3_url=none"
    )

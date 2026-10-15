"""Pytest tests result helpers for an entry."""

from __future__ import annotations

import moduscope_entry_results as entry_results
import pytest

from conftest import RUN_IDENTITY_FIELDS, read_rows_at, write_rows

RUNNING_GRID_ROWS = [
    {"grid_entry_name": "Titan_b_Async", "grid_entry_index": 1, "status": "running"}
]
INFRA_FAIL_ROW = {
    "grid_entry_name": "grid_entry_3",
    "grid_entry_index": 3,
    "status": "infra_fail",
}
INFRA_FAIL_ROW.update({"mfu(%)": -1, "error": "exit 1"})
LONG_ERROR_TEXT = "first line\n" + "x" * 5000 + "\nlast line"
NAMED_INFRA_FAIL_ROW = {
    **INFRA_FAIL_ROW,
    "grid_entry_name": "t5/11b",
    "error": LONG_ERROR_TEXT,
    **RUN_IDENTITY_FIELDS,
}
MODEL_CONFIG_TOKEN = "allenai_OLMo-1B-hf_config"
XFAIL_ROW = {"grid_entry_name": "Titan_b_Async", "status": "xfail", "error": "loss is nan"}
XPASS_ROW_FIELDS = {
    "grid_entry_name": "Hf| neuron",
    "module_configs_str": "google_gemma-4-E2B-it_config X",
}
PASSING_ROW = {"grid_entry_name": "Hf.CausalLM| tiny_config X| neuron", "status": "pass"}
PASSING_ROW["module_configs_str"] = "smeoni_tiny-random-LlamaForCausalLM_config X"
EXPECTED_FAILURE_CASES = [
    ("a", "pass"),
    ("b", "fail", True),
    ("c", "fail", False),
    ("d", None),
]
EXPECTED_FAILURE_CASES += [("e", "infra_fail"), ("f", "running")]


def finalize_test_grid(grid_path, **finalize_overrides):
    default_arguments = {
        "index": 0,
        "name": None,
        "returncode": 0,
        "reason": "",
        "identity": None,
    }
    return entry_results.finalize_entry_grid(
        str(grid_path), **{**default_arguments, **finalize_overrides}
    )


def test_finalize_preserves_unchanged_bytes_stamps_rows_and_fails_unfinished_rows(
    tmp_path,
):
    finished_row = {"grid_entry_name": "done", "status": "pass"}
    grid_path = write_rows(tmp_path / "grid_metrics.json", [finished_row])
    original_bytes = grid_path.read_bytes()
    assert (
        finalize_test_grid(grid_path)
        == "ENTRY_GRID entry=0 rows=1 marked_failed=0 profile_s3_url=none"
    )
    assert grid_path.read_bytes() == original_bytes
    write_rows(
        grid_path,
        [finished_row, {"grid_entry_name": "half", "status": "running"}, {"status": None}],
    )
    token_line = finalize_test_grid(
        grid_path,
        index=4,
        returncode=137,
        reason="exit 137",
        identity=RUN_IDENTITY_FIELDS,
    )
    assert read_rows_at(grid_path) == [
        {**finished_row, **RUN_IDENTITY_FIELDS},
        {
            "grid_entry_name": "half",
            "status": "fail",
            "error": "exit 137",
            **RUN_IDENTITY_FIELDS,
        },
        {
            "grid_entry_name": "grid_entry_4",
            "status": "fail",
            "error": "exit 137",
            **RUN_IDENTITY_FIELDS,
        },
    ]
    assert token_line == "ENTRY_GRID entry=4 rows=3 marked_failed=2 profile_s3_url=none"
    assert [child_path.name for child_path in tmp_path.iterdir()] == [
        "grid_metrics.json"
    ], "a staging file stayed"


@pytest.mark.parametrize("initial_rows", [None, []], ids=["missing-grid", "empty-grid"])
def test_zero_return_process_without_rows_is_infra_fail(tmp_path, initial_rows):
    grid_path = tmp_path / "e0" / "grid_metrics.json"
    if initial_rows is not None:
        write_rows(grid_path, initial_rows)

    token_line = finalize_test_grid(
        grid_path,
        name="t5/11b",
        identity=RUN_IDENTITY_FIELDS,
    )

    assert read_rows_at(grid_path) == [
        {
            "grid_entry_name": "t5/11b",
            "grid_entry_index": 0,
            "mfu(%)": -1,
            "status": "infra_fail",
            "error": "ModuScope produced no usable grid rows and exited with code 0.",
            **RUN_IDENTITY_FIELDS,
        }
    ]
    assert token_line == "ENTRY_GRID entry=0 rows=1 marked_failed=1 profile_s3_url=none"


def test_only_unfinished_rows_are_failed(tmp_path):
    status_by_entry_name = {"done": "pass", "half": "running", "never": "", "none": None}
    grid_rows = [
        {"grid_entry_name": name, "status": status} for name, status in status_by_entry_name.items()
    ]
    grid_path = write_rows(tmp_path / "grid_metrics.json", grid_rows)
    finalize_test_grid(grid_path, returncode=1, reason="exit 1")
    assert [row["status"] for row in read_rows_at(grid_path)] == [
        "pass",
        "fail",
        "fail",
        "fail",
    ]


@pytest.mark.parametrize(
    ("error_case", "expected_to_fail", "expected_status", "existing_error"),
    [
        ("timeout_failure", False, "fail", None),
        ("timeout_failure", True, "xfail", None),
        ("build_failure", True, "xfail", "build command failed"),
    ],
    ids=["timeout-fails", "listed-timeout-xfails", "listed-build-failure-xfails"],
)
def test_finalize_marks_handled_moduscope_error_rows(
    tmp_path, error_case, expected_to_fail, expected_status, existing_error
):
    error_message = "Crash: Local process timed out after 3600 seconds"
    grid_row = {"error_case": error_case, "error_message": error_message}
    if existing_error is not None:
        grid_row["error"] = existing_error
    grid_path = write_rows(
        tmp_path / "grid_metrics.json",
        [grid_row],
    )

    token_line = finalize_test_grid(grid_path, expected_to_fail=expected_to_fail)

    assert read_rows_at(grid_path) == [
        {
            **grid_row,
            "status": expected_status,
            "error": existing_error if existing_error is not None else error_message,
        }
    ]
    assert token_line == "ENTRY_GRID entry=0 rows=1 marked_failed=1 profile_s3_url=none"


def test_finalize_preserves_missing_status_without_moduscope_error(tmp_path):
    grid_row = {"grid_entry_name": "unknown"}
    grid_path = write_rows(tmp_path / "grid_metrics.json", [grid_row])
    original_bytes = grid_path.read_bytes()

    token_line = finalize_test_grid(grid_path)

    assert read_rows_at(grid_path) == [grid_row]
    assert "status" not in read_rows_at(grid_path)[0]
    assert grid_path.read_bytes() == original_bytes
    assert token_line == "ENTRY_GRID entry=0 rows=1 marked_failed=0 profile_s3_url=none"


def test_failed_grid_write_keeps_original_bytes_and_removes_staging_file(tmp_path, monkeypatch):
    grid_path = write_rows(tmp_path / "e1" / "grid_metrics.json", RUNNING_GRID_ROWS)
    original_bytes = grid_path.read_bytes()

    def raise_no_space_error(*_arguments, **_keyword_arguments):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(entry_results.json, "dump", raise_no_space_error)
    token_line = finalize_test_grid(
        grid_path,
        index=1,
        returncode=1,
        reason="exit 1",
        identity=RUN_IDENTITY_FIELDS,
    )
    assert token_line == (
        "ENTRY_GRID entry=1 write_failed reason=OSError: [Errno 28] No space left on device"
    )
    assert grid_path.read_bytes() == original_bytes
    assert [child_path.name for child_path in grid_path.parent.iterdir()] == ["grid_metrics.json"]


@pytest.mark.parametrize(
    ("initial_rows", "finalize_arguments", "expected_row"),
    [
        (
            None,
            {
                "name": "t5/11b",
                "identity": RUN_IDENTITY_FIELDS,
                "reason": LONG_ERROR_TEXT,
            },
            NAMED_INFRA_FAIL_ROW,
        ),
        (
            [],
            {"module_configs_str": MODEL_CONFIG_TOKEN},
            {**INFRA_FAIL_ROW, "module_configs_str": MODEL_CONFIG_TOKEN},
        ),
        (
            RUNNING_GRID_ROWS,
            {"reason": LONG_ERROR_TEXT},
            {**RUNNING_GRID_ROWS[0], "status": "fail", "error": LONG_ERROR_TEXT},
        ),
    ],
    ids=[
        "missing-grid-uses-name-and-run-identity",
        "empty-grid-uses-model-config-token",
        "running-row-keeps-full-error",
    ],
)
def test_failed_entry_row_keeps_full_error_name_config_and_run_identity(
    tmp_path, initial_rows, finalize_arguments, expected_row
):
    grid_path = tmp_path / "e3" / "grid_metrics.json"
    if initial_rows is not None:
        write_rows(grid_path, initial_rows)
    token_line = finalize_test_grid(
        grid_path,
        index=3,
        returncode=1,
        **{"reason": "exit 1", **finalize_arguments},
    )
    assert read_rows_at(grid_path) == [expected_row]
    assert token_line.startswith("ENTRY_GRID entry=3 rows=1 marked_failed=1 ")


@pytest.mark.parametrize(
    ("status", "process_passed", "expected_status"),
    [
        ("infra_fail", False, "infra_fail"),
        (None, False, None),
        ("", False, ""),
        ("unknown", False, "unknown"),
        ("fail", False, "xfail"),
        ("pass", True, "xpass"),
        ("pass", False, "pass"),
        ("xfail", False, "xfail"),
        ("xpass", False, "xpass"),
        ("running", False, "running"),
    ],
    ids=[
        "infra-fail",
        "missing",
        "empty",
        "unknown",
        "fail",
        "process-pass",
        "row-pass-process-fail",
        "xfail",
        "xpass",
        "running",
    ],
)
def test_status_after_expected_failure_only_converts_model_outcomes(
    status, process_passed, expected_status
):
    assert (
        entry_results.status_after_expected_failure(status, True, process_passed) == expected_status
    )
    assert entry_results.status_after_expected_failure(status, False, process_passed) == status


@pytest.mark.parametrize(
    ("returncode", "expected_to_fail", "expected_statuses"),
    [
        (1, True, ["pass", "xfail", "xfail", "xfail", "infra_fail", "xfail"]),
        (0, True, ["xpass", "xfail", "xfail", None, "infra_fail", "running"]),
        (1, False, ["pass", "xfail", "fail", "fail", "infra_fail", "fail"]),
        (0, False, ["pass", "xfail", "fail", None, "infra_fail", "running"]),
    ],
    ids=[
        "failed-process-expected",
        "successful-process-expected",
        "failed-process",
        "successful-process",
    ],
)
def test_finalize_applies_target_and_row_expected_failure_rules(
    tmp_path, returncode, expected_to_fail, expected_statuses
):
    grid_rows = [
        {"grid_entry_name": name, "status": status} for name, status, *_ in EXPECTED_FAILURE_CASES
    ]
    for row, (*_, row_expected_to_fail) in zip(grid_rows, EXPECTED_FAILURE_CASES, strict=True):
        if isinstance(row_expected_to_fail, bool):
            row["expected_to_fail"] = row_expected_to_fail
    grid_path = write_rows(tmp_path / "grid_metrics.json", grid_rows)
    finalize_test_grid(
        grid_path,
        returncode=returncode,
        reason="exit",
        expected_to_fail=expected_to_fail,
    )
    assert [row.get("status") for row in read_rows_at(grid_path)] == expected_statuses


def test_failure_annotations_sanitize_text_and_distinguish_xfail_from_xpass():
    failed_row = dict(PASSING_ROW, status="fail", error="bad %s\nvalue")
    assert entry_results.github_failure_annotations(
        [PASSING_ROW, failed_row, XFAIL_ROW, dict(XPASS_ROW_FIELDS, status="xpass")],
        "why %s",
    ) == [
        (
            "::warning::model smeoni_tiny-random-LlamaForCausalLM Hf.CausalLM| "
            + "tiny_config X| neuron "
            + "status=fail error=bad  s value"
        ),
        "::notice::model Titan_b_Async status=xfail reason=why  s error=loss is nan",
        "::warning::model google_gemma-4-E2B-it Hf| neuron xpass: prune its xfail entry",
    ]
    assert entry_results.github_failure_annotations([XFAIL_ROW]) == [
        (
            "::notice::model Titan_b_Async status=xfail "
            + "reason=xfail reason unset; using default reason error=loss is nan"
        )
    ]
    (preview_annotation,) = entry_results.github_failure_annotations(
        [dict(PASSING_ROW, status="infra_fail", error=LONG_ERROR_TEXT)]
    )
    expected_preview = LONG_ERROR_TEXT[:200].translate(entry_results.ANNOTATION_UNSAFE_CHARS)
    assert preview_annotation.endswith("status=infra_fail error=" + expected_preview)
    assert (
        entry_results.github_failure_annotations([PASSING_ROW]) == []
        and entry_results.github_failure_annotations(None) == []
    )


def test_failure_annotations_remove_newlines_that_could_inject_commands():
    injected_row = {
        "grid_entry_name": "t5\n::error::injected",
        "status": "fail\r::error::injected",
    }
    annotations = entry_results.github_failure_annotations(
        [
            injected_row,
            dict(injected_row, module_configs_str="g%_config X", error="e"),
        ]
    )
    assert len(annotations) == 2 and not any(
        "\n" in annotation or "\r" in annotation for annotation in annotations
    )
    assert annotations == [
        "::warning::model t5 ::error::injected status=fail ::error::injected error=",
        "::warning::model g  t5 ::error::injected status=fail ::error::injected error=e",
    ]

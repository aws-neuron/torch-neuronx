"""Pytest checks Hugging Face helpers for an entry.

The tests use independent literals.
"""

from __future__ import annotations

import sys

import moduscope_hugging_face_entry as hf_entry
import pytest
from moduscope_shard_planner import Shard
from moduscope_targets import Target

from conftest import (
    LLAMA_MODEL_ID,
    OLMO_MODEL_ID,
    PHI3_MODEL_ID,
    SAMPLE_HUGGING_FACE_MODEL_IDS,
    STABLELM_MODEL_ID,
    read_rows_at,
    write_rows,
)

PASS_ROW, EXPECTED_FAILURE_ROW, ERROR_ROW = (
    {"status": "pass"},
    {"status": "xfail"},
    {"status": "error"},
)
MARKED_FAIL_ROW = {"status": "fail", "expected_to_fail": True}
MARKED_EXPECTED_FAILURE_ROW = {"status": "xfail", "expected_to_fail": True}
GRID_FILE_NAME = "grid_metrics.json"
BENCHMARK_FILE_NAME = "benchmark_results.jsonl"
TEST_HUGGING_FACE_TARGET_DATA = Target(
    name="test_hf",
    instance="trn2.48xlarge",
    timeout_minutes=90,
    config="full_dense_decoder",
    shards=tuple(
        Shard(index, (model_id,)) for index, model_id in enumerate(SAMPLE_HUGGING_FACE_MODEL_IDS)
    ),
)
TINY_MODEL_ID = SAMPLE_HUGGING_FACE_MODEL_IDS[1]
TINY_MODEL_ROW = {
    "module_configs_str": "smeoni_tiny-random-LlamaForCausalLM_config X",
    "status": "pass",
}


def finish_test_entry(entry_dir, returncode, reason):
    return hf_entry.finalize_hugging_face_entry(
        TEST_HUGGING_FACE_TARGET_DATA,
        1,
        TINY_MODEL_ID,
        str(entry_dir),
        returncode=returncode,
        reason=reason,
        identity=None,
    )


@pytest.mark.parametrize(
    ("model_id", "normalized"),
    [
        (LLAMA_MODEL_ID, "meta-llama_Llama-3.1-8B"),
        (STABLELM_MODEL_ID, "stabilityai_stablelm-2-1_6b"),
        (PHI3_MODEL_ID, "microsoft_Phi-3-mini-4k-instruct"),
        ("gpt2", "gpt2"),
    ],
)  # fmt: skip
def test_grid_model_ids_replace_only_slashes_and_config_tokens_add_the_suffix(model_id, normalized):
    stored_model_id = hf_entry.grid_model_id(model_id)
    assert stored_model_id == normalized
    config_token = hf_entry.model_config_token(model_id)
    assert config_token == f"{normalized}_config"


def test_the_task_arguments_match_the_pod_contract():
    argv = hf_entry.build_moduscope_argv(
        OLMO_MODEL_ID, "/out/entries/entry-3", "full_dense_decoder"
    )
    assert argv == [
        sys.executable,
        "-m",
        "cli_scripts.task_runner",
        "--config-name=full_dense_decoder",
        "++task_config.hf_cfg_wishlist=['allenai/OLMo-1B-hf']",
        "++task_config.artifact_directory=/out/entries/entry-3",
        "++task_config.always_clean_artifacts=false",
        "++task_config.clean_artifacts_on_success=false",
    ]


@pytest.mark.parametrize(
    ("rows", "listed", "absorbed"),
    [
        (None, True, False), ([], True, False), ([PASS_ROW], False, False),
        ([PASS_ROW], True, True), ([EXPECTED_FAILURE_ROW], True, True),
        ([PASS_ROW, ERROR_ROW], True, False), ([MARKED_EXPECTED_FAILURE_ROW], False, True),
        ([MARKED_FAIL_ROW], False, False),
        ([PASS_ROW, MARKED_EXPECTED_FAILURE_ROW], False, True),
    ],
)  # fmt: skip
def test_exit_codes_are_absorbed_only_with_an_expectation_and_passing_rows(rows, listed, absorbed):
    is_absorbed = hf_entry.expected_failure_absorbs_exit(rows, listed=listed)
    assert is_absorbed is absorbed


def test_benchmark_aggregation_preserves_entry_order(tmp_path):
    expected = ['{"model":"first"}', '{"model":"first-2"}', '{"model":"second"}']
    entry_dirs = []
    for index, lines in enumerate((expected[:2], expected[2:])):
        entry_dir = tmp_path / f"entry-{index}"
        entry_dir.mkdir()
        (entry_dir / BENCHMARK_FILE_NAME).write_text("\n".join(lines) + "\n")
        entry_dirs.append(str(entry_dir))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    count = hf_entry.aggregate_benchmarks(str(out_dir), entry_dirs)

    assert count == len(expected)
    assert (out_dir / BENCHMARK_FILE_NAME).read_text().splitlines() == expected


def test_grid_aggregation_marks_a_missing_entry_as_infra_fail(tmp_path, capsys):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    missing_entry_dir = tmp_path / "entry-3"

    row_count, unreadable_entry_count = hf_entry.aggregate_grids(
        str(output_dir),
        [(3, OLMO_MODEL_ID, str(missing_entry_dir))],
        None,
    )

    missing_grid_path = missing_entry_dir / GRID_FILE_NAME
    assert row_count == unreadable_entry_count == 1
    assert read_rows_at(output_dir / GRID_FILE_NAME) == [
        {
            "grid_entry_name": OLMO_MODEL_ID,
            "grid_entry_index": 3,
            "mfu(%)": -1,
            "status": "infra_fail",
            "error": f"unreadable grid at {missing_grid_path}: "
            + f"no grid at {missing_grid_path}: the file does not exist.",
            "module_configs_str": "allenai_OLMo-1B-hf_config",
        }
    ]
    assert f"UNREADABLE_GRID entry=3 path={missing_grid_path}" in capsys.readouterr().out


def test_unlisted_model_keeps_exit_unless_wheel_marks_failure(tmp_path, capsys):
    entry_dir = tmp_path / "entry-1"
    write_rows(entry_dir / GRID_FILE_NAME, [TINY_MODEL_ROW])
    returncode = finish_test_entry(entry_dir, 2, "ModuScope exited with code 2")
    assert returncode == 2
    assert read_rows_at(entry_dir / GRID_FILE_NAME) == [TINY_MODEL_ROW]
    lines = capsys.readouterr().out.splitlines()
    assert f"MODUSCOPE_PROCESS_EXIT_CODE=2 entry=1 model={TINY_MODEL_ID}" in lines
    assert not any(line.startswith("XFAIL_ABSORBED") for line in lines)
    marked = dict(TINY_MODEL_ROW, status="fail", expected_to_fail=True)
    write_rows(entry_dir / GRID_FILE_NAME, [marked])
    returncode = finish_test_entry(entry_dir, 1, "ModuScope exited with code 1")
    assert returncode == 0
    assert read_rows_at(entry_dir / GRID_FILE_NAME) == [dict(marked, status="xfail")]
    lines = capsys.readouterr().out.splitlines()
    assert f"XFAIL_ABSORBED=1 model={TINY_MODEL_ID}" in lines
    (notice,) = [line for line in lines if line.startswith("::notice::")]
    assert "reason=xfail reason unset; using default reason" in notice


def test_target_xfail_marks_failure_and_absorbs_exit(tmp_path, capsys):
    entry_dir = tmp_path / "entry-1"
    failed_row = dict(TINY_MODEL_ROW, status="fail")
    write_rows(entry_dir / GRID_FILE_NAME, [failed_row])
    target = TEST_HUGGING_FACE_TARGET_DATA._replace(xfail={TINY_MODEL_ID: "known model failure"})

    returncode = hf_entry.finalize_hugging_face_entry(
        target,
        1,
        TINY_MODEL_ID,
        str(entry_dir),
        returncode=1,
        reason="ModuScope exited with code 1",
        identity=None,
    )

    assert returncode == 0
    assert read_rows_at(entry_dir / GRID_FILE_NAME) == [dict(failed_row, status="xfail")]
    lines = capsys.readouterr().out.splitlines()
    assert f"XFAIL_ABSORBED=1 model={TINY_MODEL_ID}" in lines
    (notice,) = [line for line in lines if line.startswith("::notice::")]
    assert "reason=known model failure" in notice


def test_cleanup_removes_the_individual_directory_after_a_timeout(tmp_path, capsys):
    entry_dir = tmp_path / "entry-1"
    (entry_dir / "individual/x/_profile_scratch").mkdir(parents=True)
    reason = "killed by the runner after 3s (exit code 124)"
    returncode = finish_test_entry(entry_dir, 124, reason)
    assert returncode == 124
    assert not (entry_dir / "individual").exists()
    assert "INDIVIDUAL_REMOVED=1" in capsys.readouterr().out.splitlines()
    (row,) = read_rows_at(entry_dir / GRID_FILE_NAME)
    assert (row["status"], row["error"]) == ("infra_fail", reason)

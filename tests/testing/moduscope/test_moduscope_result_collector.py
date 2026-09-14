"""Test relay."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import moduscope_result_collector as result_collector
import pytest
from moduscope_artifact_layout import (
    BENCHMARK_FILE_NAME,
    GRID_FILE_NAME,
    RUN_ID_ENV_VAR,
    STAGED_GRID_RENDERER_RELATIVE_PATH,
    run_id_filename,
)
from moduscope_artifact_layout import OUTPUT_SUBDIR as POD_OUTPUT_SUBDIR
from moduscope_result_collector import CACHE_KEYS_FILE_NAME, TOKENS_FILE_NAME

from conftest import RUN_ID, write_rows

RESULT_COLLECTOR_SCRIPT = Path(result_collector.__file__)
MODUSCOPE_WHEEL_TOKEN = "MODUSCOPE_WHEEL version=2.0 sha256=0123456789abcdef"
MODUSCOPE_WHEEL = {"version": "2.0", "sha256": "0123456789abcdef"}
AGGREGATE_GRID_HTML_FILE_NAME = run_id_filename(RUN_ID, "aggregate_grid_metrics.html")
STAMPED_RENDERER_RELATIVE_PATH = STAGED_GRID_RENDERER_RELATIVE_PATH.parent / run_id_filename(
    RUN_ID, STAGED_GRID_RENDERER_RELATIVE_PATH.name
)
AGGREGATE_FILE_NAMES = (
    run_id_filename(RUN_ID, f"aggregate_{GRID_FILE_NAME}"),
    run_id_filename(RUN_ID, f"aggregate_{BENCHMARK_FILE_NAME}"),
    run_id_filename(RUN_ID, f"aggregate_{CACHE_KEYS_FILE_NAME}"),
)
PASSING_GRID_ROW = {"grid_entry_name": "m0", "status": "pass", "mfu(%)": 16.79}
AGGREGATE_COUNTS_TEMPLATE = (
    "shards_read={shards} grid_rows={rows} benchmark_records={bench} cache_keys={keys}"
)
POD_OUTPUT_FILES = (
    run_id_filename(RUN_ID, GRID_FILE_NAME),
    run_id_filename(RUN_ID, "target_check.json"),
    run_id_filename(RUN_ID, "tokens.txt"),
    f"entries/entry-0/xla_dump/{run_id_filename(RUN_ID, 'module_0000.txt')}",
)
RENDERER_FUNCTION_PREFIX = 'def generate_report_html(rows, title="unset"):\n    '
WORKING_RENDERER_SOURCE = (
    RENDERER_FUNCTION_PREFIX + 'return "title=" + title + " rows=" + str(len(rows))\n'
)
LATER_RENDERER_SOURCE = RENDERER_FUNCTION_PREFIX + 'return "the later shard rendered"\n'
RAISING_RENDERER_SOURCE = RENDERER_FUNCTION_PREFIX + 'raise RuntimeError("boom")\n'
EXPECTED_STEP_SUMMARY = (
    "## Earlier step\n## ModuScope Grid Metrics\n\n`{counts}`\n\n`RELAY_ENTRIES_FILES=40`\n\n"
    + "| grid_entry_name | status | mfu(%) |\n| --- | --- | --- |\n"
    + "| m0 | pass | 16.79 |\n| m\\|1 | fail |  |\n| m2 | xfail | 16.79 |\n"
    + "\nModel statuses: m|1 (fail), m2 (xfail)\n"
)


pytestmark = pytest.mark.usefixtures("run_identity_env")


def _write_text_file(file_path: Path, text: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text)


def _aggregate_path(output_dir: Path, file_name: str) -> Path:
    return output_dir / run_id_filename(RUN_ID, f"aggregate_{file_name}")


def _write_shard_output(
    results_root: Path,
    shard_index: int,
    grid_rows,
    benchmark_lines=(),
    cache_keys=(),
    renderer_source=None,
) -> Path:
    shard_output_dir = results_root / f"moduscope-results-{shard_index}" / "moduscope"
    write_rows(shard_output_dir / run_id_filename(RUN_ID, GRID_FILE_NAME), grid_rows)
    _write_text_file(
        shard_output_dir / run_id_filename(RUN_ID, BENCHMARK_FILE_NAME),
        "".join(benchmark_line + "\n" for benchmark_line in benchmark_lines),
    )
    entry_dir = shard_output_dir / "entries" / f"entry-{shard_index}"
    _write_text_file(
        entry_dir / run_id_filename(RUN_ID, CACHE_KEYS_FILE_NAME),
        "".join(cache_key + "\n" for cache_key in cache_keys),
    )
    if renderer_source is not None:
        _write_text_file(shard_output_dir / STAMPED_RENDERER_RELATIVE_PATH, renderer_source)
    return shard_output_dir


def _run_aggregate_command(results_root: Path, output_dir: Path, capsys) -> list[str]:
    assert result_collector.main(["aggregate", str(results_root), "-d", str(output_dir)]) == 0
    return capsys.readouterr().out.splitlines()


def test_aggregate_requires_output_directory(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        result_collector.main(["aggregate", str(tmp_path)])
    assert error.value.code == 2
    assert "the following arguments are required: -d/--output-dir" in capsys.readouterr().err


def test_aggregate_requires_the_existing_run_id_variable(tmp_path, monkeypatch):
    monkeypatch.delenv(RUN_ID_ENV_VAR)
    with pytest.raises(ValueError, match=RUN_ID_ENV_VAR):
        result_collector.aggregate_shard_results(tmp_path, tmp_path / "out")


def test_shard_output_dirs_sorts_the_downloaded_shards(tmp_path):
    first_shard_dir = tmp_path / "moduscope-results-2" / "moduscope"
    second_shard_dir = tmp_path / "moduscope-results-10" / "moduscope"
    first_shard_dir.mkdir(parents=True)
    second_shard_dir.mkdir(parents=True)

    class ReverseOrderShardRoot:
        def glob(self, pattern):
            assert pattern == f"*/{result_collector.SHARD_SUBDIR}"
            return [second_shard_dir, first_shard_dir]

    assert result_collector.shard_output_dirs(ReverseOrderShardRoot()) == [
        first_shard_dir,
        second_shard_dir,
    ]


def test_grid_and_benchmark_aggregation_skips_missing_files_and_directories(tmp_path):
    grid_only = tmp_path / "grid-only"
    benchmark_only = tmp_path / "benchmark-only"
    directory_grid = tmp_path / "directory-grid"
    directory_benchmark = tmp_path / "directory-benchmark"
    write_rows(grid_only / run_id_filename(RUN_ID, GRID_FILE_NAME), [PASSING_GRID_ROW])
    _write_text_file(benchmark_only / run_id_filename(RUN_ID, BENCHMARK_FILE_NAME), '{"ok":1}\n')
    (directory_grid / run_id_filename(RUN_ID, GRID_FILE_NAME)).mkdir(parents=True)
    (directory_benchmark / run_id_filename(RUN_ID, BENCHMARK_FILE_NAME)).mkdir(parents=True)
    shard_dirs = [grid_only, benchmark_only, directory_grid, directory_benchmark]
    assert result_collector.aggregate_grids(shard_dirs) == [PASSING_GRID_ROW]
    assert result_collector.aggregate_benchmarks(shard_dirs) == ['{"ok":1}']


@pytest.mark.parametrize(
    ("token_text", "wheel"),
    [
        (None, None),
        ("RUN target=test\n", None),
        (f"RUN target=test\n{MODUSCOPE_WHEEL_TOKEN}\n", MODUSCOPE_WHEEL),
    ],
    ids=["no-file", "no-token", "token"],
)
def test_aggregate_grids_adds_the_shard_moduscope_wheel_to_every_row(tmp_path, token_text, wheel):
    grid_rows = [PASSING_GRID_ROW, dict(PASSING_GRID_ROW, grid_entry_name="m1")]
    shard_output_dir = _write_shard_output(tmp_path, 0, grid_rows)
    if token_text is not None:
        _write_text_file(shard_output_dir / run_id_filename(RUN_ID, TOKENS_FILE_NAME), token_text)
    expected = [{**row, "moduscope_wheel": wheel} for row in grid_rows] if wheel else grid_rows

    assert result_collector.aggregate_grids([shard_output_dir]) == expected


def test_aggregate_grids_keeps_rows_when_wheel_metadata_is_ambiguous(tmp_path, capsys):
    shard_output_dir = _write_shard_output(tmp_path, 0, [PASSING_GRID_ROW])
    for run_id in (RUN_ID, "other-run"):
        _write_text_file(
            shard_output_dir / run_id_filename(run_id, TOKENS_FILE_NAME),
            MODUSCOPE_WHEEL_TOKEN + "\n",
        )

    assert result_collector.aggregate_grids([shard_output_dir]) == [PASSING_GRID_ROW]
    warning = capsys.readouterr().out
    assert "::warning::Wheel metadata is ambiguous:" in warning
    assert "multiple run-ID files for tokens.txt" in warning
    assert "The collector omits the wheel metadata." in warning


def test_aggregate_command_sorts_cache_keys_before_writing(tmp_path, monkeypatch):
    class ReverseIteratingSet(set):
        def __iter__(self):
            return iter(("aten::mm", "aten::add"))

    monkeypatch.setattr(
        result_collector,
        "aggregate_cache_keys",
        lambda _: ReverseIteratingSet(("aten::add", "aten::mm")),
    )
    output_dir = tmp_path / "out"
    result_collector.aggregate_shard_results(tmp_path, output_dir)
    assert _aggregate_path(output_dir, CACHE_KEYS_FILE_NAME).read_text() == "aten::add\naten::mm\n"


def test_cache_key_aggregation_strips_whitespace_and_ignores_blank_lines(tmp_path):
    shard_output_dir = tmp_path / "moduscope"
    _write_text_file(
        shard_output_dir / "entries" / "entry-0" / run_id_filename(RUN_ID, CACHE_KEYS_FILE_NAME),
        "  aten::mm  \n \t \n",
    )
    assert result_collector.aggregate_cache_keys([shard_output_dir]) == {"aten::mm"}


def test_gather_copies_shard_outputs_and_reports_missing_outputs_and_file_count(tmp_path, capsys):
    results_root, destination_root = (tmp_path / "results", tmp_path / "relay")
    for shard_index in (0, 1):
        for relative_path in POD_OUTPUT_FILES:
            _write_text_file(
                results_root / f"moduscope-results-{shard_index}" / "moduscope" / relative_path,
                f"{shard_index}:{relative_path}",
            )
    _write_text_file(
        results_root
        / "moduscope-results-2"
        / POD_OUTPUT_SUBDIR
        / run_id_filename(RUN_ID, GRID_FILE_NAME),
        "the pod's own layout",
    )
    (results_root / "moduscope-results-3" / "test-reports").mkdir(parents=True)
    assert result_collector.main(["gather", str(results_root), str(destination_root)]) == 0
    copied_files = sorted(
        file_path.relative_to(destination_root).as_posix()
        for file_path in destination_root.rglob("*")
        if file_path.is_file()
    )
    assert copied_files == sorted(
        f"moduscope-entries/moduscope-results-{shard_index}/{relative_path}"
        for shard_index in (0, 1)
        for relative_path in POD_OUTPUT_FILES
    )
    assert (
        destination_root / "moduscope-entries" / "moduscope-results-1" / POD_OUTPUT_FILES[-1]
    ).read_text() == f"1:{POD_OUTPUT_FILES[-1]}"
    source_files = [
        results_root / f"moduscope-results-{shard_index}" / "moduscope" / relative_path
        for shard_index in (0, 1)
        for relative_path in POD_OUTPUT_FILES
    ]
    assert all(source_file.is_file() for source_file in source_files)
    # The workflow reads the last line into RELAY_ENTRIES_FILES.
    assert capsys.readouterr().out.splitlines() == [
        "SHARD_WITHOUT_OUTPUTS=moduscope-results-2",
        "SHARD_WITHOUT_OUTPUTS=moduscope-results-3",
        f"RELAY_ENTRIES_FILES={2 * len(POD_OUTPUT_FILES)}",
    ]
    assert (
        result_collector.gather_shard_outputs(results_root / "absent", destination_root / "empty")
        == 0
    )
    assert (destination_root / "empty" / "moduscope-entries").is_dir()


def test_gather_command_prints_the_relay_entries_files_count(tmp_path):
    results_root, destination_root = tmp_path / "results", tmp_path / "relay"
    _write_shard_output(results_root, 0, [])
    completed_process = subprocess.run(
        [sys.executable, str(RESULT_COLLECTOR_SCRIPT), "gather", results_root, destination_root],
        capture_output=True,
        text=True,
    )
    assert (completed_process.returncode, completed_process.stdout) == (
        0,
        "RELAY_ENTRIES_FILES=3\n",
    ), completed_process.stderr


def test_aggregate_command_writes_grid_benchmarks_cache_keys_html_and_summary(
    tmp_path, monkeypatch, capsys
):
    long_error = "first line\n" + "x" * 5000 + "\nlast line"
    failed_row = dict(
        PASSING_GRID_ROW, grid_entry_name="m|1", status="fail", error=long_error, **{"mfu(%)": None}
    )
    _write_shard_output(
        tmp_path,
        0,
        [PASSING_GRID_ROW, failed_row],
        benchmark_lines=["{not json", '{"ok":1}'],
        cache_keys=["aten::mm", "aten::add"],
    )
    expected_failure_row = dict(PASSING_GRID_ROW, grid_entry_name="m2", status="xfail")
    _write_shard_output(
        tmp_path,
        1,
        [expected_failure_row],
        cache_keys=["aten::add"],
        renderer_source=WORKING_RENDERER_SOURCE,
    )
    _write_text_file(
        tmp_path
        / "moduscope-results-9"
        / POD_OUTPUT_SUBDIR
        / run_id_filename(RUN_ID, GRID_FILE_NAME),
        json.dumps([dict(PASSING_GRID_ROW, status="stray")]),
    )
    output_dir, summary_path = (tmp_path / "relay" / "integ-tests", tmp_path / "summary.md")
    summary_path.write_text("## Earlier step\n")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))
    monkeypatch.setenv("RELAY_ENTRIES_FILES", "40")
    output_lines = _run_aggregate_command(tmp_path, output_dir, capsys)
    aggregate_counts = AGGREGATE_COUNTS_TEMPLATE.format(shards=2, rows=3, bench=2, keys=2)
    renderer_path = tmp_path / "moduscope-results-1" / "moduscope" / STAMPED_RENDERER_RELATIVE_PATH
    assert aggregate_counts in output_lines
    assert f"HTML_RENDERED=1 renderer={renderer_path}" in output_lines
    aggregate_rows = json.loads(_aggregate_path(output_dir, GRID_FILE_NAME).read_text())
    assert aggregate_rows == [PASSING_GRID_ROW, failed_row, expected_failure_row]
    assert aggregate_rows[1]["error"] == long_error
    assert _aggregate_path(output_dir, BENCHMARK_FILE_NAME).read_bytes() == b'{not json\n{"ok":1}\n'
    assert (
        _aggregate_path(output_dir, CACHE_KEYS_FILE_NAME).read_bytes() == b"aten::add\naten::mm\n"
    )
    assert (
        output_dir / AGGREGATE_GRID_HTML_FILE_NAME
    ).read_text() == "title=ModuScope Grid Metrics rows=3"
    assert all(RUN_ID in output_path.name for output_path in output_dir.iterdir())
    assert summary_path.read_text() == EXPECTED_STEP_SUMMARY.format(counts=aggregate_counts)


def test_step_summary_omits_file_count_and_failure_sections_when_empty():
    assert result_collector.build_step_summary([PASSING_GRID_ROW], "counts") == (
        "## ModuScope Grid Metrics\n\n`counts`\n\n"
        + "| grid_entry_name | status | mfu(%) |\n| --- | --- | --- |\n| m0 | pass | 16.79 |\n"
    )


@pytest.mark.parametrize(
    ("status", "expected_status_line"), [("infra_fail", True), ("xpass", False)]
)
def test_step_summary_lists_infra_fail_but_not_xpass_in_statuses(status, expected_status_line):
    summary_text = result_collector.build_step_summary(
        [dict(PASSING_GRID_ROW, status=status)], "counts"
    )
    assert ("Model statuses:" in summary_text) is expected_status_line


def test_empty_github_step_summary_path_writes_no_summary_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", "")
    _write_shard_output(
        tmp_path, 0, [PASSING_GRID_ROW], benchmark_lines=['{"m":0}'], cache_keys=["aten::mm"]
    )
    output_dir = tmp_path / "out"
    _run_aggregate_command(tmp_path, output_dir, capsys)
    assert sorted(file_path.name for file_path in output_dir.iterdir()) == sorted(
        AGGREGATE_FILE_NAMES
    )


def test_html_is_rendered_by_the_first_shard_that_ships_a_renderer(tmp_path, capsys):
    _write_shard_output(tmp_path, 0, [PASSING_GRID_ROW])
    _write_shard_output(tmp_path, 1, [PASSING_GRID_ROW], renderer_source=WORKING_RENDERER_SOURCE)
    _write_shard_output(tmp_path, 2, [PASSING_GRID_ROW], renderer_source=LATER_RENDERER_SOURCE)
    output_dir = tmp_path / "out"
    renderer_path = tmp_path / "moduscope-results-1" / "moduscope" / STAMPED_RENDERER_RELATIVE_PATH
    output_lines = _run_aggregate_command(tmp_path, output_dir, capsys)
    assert f"HTML_RENDERED=1 renderer={renderer_path}" in output_lines
    assert any(line.startswith("::warning::renderer SHA-256 mismatch") for line in output_lines)
    assert (
        output_dir / AGGREGATE_GRID_HTML_FILE_NAME
    ).read_text() == "title=ModuScope Grid Metrics rows=3"


def test_empty_grid_rows_write_no_html_even_when_a_renderer_exists(tmp_path, capsys):
    _write_shard_output(tmp_path, 0, [], renderer_source=WORKING_RENDERER_SOURCE)
    assert "HTML_RENDERED=0 reason=no grid rows" in _run_aggregate_command(
        tmp_path, tmp_path, capsys
    )
    assert not (tmp_path / AGGREGATE_GRID_HTML_FILE_NAME).exists()


@pytest.mark.parametrize(
    ("renderer_source", "duplicate_renderer", "failure_reason"),
    [
        (None, False, "unavailable. Add the copied renderer to a shard artifact."),
        (RAISING_RENDERER_SOURCE, False, "RuntimeError: boom. Check the copied renderer."),
        (
            RENDERER_FUNCTION_PREFIX + "return None\n",
            False,
            "TypeError: renderer returned NoneType, not str. Check the copied renderer.",
        ),
        ("import sys\n\nsys.exit(3)\n", False, "SystemExit: 3. Check the copied renderer."),
        (WORKING_RENDERER_SOURCE, True, "ValueError:"),
    ],
    ids=["none", "raises", "no-document", "exits-at-import", "duplicate"],
)
def test_renderer_failures_print_one_status_line_and_preserve_other_outputs(
    tmp_path, monkeypatch, capsys, renderer_source, duplicate_renderer, failure_reason
):
    shard_output_dir = _write_shard_output(
        tmp_path,
        0,
        [PASSING_GRID_ROW],
        benchmark_lines=['{"m":0}'],
        cache_keys=["aten::mm"],
        renderer_source=renderer_source,
    )
    if duplicate_renderer:
        duplicate_name = run_id_filename("other-run", STAGED_GRID_RENDERER_RELATIVE_PATH.name)
        _write_text_file(
            shard_output_dir / STAGED_GRID_RENDERER_RELATIVE_PATH.parent / duplicate_name,
            WORKING_RENDERER_SOURCE,
        )
    output_dir, summary_path = (tmp_path / "out", tmp_path / "summary.md")
    _write_text_file(output_dir / AGGREGATE_GRID_HTML_FILE_NAME, "previous report")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))
    renderer_warning_lines = [
        line
        for line in _run_aggregate_command(tmp_path, output_dir, capsys)
        if line.startswith("::warning::renderer ")
    ]
    expected_status_prefix = f"::warning::renderer {failure_reason}"
    if duplicate_renderer:
        assert len(renderer_warning_lines) == 1
        assert renderer_warning_lines[0].startswith(expected_status_prefix)
        assert "multiple run-ID files for grid_report_html.py" in renderer_warning_lines[0]
    else:
        assert renderer_warning_lines == [expected_status_prefix]
    assert (output_dir / AGGREGATE_GRID_HTML_FILE_NAME).read_text() == "previous report"
    assert [(output_dir / file_name).read_bytes() for file_name in AGGREGATE_FILE_NAMES] == [
        json.dumps([PASSING_GRID_ROW], indent=2).encode(),
        b'{"m":0}\n',
        b"aten::mm\n",
    ]
    assert "| m0 | pass | 16.79 |" in summary_path.read_text()


@pytest.mark.parametrize(
    ("duplicate_file_name", "status_key", "counts", "surviving_aggregates"),
    [
        (
            GRID_FILE_NAME,
            "GRID_AGGREGATED",
            {"rows": 0, "bench": 1},
            {BENCHMARK_FILE_NAME: b'{"m":0}\n', CACHE_KEYS_FILE_NAME: b"aten::mm\n"},
        ),
        (
            BENCHMARK_FILE_NAME,
            "BENCHMARKS_AGGREGATED",
            {"rows": 1, "bench": 0},
            {
                GRID_FILE_NAME: json.dumps([PASSING_GRID_ROW], indent=2).encode(),
                CACHE_KEYS_FILE_NAME: b"aten::mm\n",
            },
        ),
    ],
    ids=["grid", "benchmark"],
)
def test_duplicate_run_id_file_prints_one_failure_line_and_writes_other_outputs(
    tmp_path, monkeypatch, capsys, duplicate_file_name, status_key, counts, surviving_aggregates
):
    shard_output_dir = _write_shard_output(
        tmp_path, 0, [PASSING_GRID_ROW], benchmark_lines=['{"m":0}'], cache_keys=["aten::mm"]
    )
    # The collector refuses two run-ID files before reading either, so the content is never read.
    _write_text_file(shard_output_dir / run_id_filename("other-run", duplicate_file_name), "x\n")
    output_dir, summary_path = tmp_path / "out", tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))

    assert result_collector.main(["aggregate", str(tmp_path), "-d", str(output_dir)]) == 0
    captured = capsys.readouterr()
    output_lines = captured.out.splitlines()
    status_lines = [line for line in output_lines if line.startswith(f"{status_key}=")]

    assert len(status_lines) == 1
    assert status_lines[0].startswith(f"{status_key}=0 reason=ValueError:")
    assert f"multiple run-ID files for {duplicate_file_name}" in status_lines[0]
    assert "Traceback" not in captured.out + captured.err
    aggregate_counts = AGGREGATE_COUNTS_TEMPLATE.format(shards=1, keys=1, **counts)
    assert aggregate_counts in output_lines
    assert {file_path.name: file_path.read_bytes() for file_path in output_dir.iterdir()} == {
        run_id_filename(RUN_ID, f"aggregate_{file_name}"): content
        for file_name, content in surviving_aggregates.items()
    }
    summary_text = summary_path.read_text()
    assert aggregate_counts in summary_text
    assert ("| m0 | pass | 16.79 |" in summary_text) is (counts["rows"] == 1)


def test_empty_inputs_preserve_outputs_and_an_unreadable_grid_stops_aggregation(tmp_path, capsys):
    output_dir = tmp_path / "out"
    for file_name in (*AGGREGATE_FILE_NAMES, AGGREGATE_GRID_HTML_FILE_NAME):
        _write_text_file(output_dir / file_name, "previous\n")
    output_lines = _run_aggregate_command(tmp_path / "absent", output_dir, capsys)
    assert AGGREGATE_COUNTS_TEMPLATE.format(shards=0, rows=0, bench=0, keys=0) in output_lines
    assert {
        file_path.name: file_path.read_text() for file_path in output_dir.iterdir()
    } == dict.fromkeys((*AGGREGATE_FILE_NAMES, AGGREGATE_GRID_HTML_FILE_NAME), "previous\n")
    _write_shard_output(tmp_path, 0, {"not": "rows"})
    with pytest.raises(ValueError, match="not a list of rows"):
        result_collector.main(["aggregate", str(tmp_path), "-d", str(output_dir)])
    assert _aggregate_path(output_dir, GRID_FILE_NAME).read_text() == "previous\n"

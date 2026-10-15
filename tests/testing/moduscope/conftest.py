"""Provide shared sample data and fixtures that ModuScope test modules import directly."""

import contextlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

MODUSCOPE_MODULES_DIR = Path(__file__).resolve().parents[3] / "testing" / "moduscope"
sys.path.insert(0, str(MODUSCOPE_MODULES_DIR))
TEST_COMPONENT_EQUIVALENCE_TARGET = "component-equiv-dense"
RUN_ID = "33700000001"
REPOSITORY = "aws-neuron/private-torch-neuronx"
RUN_IDENTITY_ENV = {
    "MODUSCOPE_GITHUB_RUN_ID": RUN_ID,
    "MODUSCOPE_GITHUB_REPOSITORY": REPOSITORY,
    "MODUSCOPE_GITHUB_RUN_ATTEMPT": "1",
}
WORKLOAD_LINK = f"https://github.com/{REPOSITORY}/actions/runs/{RUN_ID}"
RUN_IDENTITY_FIELDS = {"workload_id": RUN_ID, "workload_link": WORKLOAD_LINK}
OLMO_MODEL_ID = "allenai/OLMo-1B-hf"
PHI3_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
STABLELM_MODEL_ID = "stabilityai/stablelm-2-1_6b"
LLAMA_MODEL_ID = "meta-llama/Llama-3.1-8B"
SAMPLE_HUGGING_FACE_MODEL_IDS = (
    "cookecd1/llama-2-7b-miniguanaco",
    "smeoni/tiny-random-LlamaForCausalLM",
    "0xWe11es/camel-llama2-h1024-w1",
)
SAMPLE_TITAN_ENTRY_NAMES = (
    "Qwen3_8B_MultiCore_TP4_FSDP16_36L_SAC",
    "Qwen3_8B_MultiCore_TP1_FSDP64_36L_FAC",
    "Llama3_8B_MultiCore_TP4_FSDP16_32L_SAC",
)
PROCESS_TERMINATION_GRACE_SECONDS = 1.0
# Keep this slack no larger than one termination grace period.
# A second grace period must fail the test.
TIMING_SLACK_SECONDS = 1.0


def run_test_shard(runner, target, shard, output_dir, timeout_seconds, clock=None):
    """Dispatch a shard runner test through the shared context."""
    from moduscope_artifact_layout import read_run_identity
    from moduscope_targets import ShardRunContext

    clock = time if clock is None else clock
    return runner.run_shard(
        ShardRunContext(
            target, shard, str(output_dir), read_run_identity(), clock.monotonic() + timeout_seconds
        )
    )


def write_rows(grid_path, grid_rows):
    """Write grid rows as JSON and return the path."""
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path.write_text(json.dumps(grid_rows))
    return grid_path


def read_rows_at(grid_path):
    """Read and return grid rows from JSON."""
    return json.loads(grid_path.read_text())


def write_bytes(file_path, size_bytes):
    """Write a file with the specified size in bytes."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(b"x" * size_bytes)


def write_device_results(results_dir, threshold_result="PASS"):
    """Write sample component-equivalence results for one device."""
    device_name, _, compiled_marker = results_dir.name.partition("-")
    results_dir.mkdir(parents=True)
    (results_dir / "report.json").write_text('{"summary": {"collected": 60}}')
    summary_text = (
        f"  Total: 60   Passed: 60\n\nTHRESHOLD_RESULT: {threshold_result}\n\n{'=' * 40}\n"
    )
    (results_dir / "summary.txt").write_text(summary_text)
    grid_row = {
        "device": device_name,
        "compile": bool(compiled_marker),
        "threshold_result": threshold_result,
    }
    write_rows(results_dir / "grid_metrics.json", [grid_row])


@pytest.fixture(autouse=True)
def clear_run_identity_env(monkeypatch):
    """Clear run identity variables before each test."""
    for variable_name in (*RUN_IDENTITY_ENV, "GITHUB_STEP_SUMMARY", "RELAY_ENTRIES_FILES"):
        monkeypatch.delenv(variable_name, raising=False)


@pytest.fixture
def run_identity_env(monkeypatch):
    """Set the complete run identity environment for one test."""
    for variable_name, variable_value in RUN_IDENTITY_ENV.items():
        monkeypatch.setenv(variable_name, variable_value)


@pytest.fixture
def compile_cache_directories(tmp_path, monkeypatch):
    """Redirect shared compile caches into the test directory."""
    import moduscope_runner_support as runner_support

    monkeypatch.setattr(runner_support, "HLO_CACHE_DIR", str(tmp_path / "hlo_cache"))
    monkeypatch.setattr(runner_support, "NEFF_CACHE_DIR", str(tmp_path / "neff_cache"))
    monkeypatch.setattr(runner_support, "INDUCTOR_CACHE_GLOB", str(tmp_path / "torchinductor_*"))
    return tmp_path


@pytest.fixture
def child_process_id_file(tmp_path):
    """Provide a process ID file and stop each listed child after the test."""
    process_id_path = tmp_path / "pids.txt"
    yield process_id_path
    with contextlib.suppress(FileNotFoundError, ValueError):
        for process_id_text in process_id_path.read_text().split():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(process_id_text), signal.SIGKILL)


@pytest.fixture(autouse=True)
def clear_recorded_token_lines():
    """Clear recorded token lines before each test."""
    import moduscope_process as entry_results

    entry_results._token_lines.clear()

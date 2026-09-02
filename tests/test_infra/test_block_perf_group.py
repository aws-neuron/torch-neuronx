"""Unit tests for platform-specific block performance thresholds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BLOCK_PERF_DIR = Path(__file__).resolve().parent.parent / "blocks" / "block_perf"
sys.path.insert(0, str(BLOCK_PERF_DIR))

import run_group  # noqa: E402
from run_group import build_command  # noqa: E402


@pytest.fixture
def torchrun_test():
    return {
        "benchmark_runs": 20,
        "block": "qwen3_torchtitan",
        "id": "qwen3-test",
        "mfu": {
            "trn2": {"threshold": 10.0},
            "trn3pds-b1": {"threshold": 20.0},
        },
        "n_layers": 4,
        "nproc": 4,
        "preset": "qwen3-8b-tp",
        "runner": "torchrun",
        "tp_size": 4,
        "warmup_runs": 20,
    }


@pytest.mark.parametrize(
    "platform, expected",
    [
        ("trn2", "9.500000"),
        ("trn3pds-b1", "19.000000"),
    ],
)
def test_build_command_selects_platform_threshold(torchrun_test, platform, expected):
    cmd = build_command(
        torchrun_test,
        platform,
        skip_mfu_threshold=False,
        mfu_threshold_multiplier=0.95,
    )

    threshold_index = cmd.index("--mfu-threshold")
    assert cmd[threshold_index + 1] == expected


def test_build_command_skips_threshold(torchrun_test):
    cmd = build_command(
        torchrun_test,
        "trn3pds-b1",
        skip_mfu_threshold=True,
    )

    assert "--mfu-threshold" not in cmd


def test_build_command_rejects_missing_platform_threshold(torchrun_test):
    del torchrun_test["mfu"]["trn3pds-b1"]

    with pytest.raises(
        ValueError,
        match="Missing MFU threshold for test 'qwen3-test' on platform 'trn3pds-b1'",
    ):
        build_command(
            torchrun_test,
            "trn3pds-b1",
            skip_mfu_threshold=False,
        )


def test_main_rejects_all_missing_thresholds_before_running(tmp_path, monkeypatch, capsys):
    config = tmp_path / "group.json"
    config.write_text(
        json.dumps(
            {
                "tests": [
                    {"id": "missing-explicit", "runner": "torchrun"},
                    {"id": "missing-default"},
                    {"id": "present", "mfu": {"trn3pds-b1": {"threshold": 10.0}}},
                    {"id": "skipped", "runner": "torchrun", "skip": True},
                    {"id": "singlecore", "runner": "singlecore"},
                ]
            }
        )
    )
    monkeypatch.setattr(
        run_group,
        "run_test",
        lambda *args, **kwargs: pytest.fail("validation must happen before the run loop"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_group.py",
            "--config",
            str(config),
            "--platform",
            "trn3pds-b1",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        run_group.main()

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "missing-explicit/trn3pds-b1" in stderr
    assert "missing-default/trn3pds-b1" in stderr
    assert "skipped/trn3pds-b1" not in stderr
    assert "singlecore/trn3pds-b1" not in stderr


@pytest.mark.parametrize("config_name", ["qwen3.json", "gpt_oss.json"])
def test_active_mfu_tests_define_each_platform(config_name):
    config_path = BLOCK_PERF_DIR / "group_configs" / config_name
    config = json.loads(config_path.read_text())

    for test in config["tests"]:
        if test.get("runner", "torchrun") != "torchrun" or test.get("skip"):
            continue

        assert set(test["mfu"]) == {"trn2", "trn3pds-b1"}, test["id"]
        for platform in ("trn2", "trn3pds-b1"):
            golden = test["mfu"][platform]
            assert set(golden) == {"baseline", "baseline_date", "threshold"}
            assert 0 < golden["threshold"] <= golden["baseline"]

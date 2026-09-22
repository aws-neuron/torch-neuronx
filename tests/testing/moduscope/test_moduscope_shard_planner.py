"""Test generic shard planning."""

from __future__ import annotations

import inspect

import moduscope_shard_planner as shard_planner
import moduscope_targets as targets


def test_build_shards_keeps_group_and_member_order():
    planned_shards = shard_planner.build_shards(
        (("a/three", "a/one"), ("a/two",)),
    )
    assert planned_shards == (
        shard_planner.Shard(0, ("a/three", "a/one")),
        shard_planner.Shard(1, ("a/two",)),
    )
    assert [shard.label for shard in planned_shards] == ["a/three +1", "a/two"]


def test_hand_built_shards_use_target_agnostic_planning():
    shards = (
        shard_planner.Shard(0, ("alpha", "beta")),
        shard_planner.Shard(1, ("gamma",)),
    )
    target = targets.Target(
        name="synthetic",
        timeout_minutes=45,
        adapter="hugging-face",
        shards=shards,
    )

    assert targets.shard_plan(target) == list(shards)
    assert targets.shard_matrix(target) == [
        {"index": 0, "label": "alpha +1"},
        {"index": 1, "label": "gamma"},
    ]
    assert targets.plan(target, 0, 10) == {
        "shards": [
            {"index": 0, "label": "alpha +1"},
            {"index": 1, "label": "gamma"},
        ],
        "timeout_minutes": 45,
        "inner_timeout_seconds": 2100,
        "instance": "trn2.48xlarge",
        "relay_artifact": "custom-test-artifacts",
    }

    source = inspect.getsource(shard_planner).lower()
    for forbidden_text in (
        "models",
        "entries",
        "model_config_shards",
        "hf",
        "titan",
        "component",
        "hugging_face",
        "grid_model_id",
    ):
        assert forbidden_text not in source

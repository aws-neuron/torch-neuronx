"""
Tests for io_mutation.py.

Organised into:
  1. Parametrized unit tests for every helper (pure function, all edge-cases).
  2. Parametrized integration tests that verify the full pipeline behaviour using
     custom-constructed PassMutations (no production FX passes):
       - apply_input_mutations ↔ apply_output_mutations roundtrip
       - get_alias_output_buffer_map and get_mutation_copy_back_inputs
"""

import pytest
import torch

from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
    IOMutation,
    MutationKind,
    PassMutations,
    _shift_index_through_mutation,
    _translate_indices,
    apply_input_mutations,
    apply_output_mutations,
    get_alias_output_buffer_map,
    get_mutation_copy_back_inputs,
    rebase_pre_aot_pass_mutations,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def inp_remove(indices, name="P"):
    return IOMutation(kind=MutationKind.REMOVE, indices=indices, pass_name=name)


def inp_add(indices, key="none", params=None, name="P"):
    return IOMutation(
        kind=MutationKind.ADD,
        indices=indices,
        transform_key=key,
        transform_params=params or {},
        pass_name=name,
    )


def out_remove(indices, key="none", params=None, name="P"):
    return IOMutation(
        kind=MutationKind.REMOVE,
        indices=indices,
        transform_key=key,
        transform_params=params or {},
        pass_name=name,
    )


def out_add(indices, key="none", params=None, name="P"):
    return IOMutation(
        kind=MutationKind.ADD,
        indices=indices,
        transform_key=key,
        transform_params=params or {},
        pass_name=name,
    )


def alias(out_indices, in_indices, is_mutation=False, name="P"):
    return IOMutation(
        kind=MutationKind.ALIAS,
        indices=out_indices,
        alias_input_indices=in_indices,
        is_mutation=is_mutation,
        pass_name=name,
    )


def pm(name, in_muts=None, out_muts=None):
    return PassMutations(
        pass_name=name, input_mutations=in_muts or [], output_mutations=out_muts or []
    )


# A transform that injects a sentinel string value, used to test ADD output transforms
SENTINEL = "injected"


def _make_sentinel_pass(indices):
    """Custom PassMutations: input ADD that injects SENTINEL strings."""
    from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import TRANSFORM_REGISTRY

    TRANSFORM_REGISTRY["sentinel"] = lambda idx, dev, params, orig: SENTINEL
    return pm("SentinelAdd", in_muts=[inp_add(indices, key="sentinel")])


# ===========================================================================
# 1. _shift_index_through_mutation
# ===========================================================================


@pytest.mark.parametrize(
    "idx,mutation,forward,expected",
    [
        # ALIAS is always no-op
        (3, alias([0], [0]), True, 3),
        (3, alias([0], [0]), False, 3),
        # None propagates
        (None, inp_remove([1]), True, None),
        (None, inp_remove([1]), False, None),
        # ADD forward: shift up when add_pos <= idx
        (3, inp_add([1]), True, 4),  # 1 <= 3: +1
        (3, inp_add([3]), True, 4),  # 3 <= 3: +1
        (3, inp_add([4]), True, 3),  # 4 > 3: no shift
        (3, inp_add([1, 2]), True, 5),  # two adds <= 3
        (0, inp_add([0]), True, 1),  # add at 0
        # REMOVE forward: None if removed; else shift down
        (2, inp_remove([2]), True, None),
        (3, inp_remove([1]), True, 2),
        (3, inp_remove([4]), True, 3),
        (3, inp_remove([0, 2]), True, 1),
        (0, inp_remove([1]), True, 0),
        # ADD backward: undo (shift down; None if idx was the added slot)
        (2, inp_add([2]), False, None),
        (3, inp_add([1]), False, 2),
        (3, inp_add([4]), False, 3),
        (0, inp_add([1]), False, 0),
        # REMOVE backward: undo (shift up)
        (2, inp_remove([2]), False, 3),
        (2, inp_remove([3]), False, 2),
        (1, inp_remove([0, 1]), False, 3),
    ],
)
def test_shift_index(idx, mutation, forward, expected):
    assert _shift_index_through_mutation(idx, mutation, forward=forward) == expected


# ===========================================================================
# 2. _translate_indices
# ===========================================================================


@pytest.mark.parametrize(
    "in_idx,out_idx,passes,forward,exp_in,exp_out",
    [
        ([1, 3], [0, 2], [], True, [1, 3], [0, 2]),
        ([1, 3], [0, 2], [], False, [1, 3], [0, 2]),
        ([0, 2], [0], [pm("P", in_muts=[inp_add([1])])], True, [0, 3], [0]),
        ([0, 2, 1], [], [pm("P", in_muts=[inp_remove([1])])], True, [0, 1, None], []),
        ([], [1, 3], [pm("P", out_muts=[out_add([2])])], True, [], [1, 4]),
        ([], [0, 2, 1], [pm("P", out_muts=[out_remove([1])])], True, [], [0, 1, None]),
        (
            [3],
            [3],
            [pm("P", in_muts=[alias([0], [0])], out_muts=[alias([1], [1])])],
            True,
            [3],
            [3],
        ),
        (
            [2],
            [],
            [pm("P1", in_muts=[inp_remove([2])]), pm("P2", in_muts=[inp_add([0])])],
            True,
            [None],
            [],
        ),
        ([0, 1], [], [pm("P", in_muts=[inp_remove([1])])], False, [0, 2], []),
    ],
)
def test_translate_indices(in_idx, out_idx, passes, forward, exp_in, exp_out):
    got_in, got_out = _translate_indices(in_idx, out_idx, passes, forward=forward)
    assert got_in == exp_in
    assert got_out == exp_out


# ===========================================================================
# 3. apply_input_mutations
# ===========================================================================


@pytest.mark.parametrize(
    "inputs,passes,expected",
    [
        ([1, 2, 3], [], [1, 2, 3]),
        (["a", "b", "c"], [pm("P", in_muts=[inp_remove([1])])], ["a", "c"]),
        (["a", "b", "c", "d"], [pm("P", in_muts=[inp_remove([0, 2])])], ["b", "d"]),
        ([1, 2, 3], [pm("P", in_muts=[alias([0], [0])])], [1, 2, 3]),
    ],
)
def test_apply_input_mutations_simple(inputs, passes, expected):
    assert apply_input_mutations(list(inputs), passes, "cpu") == expected


def test_apply_input_mutations_add_shard():
    from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import TRANSFORM_REGISTRY

    passes = [pm("P", in_muts=[inp_add([2], "shard_index", {"shard_index": 5})])]
    result = apply_input_mutations(["a", "b"], passes, "cpu")
    assert result[0] == "a" and result[1] == "b" and result[2].item() == 5


def test_apply_input_mutations_multi_pass():
    passes = [
        pm("P1", in_muts=[inp_remove([1])]),
        pm("P2", in_muts=[inp_add([2], "shard_index", {"shard_index": 9})]),
    ]
    result = apply_input_mutations(["a", "b", "c"], passes, "cpu")
    assert result[0] == "a" and result[1] == "c" and result[2].item() == 9


# ===========================================================================
# 4. apply_output_mutations
# ===========================================================================


@pytest.mark.parametrize(
    "outputs,passes,original_inputs,expected",
    [
        ([1, 2, 3], [], None, [1, 2, 3]),
        (
            ["a", "c"],
            [pm("P", out_muts=[out_remove([1, 3], "none")])],
            None,
            ["a", None, "c", None],
        ),
        (["copy", "real"], [pm("P", out_muts=[out_add([0])])], None, ["real"]),
        (["a", "b", "c"], [pm("P", out_muts=[alias([1], [0])])], None, ["a", "b", "c"]),
        # Multi-pass reverse: P1=ADD filter, P2=REMOVE re-insert
        # Undo P2 first (re-insert Nones at [1,3]): ["copy",None,"r0",None,"r1"]
        # Then undo P1 (filter out [0]): [None,"r0",None,"r1"]
        (
            ["copy", "r0", "r1"],
            [pm("P1", out_muts=[out_add([0])]), pm("P2", out_muts=[out_remove([1, 3], "none")])],
            None,
            [None, "r0", None, "r1"],
        ),
        # input_passthrough
        (
            ["real"],
            [pm("P", out_muts=[out_remove([1], "input_passthrough", {"index_to_input": {1: 0}})])],
            ["restored"],
            ["real", "restored"],
        ),
    ],
)
def test_apply_output_mutations(outputs, passes, original_inputs, expected):
    result = apply_output_mutations(list(outputs), passes, "cpu", original_inputs)
    assert result == expected


# ===========================================================================
# 5. get_alias_output_buffer_map
# ===========================================================================


@pytest.mark.parametrize(
    "passes,expected",
    [
        ([], {}),
        ([pm("P", out_muts=[out_remove([1], "none")])], {}),
        ([pm("P", out_muts=[alias([2], [1])])], {2: 1}),
        ([pm("P", out_muts=[alias([0], [0], is_mutation=True)])], {0: 0}),
        ([pm("P", out_muts=[alias([0, 2], [1, 3])])], {0: 1, 2: 3}),
        # Subsequent input REMOVE shifts neff_in down
        ([pm("A", out_muts=[alias([1], [2])]), pm("N", in_muts=[inp_remove([0])])], {1: 1}),
        # Subsequent input ADD shifts neff_in up
        (
            [
                pm("A", out_muts=[alias([0], [1])]),
                pm("N", in_muts=[inp_add([0], "shard_index", {"shard_index": 0})]),
            ],
            {0: 2},
        ),
        # Subsequent output ADD shifts neff_out up
        ([pm("A", out_muts=[alias([1], [0])]), pm("N", out_muts=[out_add([0])])], {2: 0}),
        # Subsequent output REMOVE shifts neff_out down
        (
            [pm("A", out_muts=[alias([2], [0])]), pm("N", out_muts=[out_remove([0], "none")])],
            {1: 0},
        ),
        # Alias output eliminated by subsequent pass → not in result
        ([pm("A", out_muts=[alias([1], [0])]), pm("N", out_muts=[out_remove([1], "none")])], {}),
        # Alias input eliminated by subsequent pass → not in result
        ([pm("A", out_muts=[alias([0], [1])]), pm("N", in_muts=[inp_remove([1])])], {}),
        # Alias in middle pass
        (
            [
                pm("P1", in_muts=[inp_remove([0])]),
                pm("P2", out_muts=[alias([1], [1])]),
                pm("P3", in_muts=[inp_add([2], "shard_index", {"shard_index": 0})]),
            ],
            {1: 1},
        ),
    ],
)
def test_get_alias_output_buffer_map(passes, expected):
    assert get_alias_output_buffer_map(passes) == expected


def test_duplicate_alias_output_first_wins():
    """When two passes emit an alias for the same NEFF output index, the first one wins."""
    passes = [
        pm("PassA", out_muts=[alias([0], [1])]),
        pm("PassB", out_muts=[alias([0], [2])]),
    ]
    result = get_alias_output_buffer_map(passes)
    assert result == {0: 1}, f"First alias should win, got {result}"


def test_duplicate_alias_within_same_pass_first_wins():
    """Two ALIAS entries in the same pass targeting the same output: first wins."""
    passes = [
        pm("Pass", out_muts=[alias([0], [1], is_mutation=True), alias([0], [3])]),
    ]
    result = get_alias_output_buffer_map(passes)
    assert result == {0: 1}, f"First alias should win, got {result}"


# ===========================================================================
# 6. get_mutation_copy_back_inputs
# ===========================================================================


@pytest.mark.parametrize(
    "passes,expected",
    [
        ([], {}),
        ([pm("P", out_muts=[alias([1], [0], is_mutation=False)])], {}),
        ([pm("P", out_muts=[alias([0], [0], is_mutation=True)])], {0: 0}),
        # Predecessor REMOVE shifts caller_in up
        (
            [
                pm("Pre", in_muts=[inp_remove([0])]),
                pm("A", out_muts=[alias([0], [1], is_mutation=True)]),
            ],
            {0: 2},
        ),
        # Predecessor ADD shifts caller_in down
        (
            [
                pm("Pre", in_muts=[inp_add([0], "shard_index", {"shard_index": 0})]),
                pm("A", out_muts=[alias([0], [2], is_mutation=True)]),
            ],
            {0: 1},
        ),
        # Successor output ADD shifts neff_out up
        (
            [
                pm("A", out_muts=[alias([1], [0], is_mutation=True)]),
                pm("S", out_muts=[out_add([0])]),
            ],
            {2: 0},
        ),
        # Successor output REMOVE eliminates neff_out
        (
            [
                pm("A", out_muts=[alias([0], [0], is_mutation=True)]),
                pm("S", out_muts=[out_remove([0], "none")]),
            ],
            {},
        ),
        # Predecessor ADD on the exact slot → caller_in None → not in result
        (
            [
                pm("Pre", in_muts=[inp_add([1], "shard_index", {"shard_index": 0})]),
                pm("A", out_muts=[alias([0], [1], is_mutation=True)]),
            ],
            {},
        ),
        ([pm("P", out_muts=[alias([0, 1], [0, 1], is_mutation=True)])], {0: 0, 1: 1}),
        (
            [
                pm("P1", in_muts=[inp_remove([0])]),
                pm("P2", out_muts=[alias([0], [0], is_mutation=True)]),
                pm("P3", in_muts=[inp_add([2], "shard_index", {"shard_index": 0})]),
            ],
            {0: 1},
        ),
    ],
)
def test_get_mutation_copy_back_inputs(passes, expected):
    assert get_mutation_copy_back_inputs(passes) == expected


def test_duplicate_copy_back_output_first_wins():
    """When two passes emit a mutation alias for the same NEFF output, the first wins."""
    passes = [
        pm("PassA", out_muts=[alias([0], [1], is_mutation=True)]),
        pm("PassB", out_muts=[alias([0], [2], is_mutation=True)]),
    ]
    result = get_mutation_copy_back_inputs(passes)
    assert result == {0: 1}, f"First copy-back should win, got {result}"


def test_duplicate_copy_back_within_same_pass_first_wins():
    """Two mutation ALIAS entries in the same pass for the same output: first wins."""
    passes = [
        pm("Pass", out_muts=[alias([0], [1], is_mutation=True), alias([0], [3], is_mutation=True)]),
    ]
    result = get_mutation_copy_back_inputs(passes)
    assert result == {0: 1}, f"First copy-back should win, got {result}"


# ===========================================================================
# 7. Full pipeline roundtrip using custom PassMutations
#
# Each case defines a sequence of custom passes that manipulate inputs and
# outputs, then verifies:
#   (a) apply_input_mutations produces the correct NEFF inputs
#   (b) apply_output_mutations restores the correct caller outputs
# ===========================================================================


@pytest.mark.parametrize(
    "case",
    [
        {
            "desc": "input remove + output re-insert (None)",
            "original_inputs": ["a", "b", "c"],
            "neff_outputs": ["r0", "r1"],
            "passes": [
                pm("PassA", in_muts=[inp_remove([1])]),  # removes "b"
                pm("PassB", out_muts=[out_remove([2], "none")]),  # re-inserts None at pos 2
            ],
            "exp_neff_inputs": ["a", "c"],
            "exp_caller_outputs": ["r0", "r1", None],
        },
        {
            "desc": "output add filter (copy_ pattern)",
            "original_inputs": ["inp"],
            "neff_outputs": ["copy_result", "real_out"],
            "passes": [
                pm("PassA", out_muts=[out_add([0])]),  # copy_ result added at 0 → filter at exec
            ],
            "exp_neff_inputs": ["inp"],
            "exp_caller_outputs": ["real_out"],
        },
        {
            "desc": "multi-pass: input remove + output add + output re-insert",
            "original_inputs": ["x", "y", "z"],
            "neff_outputs": ["copy_res", "out0", "out1"],
            # Pass1: remove "y" (input[1])
            # Pass2: copy_ added output at [0] → filter at exec
            # Pass3: None removed from output at [3] → re-insert at exec
            "passes": [
                pm("PassA", in_muts=[inp_remove([1])]),
                pm("PassB", out_muts=[out_add([0])]),
                pm("PassC", out_muts=[out_remove([3], "none")]),
            ],
            "exp_neff_inputs": ["x", "z"],
            "exp_caller_outputs": ["out0", "out1", None],
        },
        {
            "desc": "input add (shard) + no output change",
            "original_inputs": ["a", "b"],
            "neff_outputs": ["result"],
            "passes": [
                pm("PassA", in_muts=[inp_add([2], "shard_index", {"shard_index": 4})]),
            ],
            "exp_neff_inputs_check": lambda r: r[0] == "a" and r[1] == "b" and r[2].item() == 4,
            "exp_caller_outputs": ["result"],
        },
        {
            "desc": "output passthrough re-inserted from original_inputs",
            "original_inputs": ["orig"],
            "neff_outputs": ["out"],
            "passes": [
                pm(
                    "PassA",
                    out_muts=[out_remove([1], "input_passthrough", {"index_to_input": {1: 0}})],
                ),
            ],
            "exp_neff_inputs": ["orig"],
            "exp_caller_outputs": ["out", "orig"],
        },
        {
            "desc": "alias mutations are no-ops in apply_input/output",
            "original_inputs": ["x", "y"],
            "neff_outputs": ["r"],
            "passes": [
                pm("PassA", in_muts=[alias([0], [0])], out_muts=[alias([0], [0])]),
            ],
            "exp_neff_inputs": ["x", "y"],
            "exp_caller_outputs": ["r"],
        },
    ],
)
def test_full_pipeline_roundtrip(case):
    passes = case["passes"]
    neff_inputs = apply_input_mutations(list(case["original_inputs"]), passes, "cpu")

    if "exp_neff_inputs" in case:
        assert neff_inputs == case["exp_neff_inputs"]
    if "exp_neff_inputs_check" in case:
        assert case["exp_neff_inputs_check"](neff_inputs)

    caller_outputs = apply_output_mutations(
        list(case["neff_outputs"]), passes, "cpu", list(case["original_inputs"])
    )
    assert caller_outputs == case["exp_caller_outputs"]


# ===========================================================================
# 8. Alias maps integration: verify buffer_map and copy_back together
# ===========================================================================


@pytest.mark.parametrize(
    "passes,exp_buffer_map,exp_copy_back",
    [
        ([], {}, {}),
        # Passthrough only
        ([pm("P", out_muts=[alias([2], [1], is_mutation=False)])], {2: 1}, {}),
        # Mutation only
        ([pm("P", out_muts=[alias([0], [0], is_mutation=True)])], {0: 0}, {0: 0}),
        # Both in same pass
        (
            [
                pm(
                    "P",
                    out_muts=[
                        alias([0, 2], [0, 1], is_mutation=True),
                        alias([3], [2], is_mutation=False),
                    ],
                )
            ],
            {0: 0, 2: 1, 3: 2},
            {0: 0, 2: 1},
        ),
        # Preceding input removal: buffer_map neff_in unchanged (no predecessors), copy_back shifted
        (
            [
                pm("Pre", in_muts=[inp_remove([0])]),
                pm("Alias", out_muts=[alias([1], [1], is_mutation=True)]),
            ],
            {1: 1},
            {1: 2},
        ),
        # Subsequent input add shifts neff_in for buffer_map but not copy_back (no predecessors)
        (
            [
                pm("Alias", out_muts=[alias([0], [0], is_mutation=False)]),
                pm("Shard", in_muts=[inp_add([0], "shard_index", {"shard_index": 0})]),
            ],
            {0: 1},
            {},
        ),
    ],
)
def test_alias_maps_combined(passes, exp_buffer_map, exp_copy_back):
    assert get_alias_output_buffer_map(passes) == exp_buffer_map
    assert get_mutation_copy_back_inputs(passes) == exp_copy_back


# ===========================================================================
# 9. rebase_pre_aot_pass_mutations
# ===========================================================================


@pytest.mark.parametrize(
    "pre_aot_pm,passes_before,exp_buffer_map",
    [
        # None input → None output
        (None, [], None),
        # No passes before: indices unchanged
        (
            pm("AliasingAnalysis", out_muts=[alias([1], [0], is_mutation=False)]),
            [],
            {1: 0},
        ),
        # Preceding input REMOVE shifts alias input index forward
        # passes_before has REMOVE [0] on input → input[0] shifts to None but alias is input[1]
        # input[1] after REMOVE [0] → still 0 (shifts down by 1 since removal at 0 < 1)
        (
            pm("AliasingAnalysis", out_muts=[alias([1], [1], is_mutation=False)]),
            [pm("Flex", in_muts=[inp_remove([0])])],
            {1: 0},  # neff_in: input[1] - 1 removal before = 0
        ),
        # Preceding output ADD shifts alias output index forward
        # passes_before has ADD [0] on output → output[1] shifts up to 2
        (
            pm("AliasingAnalysis", out_muts=[alias([1], [0], is_mutation=False)]),
            [pm("Flex", out_muts=[out_add([0])])],
            {2: 0},  # neff_out: output[1] + 1 insertion at 0 = 2
        ),
        # Preceding input ADD shifts alias input index forward
        # passes_before ADD [0] on input → input[1] shifts up to 2
        (
            pm("AliasingAnalysis", out_muts=[alias([0], [1], is_mutation=False)]),
            [pm("Flex", in_muts=[inp_add([0], "shard_index", {"shard_index": 0})])],
            {0: 2},  # neff_in: input[1] + 1 insertion at 0 = 2
        ),
        # Preceding output REMOVE eliminates alias output → not in result
        (
            pm("AliasingAnalysis", out_muts=[alias([0], [0], is_mutation=False)]),
            [pm("Flex", out_muts=[out_remove([0], "none")])],
            {},  # output[0] removed → None → dropped
        ),
        # Mutation alias (is_mutation=True) also rebased
        (
            pm("AliasingAnalysis", out_muts=[alias([0], [1], is_mutation=True)]),
            [pm("Flex", in_muts=[inp_remove([0])])],
            {0: 0},  # input[1] - 1 removal = 0
        ),
        # Multiple aliases in same PassMutations, mixed directions
        (
            pm(
                "AliasingAnalysis",
                out_muts=[
                    alias([0, 2], [0, 1], is_mutation=False),
                ],
            ),
            [pm("Flex", in_muts=[inp_remove([0])])],
            # input[0] removed → None (dropped); input[1] - 1 = 0
            {2: 0},
        ),
        # Two alias mutations in separate IOMutations
        (
            pm(
                "AliasingAnalysis",
                out_muts=[
                    alias([0], [0], is_mutation=True),
                    alias([2], [1], is_mutation=False),
                ],
            ),
            [],
            {0: 0, 2: 1},
        ),
    ],
)
def test_rebase_pre_aot_pass_mutations(pre_aot_pm, passes_before, exp_buffer_map):
    rebased = rebase_pre_aot_pass_mutations(pre_aot_pm, passes_before)
    if exp_buffer_map is None:
        assert rebased is None
    else:
        assert rebased is not None
        rebased_list = [rebased]
        assert get_alias_output_buffer_map(rebased_list) == exp_buffer_map

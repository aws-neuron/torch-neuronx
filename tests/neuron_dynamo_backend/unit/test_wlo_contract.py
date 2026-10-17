"""
Unit tests for wlo_contract.validate_transform -- the static shape of the WLO
weight-relayout contract shared by the compile (wlo_hlo_parser) and execute
(Executor) paths.
"""

import pytest

from torch_neuronx.neuron_dynamo_backend.wlo_contract import (
    WLOContractError,
    validate_transform,
)


class TestValidateTransform:
    def test_valid_transforms_pass(self):
        """Well-formed transforms (including the identity permutation) validate."""
        validate_transform({"input_index": 0, "reshape": [4, 2, 3], "transpose": [0, 2, 1]})
        validate_transform({"input_index": 7, "reshape": [8, 4], "transpose": [1, 0]})
        validate_transform({"input_index": 1, "reshape": [2, 3], "transpose": [0, 1]})

    def test_not_a_dict_raises(self):
        with pytest.raises(WLOContractError, match="must be a dict"):
            validate_transform([0, [4], [0]])

    def test_missing_key_raises(self):
        with pytest.raises(WLOContractError, match="missing key"):
            validate_transform({"input_index": 0, "reshape": [4]})

    def test_negative_index_raises(self):
        with pytest.raises(WLOContractError, match="input_index"):
            validate_transform({"input_index": -1, "reshape": [2, 2], "transpose": [1, 0]})

    def test_empty_reshape_raises(self):
        with pytest.raises(WLOContractError, match="non-empty"):
            validate_transform({"input_index": 0, "reshape": [], "transpose": []})

    def test_non_positive_reshape_dim_raises(self):
        with pytest.raises(WLOContractError, match="positive"):
            validate_transform({"input_index": 0, "reshape": [4, 0], "transpose": [1, 0]})

    def test_transpose_rank_mismatch_raises(self):
        with pytest.raises(WLOContractError, match="same rank"):
            validate_transform({"input_index": 0, "reshape": [4, 2, 3], "transpose": [1, 0]})

    def test_transpose_not_a_permutation_raises(self):
        """Right length but not a permutation of range(rank): repeats or skips an axis."""
        with pytest.raises(WLOContractError, match="permutation"):
            validate_transform({"input_index": 0, "reshape": [4, 2, 3], "transpose": [0, 0, 1]})
        with pytest.raises(WLOContractError, match="permutation"):
            validate_transform({"input_index": 0, "reshape": [4, 2], "transpose": [1, 2]})

    def test_error_is_runtime_error_subclass(self):
        """Callers (parser/executor) surface violations as RuntimeError; preserve that."""
        assert issubclass(WLOContractError, RuntimeError)

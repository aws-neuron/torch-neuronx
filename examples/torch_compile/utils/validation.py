# ruff: noqa: N806

# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
"""
Validation
==========
Functions to help with data validation
"""

import collections
import json
import math
from collections.abc import Callable
from typing import Optional

import numpy as np
import torch
from typing_extensions import deprecated


@deprecated(
    "Use torch_neuronx.testing.assert_close to check tensors on Neuron, "
    "or use torch.testing.assert_close to check tensors on CPU."
)
def assert_allclose(
    expected: list | tuple | torch.Tensor,
    actual: list | tuple | torch.Tensor,
    rtol: float | None = 1e-5,
    atol: float | None = 1e-5,
    check_dtype: bool = False,
):
    """
    Assert that an actual torch output is equal to the expected value.

    Unlike normal torch equality checking, this recursively traverses
    structured outputs to ensure that structures and their internal values are
    equal.

    Args:
        expected: The expected network output.
        actual: The actual network output.
        rtol: Relative tolerance when checking result equality.
        rtol: Absolute tolerance when checking result equality.

    Raises:
        AssertionError: Error when either the type, shape, or values differ.
    """
    assert isinstance(expected, type(actual)), f"Type Mismatch {type(expected)} != {type(actual)}"

    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(expected, actual, rtol=rtol, atol=atol, check_dtype=check_dtype)

    elif isinstance(expected, tuple | list):
        for items in zip(expected, actual, strict=False):
            assert_allclose(*items, rtol=rtol, atol=atol, check_dtype=check_dtype)

    else:
        assert expected == actual, f"Equality Failure {expected} != {actual}"


# -----------------------------------------------------------------------------
# Logit Validation
# -----------------------------------------------------------------------------


def get_divergence_idx(expected_sequences: torch.Tensor, actual_sequences: torch.Tensor) -> int:
    # get the index of the first divergent token across all batches
    min_seq_len = min(actual_sequences.shape[1], expected_sequences.shape[1])
    diff = torch.ne(actual_sequences[:, :min_seq_len], expected_sequences[:, :min_seq_len])
    if torch.sum(diff) == 0:
        return min_seq_len
    else:
        return torch.min(torch.nonzero(diff), 0).values[1].item() + 1


def preprocess_logits(
    expected_logits: torch.Tensor,
    actual_logits: torch.Tensor,
    remove_shift: bool,
    return_removed_indices: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, float]
    | tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]
):
    """
    This function handles two preprocessing tasks: omitting -inf values and
    removing the shift. -inf values sometimes appear in logits when a token is
    prohibited, and they cause issues in some downstream tasks. This function
    identifies indices at which either the actual or expected logits are -inf
    and omits them. To return the indices of the removed values, enable
    return_removed_indices.

    For instance, the input tensors:
        actual = [1, -inf, 3, 4], expected = [5, 6, 7, -inf]
    would be output as:
        actual = [1, 3], expected = [5, 7]

    This function also optionally finds and removes a constant shift in the
    logits by finding the least-squares approximation for p in the following
    system of linear equations:
        actual_logits = A @ p
    where
        A = [expected_logits | 1]
        p = [slope, shift].T
    In other words, it does a linear regression. Then, it subtracts the shift
    from actual_logits.

    For instance, the input tensors:
        actual = [1, 2, 3, 4], expected = [5, 6, 7, 8]
    would be output as:
        actual = [5, 6, 7, 8], expected = [5, 6, 7, 8], shift = -4
    """
    # Omit indices at which logits are -inf
    vocab_size = len(expected_logits)
    assert vocab_size == len(actual_logits)
    ninf_idxs = torch.nonzero(
        torch.logical_or(actual_logits == float("-inf"), expected_logits == float("-inf"))
    )
    expected_logits = expected_logits[~torch.isin(torch.arange(vocab_size), ninf_idxs)]
    actual_logits = actual_logits[~torch.isin(torch.arange(vocab_size), ninf_idxs)]
    shift = 0
    if remove_shift:  # Calculate and remove shift
        A = np.vstack([expected_logits.float(), np.ones(len(expected_logits))]).T
        _, shift = np.linalg.lstsq(A, actual_logits.float(), rcond=None)[0]
        actual_logits -= shift
    if return_removed_indices:
        return expected_logits, actual_logits, shift, ninf_idxs.reshape(-1)
    else:
        return expected_logits, actual_logits, shift


def n_digits_past_decimal(x: float):
    s = str(x)
    exponent = 0
    if "e" in s:
        s, exponent = s.split("e")
        exponent = int(exponent)
    exponent = max(-exponent, 0)
    if "." in s:
        exponent += s[::-1].find(".")
    return exponent


@deprecated("Use torch_neuronx.testing.neuron_allclose instead.")
def custom_allclose(
    a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float
) -> tuple[bool, float]:
    # equivalent to torch.allclose except rtol is multiplied by absolute max, not abs
    # this matches the behavior of the compiler's birsim-to-xla_infergoldens verification
    error = (torch.abs(a - b) - atol) / torch.max(torch.abs(b))
    rtol_precision = n_digits_past_decimal(rtol)
    rounded_error = torch.round(error, decimals=rtol_precision)
    passed = torch.all(rounded_error <= rtol)
    return passed.item(), torch.max(error).item()


AllCloseSummary = collections.namedtuple(
    "AllCloseSummary",
    [
        "allclose",
        "num_mismatches",
        "max_rel_error",
        "max_abs_error",
        "max_rel_error_index",
        "max_abs_error_index",
    ],
)


# Default tolerances by dtype in format (rtol, atol), based on torch.testing.all_close.
_DEFAULT_DTYPE_TOLERANCE_MAP = {
    torch.float16: (1e-3, 1e-5),
    torch.bfloat16: (1.6e-2, 1e-5),
    torch.float32: (1.3e-6, 1e-5),
}
_TOLERANCE_EQUAL = (0.0, 0.0)


def neuron_allclose(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float | None = None,
    atol: float | None = None,
) -> AllCloseSummary:
    """
    Checks whether two tensors are close.

    This is equivalent to torch.allclose except rtol is multiplied by absolute max, not abs
    This matches the behavior of the compiler's birsim-to-xla_infergoldens verification

    Returns a named tuple that includes whether the tensors are close, the number of mismatches,
    the greatest relative error, the greatest absolute error, and the index of each max error.

    Args:
        actual: The actual tensor to compare.
        expected: The expected tensor to compare.
        rtol: The relative tolerance to use. Defaults to the dtype-specific
            tolerance of the expected tensor.
        atol: The absolute tolerance to use. Defaults to the dtype-specific
            tolerance of the expected tensor.
    """
    if torch.equal(expected, actual):
        # All indices share the same error (zero), so we return the first value's
        # index as the max error index.
        max_error_index = torch.unravel_index(torch.tensor(0), expected.shape)
        return AllCloseSummary(
            allclose=True,
            num_mismatches=0,
            max_rel_error=0.0,
            max_abs_error=0.0,
            max_rel_error_index=max_error_index,
            max_abs_error_index=max_error_index,
        )

    rtol, atol = _get_default_tolerances(expected, rtol, atol)
    abs_diff = torch.abs(actual - expected)
    expected_abs_max = torch.max(torch.abs(expected))
    if torch.is_nonzero(expected_abs_max):
        rel_error = (abs_diff - atol) / expected_abs_max
    else:
        rel_error = torch.full(expected.shape, torch.inf)
    close = torch.where(rel_error <= rtol, 1.0, 0.0)
    allclose = torch.all(close == 1.0).item()
    num_mismatches = torch.numel(close[close == 0.0])
    max_rel_error_index = torch.unravel_index(torch.argmax(rel_error), rel_error.shape)
    max_abs_error_index = torch.unravel_index(torch.argmax(abs_diff), abs_diff.shape)
    max_rel_error = rel_error[max_rel_error_index].item()
    max_abs_error = abs_diff[max_abs_error_index].item()
    return AllCloseSummary(
        allclose=allclose,
        num_mismatches=num_mismatches,
        max_rel_error=max_rel_error,
        max_abs_error=max_abs_error,
        max_rel_error_index=max_rel_error_index,
        max_abs_error_index=max_abs_error_index,
    )


def _get_default_tolerances(tensor: torch.Tensor, rtol: float, atol: float):
    if rtol is None:
        rtol = _DEFAULT_DTYPE_TOLERANCE_MAP.get(tensor.dtype, _TOLERANCE_EQUAL)[0]
    if atol is None:
        atol = _DEFAULT_DTYPE_TOLERANCE_MAP.get(tensor.dtype, _TOLERANCE_EQUAL)[1]
    return rtol, atol


def validate_top_k_logits(
    expected_logits: torch.Tensor, actual_logits: torch.Tensor, top_k: int, atol: float, rtol: float
) -> tuple[bool, float]:
    if top_k is not None:  # filter only the top k most likely tokens
        top_k_result = torch.topk(expected_logits, top_k)
        expected_logits = top_k_result.values
        actual_logits = torch.index_select(actual_logits, 0, top_k_result.indices)
    return custom_allclose(actual_logits, expected_logits, atol=atol, rtol=rtol)


def validate_single_token_logits(
    expected_logits: torch.Tensor,
    actual_logits: torch.Tensor,
    tol_map: dict,
    divergence_difference_tol: float,
    remove_shift: bool,
    actual_token_id: int | None = None,
) -> tuple[bool, dict, str]:
    divergence_difference = 0
    status_msg = []
    error_map = {k: 0 for k in tol_map}
    divergence = False
    passed = torch.tensor(True)

    expected_logits, actual_logits, shift, removed_indices = preprocess_logits(
        expected_logits,
        actual_logits,
        remove_shift,
        return_removed_indices=True,
    )
    if actual_token_id is not None:
        actual_token_id = calculate_new_index(actual_token_id, removed_indices)
        assert actual_token_id is not None, "Provided actual token ID has -inf score"

    def get_top2_values_indices_diff(logits):
        top2_values, top2_indices = torch.topk(logits, 2)
        top1_top2_diff = top2_values[0] - top2_values[1]
        # Calculate relative difference between top1 and top2 logits (signed)
        top1_top2_relative_diff = (
            (top1_top2_diff / torch.abs(top2_values[0]))
            if torch.abs(top2_values[0]) > torch.tensor(1e-8)
            else torch.tensor(0.0)
        )
        # TODO: Shift indices to original indices based on removed_indices
        return top2_values, top2_indices, top1_top2_diff, top1_top2_relative_diff

    # expected
    (
        expected_top2_values,
        expected_top2_indices,
        expected_top1_top2_diff,
        expected_top1_top2_relative_diff,
    ) = get_top2_values_indices_diff(expected_logits)
    # actual
    (
        actual_top2_values,
        actual_top2_indices,
        actual_top1_top2_diff,
        actual_top1_top2_relative_diff,
    ) = get_top2_values_indices_diff(actual_logits)

    # Relative error = (actual - expected) / |expected| for each top2 index (signed)
    def get_relative_error(actual_val, expected_val):
        return (
            ((actual_val - expected_val) / torch.abs(expected_val))
            if torch.abs(expected_val) > 1e-8
            else torch.abs(actual_val - expected_val)
        )

    top1_relative_error = get_relative_error(actual_top2_values[0], expected_top2_values[0])
    top2_relative_error = get_relative_error(actual_top2_values[1], expected_top2_values[1])
    # Calculate actual relative difference using expected top2 indices:
    # (actual[top1] - actual[top2]) / |actual[top1]|
    actual_values_with_expected_top1_top2_indices_relative_diff = get_relative_error(
        actual_logits[expected_top2_indices[0]],
        actual_logits[expected_top2_indices[1]],
    )

    # check if the logits are in bounds for each value of k
    total_errors = collections.defaultdict(dict)
    max_abs_expected = torch.max(torch.abs(expected_logits)).item()
    total_errors[None]["mean_abs_error"] = (
        torch.nn.functional.l1_loss(actual_logits, expected_logits, reduction="mean").item()
        / max_abs_expected
    )
    total_errors[None]["mean_squared_error"] = torch.nn.functional.mse_loss(
        actual_logits, expected_logits, reduction="mean"
    ).item() / (max_abs_expected**2)
    for top_k, tols in tol_map.items():
        atol, rtol = tols
        in_bounds, error = validate_top_k_logits(expected_logits, actual_logits, top_k, atol, rtol)

        total_errors[top_k]["max_abs_error"] = abs(error)
        total_errors[top_k]["max_squared_error"] = error**2

        if not in_bounds:
            status_msg.append(f"Top k = {top_k} error {error} > {rtol}.")
        passed &= in_bounds
        error_map[top_k] = error
    #
    # determine if the sequences diverge and evaluate the divergence difference if they do
    greedy_next_token_id = expected_logits.argmax().item()
    if actual_token_id is None:
        actual_token_id = actual_logits.argmax().item()
    if greedy_next_token_id != actual_token_id:
        divergence = True
        divergence_difference = torch.abs(
            actual_logits[actual_token_id] - actual_logits[greedy_next_token_id]
        )
        in_bounds = divergence_difference < divergence_difference_tol
        if not in_bounds:
            status_msg.append(
                f"Divergence difference {divergence_difference} > {divergence_difference_tol}."
            )
        passed &= in_bounds
    passed = passed.item()
    # report the results and return
    rel_diff = actual_values_with_expected_top1_top2_indices_relative_diff.item()
    results = {
        "passed": passed,
        "divergence": divergence,
        "divergence_difference": divergence_difference,
        "total_errors": total_errors,
        "error_map": error_map,
        "shift": shift,
        "expected_logits": expected_logits,
        "actual_logits": actual_logits,
        "expected_top2_values": expected_top2_values.tolist(),
        "actual_top2_values": actual_top2_values.tolist(),
        "expected_top1_top2_diff": expected_top1_top2_diff.item(),
        "actual_top1_top2_diff": actual_top1_top2_diff.item(),
        "expected_top1_top2_relative_diff": expected_top1_top2_relative_diff.item(),
        "actual_top1_top2_relative_diff": actual_top1_top2_relative_diff.item(),
        "actual_with_expected_top1_top2_relative_diff": rel_diff,
        "expected_top2_indices": expected_top2_indices.tolist(),
        "actual_top2_indices": actual_top2_indices.tolist(),
        "top1_relative_errors": top1_relative_error,
        "top2_relative_errors": top2_relative_error,
    }
    return passed, results, " ".join(status_msg)


def calculate_new_index(original_index, removed_indices):
    """
    Calculate new index after removing elements from a tensor.

    Args:
        original_index: Index in original tensor
        removed_indices: Tensor containing indices of removed elements (all items must be unique)

    Returns:
        new_index: Index in tensor after removal, or None if element was removed
    """
    if original_index in removed_indices:
        return None

    # Count how many elements were removed before this index
    removed_before = 0
    for removed_idx in removed_indices:
        if removed_idx < original_index:
            removed_before += 1

    new_index = original_index - removed_before
    return new_index


def to_padding_side(input, pad_token_id, padding_side):
    repadded = torch.full(input.shape, pad_token_id)
    for batch_idx, sequence in enumerate(input):
        mask = sequence != pad_token_id
        seq_len = torch.sum(mask)
        if padding_side == "left":
            repadded[batch_idx, -seq_len:] = sequence[mask]
        else:  # padding_side == 'right'
            repadded[batch_idx, :seq_len] = sequence[mask]
    return repadded


# tolerances tend to be tighter at smaller top_k values because the accuracy of
# more likely tokens is more important than less likely tokens.
DEFAULT_TOLERANCE_MAP = {
    None: (1e-5, 0.05),
    1000: (1e-5, 0.03),
    50: (1e-5, 0.02),
    5: (1e-5, 0.01),
}


def get_default_tolerance_map():
    # Returns a copy of DEFAULT_TOLERANCE_MAP. Use to avoid mutating the original
    return DEFAULT_TOLERANCE_MAP.copy()


DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE = 0.001


def logit_validation(
    input_ids: torch.Tensor,
    generate_fn: Callable[[torch.Tensor], torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    expected_logits: torch.Tensor,
    tol_map: dict | None = None,
    divergence_difference_tol: float | None = None,
    short_circuit: bool = False,
    remove_shift: bool = True,
    padding_side: str = "left",
    pad_token_id: int | None = None,
    return_summary: bool = False,
) -> tuple[bool, dict, str]:
    """
    Compares the output of the generate_fn to the expected_logits using the
    following methodolody. This function assumes greedy sampling.
    For each set of logits at each token position, it validates the top k
    highest scoring tokens for many k's. The values of k and the tolerances
    of the validation are given by tol_map (described in more detail below).
    If the greedy next token (argmax) of the actual and expected logits differ
    at any token position, then the difference between max(actual_logits) and
    actual_logits[expected_next_token_idx] is taken and thresholded to
    divergence_difference_tol. Also, since this divergence would cause the
    downstream tokens to be different, this function will take copy output up
    to the divergent index and feed it back to generate_fn as an input. Then,
    it repeats the process above until it has validated every token position.

    Args:
        input_ids: Encoded inputs with shape [batch_size, input_length].
        generate_fn: A function that accepts an argument input_ids, which is
            a tensor with shape [batch_size, input_length]. It returns a tensor
            of logits with shape [output_length, batch_size, vocab_size].
            Alternatively, to support configurations with on-device sampling
            (where the sampled token may differ when two top tokens have the
            same score), you can return a tuple where the first item is
            the logits tensor, and the second item is a tensor of token IDs with
            shape [batch_size, output_length].
        expected_logits: Expected logits with shape
            [output_length, batch_size, vocab_size].
        tol_map: A dict that maps from top k values to tuples (atol, rtol).
            A k of None indicates that the entire set of logits should be
            validated. Defaults to DEFAULT_TOLERANCE_MAP.
        divergence_difference_tol: The tolerance allowed for the difference
            between the score of the actual next token and the score of the
            expected next token. Defaults to
            DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE.
        short_circuit: Whether to end the test as soon as a failure is found.
        remove_shift: Whether to calculate a constant shift value between the
            actual and expected logits and correct for it before validation.
        padding_side: The padding side of the input, one of "left" or "right".
        pad_token_id: Must be provided if padding_side is "right".

    Returns:
        A tuple (passed, results, status_msg) where passed is a boolean value
            indicating whether or not the test passed, results is a 2D list of
            dicts with the shape [batch_size, output_length] where each element
            contains the validation results of that token, and status_msg is a
            string summary of the test.
    """
    if padding_side not in ["left", "right"]:
        msg = (
            'logit_validation: padding_side must be one of "left" or "right". '
            + f'"{padding_side}" was given.'
        )
        raise ValueError(msg)
    if padding_side == "right" and pad_token_id is None:
        raise ValueError(
            'logit_validation: If padding_side == "right", a pad_token_id must be provided.'
        )
    if tol_map is None:
        tol_map = DEFAULT_TOLERANCE_MAP
    if divergence_difference_tol is None:
        divergence_difference_tol = DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE

    if padding_side == "right":
        input_ids = to_padding_side(input_ids, pad_token_id, "left")

    batch_size, input_length = input_ids.shape
    current_output_start_idx = 0
    expected_sequences = expected_logits.argmax(dim=2).T
    expected_sequence_length = expected_sequences.shape[1]

    passed = True
    results = [[] for _ in range(batch_size)]
    status_msg = []

    while current_output_start_idx < expected_sequence_length:
        repadded_inputs = (
            input_ids
            if padding_side == "left"
            else to_padding_side(input_ids, pad_token_id, "right")
        )
        generate_result = generate_fn(repadded_inputs)
        client_sampling = False
        if isinstance(generate_result, tuple):
            actual_logits, actual_sequences = generate_result
            assert actual_logits.shape[:-1] == actual_sequences.T.shape, (
                f"Shape mismatch between logits and sequences returned by generate_fn: "
                f"{actual_logits.shape=}, {actual_sequences.shape=}"
            )
            client_sampling = True
        else:
            actual_logits = generate_result
            actual_sequences = actual_logits.argmax(dim=2).T
        divergence_idx = get_divergence_idx(
            expected_sequences[:, current_output_start_idx:], actual_sequences
        )
        divergence_idx += current_output_start_idx
        if divergence_idx == expected_sequence_length:
            status_msg.append(
                "No divergence. "
                f"Validating the remaining {divergence_idx - current_output_start_idx} "
                f"tokens in each batch."
            )
        else:
            status_msg.append(
                f"Divergence at index {divergence_idx}. "
                f"Validating {divergence_idx - current_output_start_idx} tokens in each batch."
            )

        for batch_idx in range(batch_size):
            for token_idx in range(divergence_idx - current_output_start_idx):
                actual_token_id = None
                if client_sampling:
                    actual_token_id = actual_sequences[batch_idx, token_idx].item()
                single_token_passed, single_token_results, single_token_status_msg = (
                    validate_single_token_logits(
                        expected_logits=expected_logits[
                            token_idx + current_output_start_idx, batch_idx, :
                        ],
                        actual_logits=actual_logits[token_idx, batch_idx, :],
                        tol_map=tol_map,
                        divergence_difference_tol=divergence_difference_tol,
                        remove_shift=remove_shift,
                        actual_token_id=actual_token_id,
                    )
                )

                results[batch_idx].append(single_token_results)
                passed &= single_token_passed
                if not single_token_passed:
                    status_msg.append(
                        f"Test failed at batch {batch_idx} token "
                        f"{token_idx + current_output_start_idx}. "
                        f"{single_token_status_msg}"
                    )
                    if short_circuit:
                        return passed, results, status_msg
        input_ids = torch.cat(
            [input_ids, expected_sequences[:, current_output_start_idx:divergence_idx]], dim=1
        )
        current_output_start_idx = divergence_idx
    results_summary = _get_logit_validation_results_summary(results)
    status_msg.append(_format_logit_validation_results_summary(results_summary))
    status_msg.append(_format_logit_validation_results_dictionary(results_summary))
    if passed:
        status_msg.append("Test passes logit validation.")
    else:
        status_msg.append("Test fails logit validation.")
    if return_summary:
        return passed, results, "\n".join(status_msg), results_summary
    else:
        return passed, results, "\n".join(status_msg)


def _get_logit_validation_max_topk_error_results_summary(results):
    summary = {
        "max_divergence": {"error": -1},
        "max_top_k_errors": {},
    }
    for batch_index in range(len(results)):
        for token_index in range(len(results[batch_index])):
            token_results = results[batch_index][token_index]
            if token_results["divergence_difference"] > summary["max_divergence"]["error"]:
                summary["max_divergence"]["error"] = token_results["divergence_difference"]
                summary["max_divergence"]["batch_index"] = batch_index
                summary["max_divergence"]["token_index"] = token_index
            for top_k, error in token_results["error_map"].items():
                if top_k not in summary["max_top_k_errors"]:
                    summary["max_top_k_errors"][top_k] = {"error": -1}
                if error > summary["max_top_k_errors"][top_k]["error"]:
                    summary["max_top_k_errors"][top_k]["error"] = error
                    summary["max_top_k_errors"][top_k]["batch_index"] = batch_index
                    summary["max_top_k_errors"][top_k]["token_index"] = token_index
    return summary


def _get_logit_validation_average_over_tokens_results_summary(results):
    summary = {
        "average_over_tokens": collections.defaultdict(dict),
    }

    count = 0
    total_error_dict = collections.defaultdict(dict)
    for batch_index in range(len(results)):
        for token_index in range(len(results[batch_index])):
            token_results = results[batch_index][token_index]
            count += 1
            for top_k, top_k_errors in token_results["total_errors"].items():
                for error_key, error in top_k_errors.items():
                    total_error_dict[top_k][error_key] = (
                        total_error_dict.get(top_k, {}).get(error_key, 0) + error
                    )

    if count != 0:
        for top_k, error_dict in total_error_dict.items():
            for error_key, error in error_dict.items():
                summary["average_over_tokens"][top_k][error_key] = error / count

    return summary


def _get_logit_validation_results_summary(results):
    max_error_summary = _get_logit_validation_max_topk_error_results_summary(results)
    average_over_tokens_summary = _get_logit_validation_average_over_tokens_results_summary(results)

    summary = max_error_summary | average_over_tokens_summary

    return summary


def _format_logit_validation_results_dictionary(summary):
    banner = "Complete Error Summary"
    keys_to_keep = ("max_top_k_errors", "average_over_tokens")
    clarification = (
        "These errors are normalized for each token by the largest expected logit for that token"
    )

    logit_error_dict = {k: v for k, v in summary.items() if k in keys_to_keep}
    dict_info = json.dumps(logit_error_dict, indent=4)

    return "\n".join([banner, clarification, dict_info])


def _format_logit_validation_results_summary(summary):
    message = [
        f"Summary: Max divergence difference = {summary['max_divergence']['error']} "
        f"at index (batch {summary['max_divergence']['batch_index']} "
        f"token {summary['max_divergence']['token_index']})"
    ]
    for top_k, max_top_k_error in summary["max_top_k_errors"].items():
        message.append(
            f", Top k = {top_k} max error = {max_top_k_error['error']} "
            f"at index (batch {max_top_k_error['batch_index']} "
            f"token {max_top_k_error['token_index']})"
        )
    return "".join(message)


class TensorMismatchInfo:
    """
    Contains information about a pair of tensors that aren't close.
    """

    def __init__(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        pair_id: str,
        num_mismatches: int,
        max_rel_error: float,
        max_abs_error: float,
        max_rel_error_index: torch.Tensor,
        max_abs_error_index: torch.Tensor,
    ):
        self.actual = actual
        self.expected = expected
        self.pair_id = pair_id
        self.num_mismatches = num_mismatches
        self.max_rel_error = max_rel_error
        self.max_abs_error = max_abs_error
        self.max_rel_error_index = max_rel_error_index
        self.max_abs_error_index = max_abs_error_index


class TensorCloseAssertionError(AssertionError):
    """
    Raised when tensors aren't close.
    """

    def __init__(
        self,
        tensor_mismatches: list[TensorMismatchInfo],
        num_elements: int,
        rtol: float,
        atol: float,
    ):
        self.tensor_mismatches = tensor_mismatches
        self.num_elements = num_elements
        self.rtol = rtol
        self.atol = atol
        self._process_mismatches()
        message = self._get_message()
        super().__init__(message)

    def _process_mismatches(self):
        """
        Identity the total number of elements, total number of mismatched elements,
        and maximum errors.
        """
        num_mismatched_elements = 0
        max_rel_error_mismatch = None
        max_abs_error_mismatch = None
        for mismatch in self.tensor_mismatches:
            num_mismatched_elements += mismatch.num_mismatches
            if (
                max_rel_error_mismatch is None
                or mismatch.max_rel_error >= max_rel_error_mismatch.max_rel_error
            ):
                max_rel_error_mismatch = mismatch
            if (
                max_abs_error_mismatch is None
                or mismatch.max_abs_error >= max_abs_error_mismatch.max_abs_error
            ):
                max_abs_error_mismatch = mismatch
        self.num_mismatched_elements = num_mismatched_elements
        self.max_rel_error_mismatch = max_rel_error_mismatch
        self.max_abs_error_mismatch = max_abs_error_mismatch

    def _get_message(self):
        """
        Creates a summary of the errors, including mismatched element count
        and greatest relative/absolute errors.
        """
        messages = ["Tensors are not close."]
        if self.num_elements == 1:
            messages.append(
                f"Relative error: "
                f"{_format_error(self.tensor_mismatches[0].max_rel_error, self.rtol)} "
                f"(up to {self.rtol} allowed)"
            )
            messages.append(
                f"Absolute error: "
                f"{_format_error(self.tensor_mismatches[0].max_abs_error, self.atol)} "
                f"(up to {self.atol} allowed)"
            )
        else:
            mismatch_percent = self.num_mismatched_elements / self.num_elements * 100
            messages.append(
                f"Mismatched elements: {self.num_mismatched_elements} / {self.num_elements} "
                f"({mismatch_percent:.1f}%)"
            )

            max_rel_error = _format_error(self.max_rel_error_mismatch.max_rel_error, self.rtol)
            max_rel_error_index = _format_index(self.max_rel_error_mismatch.max_rel_error_index)
            messages.append(
                _format_max_error_message(
                    "relative",
                    max_rel_error,
                    max_rel_error_index,
                    self.rtol,
                    self.max_rel_error_mismatch.pair_id,
                )
            )

            max_abs_error = _format_error(self.max_abs_error_mismatch.max_abs_error, self.atol)
            max_abs_error_index = _format_index(self.max_abs_error_mismatch.max_abs_error_index)
            messages.append(
                _format_max_error_message(
                    "absolute",
                    max_abs_error,
                    max_abs_error_index,
                    self.atol,
                    self.max_abs_error_mismatch.pair_id,
                )
            )
        return "\n".join(messages)


def _format_max_error_message(label, error, index, tol, pair_id):
    message = f"Greatest {label} error: {error} "
    if index != "":
        message += f"at index {index} "
    if pair_id != "":
        message += f"in tensor {pair_id} "
    message += f"(up to {tol} allowed)"
    return message


def _format_index(indices_tuple: tuple[torch.Tensor]):
    """
    Formats a multidimensional index tensor for display.

    Examples:
        () -> ""
        (torch.tensor(0)) -> "(0,)"
        (torch.tensor(0), torch.tensor(1)) -> "(0, 1)"
    """
    if len(indices_tuple) == 0:
        return ""
    elif len(indices_tuple) == 1:
        return f"({indices_tuple[0].item()},)"
    else:
        index_str = ", ".join([str(index.item()) for index in indices_tuple])
        return f"({index_str})"


def _format_error(error: float, tol: float):
    """
    Formats an error for display.

    This formats the error to a number of digits that is one greater than the absolute magnitude of
    the tolerance's decimal part. If the tolerance is zero, this outputs all digits.
    """
    if tol == 0.0:
        return str(error)
    ndigits = abs(min(0, round(math.log10(tol)))) + 1
    return f"{error:.{ndigits}f}"


def assert_close(
    actual: torch.Tensor | collections.abc.Sequence,
    expected: torch.Tensor | collections.abc.Sequence,
    rtol: float | None = None,
    atol: float | None = None,
    check_device: bool = True,
    check_dtype: bool = True,
):
    """
    Checks whether two tensors or tensor-likes are close.

    This uses neuron_allclose to check closeness, where rtol is multipled by absolute max, not abs.

    Args:
        actual: The tensor or tensor-like to check.
        expected: The tensor or tensor-like to check against.
        rtol: The relative tolerance to use. Defaults to the dtype-specific tolerance
            of the expected tensor.
        atol: The absolute tolerance to use. Defaults to the dtype-specific tolerance
            of the expected tensor.
        check_device: Whether to raise an error if the tensors are on different devices.
        check_dtype: Whether to raise an error if the tensors have different dtypes.

    Raises:
        ValueError: If the inputs are tensor-likes with different sizes.
        ValueError: If the inputs are unsupported types for comparison.
        AssertionError: If check_device=True and the inputs are on different devices.
        AssertionError: If check_dtype=True and the inputs have different dtypes.
        TensorCloseAssertionError: If the inputs aren't close.
    """

    mismatches = []
    num_elements = 0
    tensor_pairs = _get_tensor_pairs(actual, expected)
    if len(tensor_pairs) == 0:
        return

    # Get default tolerances based on first expected tensor.
    rtol, atol = _get_default_tolerances(tensor_pairs[0][1], rtol, atol)
    for tensor_pair in tensor_pairs:
        actual_tensor, expected_tensor, pair_id = tensor_pair
        mismatch = _compare_tensors(
            actual_tensor,
            expected_tensor,
            pair_id,
            rtol=rtol,
            atol=atol,
            check_device=check_device,
            check_dtype=check_dtype,
        )
        if mismatch is not None:
            mismatches.append(mismatch)
        num_elements += torch.numel(expected_tensor)

    if len(mismatches) > 0:
        raise TensorCloseAssertionError(mismatches, num_elements, rtol, atol)


def _get_tensor_pairs(
    actual: torch.Tensor | collections.abc.Sequence,
    expected: torch.Tensor | collections.abc.Sequence,
    pair_id: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = []
    if isinstance(actual, collections.abc.Sequence) and isinstance(
        expected, collections.abc.Sequence
    ):
        if len(actual) != len(expected):
            raise ValueError(
                f"Actual and expected have different sizes. actual = {len(actual)}, "
                f"expected = {len(expected)}"
            )

        for i in range(len(actual)):
            pairs.extend(_get_tensor_pairs(actual[i], expected[i], pair_id + f"[{i}]"))
    elif isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        pairs.append((actual, expected, pair_id))
    else:
        raise ValueError(
            f"Unsupported tensor comparison types: actual = {type(actual)}, "
            f"expected = {type(expected)}"
        )
    return pairs


def _compare_tensors(
    actual: torch.Tensor,
    expected: torch.Tensor,
    pair_id: str,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    check_device: bool = True,
    check_dtype: bool = True,
):
    pair_id_info = f"at {pair_id} " if pair_id != "" else ""
    if check_device and actual.device != expected.device:
        raise AssertionError(
            f"Tensors {pair_id_info}are on different devices. actual = {actual.device}, "
            f"expected = {expected.device}"
        )

    if check_dtype and actual.dtype != expected.dtype:
        raise AssertionError(
            f"Tensors {pair_id_info}have different dtypes. actual = {actual.dtype}, "
            f"expected = {expected.dtype}"
        )

    allclose_summary = neuron_allclose(actual, expected, rtol=rtol, atol=atol)
    if not allclose_summary.allclose:
        return TensorMismatchInfo(
            actual,
            expected,
            pair_id,
            num_mismatches=allclose_summary.num_mismatches,
            max_rel_error=allclose_summary.max_rel_error,
            max_abs_error=allclose_summary.max_abs_error,
            max_rel_error_index=allclose_summary.max_rel_error_index,
            max_abs_error_index=allclose_summary.max_abs_error_index,
        )
    return None


# -----------------------------------------------------------------------------
# Tutorials
# -----------------------------------------------------------------------------


def _logit_validation_tutorial():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1. load your model
    model_name = "openlm-research/open_llama_3b"
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 2. prepare your input
    prompt = "I am a fun tutorial."
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    initial_input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # 3. retrieve your goldens (in a real example, you wouldn't use exactly the
    #    same model in steps 3 and 4)
    generation_result = model.generate(
        initial_input_ids, do_sample=False, return_dict_in_generate=True, output_scores=True
    )
    expected_logits = torch.stack(generation_result["scores"])

    # 4. build your generate function
    def generate_fn(input_ids):
        generation_result = model.generate(
            input_ids, do_sample=False, return_dict_in_generate=True, output_scores=True
        )
        return torch.stack(generation_result["scores"])

    # 5. validate
    passed, result, status_msg = logit_validation(initial_input_ids, generate_fn, expected_logits)

    # 6. postprocess
    assert passed, status_msg

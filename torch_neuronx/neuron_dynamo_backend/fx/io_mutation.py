"""
Unified IO mutation interface for FX passes that modify graph signatures.

Each FX pass records its mutations atomically as a ``PassMutations`` entry
appended to ``NeuronFXPassState.pass_mutations``.  All indices within a
``PassMutations`` entry are relative to the graph's IO space **before** that
pass's mutations have been applied (pass-local space).

Input mutation semantics (executor perspective == FX graph perspective):
  - ADD:    input was added to the graph   → inject it at execution
  - REMOVE: input was removed from graph   → drop it at execution

Output mutation semantics (FX graph perspective; executor does the inverse):
  - REMOVE: output was removed from graph  → re-insert at execution using transform
  - ADD:    output was added to the graph  → filter it out at execution

``apply_input_mutations`` iterates forward through all ``PassMutations``
entries applying each pass's input mutations in order.
``apply_output_mutations`` iterates in reverse applying each pass's output
mutations in reverse order (last pass undone first).
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

logger = logging.getLogger(__name__)


class MutationKind(Enum):
    ADD = "add"
    REMOVE = "remove"
    ALIAS = "alias"


@dataclass
class IOMutation:
    """A single IO transformation for one direction (input or output).

    Input mutations (ADD/REMOVE): applied directly at execution.
    Output mutations (ADD/REMOVE): applied inverted at execution (see module docstring).
    ALIAS mutations: metadata only — skipped by apply_input/output_mutations.

    For kind == ALIAS:
      - indices[i]:             output index in pass-local space
      - alias_input_indices[i]: input  index in pass-local space
      - is_mutation=True:  result must be copied back to original input after execution
      - is_mutation=False: pure buffer reuse (passthrough, no copy-back needed)
    One ALIAS IOMutation per pass per direction (mutations vs passthroughs).
    """

    kind: MutationKind  # Type of mutation (add, remove, or alias)
    indices: list[int]  # Affected IO indices in pass-local space
    transform_key: str | None = None  # TRANSFORM_REGISTRY key for generating values (ADD/REMOVE)
    transform_params: dict | None = None  # Parameters passed to the transform function
    pass_name: str = ""  # Name of the FX pass that emitted this mutation
    alias_input_indices: list[int] | None = None  # ALIAS only: input indices paired with indices
    is_mutation: bool | None = None  # ALIAS only: True=copy-back needed, False=buffer reuse


@dataclass
class PassMutations:
    """Groups the input and output mutations emitted by a single FX pass.

    All indices in both lists are in the graph's IO space **before** this
    pass's mutations have been applied (pass-local space).
    """

    pass_name: str
    input_mutations: list[IOMutation] = field(default_factory=list)
    output_mutations: list[IOMutation] = field(default_factory=list)


TRANSFORM_REGISTRY: dict[str, Callable[[int, str, dict, list], Any]] = {
    "none": lambda index, device, params, original_inputs: None,
    "shard_index": lambda index, device, params, original_inputs: torch.tensor(
        params["shard_index"], dtype=torch.int64, device=device
    ),
    "input_passthrough": lambda index, device, params, original_inputs: (
        original_inputs[params["index_to_input"][index]] if original_inputs is not None else None
    ),
}


def _apply_input_mutation(values: list, m: IOMutation, device: str) -> list:
    """Apply one input ADD/REMOVE mutation. ALIAS mutations are skipped."""
    if m.kind == MutationKind.REMOVE:
        remove_set = set(m.indices)
        return [v for i, v in enumerate(values) if i not in remove_set]
    if m.kind == MutationKind.ADD:
        fn = TRANSFORM_REGISTRY[m.transform_key]
        params = m.transform_params or {}
        add_set = set(m.indices)
        result = []
        val_iter = iter(values)
        for i in range(len(values) + len(m.indices)):
            if i in add_set:
                result.append(fn(i, device, params, None))
            else:
                result.append(next(val_iter))
        return result
    return values  # ALIAS: no-op


def _apply_output_mutation(
    values: list,
    m: IOMutation,
    device: str,
    original_inputs: list | None = None,
) -> list:
    """Undo one output mutation, restoring the original output signature.

    Output mutation semantics are from the FX graph's perspective:
      - REMOVE (pass removed the output from the graph): re-insert at execution using transform
      - ADD    (pass added the output to the graph):     filter it out at execution
    """
    if m.kind == MutationKind.REMOVE:
        # Output was removed from graph → re-insert using transform
        fn = TRANSFORM_REGISTRY[m.transform_key]
        params = m.transform_params or {}
        add_set = set(m.indices)
        result = []
        val_iter = iter(values)
        for i in range(len(values) + len(m.indices)):
            if i in add_set:
                result.append(fn(i, device, params, original_inputs))
            else:
                result.append(next(val_iter))
        return result
    if m.kind == MutationKind.ADD:
        # Output was added to graph → filter it out
        remove_set = set(m.indices)
        return [v for i, v in enumerate(values) if i not in remove_set]
    return values  # ALIAS: no-op


def _shift_index_through_mutation(idx: int | None, m: IOMutation, *, forward: bool) -> int | None:
    """Shift one index through a single ADD/REMOVE mutation.

    Forward: ADD grows space (+shift); REMOVE shrinks it (-shift or None if removed).
    Backward: inverse — undo a forward ADD (-shift) or REMOVE (+shift).
    ALIAS mutations are no-ops.
    """
    if m.kind == MutationKind.ALIAS:
        return idx
    if idx is None:
        return idx
    if forward:
        if m.kind == MutationKind.ADD:
            return idx + sum(1 for a in m.indices if a <= idx)
        if m.kind == MutationKind.REMOVE:
            if idx in m.indices:
                # idx was removed by a successor FX pass -> replace with None
                return None
            return idx - sum(1 for r in m.indices if r < idx)
    else:
        if m.kind == MutationKind.ADD:
            if idx in m.indices:
                # idx was added by a predecessor FX pass -> replace with None
                return None
            return idx - sum(1 for a in m.indices if a <= idx)
        if m.kind == MutationKind.REMOVE:
            return idx + sum(1 for r in m.indices if r <= idx)
    return idx


def _translate_indices(
    input_indices: list[int],
    output_indices: list[int],
    mutations: list[PassMutations],
    *,
    forward: bool,
    source: IOMutation | None = None,
) -> tuple[list[int | None], list[int | None]]:
    """Translate pass-local indices through an ordered list of ADD/REMOVE mutations.

    forward=True:  left-to-right (pass-local → NEFF-space)
    forward=False: right-to-left reversed (pass-local → caller-space)
    Returns a parallel list; None means the index was eliminated by a REMOVE.
    Logs at INFO level when an index is eliminated, including which pass did it.

    Args:
        source: The IOMutation whose indices are being translated. Used for logging.
    """
    curr_input_indices: list[int | None] = list(input_indices)
    curr_output_indices: list[int | None] = list(output_indices)
    ordered_mutations = mutations if forward else list(reversed(mutations))
    for pm in ordered_mutations:
        for im in pm.input_mutations:
            new_input_indices = [
                _shift_index_through_mutation(idx, im, forward=forward)
                for idx in curr_input_indices
            ]
            for i, (old, new) in enumerate(zip(curr_input_indices, new_input_indices, strict=True)):
                if old is not None and new is None:
                    logger.info(
                        f"Alias IOMutation {source} generated by pass "
                        f"{source.pass_name if source else 'unknown'} on "
                        f"in={input_indices[i]} "
                        f"was removed by {pm.pass_name} "
                        f"(input {im.kind.value} at indices={im.indices}) - "
                        f"the corresponding alias will be ignored and not applied "
                        f"to the stablehlo passed to neuronx-cc compilation or to "
                        f"the NEFF output->input aliased buffer allocation"
                    )
            curr_input_indices = new_input_indices
        for om in pm.output_mutations:
            new_output_indices = [
                _shift_index_through_mutation(idx, om, forward=forward)
                for idx in curr_output_indices
            ]
            for i, (old, new) in enumerate(
                zip(curr_output_indices, new_output_indices, strict=True)
            ):
                if old is not None and new is None:
                    logger.info(
                        f"Alias IOMutation {source} generated by pass "
                        f"{source.pass_name if source else 'unknown'} on "
                        f"out={output_indices[i]} "
                        f"was removed by {pm.pass_name} "
                        f"(output {om.kind.value} at indices={om.indices}) - "
                        f"the corresponding alias will be ignored and not applied "
                        f"to the stablehlo passed to neuronx-cc compilation or to "
                        f"the NEFF output->input aliased buffer allocation"
                    )
            curr_output_indices = new_output_indices
    return curr_input_indices, curr_output_indices


def get_alias_output_buffer_map(
    pass_mutations: list[PassMutations],
) -> dict[int, int]:
    """Return {neff_output_index: neff_input_index} for buffer reuse and StableHLO hints.

    Covers both is_mutation=True and is_mutation=False ALIAS entries.
    Translates pass-local output indices forward through subsequent output mutations
    and pass-local input indices forward through subsequent input mutations.
    """
    alias_output_to_input_map: dict[int, int] = {}
    for i, pm in enumerate(pass_mutations):
        for m in pm.output_mutations:
            if m.kind != MutationKind.ALIAS:
                continue
            successor_mutations = [] if i == len(pass_mutations) - 1 else pass_mutations[i + 1 :]
            assert m.alias_input_indices is not None, (
                f"Alias IOMutation {m} generated by pass {m.pass_name}"
                f"is missing alias_input_indices"
            )
            neff_in, neff_out = _translate_indices(
                m.alias_input_indices,
                m.indices,
                successor_mutations,
                forward=True,
                source=m,
            )
            for out_idx, in_idx in zip(neff_out, neff_in, strict=True):
                if out_idx is None or in_idx is None:
                    continue
                if out_idx in alias_output_to_input_map:
                    logger.debug(
                        f"Skipping duplicate alias for output[{out_idx}]: "
                        f"already mapped to input[{alias_output_to_input_map[out_idx]}], "
                        f"ignoring mapping to input[{in_idx}] from {pm.pass_name}"
                    )
                    continue
                alias_output_to_input_map[out_idx] = in_idx
    logger.debug(f"Generated alias output->input buffers: {alias_output_to_input_map}")
    return alias_output_to_input_map


def get_mutation_copy_back_inputs(
    pass_mutations: list[PassMutations],
) -> dict[int, int]:
    """Return {neff_output_index: caller_space_input_index} for is_mutation=True ALIAS entries.

    Maps each NEFF output index that holds a mutation result to the caller-space input
    index whose tensor should receive the copy-back (for retain_device support).
    """
    copy_back_output_to_input_buffer_map: dict[int, int] = {}
    for i, pm in enumerate(pass_mutations):
        predecessor_mutations = [] if i == 0 else pass_mutations[:i]
        successor_mutations = [] if i == len(pass_mutations) - 1 else pass_mutations[i + 1 :]
        for m in pm.output_mutations:
            if m.kind != MutationKind.ALIAS or not m.is_mutation:
                continue
            assert m.alias_input_indices is not None, (
                f"Alias IOMutation {m} generated by pass {m.pass_name}"
                f"is missing alias_input_indices"
            )
            # Translate output indices forward to NEFF space
            _, neff_out = _translate_indices(
                m.alias_input_indices,
                m.indices,
                successor_mutations,
                forward=True,
                source=m,
            )
            # Translate input indices backward to caller space
            caller_in, _ = _translate_indices(
                m.alias_input_indices,
                m.indices,
                predecessor_mutations,
                forward=False,
                source=m,
            )
            for out_idx, in_idx in zip(neff_out, caller_in, strict=True):
                if out_idx is None or in_idx is None:
                    continue
                if out_idx in copy_back_output_to_input_buffer_map:
                    logger.debug(
                        f"Skipping duplicate copy-back for output[{out_idx}]: "
                        f"already mapped to "
                        f"input[{copy_back_output_to_input_buffer_map[out_idx]}], "
                        f"ignoring mapping to input[{in_idx}] from {pm.pass_name}"
                    )
                    continue
                copy_back_output_to_input_buffer_map[out_idx] = in_idx
    logger.debug(
        f"Generated copy back output->input buffers: {copy_back_output_to_input_buffer_map}"
    )
    return copy_back_output_to_input_buffer_map


def rebase_pre_aot_pass_mutations(
    pm: PassMutations | None,
    passes_before_functionalize: list[PassMutations],
) -> PassMutations | None:
    """Translate pre-AOT ALIAS mutation indices through passes that ran before
    FunctionalizeCopyInplacePass.

    The pre-AOT AliasingAnalysis sees the already-functionalized graph, so its
    ALIAS indices are in the graph space *after* FunctionalizeCopyInplacePass has
    run.  However, passes like FlexAttentionNKILowering and FlexAttentionLegalization
    execute *before* FunctionalizeCopyInplacePass and may shift input/output indices.

    This utility replays those earlier passes' ADD/REMOVE mutations forward onto both
    the ALIAS output and input indices, producing a new PassMutations list whose indices
    are valid in the graph space immediately after FunctionalizeCopyInplacePass — which
    is where the rebased mutations should be inserted.

    Args:
        pre_aot_pass_mutations: The PassMutations emitted by AliasingAnalysis (pre-AOT).
        passes_before_functionalize: PassMutations from post-AOT passes that ran before
            FunctionalizeCopyInplacePass (e.g. FlexAttentionNKILowering,
            FlexAttentionLegalization).

    Returns:
        A new list of PassMutations with translated ALIAS indices, ready to be inserted
        immediately after the FunctionalizeCopyInplacePass entry in the global
        pass_mutations list.
    """
    if not pm:
        return pm

    assert not pm.input_mutations
    assert all(m.kind == MutationKind.ALIAS for m in pm.output_mutations)
    rebased_out_muts: list[IOMutation] = []
    for m in pm.output_mutations:
        # Replay both input and output indices through the preceding passes
        translated_in, translated_out = _translate_indices(
            m.alias_input_indices,
            m.indices,
            passes_before_functionalize,
            forward=True,
            source=m,
        )

        valid_pairs = []
        for out_idx, in_idx in zip(translated_out, translated_in, strict=True):
            if out_idx is None or in_idx is None:
                continue
            valid_pairs.append((out_idx, in_idx))
        if not valid_pairs:
            continue
        filtered_out, filtered_in = zip(*valid_pairs, strict=False)
        rebased_out_muts.append(
            IOMutation(
                kind=MutationKind.ALIAS,
                indices=list(filtered_out),
                alias_input_indices=list(filtered_in),
                is_mutation=m.is_mutation,
                pass_name=m.pass_name,
            )
        )

    return PassMutations(
        pass_name=pm.pass_name,
        input_mutations=pm.input_mutations,
        output_mutations=rebased_out_muts,
    )


def apply_input_mutations(
    inputs: list,
    pass_mutations: list[PassMutations],
    device: str,
) -> list:
    """Apply input mutations in forward order (original inputs -> NEFF inputs).

    Iterates through each PassMutations entry in order and applies its
    input ADD/REMOVE mutations sequentially.
    """
    values = inputs
    for pm in pass_mutations:
        for m in pm.input_mutations:
            values = _apply_input_mutation(values, m, device)
    return values


def apply_output_mutations(
    outputs: list,
    pass_mutations: list[PassMutations],
    device: str,
    original_inputs: list | None = None,
) -> list:
    """Apply output mutations in reverse order (NEFF outputs -> original outputs).

    Iterates through PassMutations entries in reverse and undoes each pass's
    output mutations in reverse order (last mutation of last pass first).
    """
    values = outputs
    for pm in reversed(pass_mutations):
        for m in reversed(pm.output_mutations):
            values = _apply_output_mutation(values, m, device, original_inputs)
    return values


__all__ = [
    "TRANSFORM_REGISTRY",
    "IOMutation",
    "MutationKind",
    "PassMutations",
    "apply_input_mutations",
    "apply_output_mutations",
    "get_alias_output_buffer_map",
    "get_mutation_copy_back_inputs",
    "rebase_pre_aot_pass_mutations",
]

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
    """Identify supported IO signature and alias mutation kinds.

    Note:
        ``ADD`` and ``REMOVE`` alter an IO signature, while ``ALIAS`` records
        output-to-input buffer reuse metadata.
    """

    ADD = "add"
    REMOVE = "remove"
    ALIAS = "alias"


@dataclass
class IOMutation:
    """Represent one pass-local input or output mutation.

    Args:
        kind (MutationKind): Mutation kind to apply or record.
        indices (list[int]): Affected IO indices in pass-local space.
        transform_key (str | None, optional): Registry key used to generate added
            values. Defaults to ``None``.
        transform_params (dict | None, optional): Parameters passed to the registered
            transform. Defaults to ``None``.
        pass_name (str, optional): Name of the emitting FX pass. Defaults to ``""``.
        alias_input_indices (list[int] | None, optional): Input indices paired with
            output ``indices`` for an ``ALIAS`` mutation. Defaults to ``None``.
        is_mutation (bool | None, optional): Whether an alias requires copy-back;
            ``False`` denotes buffer reuse. Defaults to ``None``.

    Note:
        Input additions and removals apply directly, output mutations apply
        inversely, and alias mutations are metadata-only during value transforms.
        Each pass records one alias mutation per mutation or passthrough direction.
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
    """Group the input and output mutations emitted by one FX pass.

    Args:
        pass_name (str): Name of the FX pass that emitted the mutations.
        input_mutations (list[IOMutation], optional): Input mutations in application
            order. Defaults to an empty list.
        output_mutations (list[IOMutation], optional): Output mutations in application
            order. Defaults to an empty list.

    Note:
        Every index uses the graph IO space before this pass runs.
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


def _log_dropped_alias(
    source: IOMutation | None,
    source_indices: list[int],
    old_indices: list[int | None],
    new_indices: list[int | None],
    remover: IOMutation,
    remover_pass_name: str,
    direction: str,
) -> None:
    """Log one message for each pass that eliminates alias indices.

    The message is at DEBUG level. A removed alias is expected, for example when
    ReplaceDeviceMeshPlaceholders drops a device mesh input.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    dropped = dict.fromkeys(
        source_indices[i]
        for i, (old, new) in enumerate(zip(old_indices, new_indices, strict=True))
        if old is not None and new is None
    )
    if not dropped:
        return
    logger.debug(
        f"The alias from pass {source.pass_name if source else 'unknown'} "
        f"on {direction} indices={list(dropped)} is not applied. "
        f"Pass {remover_pass_name} applied {remover.kind.value} "
        f"to the {direction} indices={remover.indices}. "
        f"The StableHLO for neuronx-cc and the NEFF aliased buffer allocation "
        f"do not get this alias."
    )


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
            _log_dropped_alias(
                source,
                input_indices,
                curr_input_indices,
                new_input_indices,
                im,
                pm.pass_name,
                "input",
            )
            curr_input_indices = new_input_indices
        for om in pm.output_mutations:
            new_output_indices = [
                _shift_index_through_mutation(idx, om, forward=forward)
                for idx in curr_output_indices
            ]
            _log_dropped_alias(
                source,
                output_indices,
                curr_output_indices,
                new_output_indices,
                om,
                pm.pass_name,
                "output",
            )
            curr_output_indices = new_output_indices
    return curr_input_indices, curr_output_indices


def get_alias_output_buffer_map(
    pass_mutations: list[PassMutations],
) -> dict[int, int]:
    """Return NEFF output-to-input aliases for buffer reuse and StableHLO hints.

    Args:
        pass_mutations (list[PassMutations]): Ordered pass-local mutation records.

    Returns:
        dict[int, int]: NEFF output indices mapped to NEFF input indices.

    Raises:
        AssertionError: If an alias mutation has no paired input indices.

    Note:
        Both mutation and passthrough aliases are translated through subsequent
        passes. Eliminated aliases are dropped, and the first duplicate mapping wins.
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
    """Return copy-back destinations for mutation aliases.

    Args:
        pass_mutations (list[PassMutations]): Ordered pass-local mutation records.

    Returns:
        dict[int, int]: NEFF output indices mapped to caller-space input indices.

    Raises:
        AssertionError: If a mutation alias has no paired input indices.

    Note:
        Only aliases marked as mutations are included for retain-device copy-back.
        Eliminated aliases are dropped, and the first duplicate mapping wins.
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
    """Rebase pre-AOT alias indices through passes before functionalization.

    Args:
        pm (PassMutations | None): Alias mutations emitted by pre-AOT alias analysis.
        passes_before_functionalize (list[PassMutations]): Mutations from passes that
            run before ``FunctionalizeCopyInplacePass``.

    Returns:
        PassMutations | None: New record with translated aliases, or the unchanged
        falsey input.

    Raises:
        AssertionError: If ``pm`` contains input or non-alias output mutations.

    Note:
        Pre-AOT alias analysis observes post-functionalization graph space. Replay
        earlier ``ADD`` and ``REMOVE`` mutations before inserting the result after
        functionalization; aliases eliminated by those changes are omitted.
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
    """Apply input mutations from caller space to NEFF space.

    Args:
        inputs (list): Values matching the original caller signature.
        pass_mutations (list[PassMutations]): Ordered pass-local mutation records.
        device (str): Device used by transforms that generate values.

    Returns:
        list: Values matching the post-pass NEFF input signature.

    Note:
        Passes and their input additions or removals are applied in forward order;
        alias metadata does not alter values.
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
    """Restore caller outputs from the NEFF output signature.

    Args:
        outputs (list): Values matching the post-pass NEFF output signature.
        pass_mutations (list[PassMutations]): Ordered pass-local mutation records.
        device (str): Device used by transforms that restore values.
        original_inputs (list | None, optional): Caller inputs available to passthrough
            transforms. Defaults to ``None``.

    Returns:
        list: Values matching the original caller output signature.

    Note:
        Passes and their output mutations are undone in reverse order; alias metadata
        does not alter values.
    """
    values = outputs
    for pm in reversed(pass_mutations):
        for m in reversed(pm.output_mutations):
            values = _apply_output_mutation(values, m, device, original_inputs)
    return values


def translate_input_indices_to_neff(
    input_indices: list[int],
    pass_mutations: list[PassMutations],
) -> list[int]:
    """Translate caller-space input indices to NEFF-space indices.

    Args:
        input_indices (list[int]): Original input positions to translate.
        pass_mutations (list[PassMutations]): Ordered pass-local mutation records.

    Returns:
        list[int]: Surviving input positions after applying input mutations forward.

    Note:
        Removed indices are dropped, and an empty mutation sequence is the identity.
    """
    translated, _ = _translate_indices(list(input_indices), [], pass_mutations, forward=True)
    return [idx for idx in translated if idx is not None]


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
    "translate_input_indices_to_neff",
]

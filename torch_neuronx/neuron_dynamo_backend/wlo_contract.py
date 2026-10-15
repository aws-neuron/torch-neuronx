"""
The WLO weight-relayout contract shared by the compile and execute paths.

A transform is a dict describing how one NEFF input weight is relaid out:

    {"input_index": int, "reshape": list[int], "transpose": list[int]}

- input_index: which NEFF input (0-based) the rule applies to.
- reshape: target dims; the weight is reshaped to these first.
- transpose: a permutation of range(len(reshape)), applied after the reshape.

This mirrors the neuronx-cc emitter (HloNeffWrapper.cc::createTransposeComputation),
which builds reshape(transpose(reshape(weight))). Its trailing reshape-back (to the
original weight shape, only to keep the computation shape-preserving) is NOT part of
the contract -- the executor reconstructs it from the runtime shape, applying
weight.reshape(reshape).permute(transpose).contiguous().reshape(weight.shape), so the
NEFF sees the original shape with only the bytes relaid out.

validate_transform enforces the static shape at both ends -- the parser
(wlo_hlo_parser) and the Executor -- because transforms also reach the Executor from a
pickled precompile cache artifact without passing back through the parser.
"""

# The exact keys of a WLO transform dict.
TRANSFORM_KEYS = ("input_index", "reshape", "transpose")


class WLOContractError(RuntimeError):
    """A WLO transform violates the compile<->execute relayout contract."""


def validate_transform(t: object) -> None:
    """Raise WLOContractError unless ``t`` is a structurally valid WLO transform.

    Checks only the static shape: keys present, non-negative input_index, non-empty
    positive-dim reshape, and transpose a permutation of range(len(reshape)). Callers
    holding the input tensor also check element count and input-index range.
    """
    if not isinstance(t, dict):
        raise WLOContractError(f"WLO transform must be a dict, got {type(t).__name__}: {t!r}")

    missing = [k for k in TRANSFORM_KEYS if k not in t]
    if missing:
        raise WLOContractError(f"WLO transform missing key(s) {missing}: {t!r}")

    idx, reshape, transpose = t["input_index"], t["reshape"], t["transpose"]

    if not isinstance(idx, int) or idx < 0:
        raise WLOContractError(f"WLO transform input_index must be a non-negative int, got {idx!r}")

    if not isinstance(reshape, list | tuple) or not reshape:
        raise WLOContractError(f"WLO transform reshape must be a non-empty list, got {reshape!r}")
    if any(not isinstance(d, int) or d <= 0 for d in reshape):
        raise WLOContractError(f"WLO transform reshape dims must be positive ints, got {reshape!r}")

    if not isinstance(transpose, list | tuple) or len(transpose) != len(reshape):
        raise WLOContractError(
            f"WLO transform transpose must have the same rank as reshape "
            f"(len {len(reshape)}), got {transpose!r}"
        )
    if sorted(transpose) != list(range(len(reshape))):
        raise WLOContractError(
            f"WLO transform transpose must be a permutation of range({len(reshape)}), "
            f"got {transpose!r}"
        )

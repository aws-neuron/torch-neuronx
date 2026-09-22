"""Public, versioned data models for the compiler API.

The CompileOptions input plus the artifact type that fx_to_stablehlo /
stablehlo_to_neff / fx_to_neff return. Kept separate from api.py so the surface
is easy to see and import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import PassMutations


@dataclass
class CompileOptions:
    """Compilation options.

    `target` makes the platform a *compile input* rather than a live-hardware
    query: when set (e.g. "trn2"), the compile path does not query NRT for the
    instance type, so it runs on a host with no Neuron device. When None, the
    behaviour is unchanged (live-instance query on hardware).
    """

    target: str | None = None  # "trn2" / "trn3" / "inf2"; None -> query live instance
    logical_nc: str | None = None  # None -> default for target
    optimization_level: str = "-O2"
    model_name: str = "model"
    use_tensorizer: bool = False
    # False: graph is already post-pass; caller supplies alias analysis.
    run_fx_passes: bool = True


@dataclass
class TensorSpec:
    """Tensor shape + dtype specification."""

    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass
class InputOutputAlias:
    """One input->output storage alias.

    When an output is produced by writing in place into an input buffer (e.g. a
    KV cache that the graph updates and returns), the two share storage. A
    runtime should bind that output to the input's buffer instead of allocating
    a fresh tensor.
    """

    input_index: int  # NEFF input position whose buffer is reused
    output_index: int  # NEFF output position written into that input


@dataclass
class IOSpec:
    """Ordered input/output specs + aliases — the runtime handoff."""

    inputs: list[TensorSpec] = field(default_factory=list)
    outputs: list[TensorSpec] = field(default_factory=list)
    aliases: list[InputOutputAlias] = field(default_factory=list)


@dataclass
class CompilationArtifact:
    """Single artifact holding all compilation outputs for a graph.

    `io_spec`, `target`, `stablehlo`, and `has_collectives` form the serializable
    handoff. The underscore-prefixed fields are optional process-local state.

    Progressively populated through the pipeline stages:
    - After fx_to_stablehlo: stablehlo, io_spec set
    - After fx_to_neff / stablehlo_to_neff: cache_key also set
    - After synchronize + get_neff_info: neff bytes fetched by caller

        artifacts = [fx_to_neff(gm, opts) for gm in graphs]
        torch.neuron.synchronize()
        for art in artifacts:
            _, neff = torch_neuronx._C.get_neff_info(art.cache_key)
    """

    io_spec: IOSpec
    cache_key: str | None = None  # set after NEFF compile submitted
    target: str | None = None
    stablehlo: bytes | None = None  # set after FX -> StableHLO
    neff: bytes | None = None  # caller-populated after synchronize + get_neff_info
    has_collectives: bool = False
    pass_mutations: list[PassMutations] = field(default_factory=list)
    _mlir_module: object = None
    _lowering: object = None

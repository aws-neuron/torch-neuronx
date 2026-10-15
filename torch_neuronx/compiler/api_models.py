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
    """Configure compiler API behavior.

    Args:
        target (str, optional): Platform target override, such as ``"trn2"``.
            Defaults to ``None``, which leaves platform discovery unchanged.
        logical_nc (str, optional): Logical NeuronCore override. Defaults to
            ``None``, which leaves logical-core discovery unchanged.
        optimization_level (str, optional): Neuron compiler optimization level. Defaults
            to ``"-O2"``.
        model_name (str, optional): Name used for compilation artifacts. Defaults to
            ``"model"``.
        use_tensorizer (bool, optional): Whether to use tensorizer lowering. Defaults to
            ``False``.
        run_fx_passes (bool, optional): Whether to run the post-AOT FX passes. Defaults to
            ``True``; set to ``False`` for an already processed graph.

    Note:
        Set both ``target`` and ``logical_nc`` to avoid live platform discovery.
        Device-free NRT initialization also requires fake instance configuration.
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
    """Describe a tensor shape and data type.

    Args:
        shape (tuple): Tensor dimensions, including dynamic or scalar shapes.
        dtype (torch.dtype): Tensor data type.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass
class InputOutputAlias:
    """Describe one input-to-output storage alias.

    Args:
        input_index (int): NEFF input position whose buffer is reused.
        output_index (int): NEFF output position written into that input.

    Note:
        An output produced by an in-place input update, such as a returned KV
        cache, shares storage with that input. The runtime should bind the output
        to the input buffer instead of allocating a new tensor.
    """

    input_index: int  # NEFF input position whose buffer is reused
    output_index: int  # NEFF output position written into that input


@dataclass
class IOSpec:
    """Describe the ordered runtime IO handoff.

    Args:
        inputs (list, optional): Ordered NEFF input tensor specifications. Defaults to an
            empty list.
        outputs (list, optional): Ordered NEFF output tensor specifications. Defaults to an
            empty list.
        aliases (list, optional): Input-to-output storage aliases. Defaults to an empty list.
    """

    inputs: list[TensorSpec] = field(default_factory=list)
    outputs: list[TensorSpec] = field(default_factory=list)
    aliases: list[InputOutputAlias] = field(default_factory=list)


@dataclass
class CompilationArtifact:
    """Hold compilation outputs for one graph.

    Args:
        io_spec (IOSpec): Ordered input, output, and alias metadata.
        cache_key (str, optional): Submitted NEFF compilation cache key. Defaults to ``None``.
        target (str, optional): Platform target used for compilation. Defaults to ``None``.
        stablehlo (bytes, optional): Serialized StableHLO module. Defaults to ``None``.
        neff (bytes, optional): Caller-populated compiled NEFF bytes. Defaults to ``None``.
        has_collectives (bool, optional): Whether the graph contains collective operations.
            Defaults to ``False``.
        pass_mutations (list, optional): IO mutations introduced by graph passes. Defaults to an
            empty list.
        _mlir_module (object, optional): Process-local MLIR module. Defaults to ``None``.
        _lowering (object, optional): Process-local lowering state. Defaults to ``None``.

    Note:
        ``io_spec``, ``target``, ``stablehlo``, ``has_collectives``, and
        ``pass_mutations`` form the detached handoff. Graph lowering populates
        StableHLO and IO metadata; NEFF submission populates ``cache_key`` before
        returning, while compilation completes asynchronously. After
        synchronization, callers fetch and populate ``neff`` from the cache.
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

"""torch_neuronx.compiler — public compile API (FX → StableHLO → NEFF).

Surfaces the lowering pipeline as a stable API so a consumer (vLLM-Neuron, a
standalone runtime, a custom dynamo backend) can obtain a NEFF from an FX graph
without reimplementing the lowering. The implementation lives in:

    api_models.py — CompileOptions and the artifact/handoff data models
    api.py      — fx_to_stablehlo / stablehlo_to_neff / fx_to_neff / compile_only

Trace the FX graph with `decompositions=get_neuron_decompositions()` (via
torch.export / AOTAutograd) so composite ops lower correctly.
"""

from torch_neuronx.compiler.api import (
    compile_only,
    fx_to_neff,
    fx_to_stablehlo,
    get_neuron_decompositions,
    stablehlo_to_neff,
)
from torch_neuronx.compiler.api_models import (
    CompilationArtifact,
    CompileOptions,
    InputOutputAlias,
    IOSpec,
    TensorSpec,
)

__all__ = [
    "CompilationArtifact",
    "CompileOptions",
    "IOSpec",
    "InputOutputAlias",
    "TensorSpec",
    "compile_only",
    "fx_to_neff",
    "fx_to_stablehlo",
    "get_neuron_decompositions",
    "stablehlo_to_neff",
]

"""
Pre-AOTAutograd graph preprocessing pipeline.

This module applies FX graph analysis and transformations before the graph
is processed by AOTAutograd. Currently performs aliasing analysis to detect
input-output relationships for buffer optimization.
"""

from typing import NamedTuple

import torch
from torch.fx.passes.infra.pass_manager import PassManager

from torch_neuronx.neuron_dynamo_backend.fx.passes.aliasing_analysis import AliasingAnalysis
from torch_neuronx.neuron_dynamo_backend.fx.passes.contiguous_memory import (
    EnsureContiguousInputs,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.dynamic_shape_analysis import (
    DynamicShapeAnalysis,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.sdpa_causal_mask import (
    SdpaCausalMaskRewrite,
)


class PreprocessResult(NamedTuple):
    """Store pre-AOTAutograd graph preprocessing results.

    Args:
        graph_module (torch.fx.GraphModule): Graph after preprocessing passes.
        analysis (dict): Mapping whose ``"pre_aot_pass_mutations"`` entry contains
            the aliasing pass's ``PassMutations`` record.
    """

    graph_module: torch.fx.GraphModule
    analysis: dict


def preprocess_graph(gm: torch.fx.GraphModule) -> PreprocessResult:
    """Apply FX passes before AOTAutograd processing.

    Args:
        gm (torch.fx.GraphModule): Graph module to transform.

    Returns:
        PreprocessResult: Transformed graph and a mapping whose
            ``"pre_aot_pass_mutations"`` entry is the aliasing pass's
            ``PassMutations`` record.
    """
    aliasing_analysis = AliasingAnalysis()
    dynamic_shape_analysis = DynamicShapeAnalysis()
    ensure_contiguous = EnsureContiguousInputs()
    sdpa_causal_mask = SdpaCausalMaskRewrite()
    pm = PassManager(
        passes=[ensure_contiguous, sdpa_causal_mask, aliasing_analysis, dynamic_shape_analysis],
        run_checks_after_each_pass=True,
    )
    pass_result = pm(gm)

    analysis_results = {
        "pre_aot_pass_mutations": aliasing_analysis._pass_mutations,
    }

    return PreprocessResult(pass_result.graph_module, analysis_results)

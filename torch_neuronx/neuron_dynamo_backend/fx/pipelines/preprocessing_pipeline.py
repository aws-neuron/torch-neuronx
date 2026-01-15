from typing import NamedTuple

import torch
from torch.fx.passes.pass_manager import PassManager

from torch_neuronx.neuron_dynamo_backend.fx.passes.aliasing_analysis import AliasingAnalysis


class PreprocessResult(NamedTuple):
    """Result of graph preprocessing."""

    graph_module: torch.fx.GraphModule
    analysis: dict


def preprocess_graph(gm: torch.fx.GraphModule) -> PreprocessResult:
    """
    Apply FX graph analysis and transformations before AOTAutograed.

    Args:
        gm: Input GraphModule

    Returns:
        PreprocessResult containing the graph and analysis results
    """
    aliasing_analysis = AliasingAnalysis()
    PassManager(passes=[aliasing_analysis])(gm)

    analysis_results = {
        "aliasing_analysis": aliasing_analysis.result,
    }

    return PreprocessResult(gm, analysis_results)

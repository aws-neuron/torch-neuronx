"""
Post-AOTAutograd graph processing pipeline.

This module applies FX graph transformations after AOTAutograd has processed
the graph. Includes flex attention legalization, in-place mutation handling,
and None output removal.
"""

from typing import NamedTuple

import torch
from torch.fx.passes.infra.pass_manager import PassManager

from torch_neuronx.neuron_dynamo_backend.fx.passes.collective_legalization import (
    CollectiveLegalization,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.dtensor_redistribute_optimization import (
    DTensorRedistributeOptimization,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.flex_attention_legalization import (
    FlexAttentionLegalization,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.flex_attention_nki_lowering import (
    FlexAttentionNKILowering,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.functionalize_copy_inplace_result import (
    FunctionalizeCopyInplacePass,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.legalize_bool_scalars import (
    LegalizeBoolScalars,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.legalize_index_indices import (
    LegalizeIndexIndices,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.random_op_legalization import (
    RandomOpLegalization,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.remove_none_outputs import RemoveNoneOutputs


class PostAOTProcessingResults(NamedTuple):
    """Result of graph preprocessing."""

    graph_module: torch.fx.GraphModule
    analysis: dict


def post_aot_processing(gm: torch.fx.GraphModule) -> PostAOTProcessingResults:
    """
    Apply FX graph analysis and transformations after AOTAutograd.

    Args:
        gm: Input GraphModule

    Returns:
        PostAOTProcessingResults containing the graph and analysis results
    """
    flex_lowering_pass = FlexAttentionNKILowering()
    flex_attention_pass = FlexAttentionLegalization()
    functionalize_pass = FunctionalizeCopyInplacePass()
    legalize_bool_scalars_pass = LegalizeBoolScalars()
    legalize_index_indices_pass = LegalizeIndexIndices()
    random_op_pass = RandomOpLegalization()
    remove_none_outputs = RemoveNoneOutputs()
    collective_legalization = CollectiveLegalization()
    dtensor_redistribute = DTensorRedistributeOptimization()

    pm = PassManager(
        passes=[
            flex_lowering_pass,
            flex_attention_pass,
            functionalize_pass,
            legalize_bool_scalars_pass,
            legalize_index_indices_pass,
            random_op_pass,
            remove_none_outputs,
            dtensor_redistribute,  # Before collective legalization
            collective_legalization,
        ],
        run_checks_after_each_pass=True,
        suppress_check_failures=False,
    )
    pass_result = pm(gm)

    analysis_results = {
        "mutation_info": functionalize_pass.get_mutation_info(),
        "none_output_info": remove_none_outputs.result,
        "random_input_info": random_op_pass.result,
    }

    return PostAOTProcessingResults(pass_result.graph_module, analysis_results)

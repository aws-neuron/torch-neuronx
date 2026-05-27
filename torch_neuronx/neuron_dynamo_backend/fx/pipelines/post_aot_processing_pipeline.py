"""
Post-AOTAutograd graph processing pipeline.

This module applies FX graph transformations after AOTAutograd has processed
the graph. Includes flex attention legalization, in-place mutation handling,
and None output removal.
"""

from typing import NamedTuple

import torch
from torch.fx.passes.infra.pass_manager import PassManager

from torch_neuronx.kernels.cache_utils import CompilationJob
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
from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import NeuronFXPassState
from torch_neuronx.neuron_dynamo_backend.fx.passes.remove_none_outputs import RemoveNoneOutputs
from torch_neuronx.neuron_dynamo_backend.fx.passes.replace_device_mesh_placeholders import (
    ReplaceDeviceMeshPlaceholders,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.vocab_sharding_spmd import (
    ShardIndexInjection,
    VocabShardingSPMD,
)


class PostAOTProcessingResults(NamedTuple):
    """Result of graph preprocessing."""

    graph_module: torch.fx.GraphModule
    analysis: dict


def post_aot_processing(
    gm: torch.fx.GraphModule, job: CompilationJob, model_name: str
) -> PostAOTProcessingResults:
    """
    Apply FX graph analysis and transformations after AOTAutograd.

    Args:
        gm: Input GraphModule
        job: CompilationJob for dumping FX graph after each pass.
        model_name: model name for artifact filenames.

    Returns:
        PostAOTProcessingResults containing the graph and analysis results
    """
    config = NeuronFXPassState(job=job, model_name=model_name, dumping_enabled=job.preserve)

    flex_lowering_pass = FlexAttentionNKILowering(config)
    flex_attention_pass = FlexAttentionLegalization(config)
    functionalize_pass = FunctionalizeCopyInplacePass(config)
    legalize_bool_scalars_pass = LegalizeBoolScalars(config)
    legalize_index_indices_pass = LegalizeIndexIndices(config)
    remove_none_outputs = RemoveNoneOutputs(config)
    replace_device_mesh = ReplaceDeviceMeshPlaceholders(config)
    collective_legalization = CollectiveLegalization(config)
    dtensor_redistribute = DTensorRedistributeOptimization(config)

    # Vocab sharding SPMD passes
    shard_injection = ShardIndexInjection(config)
    vocab_sharding = VocabShardingSPMD(config)
    spmd_vocab_passes = [shard_injection, vocab_sharding]

    pm = PassManager(
        passes=[
            flex_lowering_pass,
            flex_attention_pass,
            functionalize_pass,
            legalize_bool_scalars_pass,
            legalize_index_indices_pass,
            remove_none_outputs,
            replace_device_mesh,
            dtensor_redistribute,  # Before collective legalization
            *spmd_vocab_passes,
            collective_legalization,
        ],
        run_checks_after_each_pass=True,
        suppress_check_failures=False,
    )
    pass_result = pm(gm)

    analysis_results = {
        "mutation_info": functionalize_pass.get_mutation_info(),
        "passthrough_info": functionalize_pass.get_passthrough_info(),
        "none_output_info": remove_none_outputs.result,
        "shard_input_info": shard_injection.result,
        "removed_device_mesh_indices": replace_device_mesh.removed_indices,
    }

    return PostAOTProcessingResults(pass_result.graph_module, analysis_results)

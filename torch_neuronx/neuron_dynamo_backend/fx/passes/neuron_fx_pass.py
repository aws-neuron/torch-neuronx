"""Base class for Neuron FX passes with optional artifact dumping."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch.fx.passes.infra.pass_base import PassBase, PassResult

from torch_neuronx.kernels.cache_utils import CompilationJob
from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import IOMutation, PassMutations

logger = logging.getLogger(__name__)


@dataclass
class NeuronFXPassState:
    job: CompilationJob
    model_name: str
    dumping_enabled: bool
    pass_mutations: list[PassMutations] = field(default_factory=list)
    pre_aot_pass_mutations: PassMutations | None = None

    def add_pass_mutations(
        self,
        pass_name: str,
        input_mutations: list[IOMutation] | None = None,
        output_mutations: list[IOMutation] | None = None,
    ) -> None:
        """Record the input and output mutations emitted by a single FX pass."""
        pm = PassMutations(
            pass_name=pass_name,
            input_mutations=input_mutations or [],
            output_mutations=output_mutations or [],
        )
        logger.debug(f"Added pass mutations from {pass_name}: {pm}")
        self.pass_mutations.append(pm)


class NeuronFXPass(PassBase):
    """PassBase subclass that optionally dumps FX artifacts after execution."""

    def __init__(self, state: NeuronFXPassState | None = None):
        super().__init__()
        self.state = state

    def __call__(self, graph_module: torch.fx.GraphModule) -> PassResult | None:
        result = super().__call__(graph_module)
        if result is not None and self.state is not None and self.state.dumping_enabled:
            self.state.job.save_pass_fx(
                self.state.model_name, result.graph_module, type(self).__name__
            )
        return result

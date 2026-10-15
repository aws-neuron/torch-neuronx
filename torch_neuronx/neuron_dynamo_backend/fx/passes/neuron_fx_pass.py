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
    """Store shared state for Neuron FX passes.

    Args:
        job (CompilationJob): Compilation job used to save pass artifacts.
        model_name (str): Model name used when saving artifacts.
        dumping_enabled (bool): Whether passes save their resulting FX graphs.
        pass_mutations (list[PassMutations], optional): Recorded pass mutations.
            Defaults to an empty list.
        pre_aot_pass_mutations (PassMutations, optional): Mutations recorded before AOT.
            Defaults to ``None``.
    """

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
        """Record input and output mutations emitted by one FX pass.

        Args:
            pass_name (str): Name of the pass that emitted the mutations.
            input_mutations (list[IOMutation], optional): Input mutations to record.
                Defaults to ``None``.
            output_mutations (list[IOMutation], optional): Output mutations to record.
                Defaults to ``None``.
        """
        pm = PassMutations(
            pass_name=pass_name,
            input_mutations=input_mutations or [],
            output_mutations=output_mutations or [],
        )
        logger.debug(f"Added pass mutations from {pass_name}: {pm}")
        self.pass_mutations.append(pm)


class NeuronFXPass(PassBase):
    """Run an FX pass and optionally save its resulting graph artifact.

    Args:
        state (NeuronFXPassState, optional): Shared pass state. Defaults to ``None``.
    """

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

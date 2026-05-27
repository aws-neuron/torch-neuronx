"""Base class for Neuron FX passes with optional artifact dumping."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.fx.passes.infra.pass_base import PassBase, PassResult

from torch_neuronx.kernels.cache_utils import CompilationJob


@dataclass
class NeuronFXPassState:
    job: CompilationJob
    model_name: str
    dumping_enabled: bool


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

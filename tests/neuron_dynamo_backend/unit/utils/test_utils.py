from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.fx as fx
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

# ============== Graph Capture Infrastructure ==============


@dataclass
class CapturedGraphs:
    """Stores FX graphs captured during compilation."""

    pre_aot_graph: fx.GraphModule | None = None
    post_aot_forward_graph: fx.GraphModule | None = None


def _make_capture_backend(storage: CapturedGraphs, capture: bool = False):
    """Creates a torch.compile backend that captures pre/post AOT graphs."""

    def fw_compiler(gm: fx.GraphModule, example_inputs):
        storage.post_aot_forward_graph = gm
        return gm.forward

    aot_backend = aot_autograd(fw_compiler=fw_compiler)

    def capture_backend(gm: fx.GraphModule, example_inputs):
        storage.pre_aot_graph = gm
        return aot_backend(gm, example_inputs)

    return capture_backend


def get_aot_graphs(model: torch.nn.Module, *inputs) -> CapturedGraphs:
    """Compile model and capture FX graphs during tracing."""
    captured = CapturedGraphs()
    compiled = torch.compile(model, backend=_make_capture_backend(captured))
    with torch.no_grad():
        compiled(*inputs)
    return captured


def create_capture_compiler(captured_gm_holder: list):
    """Creates a compiler function that captures the graph module."""

    def capture_compiler(gm: torch.fx.GraphModule, example_inputs):
        captured_gm_holder.clear()
        captured_gm_holder.append(gm)
        return make_boxed_func(gm.forward)

    return capture_compiler

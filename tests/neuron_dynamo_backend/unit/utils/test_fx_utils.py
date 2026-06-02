"""Unit tests for `torch_neuronx.neuron_dynamo_backend.utils.fx_utils`."""

import torch
import torch.fx as fx

from torch_neuronx.neuron_dynamo_backend.utils.fx_utils import replace_device_with_cpu


class TestReplaceDeviceWithCpu:
    """Tests for `replace_device_with_cpu`."""

    def test_call_function_kwarg_rewritten(self):
        """A `call_function` node with `device=<non-cpu>` kwarg gets rewritten to cpu."""
        graph = fx.Graph()
        b = graph.placeholder("b")
        new_ones = graph.call_function(
            torch.ops.aten.new_ones.default,
            args=(b, []),
            kwargs={
                "dtype": torch.bool,
                "device": torch.device("neuron"),
                "pin_memory": False,
            },
        )
        graph.output(new_ones)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        new_gm = replace_device_with_cpu(gm)

        rewritten = next(n for n in new_gm.graph.nodes if n.op == "call_function")
        assert rewritten.kwargs["device"] == torch.device("cpu")

    def test_call_method_kwarg_rewritten(self):
        """A `call_method` node with `device=<non-cpu>` kwarg gets rewritten to cpu."""

        class M(torch.nn.Module):
            def forward(self, b):
                return b.new_ones(size=(), dtype=torch.bool, device=torch.device("neuron"))

        gm = fx.symbolic_trace(M())
        # Sanity: ensure the tracer produced a call_method node (not constant-folded).
        assert any(n.op == "call_method" for n in gm.graph.nodes)

        new_gm = replace_device_with_cpu(gm)

        rewritten = next(n for n in new_gm.graph.nodes if n.op == "call_method")
        assert rewritten.kwargs["device"] == torch.device("cpu")

    def test_original_gm_not_mutated(self):
        """`replace_device_with_cpu` must not mutate the input graph module."""
        graph = fx.Graph()
        b = graph.placeholder("b")
        new_ones = graph.call_function(
            torch.ops.aten.new_ones.default,
            args=(b, []),
            kwargs={"dtype": torch.bool, "device": torch.device("neuron"), "pin_memory": False},
        )
        graph.output(new_ones)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        replace_device_with_cpu(gm)

        original_node = next(n for n in gm.graph.nodes if n.op == "call_function")
        assert original_node.kwargs["device"] == torch.device("neuron")

    def test_string_device_kwarg_rewritten(self):
        """Bare string device kwargs (e.g. `device="neuron"`) must also be
        rewritten to `torch.device("cpu")`.
        """
        graph = fx.Graph()
        b = graph.placeholder("b")
        new_ones = graph.call_function(
            torch.ops.aten.new_ones.default,
            args=(b, []),
            kwargs={"dtype": torch.bool, "device": "neuron", "pin_memory": False},
        )
        graph.output(new_ones)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        new_gm = replace_device_with_cpu(gm)

        rewritten = next(n for n in new_gm.graph.nodes if n.op == "call_function")
        assert rewritten.kwargs["device"] == torch.device("cpu")

"""
Legalize boolean scalar arguments in arithmetic FX graph nodes.

Replaces Python bool literals (True/False) with integer scalars (1/0)
in arithmetic op arguments. This is needed because the Neuron compiler
does not support boolean scalars in arithmetic contexts.
"""

import logging

import torch
from torch.fx.passes.infra.pass_base import PassResult

from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import NeuronFXPass

logger = logging.getLogger(__name__)

_ARITHMETIC_OPS = {
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add.Scalar,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.sub.Scalar,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.div.Scalar,
    torch.ops.aten.pow.Tensor_Scalar,
    torch.ops.aten.pow.Scalar,
    torch.ops.aten.fmod.Tensor,
    torch.ops.aten.fmod.Scalar,
    torch.ops.aten.remainder.Tensor,
    torch.ops.aten.remainder.Scalar,
    torch.ops.aten.floor_divide.default,
}


class LegalizeBoolScalars(NeuronFXPass):
    """Replace positional Boolean scalars with integers in arithmetic operations.

    Args:
        state (NeuronFXPassState, optional): Shared FX-pass state used for artifact dumping.
            Defaults to ``None``.
    """

    def call(self, gm: torch.fx.GraphModule) -> PassResult:
        """Legalize positional Boolean scalar arguments in an FX graph.

        Args:
            gm (torch.fx.GraphModule): FX graph module to update.

        Returns:
            PassResult: Graph module and whether any Boolean argument changed.
        """
        graph = gm.graph
        modified = False

        for node in graph.nodes:
            if node.op != "call_function" or node.target not in _ARITHMETIC_OPS:
                continue

            new_args = []
            args_changed = False
            for arg in node.args:
                if isinstance(arg, bool):
                    new_args.append(int(arg))
                    args_changed = True
                else:
                    new_args.append(arg)

            if args_changed:
                logger.debug(f"Node {node.name}: replaced bool scalar(s) with int in args")
                node.args = tuple(new_args)
                modified = True

        if modified:
            graph.lint()
            gm.recompile()

        return PassResult(gm, modified=modified)

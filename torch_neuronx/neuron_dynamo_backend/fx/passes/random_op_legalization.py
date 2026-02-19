"""
Random Op Legalization Pass for PyTorch FX Graphs.

Legalizes random operations (e.g., native_dropout) by converting them to
deterministic operations that take random masks as inputs. This allows
the random number generation to happen outside the compiled graph.
"""

import logging

import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult

from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import (
    NativeDropoutOp,
    RandomInputInfo,
)

logger = logging.getLogger(__name__)


class RandomOpLegalization(PassBase):
    """Random Op Legalization Pass for PyTorch FX Graphs.

    This pass legalizes random operations by converting them to deterministic
    operations that accept pre-generated random masks as graph inputs.

    For native_dropout:
    - Input: native_dropout(input, p, train) -> (output, mask)
    - Output when train=True: mul(input, random_mask) * scale, random_mask
    - Output when train=False: mul(input, random_mask), random_mask (no scaling)

    The random masks are added as new inputs to the graph, allowing the runtime
    to provide different random masks for each execution.
    """

    def __init__(self):
        super().__init__()
        self.result: RandomInputInfo | None = None

    @property
    def name(self) -> str:
        return "random_op_legalization"

    # =========================================================================
    # Helper Methods
    # =========================================================================

    @staticmethod
    def _add_metadata_to_node(
        node: fx.Node, example_tensor=None, shape=None, dtype=None
    ) -> fx.Node:
        """Helper to add metadata to a node for torch-mlir compatibility."""
        if not hasattr(node, "meta"):
            node.meta = {}
        with FakeTensorMode():
            node.meta["val"] = torch.empty(shape, dtype=dtype)
        return node

    @staticmethod
    def _is_native_dropout(node: Node) -> bool:
        """Check if node is a native_dropout operation."""
        return node.op == "call_function" and node.target == torch.ops.aten.native_dropout.default

    def _get_placeholder_nodes(self, graph: fx.Graph) -> list[fx.Node]:
        """Get all placeholder (input) nodes from the graph."""
        return [node for node in graph.nodes if node.op == "placeholder"]

    def _get_last_placeholder(self, graph: fx.Graph) -> fx.Node | None:
        """Get the last placeholder node in the graph."""
        placeholders = self._get_placeholder_nodes(graph)
        return placeholders[-1] if placeholders else None

    # =========================================================================
    # Dropout Legalization
    # =========================================================================

    def _legalize_native_dropout(
        self,
        graph: fx.Graph,
        node: fx.Node,
        new_input_idx: int,
    ) -> NativeDropoutOp | None:
        """Legalize a native_dropout node.

        Transforms: native_dropout(input, p, train) -> (output, mask)
        To: (mul(input, mask_input) * scale, mask_input) when train=True
        Or: (input, ones_mask) when train=False (no dropout, just pass through)

        Args:
            graph: The FX graph being modified
            node: The native_dropout node to legalize
            new_input_idx: Index for the new random input

        Returns:
            NativeDropoutOp descriptor for the new random input when train=True,
            None when train=False (no random input needed)

        Raises:
            ValueError: If dropout probability is not a constant
            ValueError: If input tensor lacks metadata
        """
        logger.debug(f"Legalizing native_dropout node: {node.name}")
        if len(node.args) < 3:
            raise ValueError(
                f"native_dropout {node.name} has insufficient args: "
                f"expected 3, got {len(node.args)}"
            )
        input_tensor = node.args[0]
        dropout_prob = node.args[1]
        train_mode = node.args[2]

        # Validate dropout probability is a constant
        if not isinstance(dropout_prob, int | float):
            raise ValueError(
                f"Dropout probability must be a constant (int or float), "
                f"got {type(dropout_prob).__name__} in node {node.name}"
            )
        prob_value = float(dropout_prob)

        # Get shape and dtype from input tensor metadata
        if "tensor_meta" in input_tensor.meta:
            tensor_metadata = input_tensor.meta["tensor_meta"]
            input_shape = tensor_metadata.shape
            input_dtype = tensor_metadata.dtype
        elif "val" in input_tensor.meta:
            val = input_tensor.meta["val"]
            input_shape = val.shape
            input_dtype = val.dtype
        else:
            raise ValueError(
                f"No metadata for input tensor of {node.name}. "
                "Shape propagation may not have been run."
            )

        # deterministic mask cases
        if not train_mode or prob_value == 0:
            # train=False or p=0: no dropout, output = input
            logger.debug(
                f"train=False or p=0 for {node.name}, replacing with identity (no dropout)"
            )
            self._replace_dropout_with_identity(graph, node, input_tensor, input_shape, input_dtype)
            return None
        elif prob_value == 1:
            # p=1: all elements dropped, output = zeros
            logger.debug(f"p=1 for {node.name}, replacing with zeros (all dropped)")
            self._replace_dropout_with_zeros(graph, node, input_tensor, input_shape, input_dtype)
            return None

        # Create new placeholder for the boolean random mask input
        last_placeholder = self._get_last_placeholder(graph)
        with graph.inserting_after(last_placeholder):
            mask_input_name = f"random_mask_{new_input_idx}"
            mask_input = graph.placeholder(mask_input_name)
            self._add_metadata_to_node(
                mask_input,
                example_tensor=node,
                shape=input_shape,
                dtype=torch.bool,
            )
            logger.debug(
                f"Created new placeholder: {mask_input_name} with shape {input_shape} (bool)"
            )

        # Insert operations before the dropout node
        # For forward: output = input * (mask * scale)
        scale = 1.0 / (1.0 - prob_value)
        with graph.inserting_before(node):
            # cast input mask to input dtype
            mask_float = graph.call_function(
                torch.ops.aten._to_copy.default,
                args=(mask_input,),
                kwargs={"dtype": input_dtype},
            )
            self._add_metadata_to_node(mask_float, shape=input_shape, dtype=input_dtype)

            # scaled_mask = mask * scale
            scaled_mask = graph.call_function(
                torch.ops.aten.mul.Tensor,
                args=(mask_float, scale),
            )
            self._add_metadata_to_node(scaled_mask, shape=input_shape, dtype=input_dtype)

            # output = input * scaled_mask
            masked_output = graph.call_function(
                torch.ops.aten.mul.Tensor,
                args=(input_tensor, scaled_mask),
            )
            self._add_metadata_to_node(masked_output, shape=input_shape, dtype=input_dtype)
        self._replace_dropout_users(graph, node, masked_output, mask_input)
        return NativeDropoutOp(
            input_position=new_input_idx,
            shape=input_shape,
            dtype=input_dtype,
            probability=prob_value,
            train=train_mode,
        )

    def _replace_dropout_with_identity(
        self,
        graph: fx.Graph,
        node: fx.Node,
        input_tensor: fx.Node,
        input_shape: tuple,
        input_dtype: torch.dtype,
    ):
        """Replace dropout with identity when train=False or p=0.

        When training is disabled or p=0, dropout becomes a no-op.
        The output is just the input, and the mask is all ones (True = keep all).

        Args:
            graph: The FX graph
            node: The native_dropout node being replaced
            input_tensor: The input to the dropout
            input_shape: Shape of the input tensor
            input_dtype: Dtype of the input tensor
        """
        with graph.inserting_before(node):
            ones_mask = graph.call_function(
                torch.ops.aten.ones.default,
                args=(list(input_shape),),
                kwargs={"dtype": torch.bool},
            )
            self._add_metadata_to_node(ones_mask, shape=input_shape, dtype=torch.bool)

        self._replace_dropout_users(graph, node, input_tensor, ones_mask)

    def _replace_dropout_with_zeros(
        self,
        graph: fx.Graph,
        node: fx.Node,
        input_tensor: fx.Node,
        input_shape: tuple,
        input_dtype: torch.dtype,
    ):
        """Replace dropout with zeros when p=1 (all elements dropped).

        When p=1, all elements are dropped, so output is zeros.
        The mask is all zeros (False = drop all).

        Args:
            graph: The FX graph
            node: The native_dropout node being replaced
            input_tensor: The input to the dropout
            input_shape: Shape of the input tensor
            input_dtype: Dtype of the input tensor
        """
        with graph.inserting_before(node):
            zeros_output = graph.call_function(
                torch.ops.aten.zeros.default,
                args=(list(input_shape),),
                kwargs={"dtype": input_dtype},
            )
            self._add_metadata_to_node(zeros_output, shape=input_shape, dtype=input_dtype)
            zeros_mask = graph.call_function(
                torch.ops.aten.zeros.default,
                args=(list(input_shape),),
                kwargs={"dtype": torch.bool},
            )
            self._add_metadata_to_node(zeros_mask, shape=input_shape, dtype=torch.bool)

        self._replace_dropout_users(graph, node, zeros_output, zeros_mask)

    def _replace_dropout_users(
        self,
        graph: fx.Graph,
        node: fx.Node,
        output_tensor: fx.Node,
        mask_input: fx.Node,
    ):
        """Replace all users of a native_dropout node.

        native_dropout returns (output, mask), so we need to handle getitem users.

        Args:
            graph: The FX graph
            node: The native_dropout node being replaced
            output_tensor: The new output tensor (possibly scaled)
            mask_input: The random mask input placeholder
        """
        import operator

        users_to_replace = list(node.users)

        for user in users_to_replace:
            if user.op == "call_function" and user.target == operator.getitem:
                index = user.args[1]
                if index == 0:
                    # First output (dropped tensor) -> replace with output
                    user.replace_all_uses_with(output_tensor)
                elif index == 1:
                    # Second output (mask) -> replace with input mask
                    user.replace_all_uses_with(mask_input)
                graph.erase_node(user)
            else:
                # Direct user - replace with output tensor
                user.replace_all_uses_with(output_tensor)

        # Remove the original dropout node
        graph.erase_node(node)
        logger.debug(f"Removed native_dropout node and replaced {len(users_to_replace)} users")

    # =========================================================================
    # Pass Entry Point
    # =========================================================================

    def call(self, gm: fx.GraphModule) -> PassResult:
        """Execute the random op legalization pass.

        Args:
            gm: The GraphModule to transform.

        Returns:
            PassResult with modified=True if any transformations were made.
        """
        logger.debug("Running random op legalization pass")
        graph = gm.graph
        modified = False
        original_input_count = len(self._get_placeholder_nodes(graph))
        dropout_nodes = [n for n in graph.nodes if self._is_native_dropout(n)]
        if not dropout_nodes:
            logger.debug("No random operations found in graph")
            self.result = RandomInputInfo(
                ops=[],
                original_input_count=original_input_count,
                new_input_count=original_input_count,
            )
            return PassResult(gm, modified=False)

        logger.debug(f"Found {len(dropout_nodes)} native_dropout operations to legalize")
        ops = []
        random_input_idx = 0
        for node in dropout_nodes:
            op = self._legalize_native_dropout(graph, node, random_input_idx)
            if op is not None:
                ops.append(op)
                random_input_idx += 1
            modified = True
        if modified:
            gm.recompile()
        new_input_count = len(self._get_placeholder_nodes(graph))
        self.result = RandomInputInfo(
            ops=ops,
            original_input_count=original_input_count,
            new_input_count=new_input_count,
        )

        logger.debug(f"Random op legalization complete: {self.result}")
        return PassResult(gm, modified=modified)


__all__ = ["RandomOpLegalization"]

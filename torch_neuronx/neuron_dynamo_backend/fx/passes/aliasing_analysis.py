"""
Aliasing Analysis Pass for PyTorch FX Graphs.

Identifies:
1. View/slice operations that create aliases of input tensors
2. In-place mutations (setitem, in-place methods) that modify input tensors

Output aliasing information uses AOTAutograd's output convention:
    [mutated_inputs_in_input_order..., non_aliased_explicit_outputs...]
"""

import logging
import operator
from collections import OrderedDict
from typing import ClassVar

import torch
from torch.fx import Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult

from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo

logger = logging.getLogger(__name__)


class AliasingAnalysis(PassBase):
    """Aliasing and Mutation Analysis Pass for PyTorch FX Graphs.

    This pass analyzes a PyTorch FX GraphModule to identify:
    1. Output tensors that alias input tensors (via view operations)
    2. Input tensors that are mutated in-place (via setitem or in-place methods)

    The output aliasing information is transformed to match AOTAutograd's
    output convention where mutated inputs appear first (sorted by input index),
    followed by non-aliased explicit outputs.
    """

    def __init__(self):
        super().__init__()
        self.result: AliasingInfo | None = None

    @property
    def name(self) -> str:
        return "aliasing_analysis"

    # =========================================================================
    # Operation Classification Sets
    # =========================================================================

    ALIASING_METHODS: ClassVar[set[str]] = {
        "t",
        "transpose",
        "view",
        "slice",
        "reshape",
        "permute",
        "expand",
        "expand_as",
        "squeeze",
        "unsqueeze",
        "mH",
        "mT",
        "as_strided",
        "chunk",
        "unfold",
        "narrow",
        "split",
        "unbind",
        "select",
        "unflatten",
        "flatten",
    }

    ALIASING_ATEN_OPS: ClassVar[set[str]] = {
        "view",
        "reshape",
        "transpose",
        "permute",
        "t",
        "squeeze",
        "unsqueeze",
        "expand",
        "slice",
        "select",
        "narrow",
        "chunk",
        "split",
        "unbind",
        "as_strided",
        "unfold",
        "flatten",
        "unflatten",
        "_unsafe_view",
        "view_as",
        "reshape_as",
        "expand_as",
    }

    INPLACE_METHODS: ClassVar[set[str]] = {
        "add_",
        "sub_",
        "mul_",
        "div_",
        "copy_",
        "fill_",
        "zero_",
        "scatter_",
        "index_copy_",
        "index_fill_",
        "index_put_",
        "masked_fill_",
        "masked_scatter_",
        "clamp_",
        "relu_",
        "pow_",
        "abs_",
    }

    # =========================================================================
    # Operation Detection Methods
    # =========================================================================

    def _contains_slice(self, arg) -> bool:
        """Check if an argument contains a slice object."""
        return isinstance(arg, slice) or (
            isinstance(arg, (tuple | list)) and any(isinstance(a, slice) for a in arg)
        )

    def _is_slice_operation(self, node: Node) -> bool:
        """Check if a node is a slice/getitem operation (creates a view)."""
        if node.op == "call_function" and node.target == operator.getitem:
            return any(self._contains_slice(arg) for arg in node.args)
        return False

    def _is_setitem_operation(self, node: Node) -> bool:
        """Check if a node is a setitem operation (in-place mutation)."""
        return node.op == "call_function" and node.target == operator.setitem

    def _is_inplace_method(self, node: Node) -> bool:
        """Check if a node is an in-place method call."""
        if node.op != "call_method":
            return False
        target = node.target
        if not isinstance(target, str):
            return False
        return target.endswith("_") or target in self.INPLACE_METHODS

    def _is_aten_aliasing_op(self, node: Node) -> bool:
        """
        Check if a node is an ATen aliasing operation.
        Use of make_fx or torch.export to trace graph causes the node to become
        an ATen op.
        class Model(torch.nn.Module):
            def forward(self, x):
                a = x.view(2, 6)           # ATen view
                b = x.transpose(0, 1)      # ATen transpose
                c = x[2:8, :]              # ATen slice
                d = x.reshape(3, 4)        # ATen reshape
                return a, b, c, d
        In the above example, we show what the ATen equivalent of an op is
        'view' -> 'ATen view', 'transpose' -> 'ATen transpose', ... when you
        trace a graph using 'make_fx' or use torch.export.
        """
        if node.op != "call_function":
            return False

        target = node.target
        if hasattr(target, "__module__") and "aten" in str(target.__module__):
            op_name = getattr(target, "__name__", str(target))
            base_name = op_name.split(".")[0] if "." in op_name else op_name
            return base_name in self.ALIASING_ATEN_OPS

        target_name = getattr(target, "__name__", str(target))
        return target_name in self.ALIASING_ATEN_OPS

    def _is_method_aliasing_op(self, node: Node) -> bool:
        """Check if a node is a method-style aliasing operation."""
        if node.op != "call_method":
            return False
        return node.target in self.ALIASING_METHODS

    def is_aliasing_op(self, node: Node) -> bool:
        """Determine if a node creates a view/alias."""
        return (
            self._is_slice_operation(node)
            or self._is_method_aliasing_op(node)
            or self._is_aten_aliasing_op(node)
        )

    def is_mutating_op(self, node: Node) -> bool:
        """Determine if a node mutates a tensor in-place."""
        return self._is_setitem_operation(node) or self._is_inplace_method(node)

    def _is_custom_op_with_mutations(self, node: Node) -> bool:
        """Check if node is a custom op with mutable arguments."""
        if node.op != "call_function":
            return False
        target = node.target
        # Check if it's an OpOverload with a schema
        if not hasattr(target, "_schema"):
            return False
        schema = target._schema
        # Check if any argument has is_write=True
        return any(arg.alias_info and arg.alias_info.is_write for arg in schema.arguments)

    def _get_custom_op_mutated_inputs(
        self,
        node: Node,
        alias_chain: dict[Node, Node],
        input_nodes: set[Node],
        input_nodes_list: list[Node],
        input_names_list: list[str],
    ) -> list[int]:
        """Extract mutated input indices from custom op with mutable arguments.

        Custom ops can specify mutable arguments via schema annotations like
        Tensor(a!) which sets alias_info.is_write=True.

        Args:
            node: The custom op call node
            alias_chain: Current alias chain for tracing
            input_nodes: Set of input placeholder nodes
            input_nodes_list: Ordered list of input placeholder nodes
            input_names_list: Ordered list of input names for logging

        Returns:
            List of input placeholder indices that are mutated by this op
        """
        mutated = []
        target = node.target
        if not hasattr(target, "_schema"):
            return mutated

        schema = target._schema

        # Get the args from the node
        args = node.args
        kwargs = node.kwargs

        # Find arguments with is_write=True
        for i, schema_arg in enumerate(schema.arguments):
            if schema_arg.alias_info and schema_arg.alias_info.is_write:
                # Get the actual tensor node for this argument
                arg_name = schema_arg.name
                tensor_node = args[i] if i < len(args) else kwargs.get(arg_name)

                if tensor_node is not None and isinstance(tensor_node, Node):
                    # Trace back to root placeholder
                    root_input = self._find_root_input(tensor_node, alias_chain, input_nodes)
                    if root_input is not None and root_input in input_nodes:
                        placeholder_idx = input_nodes_list.index(root_input)
                        if placeholder_idx not in mutated:
                            mutated.append(placeholder_idx)
                            logger.debug(
                                f"Custom op mutation detected: input[{placeholder_idx}] "
                                f"({input_names_list[placeholder_idx]}) via "
                                f"schema arg '{arg_name}' (is_write=True)"
                            )

        return mutated

    def _extract_placeholder_name(self, node: Node) -> str | None:
        """
        Convert Dynamo's mangled placeholder target names to human-readable names.

        Handles these Dynamo naming patterns:
        - Regular inputs:  L_<name>_           → <name>
        - Parameters:      L_self_modules_<path>_parameters_<name>_ → <path>.<name>
        - Buffers:         L_self_buffers_<name>_  → <name>

        Args:
            node: A placeholder node from a Dynamo-traced FX graph.

        Returns:
            A cleaned, human-readable name for the placeholder.
        """
        target = node.target

        if not isinstance(target, str):
            return node.name

        if "parameters" in target:
            name = target.replace("L_self_modules_", "").replace("_parameters_", ".")
            return name.rstrip("_")

        if "buffers" in target:
            return target.replace("L_self_buffers_", "").rstrip("_")

        if target.startswith("L_") and "self" not in target:
            return target[2:].rstrip("_")

        return target.rstrip("_") if target else node.name

    def _get_mutated_tensor(self, node: Node) -> Node | None:
        """Get the tensor being mutated by an in-place operation.

        For setitem and in-place method calls, returns the tensor
        that is being modified.

        Args:
            node (Node): An in-place mutation node.

        Returns:
            Node | None: Mutated tensor node, or None if not applicable.
        """
        if not node.args:
            return None

        if self._is_setitem_operation(node):
            return node.args[0] if isinstance(node.args[0], Node) else None

        if self._is_inplace_method(node):
            return node.args[0] if isinstance(node.args[0], Node) else None

        return None

    def _get_aliasing_source(self, node: Node) -> Node | None:
        """Get the source tensor for an aliasing operation.

        For view/reshape operations, returns the tensor being viewed.

        Args:
            node (Node): An aliasing operation node.

        Returns:
            Node | None: Source tensor node, or None if not applicable.
        """
        if not node.args:
            return None

        first_arg = node.args[0]
        return first_arg if isinstance(first_arg, Node) else None

    def _find_root_input(
        self,
        node: Node,
        alias_chain: dict[Node, Node],
        input_nodes: set[Node],
    ) -> Node | None:
        """Trace a node back to its root input placeholder.

        Follows the alias chain and aliasing operations to find the
        original input that a node derives from.

        Args:
            node (Node): Starting node to trace from.
            alias_chain (dict[Node, Node]): Mapping of nodes to their alias sources.
            input_nodes (set[Node]): Set of input placeholder nodes.

        Returns:
            Node | None: Root input node if found, None otherwise.
        """
        if node in input_nodes:
            return node

        current = node
        visited = set()

        while current is not None and current not in visited:
            visited.add(current)

            if current in input_nodes:
                return current

            if current in alias_chain:
                current = alias_chain[current]
                continue

            if self.is_aliasing_op(current):
                current = self._get_aliasing_source(current)
            else:
                return None

        return None

    # =========================================================================
    # Post-AOT Output Order Transformation
    # =========================================================================

    def _compute_post_aot_aliasing(
        self,
        mutated_input_indices: list[int],
        explicit_output_to_input: dict[int, int],
        explicit_output_count: int,
        input_names_list: list[str],
    ) -> AliasingInfo:
        """
        For models code:
            def forward(self, x, y):
                output = y.sum() + x.sum()
                x[8, 9] = 10.23
                return output, x

        the FxGraph is:

            def forward(self, L_y_ : torch.Tensor, L_x_ : torch.Tensor):
                l_y_ = L_y_
                l_x_ = L_x_
                sum_1 = l_y_.sum();  l_y_ = None
                sum_2 = l_x_.sum()
                output = sum_1 + sum_2;  sum_1 = sum_2 = None
                l_x_[(8, 9)] = 10.23;  setitem = l_x_;  l_x_ = setitem = None
                return (output,)

        Here, the return-value 'x' is droped in the input IR that we see.
        These reappear in the functionalized IR, but is put-in first!

            def forward(self, arg0_1, arg1_1):
                sum_1 = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
                sum_2 = torch.ops.aten.sum.default(arg1_1)
                add = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
                _tensor_constant0 = self._tensor_constant0
                lift_fresh_copy =
                    torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);
                select = torch.ops.aten.select.int(arg1_1, 0, 8)
                select_1 =
                    torch.ops.aten.select.int(select, 0, 9);  select = None
                copy = torch.ops.aten.copy.default(select_1, lift_fresh_copy);
                select_2 = torch.ops.aten.select.int(arg1_1, 0, 8)
                select_scatter =
                    torch.ops.aten.select_scatter.default(select_2, copy, 0, 9);
                select_scatter_1 =
                    torch.ops.aten.select_scatter.default(arg1_1, select_scatter, 0, 8);
                return (select_scatter_1, add)

        Transform pre-AOT analysis to match AOTAutograd's output convention.

        AOTAutograd output order:
            [mutated_inputs_sorted_by_input_index..., non_aliased_explicit_outputs...]

        Reference:
        https://github.com/pytorch/pytorch/blob/release/2.8/torch/_functorch/_aot_autograd/runtime_wrappers.py#L2069

        Explicit outputs that alias mutated inputs are deduplicated - they become
        part of the mutated input outputs, not separate outputs.

        Args:
            mutated_input_indices: Input indices that are mutated (any order)
            explicit_output_to_input: Mapping from explicit output index to aliased input index
            explicit_output_count: Number of explicit outputs in pre-AOT graph
            input_names_list: Input names for logging

        Returns:
            AliasingInfo with output indices matching post-AOT convention
        """
        aliasing_info = AliasingInfo()

        # Sort mutated inputs by input index (AOTAutograd's convention)
        sorted_mutated = sorted(set(mutated_input_indices))
        mutated_set = set(sorted_mutated)

        logger.debug(f"Mutated inputs (sorted by input index): {sorted_mutated}")

        # Mutated inputs become outputs 0..N-1 (in input order)
        for post_aot_idx, input_idx in enumerate(sorted_mutated):
            aliasing_info.add(
                parameter_number=input_idx, parameter_index=[], output_index=post_aot_idx
            )
            logger.debug(
                f"Post-AOT output[{post_aot_idx}] = mutated input[{input_idx}] "
                f"({input_names_list[input_idx]})"
            )

        # Process explicit outputs that are NOT aliases of mutated inputs
        # These appear after the mutated inputs in post-AOT order
        post_aot_idx = len(sorted_mutated)

        for explicit_idx in range(explicit_output_count):
            if explicit_idx in explicit_output_to_input:
                aliased_input = explicit_output_to_input[explicit_idx]

                if aliased_input in mutated_set:
                    # This explicit output aliases a mutated input - deduplicated
                    logger.debug(
                        f"Pre-AOT output[{explicit_idx}] aliases mutated input[{aliased_input}] "
                        f"- deduplicated (merged into post-AOT output)"
                    )
                    continue
                else:
                    # This explicit output aliases a NON-mutated input - preserve it
                    aliasing_info.add(
                        parameter_number=aliased_input,
                        parameter_index=[],
                        output_index=post_aot_idx,
                    )
                    logger.debug(
                        f"Post-AOT output[{post_aot_idx}] = pre-AOT output[{explicit_idx}] "
                        f"(aliases non-mutated input[{aliased_input}])"
                    )
            else:
                # Explicit output doesn't alias any input (pure computation)
                logger.debug(
                    f"Post-AOT output[{post_aot_idx}] = pre-AOT output[{explicit_idx}] "
                    f"(no alias, pure computation)"
                )

            post_aot_idx += 1

        return aliasing_info

    # =========================================================================
    # Pass-driver
    # =========================================================================

    def call(self, gm: torch.fx.GraphModule) -> PassResult:
        """Execute the aliasing analysis pass.

        Analyzes the graph and produces aliasing information that matches
        AOTAutograd's output convention.

        Args:
            gm: The GraphModule to analyze (not modified).

        Returns:
            PassResult with modified=False. Results stored in self.result.

        Raises:
            RuntimeError: If analysis fails.
        """
        try:
            alias_chain: dict[Node, Node] = {}
            input_placeholders: OrderedDict[str, Node] = OrderedDict()
            mutated_input_indices: list[int] = []
            explicit_output_to_input: dict[int, int] = {}

            # -----------------------------------------------------------------
            # Collect input placeholders in order
            # -----------------------------------------------------------------
            for node in gm.graph.nodes:
                if node.op == "placeholder":
                    input_name = self._extract_placeholder_name(node)
                    if input_name:
                        input_placeholders[input_name] = node
                        logger.debug(
                            f"Input[{len(input_placeholders)-1}]: {input_name} "
                            f"(node: {node.name})"
                        )

            input_nodes = set(input_placeholders.values())
            input_nodes_list = list(input_placeholders.values())
            input_names_list = list(input_placeholders.keys())

            logger.debug(f"Total inputs: {len(input_placeholders)}")

            # -----------------------------------------------------------------
            # Analyze mutations and build alias chain
            # -----------------------------------------------------------------
            for node in gm.graph.nodes:
                if self.is_mutating_op(node):
                    mutated_tensor = self._get_mutated_tensor(node)
                    if mutated_tensor is None:
                        continue

                    root_input = self._find_root_input(mutated_tensor, alias_chain, input_nodes)

                    if root_input is not None and root_input in input_nodes:
                        input_index = input_nodes_list.index(root_input)
                        if input_index not in mutated_input_indices:
                            mutated_input_indices.append(input_index)
                            logger.debug(
                                f"Mutation detected: input[{input_index}] "
                                f"({input_names_list[input_index]}) via {node.target}"
                            )

                elif self.is_aliasing_op(node):
                    source_node = self._get_aliasing_source(node)
                    if source_node is not None:
                        alias_chain[node] = source_node
                        logger.debug(f"Alias: {node.name} -> {source_node.name}")

                # Handle custom ops with mutable arguments (pre-AOT - NKI)
                elif self._is_custom_op_with_mutations(node):
                    logger.debug(f"Custom op with mutations detected: {node.name}")
                    custom_op_mutated = self._get_custom_op_mutated_inputs(
                        node,
                        alias_chain,
                        input_nodes,
                        input_nodes_list,
                        input_names_list,
                    )
                    logger.debug(f"Custom op mutated inputs: {custom_op_mutated}")
                    for idx in custom_op_mutated:
                        if idx not in mutated_input_indices:
                            mutated_input_indices.append(idx)

            # -----------------------------------------------------------------
            # Find output node and analyze explicit outputs
            # -----------------------------------------------------------------
            output_node = next((n for n in gm.graph.nodes if n.op == "output"), None)
            if output_node is None:
                raise RuntimeError('The graph has no "output"-node and hence is malformed.')

            current_output = output_node.args[0]
            if isinstance(current_output, (tuple | list)):
                output_list = list(current_output)
            else:
                output_list = [current_output]

            explicit_output_count = len(output_list)
            logger.debug(f"Explicit outputs: {explicit_output_count}")

            # -----------------------------------------------------------------
            # Map explicit outputs to their aliased inputs
            # -----------------------------------------------------------------
            for output_index, output_item in enumerate(output_list):
                if not isinstance(output_item, Node):
                    continue

                # Direct pass-through
                if output_item in input_nodes:
                    input_index = input_nodes_list.index(output_item)
                    explicit_output_to_input[output_index] = input_index
                    logger.debug(
                        f"Pre-AOT output[{output_index}] is pass-through of "
                        f"input[{input_index}] ({input_names_list[input_index]})"
                    )
                    continue

                # Traces back to input through alias chain
                root = self._find_root_input(output_item, alias_chain, input_nodes)
                if root is not None and root in input_nodes:
                    input_index = input_nodes_list.index(root)
                    explicit_output_to_input[output_index] = input_index
                    logger.debug(
                        f"Pre-AOT output[{output_index}] aliases "
                        f"input[{input_index}] ({input_names_list[input_index]})"
                    )

            # -----------------------------------------------------------------
            # Transform to post-AOT output order
            # -----------------------------------------------------------------
            self.result = self._compute_post_aot_aliasing(
                mutated_input_indices=mutated_input_indices,
                explicit_output_to_input=explicit_output_to_input,
                explicit_output_count=explicit_output_count,
                input_names_list=input_names_list,
            )

            logger.info(f"AliasingInfo (post-AOT order) = {self.result}")
            logger.info(
                f"Mutated inputs: {sorted(mutated_input_indices)}, "
                f"Explicit outputs: {explicit_output_count}"
            )

            return PassResult(gm, modified=False)

        except Exception as e:
            raise RuntimeError(f"Aliasing analysis failed: {e!s}") from e

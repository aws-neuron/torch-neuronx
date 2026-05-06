"""Dynamic shape analysis pass for torch-neuronx.

Detects dynamic shapes in FX graphs and raises errors with detailed diagnostics,
as torch-neuronx requires static shapes for compilation.
"""

import logging

import torch
from torch.fx.passes.infra.pass_base import PassBase, PassResult

logger = logging.getLogger(__name__)


class DynamicShapeAnalysis(PassBase):
    def __init__(self):
        super().__init__()

    def _dynamic_shape_checker(self, gm: torch.fx.GraphModule) -> tuple[bool, str]:
        """Check for dynamic shapes in the FX graph and generate detailed error information.

        This method analyzes the GraphModule to detect dynamic shapes by examining the
        ShapeEnv (shape environment) which tracks symbolic dimensions. Since torch-neuronx
        requires static shapes for compilation, any dynamic shapes must be reported as errors.

        Args:
            gm (torch.fx.GraphModule): The FX graph module to analyze for dynamic shapes.
                Should contain shape environment information either directly or in node metadata.

        Returns:
            tuple[bool, str]: A tuple containing:
                - bool: True if dynamic shapes are detected, False otherwise
                - str: Detailed error message with diagnostic information if dynamic shapes
                  are found, empty string if no dynamic shapes detected. The error message
        """

        def get_shape_env(gm: torch.fx.GraphModule):
            """Extract ShapeEnv from GraphModule if it exists."""
            # Check common locations for shape_env
            if hasattr(gm, "_shape_env"):
                return gm._shape_env

            # Check in graph metadata
            if hasattr(gm.graph, "_shape_env"):
                return gm.graph._shape_env

            # Try to find from FakeTensors in node metadata
            for node in gm.graph.nodes:
                if "val" in node.meta:
                    val = node.meta["val"]
                    if (
                        hasattr(val, "fake_mode")
                        and val.fake_mode is not None
                        and hasattr(val.fake_mode, "shape_env")
                    ):
                        return val.fake_mode.shape_env
            return None

        shape_env = get_shape_env(gm)
        if shape_env is None:
            return False, ""

        # Collect detailed information about dynamic shapes
        dynamic_vars = list(shape_env.var_to_val.keys() | shape_env.var_to_range.keys())
        if len(dynamic_vars) == 0:
            return False, ""

        # Find source information from stack traces
        source_locations = []
        for node in gm.graph.nodes:
            if node.meta.get("stack_trace"):
                source_locations.append(node.meta["stack_trace"])

        # Build comprehensive error message
        error_parts = [
            "\n\nDynamic shapes detected in the model, but torch-neuronx requires static shapes. ",
            f"Found {len(dynamic_vars)} dynamic dimension(s): "
            f"{dynamic_vars[:3]}{'...' if len(dynamic_vars) > 3 else ''}",
        ]

        if source_locations:
            error_parts.append(f"\nSource location: {source_locations[0]}")

        error_parts.extend(
            [
                "\nTo fix this issue:",
                "\n1. Ensure all input tensors have fixed, known shapes",
                "\n2. Avoid operations that create dynamic shapes "
                "(e.g., dynamic slicing, variable-length sequences)",
                "\nFor more information, see: https://pytorch.org/docs/stable/export.html#dynamic-shapes",
            ]
        )

        return True, "".join(error_parts)

    def call(self, gm: torch.fx.GraphModule) -> PassResult:
        """Execute the dynamic shape analysis pass on the given GraphModule.

        This is the main entry point for the pass that performs dynamic shape detection
        and validation. The pass ensures that the FX graph contains only static shapes,
        which is a requirement for torch-neuronx compilation.

        Args:
            gm (torch.fx.GraphModule): The FX graph module to analyze. This should be
                a valid GraphModule that may contain shape environment information.

        Returns:
            PassResult: A PassResult object indicating the outcome of the pass:

        Raises:
            RuntimeError: If dynamic shapes are detected in the graph. The error message
                contains detailed diagnostic information
        """
        # Check dynamic shape here for error message consistency
        logger.debug("Running DynamicShapeAnalysis pass...")
        has_dynamic, error_msg = self._dynamic_shape_checker(gm)
        if has_dynamic:
            raise RuntimeError(error_msg)
        return PassResult(gm, modified=False)

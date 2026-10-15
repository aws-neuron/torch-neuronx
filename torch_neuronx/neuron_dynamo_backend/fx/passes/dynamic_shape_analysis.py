"""Dynamic shape analysis pass for torch-neuronx.

Detects dynamic shapes in FX graphs and raises errors with detailed diagnostics,
as torch-neuronx requires static shapes for compilation.
"""

import logging

import torch
from torch.fx.passes.infra.pass_base import PassResult

from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import (
    NeuronFXPass,
)

logger = logging.getLogger(__name__)


class DynamicShapeAnalysis(NeuronFXPass):
    """Reject FX graphs that contain symbolic dynamic dimensions.

    Args:
        state (NeuronFXPassState, optional): Shared state used for pass artifact dumping.
            Defaults to ``None``.
    """

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
            """Extract the shape environment from an FX graph module.

            Args:
                gm (torch.fx.GraphModule): Graph module to inspect.

            Returns:
                object: Shape environment found on the graph or its fake tensors, or ``None``.
            """
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
        """Reject dynamic dimensions in an FX graph module.

        Args:
            gm (torch.fx.GraphModule): FX graph module to analyze.

        Returns:
            PassResult: Unmodified graph when no dynamic dimensions are found.

        Raises:
            RuntimeError: If the graph contains dynamic dimensions, with detailed
                diagnostics and remediation guidance.
        """
        # Check dynamic shape here for error message consistency
        logger.debug("Running DynamicShapeAnalysis pass...")
        has_dynamic, error_msg = self._dynamic_shape_checker(gm)
        if has_dynamic:
            raise RuntimeError(error_msg)
        return PassResult(gm, modified=False)

"""
FX graph transformation passes for the Neuron Dynamo backend.

This package contains individual passes that transform FX graphs:

- aliasing_analysis: Detect input-output aliasing relationships
- collective_legalization: Rewrite collective ops for StableHLO compatibility
- dtype_conversion: Convert f64 types to f32 for Neuron hardware
- flex_attention_legalization: Decompose flex_attention higher-order ops
- functionalize_copy_inplace_result: Handle in-place mutation patterns
- remove_none_outputs: Filter None values from graph outputs
"""

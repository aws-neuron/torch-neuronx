"""
FX graph transformation utilities for the Neuron Dynamo backend.

This package provides utilities for transforming PyTorch FX graphs during the
torch.compile pipeline, including:

- fx_transform: FX to StableHLO conversion
- fx_hooks: Custom FX importer hooks for MLIR translation
- passes/: Individual graph transformation passes
- pipelines/: Composed pass pipelines for pre/post AOTAutograd processing
"""

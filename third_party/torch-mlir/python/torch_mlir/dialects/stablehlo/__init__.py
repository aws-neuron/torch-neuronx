"""StableHLO dialect for Torch-MLIR."""

# Import the generated ops
try:
    from ._stablehlo_ops_gen import *
    from ._chlo_ops_gen import *
    from ._vhlo_ops_gen import *
    from ._check_ops_gen import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import StableHLO ops: {e}")

# Import the C++ extension module
try:
    from torch_mlir._mlir_libs import _stablehlo

    # Re-export all functions from the C++ module
    for attr_name in dir(_stablehlo):
        if not attr_name.startswith('_'):
            globals()[attr_name] = getattr(_stablehlo, attr_name)

except ImportError as e:
    import warnings
    warnings.warn(f"Could not import StableHLO C++ extension: {e}")

# Helper function to ensure StableHLO dialect is registered
def ensure_stablehlo_registered(context):
    """Ensure StableHLO dialect is registered with the given MLIR context."""
    try:
        # First try to load all available dialects
        context.load_all_available_dialects()

        # Check if stablehlo is registered
        if hasattr(context, 'dialects') and 'stablehlo' in context.dialects:
            return True

        return False
    except Exception as e:
        import warnings
        warnings.warn(f"Could not register StableHLO dialect: {e}")
        return False

"""Refactored JAX operations package with clean architecture."""

from .kernel import JaxKernel
from .op_impl import JaxOpImpl

# Re-export the main classes for backward compatibility
__all__ = ["JaxKernel", "JaxOpImpl"]

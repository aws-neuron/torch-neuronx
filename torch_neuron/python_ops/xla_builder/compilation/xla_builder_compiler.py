"""HLO function to HLO compilation utilities."""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class XLABuilderCompiler:
    """Handles compilation of functions written using HLO proto to HLO."""

    def __init__(
        self,
        static_argnums: tuple[int, ...] | None = None,
        static_argnames: tuple[str, ...] | None = None,
    ):
        """Initialize the compiler.

        Args:
            static_argnums: Indices of static arguments
            static_argnames: Names of static keyword arguments
        """
        self.static_argnums = static_argnums or ()
        self.static_argnames = static_argnames or ()
        # self.type_converter = JaxTypeConverter()
        self.hlo_cache = {}

    def compile_to_hlo(
        self, hlo_fn: Callable, sample_inputs: tuple[Any, ...], kwargs: dict[str, Any] | None = None
    ):
        """Compile HLO function to HLO module.

        Args:
            hlo_fn: HLO function to compile
            sample_inputs: Sample inputs for shape/dtype inference
            kwargs: Optional keyword arguments

        Returns:
            HLO module object
        """
        kwargs = kwargs or {}
        hlo = hlo_fn(sample_inputs, **kwargs)
        return hlo

    def get_or_compile_hlo(
        self,
        cache_key: str,
        hlo_fn: Callable,
        sample_inputs: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> bytes:
        """Get HLO from cache or compile and cache it.

        Args:
            cache_key: Unique cache key for this compilation
            hlo_fn: HLO function to compile
            sample_inputs: Sample inputs for shape/dtype inference
            kwargs: Optional keyword arguments

        Returns:
            HLO module object
        """
        kwargs = kwargs or {}
        if cache_key in self.hlo_cache:
            return self.hlo_cache[cache_key]
        hlo = hlo_fn(sample_inputs, **kwargs)
        self.hlo_cache[cache_key] = hlo
        return hlo

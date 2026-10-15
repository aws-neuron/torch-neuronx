"""Base kernel abstraction for unified NKI and XLA execution."""


class BaseNeuronKernel:
    """Provide shared behavior for NKI and XLA kernel implementations.

    Note:
        Normalize static argument indices for all kernels. Subclasses that support
        lowering through Torch-MLIR, NKI, or HLO scribe must override ``lower`` to
        produce a ``LoweringResult`` for the C++ Neuron dispatcher on a cache miss.
    """

    def __init__(self):
        self.static_argnums = ()

    def lower(self, *args, **kwargs):
        """Require subclasses to implement StableHLO lowering.

        Args:
            *args: Positional arguments accepted by the kernel call.
            **kwargs: Keyword arguments accepted by the kernel call.

        Raises:
            NotImplementedError: If the subclass does not implement lowering.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement lower(). "
            f"Subclasses that support compilation must override this method "
            f"to return a LoweringResult."
        )

    def _normalize_static_argnums(self, num_inputs: int) -> tuple[int, ...]:
        """Normalize static argument indices (convert negative to positive)."""
        static_argnums = self.static_argnums or ()
        return tuple(idx + num_inputs if idx < 0 else idx for idx in static_argnums)

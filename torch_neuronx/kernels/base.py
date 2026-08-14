"""Base kernel abstraction for unified NKI and XLA execution."""


class BaseNeuronKernel:
    """Base class for all Neuron kernels (NKI and XLA).

    This provides the static-argnum normalization shared between NKI and XLA
    kernel implementations.

    Subclasses that support lowering (torch-mlir, NKI, HLO scribe) must
    override the ``lower()`` method to produce a ``LoweringResult``.
    The C++ NeuronDispatcher calls ``kernel.lower()`` on cache miss.
    """

    def __init__(self):
        self.static_argnums = ()

    def lower(self, *args, **kwargs):
        """Lower the operation to StableHLO without executing.

        Called by the C++ NeuronDispatcher on cache miss. Subclasses that
        support compilation-based execution (TorchMlirKernel, NkiKernel,
        XLABuilderKernel) must override this method.

        Args:
            *args: Same positional arguments as __call__
            **kwargs: Same keyword arguments as __call__

        Returns:
            LoweringResult with StableHLO bytes, cache key, and metadata

        Raises:
            NotImplementedError: If the kernel subclass does not support lowering
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

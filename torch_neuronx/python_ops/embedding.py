from .base import Operation
from .xla_ops.embedding_xla import EmbeddingDenseBackwardXLAImpl


class EmbeddingDenseBackwardOp(Operation):
    """Embedding dense backward operation"""

    def _setup_implementations(self):
        """Setup available implementations for the embedding_dense_backward operation"""
        self._implementations.append(EmbeddingDenseBackwardXLAImpl())

    @property
    def op_name(self) -> str:
        """Return the operation name for caching and debugging"""
        return "embedding_dense_backward"

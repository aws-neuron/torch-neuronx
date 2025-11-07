from .base import Operation
from .xla_ops.native_multi_head_attn_prefix_xla import NativeMultiHeadAttnPrefixXLAImpl


class NativeMultiHeadAttnPrefixOp(Operation):
    """Native multi-head attention prefix operation for transforming Q, K, V tensors"""

    def _setup_implementations(self):
        """Setup available implementations"""
        self._implementations.append(NativeMultiHeadAttnPrefixXLAImpl(self.op_name))

    @property
    def op_name(self) -> str:
        """Return the operation name"""
        return "native_multi_head_attn_prefix"

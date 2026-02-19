from .base import Operation


class NativeMultiHeadAttnPrefixOp(Operation):
    """Native multi-head attention prefix operation for transforming Q, K, V tensors"""

    def _setup_implementations(self):
        """Setup available implementations"""
        from .torch_mlir.ops.native_multi_head_attn_prefix import (
            NativeMultiHeadAttnPrefixMLIRImpl,
        )

        self._implementations.append(NativeMultiHeadAttnPrefixMLIRImpl(self.op_name))

    @property
    def op_name(self) -> str:
        """Return the operation name"""
        return "native_multi_head_attn_prefix"

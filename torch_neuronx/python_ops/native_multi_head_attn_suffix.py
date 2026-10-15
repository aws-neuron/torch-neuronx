from .base import Operation


class NativeMultiHeadAttnSuffixOp(Operation):
    """Native multi-head attention suffix operation for output projection"""

    def _setup_implementations(self):
        """Setup available implementations"""
        from .torch_mlir.ops.native_multi_head_attn_suffix import (
            NativeMultiHeadAttnSuffixMLIRImpl,
        )

        self._implementations.append(NativeMultiHeadAttnSuffixMLIRImpl(self.op_name))

    @property
    def op_name(self) -> str:
        """Return the operation name"""
        return "native_multi_head_attn_suffix"

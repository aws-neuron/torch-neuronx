from .base import Operation


class NativeMultiHeadAttnPrefixOp(Operation):
    """Provide QKV projection and head-layout transformation for native attention."""

    def _setup_implementations(self):
        """Setup available implementations"""
        from .torch_mlir.ops.native_multi_head_attn_prefix import (
            NativeMultiHeadAttnPrefixMLIRImpl,
        )

        self._implementations.append(NativeMultiHeadAttnPrefixMLIRImpl(self.op_name))

    @property
    def op_name(self) -> str:
        """Return the registered operation name.

        Returns:
            str: ``"native_multi_head_attn_prefix"``.
        """
        return "native_multi_head_attn_prefix"

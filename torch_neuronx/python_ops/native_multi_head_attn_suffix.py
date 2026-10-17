from .base import Operation


class NativeMultiHeadAttnSuffixOp(Operation):
    """Provide head-layout merging and output projection for native attention."""

    def _setup_implementations(self):
        """Setup available implementations"""
        from .torch_mlir.ops.native_multi_head_attn_suffix import (
            NativeMultiHeadAttnSuffixMLIRImpl,
        )

        self._implementations.append(NativeMultiHeadAttnSuffixMLIRImpl(self.op_name))

    @property
    def op_name(self) -> str:
        """Return the registered operation name.

        Returns:
            str: ``"native_multi_head_attn_suffix"``.
        """
        return "native_multi_head_attn_suffix"

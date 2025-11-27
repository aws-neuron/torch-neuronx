from .base import Operation
from .xla_ops.native_multi_head_attn_suffix_xla import NativeMultiHeadAttnSuffixXLAImpl


class NativeMultiHeadAttnSuffixOp(Operation):
    """Native multi-head attention suffix operation for output projection"""

    def _setup_implementations(self):
        """Setup available implementations"""
        self._implementations.append(NativeMultiHeadAttnSuffixXLAImpl(self.op_name))

    @property
    def op_name(self) -> str:
        """Return the operation name"""
        return "native_multi_head_attn_suffix"

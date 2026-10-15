from typing import Any, Optional

import torch
from torch.utils._ordered_set import OrderedSet


class NKIContext:
    """Track generated names, buffers, and statements for NKI lowering."""

    nameset: OrderedSet[str]
    buffers: dict[str, "NKIArray"]
    code_body: list["NISAOpValue"]
    prefix: str = "{{prefix}}"

    def __init__(self):
        self.nameset = OrderedSet()
        self.buffers = {}
        self.code_body = []
        self.prefix = "{{prefix}}"

    def create_name(self):
        """Create and reserve the next context-prefixed buffer name.

        Returns:
            str: Newly reserved name.
        """
        tmp_idx = len(self.nameset)
        name = f"{self.prefix}_{tmp_idx}"
        while name in self.nameset:
            tmp_idx += 1
            name = f"{self.prefix}_{tmp_idx}"
        self.nameset.add(name)
        return name

    def print_readable(self):
        """Print generated buffer allocations and statements."""
        print("# Buffer allocations:")
        for bufname in self.buffers:
            print("\t" + self.buffers[bufname].codegen())
        print("# Code body:")
        for stmt in self.code_body:
            print("\t" + stmt.codegen())


class NISAOpValue:
    """Represent a symbolic NISA operation and its optional destination array.

    Args:
        ctx (NKIContext): Context that owns generated names and statements.
        op (str): NKI or NISA operation name.
        meta_tensor (torch.Tensor, optional): Destination metadata. Defaults to ``None``.
        zero (bool, optional): Emit a zero-initialized destination. Defaults to ``False``.
    """

    dst_arr: Optional["NKIArray"]
    name: str | None  # Name of the dst buffer or immediate value
    args: list["NISAOpValue"]
    kwargs: dict[str, "NISAOpValue"]
    op: str
    ctx: NKIContext

    def __init__(self, ctx, op, meta_tensor=None, zero=False):
        self.operands = []
        self.dst_arr = None
        self.ctx = ctx
        if meta_tensor is not None:
            self.allocate_buffer(meta_tensor, zero=zero)
        self.args = []
        self.kwargs = {}
        self.op = op

    def allocate_buffer(self, meta_tensor, zero=False):
        """Create and register a symbolic destination array from tensor metadata.

        Args:
            meta_tensor (torch.Tensor): Tensor supplying destination shape and dtype.
            zero (bool, optional): Emit zero initialization. Defaults to ``False``.
        """
        self.name = self.ctx.create_name()
        self.dst_arr = NKIArray.from_meta(self.ctx, meta_tensor, self.name, zero=zero)
        # zeroing must run inside the code body
        if zero:
            self.ctx.code_body.append(self.dst_arr)
        else:
            self.ctx.buffers[self.name] = self.dst_arr

    def __str__(self):
        assert isinstance(self.name, str), f"Variable of {self.op} is not defined!"
        return self.name

    def codegen(self):
        """Generate source code for the symbolic operation.

        Returns:
            str: Custom source when provided; otherwise, the operation call source.
        """
        if hasattr(self, "_custom_codegen"):
            return self._custom_codegen
        arg_string = ", ".join([f"{arg}" for arg in self.args])
        if self.kwargs:
            kwarg_string = ", ".join([f"{k}={v.name}" for k, v in self.kwargs.items()])
            arg_string += ", " + kwarg_string if arg_string != "" else kwarg_string
        return f"{self.op}({arg_string})"

    # Symbolic call that returns self
    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self


class NKIArray(NISAOpValue):
    """Represent an NKI array allocation and its metadata.

    Args:
        ctx (NKIContext): Context associated with the array.
        name (str): Generated or template array name.
        dtype (torch.dtype): Array element dtype.
        shape (Sequence[int]): Array dimensions.
        target_buffer (str, optional): Allocation buffer. Defaults to ``"nl.sbuf"``.
        zero (bool, optional): Generate ``nl.zeros``. Defaults to ``False``.
    """

    dtype: torch.dtype
    shape: tuple[int, ...]
    target_buffer: str
    meta_tensor: torch.Tensor

    def __init__(self, ctx, name, dtype, shape, target_buffer="nl.sbuf", zero=False):
        nl_op = "nl.zeros" if zero else "nl.ndarray"
        super().__init__(ctx, nl_op)
        self.name = name
        self.dtype = dtype
        self.shape = tuple(shape)
        self.target_buffer = target_buffer
        self.meta_tensor = torch.empty(shape, dtype=dtype, device="meta")
        self.dst_arr = self

    @classmethod
    def from_meta(cls, ctx, tensor, name, zero=False):
        """Create an NKI array from tensor metadata.

        Args:
            ctx (NKIContext): Context associated with the array.
            tensor (torch.Tensor): Tensor supplying shape and dtype.
            name (str): Generated or template array name.
            zero (bool, optional): Generate a zero-initialized array. Defaults to ``False``.

        Returns:
            NKIArray: Array with the tensor's shape and dtype.
        """
        return cls(ctx, name, tensor.dtype, tensor.shape, zero=zero)

    def codegen(self):
        """Generate source code for the NKI array allocation.

        Returns:
            str: NKI allocation source.
        """
        nl_dtype = str(self.dtype).replace("torch.", "nl.")
        if self.dtype == torch.bool:
            nl_dtype = "nl.bool_"
        return f"{self.name} = {self.op}({self.shape}, {nl_dtype}, {self.target_buffer})"


# Immediate constant value
class NISAConstant(NISAOpValue):
    """Represent an immediate constant in generated NISA code.

    Args:
        ctx (NKIContext): Context associated with the constant.
        value (Any): Immediate constant value.
    """

    value: Any

    def __init__(self, ctx, value):
        super().__init__(ctx, "constant")
        self.value = value
        self.name = str(value)

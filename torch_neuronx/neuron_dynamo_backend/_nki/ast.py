from typing import Any, Optional

import torch
from torch.utils._ordered_set import OrderedSet


class NKIContext:
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
        tmp_idx = len(self.nameset)
        name = f"{self.prefix}_{tmp_idx}"
        while name in self.nameset:
            tmp_idx += 1
            name = f"{self.prefix}_{tmp_idx}"
        self.nameset.add(name)
        return name

    def print_readable(self):
        print("# Buffer allocations:")
        for bufname in self.buffers:
            print("\t" + self.buffers[bufname].codegen())
        print("# Code body:")
        for stmt in self.code_body:
            print("\t" + stmt.codegen())


class NISAOpValue:
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
        return cls(ctx, name, tensor.dtype, tensor.shape, zero=zero)

    def codegen(self):
        nl_dtype = str(self.dtype).replace("torch.", "nl.")
        if self.dtype == torch.bool:
            nl_dtype = "nl.bool_"
        return f"{self.name} = {self.op}({self.shape}, {nl_dtype}, {self.target_buffer})"


# Immediate constant value
class NISAConstant(NISAOpValue):
    value: Any

    def __init__(self, ctx, value):
        super().__init__(ctx, "constant")
        self.value = value
        self.name = str(value)

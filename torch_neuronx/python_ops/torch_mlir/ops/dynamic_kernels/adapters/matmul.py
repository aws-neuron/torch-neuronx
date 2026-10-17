"""mm dynamic NKI adapter  (lhs[M,K] @ rhs[K,N] -> [M,N]; M dynamic, K/N baked).

bf16-only (the dynamic-M matmul kernel is bf16-specialized). compile_key=(K,N).
"""

import torch

from torch_neuronx.utils import get_logical_neuron_cores

from ..adapter_utils import (
    _MM_BLOCK_N_CANDIDATES,
    _MM_RESIDENT_BUDGET,
    _TILE_M,
    _ceil_to_tile,
    _compile_or_load_neff,
    _num_tiles_tensor,
    _pad_rows_to,
)
from ..registry import DynamicNkiEntry, register_dynamic_nki

_MM_KTILE = 128


def _can_handle_mm(mat1, mat2, out=None) -> bool:
    if not isinstance(mat1, torch.Tensor) or not isinstance(mat2, torch.Tensor):
        return False
    if mat1.device.type != "neuron" or mat2.device.type != "neuron":
        return False
    if mat1.dtype != torch.bfloat16 or mat2.dtype != torch.bfloat16:
        return False
    if mat1.dim() != 2 or mat2.dim() != 2:
        return False
    M, K = mat1.shape  # noqa: N806
    K2, N = mat2.shape  # noqa: N806
    if K != K2 or M == 0:  # any M > 0 — prepare_io pads rows up to a 128-multiple
        return False
    if K <= 0 or N <= 0:  # K any positive size (ragged final k-tile handled in-kernel)
        return False
    # N is LNC-sharded into num_programs equal column blocks, so it must divide evenly.
    # N not divisible (e.g. odd N under LNC=2) is declined here -> correct native fallback.
    lnc = int(get_logical_neuron_cores())
    if N % lnc != 0:
        return False
    # decline only if even the narrowest block over-subscribes SBUF (else silent corruption)
    shard_n = N // lnc
    backstop_width = min(_MM_BLOCK_N_CANDIDATES[-1], shard_n)
    num_tile_k = -(-K // _MM_KTILE)  # ceil-div: ragged final k-tile counts as a full tile
    if num_tile_k * backstop_width * 2 > _MM_RESIDENT_BUDGET:  # noqa: SIM103
        return False
    return True


def _mm_compile_key(mat1, mat2, out=None, **_):
    """compile_key = (K, N), both baked into the NEFF."""
    return (int(mat1.shape[1]), int(mat2.shape[1]))


def _compile_mm(compile_key) -> tuple[bytes, str]:
    """Compile matmul at [128, K] @ [K, N] (K, N baked; M dynamic)."""
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    from ..wrappers import dynamic_matmul_nki

    K, N = compile_key  # noqa: N806
    lhs = torch.empty(_TILE_M, K, dtype=torch.bfloat16, device="meta")
    rhs = torch.empty(K, N, dtype=torch.bfloat16, device="meta")
    nm = torch.tensor([[1]], dtype=torch.int32, device="meta")
    out = torch.empty(_TILE_M, N, dtype=torch.bfloat16, device="meta")

    kernel = TorchMlirKernel(torch_fn=dynamic_matmul_nki, op_name="dynamic_nki_matmul")
    return _compile_or_load_neff(kernel, "dynamic_nki_matmul", (lhs, rhs, nm), out)


def _prepare_mm_io(mat1, mat2, dest=None, **_) -> tuple[tuple, tuple, int]:
    """Pad lhs rows to a 128-multiple (M_ceil); zero rows give zero output rows,
    sliced off in reshape_output. K, N baked."""
    M, _K = mat1.shape  # noqa: N806
    N = mat2.shape[1]  # noqa: N806
    M_ceil = _ceil_to_tile(M)  # noqa: N806
    device = mat1.device
    lhs = _pad_rows_to(mat1.contiguous(), M_ceil)
    rhs = mat2.contiguous()
    nm = _num_tiles_tensor(M_ceil // _TILE_M, device)
    inputs = (lhs, rhs, nm)
    outputs = (torch.empty((M_ceil, N), dtype=mat1.dtype, device=device),)
    return inputs, outputs, M


def _reshape_mm_output(outputs, original_args):
    """mm output is [M_ceil, N]; slice off padded rows -> [M, N]."""
    M = original_args[0].shape[0]  # noqa: N806
    N = original_args[1].shape[1]  # noqa: N806
    return outputs[0][:M].reshape(M, N)


def register():
    """Register the dynamic-row matrix multiplication adapter."""
    register_dynamic_nki(
        "mm",
        DynamicNkiEntry(
            compile_fn=_compile_mm,
            can_handle=_can_handle_mm,
            prepare_io=_prepare_mm_io,
            reshape_output=_reshape_mm_output,
            compile_key_fn=_mm_compile_key,
        ),
    )

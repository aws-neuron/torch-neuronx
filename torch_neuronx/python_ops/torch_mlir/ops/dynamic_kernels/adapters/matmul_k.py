"""Dynamic-K mm adapter: grad_weight = mat1[M,K] @ mat2[K,N], contraction K dynamic.

K (per-expert token count) is the dynamic dim; M, N are baked. bf16-only. The
K-major kernel takes x_e=[K,M], dout=[K,N] and contracts the K partition dim via a
dynamic-count loop, so one NEFF serves any 128-multiple K. mat1=[M,K] is passed as
x_e=mat1ᵀ=[K,M]. compile_key=(M,N). Registered as "mm_k"; the dispatcher routes an
aten::mm with a transposed mat1 (the grad_weight signature) here.

M and N do not need tile alignment. The kernel peels a short final tile. A decline is not
free here. This op is the MoE backward dW, so a decline becomes a `prologue` recompile.
"""

import torch

from ..adapter_utils import (
    _MM_RESIDENT_BUDGET,
    _TILE_M,
    _ceil_to_tile,
    _compile_or_load_neff,
    _pad_rows_to,
)
from ..registry import DynamicNkiEntry, register_dynamic_nki

_K_COMPILE = 128  # one K-tile; runtime K supplied via num_k_tiles


def _can_handle_mm_k(mat1, mat2, out=None) -> bool:
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
    if K != K2 or K == 0:  # K (contraction) dynamic — any K > 0, prepare_io pads it
        return False
    if M == 0 or N == 0:
        return False
    # The kernel stages the output zero-init in one [_TILE_M, N] SBUF buffer.
    if N * mat1.element_size() > _MM_RESIDENT_BUDGET:  # noqa: SIM103
        return False
    return True


def _mm_k_compile_key(mat1, mat2, out=None):
    """compile_key = (M, N), both baked into the NEFF. K is runtime-dynamic."""
    return (int(mat1.shape[0]), int(mat2.shape[1]))


def _compile_mm_k(compile_key) -> tuple[bytes, str]:
    """Compile at K_compile=128: x_e[128,M] @-contract dout[128,N] -> out[M,N]."""
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    from ..wrappers import dynamic_matmul_k_nki

    M, N = compile_key  # noqa: N806
    x_e = torch.empty(_K_COMPILE, M, dtype=torch.bfloat16, device="meta")
    dout = torch.empty(_K_COMPILE, N, dtype=torch.bfloat16, device="meta")
    num_k_tiles = torch.tensor([[1]], dtype=torch.int32, device="meta")
    out = torch.empty(M, N, dtype=torch.bfloat16, device="meta")

    kernel = TorchMlirKernel(torch_fn=dynamic_matmul_k_nki, op_name="dynamic_nki_matmul_k")
    return _compile_or_load_neff(kernel, "dynamic_nki_matmul_k", (x_e, dout, num_k_tiles), out)


def _prepare_mm_k_io(mat1, mat2, dest=None) -> tuple[tuple, tuple, int]:
    """out[M,N] = mat1[M,K] @ mat2[K,N], contracting K (dynamic).

    Kernel is K-major: x_e=[K,M], dout=[K,N]. mat1=[M,K] -> x_e = mat1.T = [K,M];
    mat2=[K,N] is already dout. K padded to K_ceil (128-multiple) with zero rows in
    both operands -> 0 contribution to the contraction, so the output needs no slicing.
    """
    M, K = mat1.shape  # noqa: N806
    N = mat2.shape[1]  # noqa: N806
    K_ceil = _ceil_to_tile(K)  # noqa: N806
    device = mat1.device
    x_e = _pad_rows_to(mat1.t().contiguous(), K_ceil)  # [K_ceil, M]
    dout = _pad_rows_to(mat2.contiguous(), K_ceil)  # [K_ceil, N]
    num_k_tiles = torch.tensor([[K_ceil // _TILE_M]], dtype=torch.int32, device=device)
    inputs = (x_e, dout, num_k_tiles)
    outputs = (torch.empty((M, N), dtype=mat1.dtype, device=device),)
    return inputs, outputs, K


def _reshape_mm_k_output(outputs, original_args):
    M = original_args[0].shape[0]  # noqa: N806
    N = original_args[1].shape[1]  # noqa: N806
    return outputs[0].reshape(M, N)


def register():
    register_dynamic_nki(
        "mm_k",
        DynamicNkiEntry(
            compile_fn=_compile_mm_k,
            can_handle=_can_handle_mm_k,
            prepare_io=_prepare_mm_k_io,
            reshape_output=_reshape_mm_k_output,
            compile_key_fn=_mm_k_compile_key,
        ),
    )

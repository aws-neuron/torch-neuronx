"""Dynamic-K mm adapter: grad_weight = mat1[M,K] @ mat2[K,N], contraction K dynamic.

K (per-expert token count) is the dynamic dim; M, N are baked. bf16-only. The
K-major kernel takes x_e=[K,M], dout=[K,N] and contracts the K partition dim via a
dynamic-count loop, so one NEFF serves any 128-multiple K. mat1=[M,K] is passed as
x_e=mat1ᵀ=[K,M]. compile_key=(M,N). Registered as "mm_k"; the dispatcher routes an
aten::mm with a transposed mat1 (the grad_weight signature) here.
"""

import torch

from ..adapter_utils import _TILE_M, _compile_or_load_neff
from ..registry import DynamicNkiEntry, register_dynamic_nki

_MM_NBLOCK = 512  # dynamic_k kernel n-tile
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
    if K != K2 or K == 0 or K % _TILE_M != 0:  # K (contraction) is the dynamic dim
        return False
    if M % _TILE_M != 0 or N % _MM_NBLOCK != 0:  # noqa: SIM103  (M, N baked)
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
    mat2=[K,N] is already dout. num_k_tiles = K // 128.
    """
    M, K = mat1.shape  # noqa: N806
    N = mat2.shape[1]  # noqa: N806
    device = mat1.device
    x_e = mat1.t().contiguous()  # [K, M]
    dout = mat2.contiguous()  # [K, N]
    num_k_tiles = torch.tensor([[K // _TILE_M]], dtype=torch.int32, device=device)
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

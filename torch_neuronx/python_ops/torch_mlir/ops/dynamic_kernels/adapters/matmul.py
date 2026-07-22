"""mm dynamic NKI adapter  (lhs[M,K] @ rhs[K,N] -> [M,N]; M dynamic, K/N baked).

bf16-only (the dynamic-M matmul kernel is bf16-specialized). compile_key=(K,N).
"""

import torch

from ..adapter_utils import _TILE_M, _compile_or_load_neff, _num_tiles_tensor
from ..registry import DynamicNkiEntry, register_dynamic_nki

# dynamic_m_matmul_kernel block sizes: k_block=8*128=1024, n_block=2*512=1024.
_MM_KBLOCK = 1024
_MM_NBLOCK = 1024


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
    if K != K2 or M == 0 or M % _TILE_M != 0:
        return False
    if K % _MM_KBLOCK != 0 or N % _MM_NBLOCK != 0:  # noqa: SIM103
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
    lhs = torch.empty(_TILE_M, K, dtype=torch.bfloat16, device="neuron")
    rhs = torch.empty(K, N, dtype=torch.bfloat16, device="neuron")
    nm = torch.tensor([[1]], dtype=torch.int32, device="neuron")
    out = torch.empty(_TILE_M, N, dtype=torch.bfloat16, device="neuron")

    kernel = TorchMlirKernel(torch_fn=dynamic_matmul_nki, op_name="dynamic_nki_matmul")
    return _compile_or_load_neff(kernel, "dynamic_nki_matmul", (lhs, rhs, nm), out)


def _prepare_mm_io(mat1, mat2, dest=None, **_) -> tuple[tuple, tuple, int]:
    """Actual-sized IO (M is a 128-multiple; no padding). K, N baked."""
    M, _K = mat1.shape  # noqa: N806
    N = mat2.shape[1]  # noqa: N806
    device = mat1.device
    lhs = mat1.contiguous()
    rhs = mat2.contiguous()
    nm = _num_tiles_tensor(M // _TILE_M, device)
    inputs = (lhs, rhs, nm)
    outputs = (torch.empty((M, N), dtype=mat1.dtype, device=device),)
    return inputs, outputs, M


def _reshape_mm_output(outputs, original_args):
    """mm output is [M, N] (differs from lhs shape, so reshape explicitly)."""
    M = original_args[0].shape[0]  # noqa: N806
    N = original_args[1].shape[1]  # noqa: N806
    return outputs[0].reshape(M, N)


def register():
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

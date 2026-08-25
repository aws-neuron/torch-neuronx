"""Dynamic block-strided gather NKI kernel.

Copies contiguous blocks at regular offsets — the "row-gap" pattern from
column-slice views (e.g., ``gate_up[:, :1024]`` where ``gate_up`` is ``[M, 2048]``).

Compiled ONCE per ``(block_size, row_stride, dtype)``; ``num_rows`` (M) is
dynamic, passed at runtime via a control tensor using ``fori_loop``.

Each iteration loads one ``[1, block_size]`` row from src at ``src_offset``,
stores to dst at ``dst_offset``, and advances both offsets by ``row_stride``
and ``block_size`` respectively. SBUF usage is bounded by ``block_size``
regardless of ``row_stride``.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from torch_neuronx.nki_hop import nki_op, wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

from .adapter_utils import _compile_or_load_neff

_LNC = int(get_logical_neuron_cores())
_SUPPORTED_LNCS = (1, 2)


def _make_block_gather_kernel(block_size: int, row_stride: int):
    """Create a block-strided gather NKI kernel for specific (block_size, row_stride).

    Returns a @nki.jit kernel function that takes (src_flat, control) and uses
    fori_loop with dynamic num_rows from control tensor.

    control[prog_id, 0:3] = [num_rows, src_offset_start, dst_offset_start]
    """
    _bs = int(block_size)
    _rs = int(row_stride)

    @nki.jit
    def kernel(src_flat, control, _block_size=_bs, _row_stride=_rs):
        prog_id = nl.program_id(0)

        # Allocate output in shared HBM (out-of-place for safety)
        max_rows = src_flat.shape[0] // _row_stride
        result = nl.ndarray((max_rows * _block_size,), dtype=src_flat.dtype, buffer=nl.shared_hbm)

        # Load control: [num_rows, src_offset, dst_offset]
        ctrl = nl.ndarray((1, 3), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ctrl, src=control[prog_id : prog_id + 1, 0:3])

        src_offset = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        dst_offset = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=src_offset, src=ctrl[0:1, 1:2])
        nisa.tensor_copy(dst=dst_offset, src=ctrl[0:1, 2:3])

        num_rows_reg = nisa.register_alloc()
        nisa.register_load(num_rows_reg, ctrl[0:1, 0:1])

        def body(_i):
            row_data = nl.ndarray((1, _block_size), dtype=src_flat.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=row_data,
                src=src_flat.ap(
                    pattern=[[_block_size, 1], [1, _block_size]],
                    offset=0,
                    scalar_offset=src_offset,
                    indirect_dim=0,
                ),
            )
            nisa.dma_copy(
                dst=result.ap(
                    pattern=[[_block_size, 1], [1, _block_size]],
                    offset=0,
                    scalar_offset=dst_offset,
                    indirect_dim=0,
                ),
                src=row_data,
            )
            nisa.tensor_scalar(src_offset, src_offset, nl.add, _row_stride)
            nisa.tensor_scalar(dst_offset, dst_offset, nl.add, _block_size)

        nl.fori_loop(0, num_rows_reg, body)
        return result

    return kernel


_BLOCK_GATHER_KERNELS: dict[tuple[int, int], object] = {}
_BLOCK_GATHER_CACHE: dict[tuple[int, int, torch.dtype], tuple[str, str]] = {}
_BLOCK_GATHER_CONTROL_CACHE: dict[tuple[int, str, int], torch.Tensor] = {}


def _get_block_gather_kernel(block_size: int, row_stride: int):
    key = (block_size, row_stride)
    kernel = _BLOCK_GATHER_KERNELS.get(key)
    if kernel is None:
        kernel = _make_block_gather_kernel(block_size, row_stride)
        _BLOCK_GATHER_KERNELS[key] = kernel
    return kernel


def _compile_block_strided_gather(block_size: int, row_stride: int, dtype: torch.dtype):
    """Compile the dynamic block-strided gather kernel once per (block_size, row_stride, dtype)."""
    key = (block_size, row_stride, dtype)
    cached = _BLOCK_GATHER_CACHE.get(key)
    if cached is not None:
        return cached

    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    graph_name = f"dynamic_block_gather_H{block_size}_S{row_stride}"
    kernel = _get_block_gather_kernel(block_size, row_stride)
    wrapped = wrap_nki(kernel)

    @nki_op(f"nki_kernels::{graph_name}", mutates_args={})
    def _block_gather_op(src: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        return wrapped[(_LNC,)](src, control)

    # Compile with a reference src of 2 rows (minimum meaningful)
    ref_rows = 2
    src_flat = torch.empty(ref_rows * row_stride, dtype=dtype, device="meta")
    control = torch.zeros((_LNC, 3), dtype=torch.int32, device="meta")
    output = torch.empty(ref_rows * block_size, dtype=dtype, device="meta")

    mlir_kernel = TorchMlirKernel(torch_fn=_block_gather_op, op_name=graph_name)
    _, cache_key = _compile_or_load_neff(mlir_kernel, graph_name, (src_flat, control), output)
    _BLOCK_GATHER_CACHE[key] = (graph_name, cache_key)
    return graph_name, cache_key


def _block_gather_control(num_rows: int, device: torch.device) -> torch.Tensor:
    """Build per-program control tensor for the dynamic block-strided gather."""
    device_index = device.index if device.index is not None else -1
    key = (num_rows, device.type, device_index)
    control = _BLOCK_GATHER_CONTROL_CACHE.get(key)
    if control is not None:
        return control

    # Single-program path (LNC=1) or broadcast to all programs (LNC>1).
    # TODO: proper LNC>1 splitting requires passing row_stride/block_size to compute
    # per-program src/dst offsets.
    control = torch.tensor([[num_rows, 0, 0]], dtype=torch.int32, device=device)
    if _LNC > 1:
        control = control.expand(_LNC, -1).contiguous()

    _BLOCK_GATHER_CONTROL_CACHE[key] = control
    return control


def try_run_dynamic_block_strided_gather(
    src_flat: torch.Tensor,
    dst_flat: torch.Tensor,
    *,
    num_rows: int,
    block_size: int,
    row_stride: int,
) -> bool:
    """Run the dynamic block-strided gather. Returns True if handled.

    The kernel is compiled once per (block_size, row_stride, dtype) and reused
    for any num_rows via fori_loop + control tensor.
    """
    if _LNC not in _SUPPORTED_LNCS:
        return False
    if num_rows < 1 or block_size < 1:
        return False

    import torch_neuronx._C as _C

    graph_name, cache_key = _compile_block_strided_gather(block_size, row_stride, src_flat.dtype)
    control = _block_gather_control(num_rows, src_flat.device)

    _C.execute_compiled_graph(
        graph_name,
        cache_key,
        [src_flat, control],
        [dst_flat],
        False,
    )
    return True

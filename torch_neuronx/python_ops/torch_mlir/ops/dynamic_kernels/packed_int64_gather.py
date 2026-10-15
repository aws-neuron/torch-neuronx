"""Dynamic packed-int64 low-word gather.

Compiles fixed-shape small and tiled kernels once, then executes them against
runtime-sized packed storage using device control tensors.
"""

from functools import lru_cache

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from torch_neuronx.nki_hop import nki_op, wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

from .adapter_utils import _compile_or_load_neff

_TILE_SIZE = 128
_LNC = int(get_logical_neuron_cores())
_SUPPORTED_LNCS = (1, 2)


@nki.jit
def packed_int64_dynamic_gather_kernel(grouped_in, control):
    """Gather packed int64 low words in fixed per-program tiles.

    Args:
        grouped_in (nl.NkiTensor): Interleaved int32 low and high words in shared HBM.
        control (nl.NkiTensor): Per-program bulk counts and packed/output offsets.

    Returns:
        nl.NkiTensor: Contiguous int32 tensor containing the gathered low words.

    Note:
        With two programs, the programs write equal disjoint ranges around the center.
        For odd lengths, an extra synchronized iteration writes the center on one program
        and rewrites the last element on the other. With one program, the normal bulk and
        tail paths gather every low word without the extra iteration.
    """
    num_progs = nl.num_programs(0)
    prog_id = nl.program_id(0)
    tile_size = _TILE_SIZE // num_progs
    result = nl.ndarray((grouped_in.shape[0] // 2,), dtype=nl.int32, buffer=nl.shared_hbm)

    control_sbuf = nl.ndarray((1, 8), dtype=nl.int32, buffer=nl.sbuf)
    packed_offset = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    output_offset = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=control_sbuf, src=control[prog_id : prog_id + 1, 0:8])
    nisa.tensor_copy(dst=packed_offset, src=control_sbuf[0:1, 1:2])
    nisa.tensor_copy(dst=output_offset, src=control_sbuf[0:1, 2:3])
    bulk_tiles_reg = nisa.register_alloc()
    nisa.register_load(bulk_tiles_reg, control_sbuf[0:1, 0:1])
    extra_count_reg = nisa.register_alloc()
    nisa.register_load(extra_count_reg, control_sbuf[0:1, 5:6])

    def bulk_body(_i):
        """Copy one packed bulk tile into the contiguous output.

        Args:
            _i (int): Unused loop iteration index supplied by ``nl.fori_loop``.
        """
        packed_tile = nl.ndarray((tile_size, 2), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=packed_tile,
            src=grouped_in.ap(
                pattern=[[2, tile_size], [1, 2]],
                offset=0,
                scalar_offset=packed_offset,
                indirect_dim=0,
            ),
        )
        low_words = nl.ndarray((tile_size, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=low_words,
            src=packed_tile[:, 0:1],
            engine=nisa.engine.vector,
        )
        nisa.dma_copy(
            dst=result.ap(
                pattern=[[1, tile_size], [1, 1]],
                offset=0,
                scalar_offset=output_offset,
                indirect_dim=0,
            ),
            src=low_words,
        )
        nisa.tensor_scalar(packed_offset, packed_offset, nl.add, 2 * tile_size)
        nisa.tensor_scalar(output_offset, output_offset, nl.add, tile_size)

    nl.fori_loop(0, bulk_tiles_reg, bulk_body)

    packed_tail = nl.ndarray((tile_size, 2), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=packed_tail,
        src=grouped_in.ap(
            pattern=[[2, tile_size], [1, 2]],
            offset=0,
            scalar_offset=control_sbuf[0:1, 3:4],
            indirect_dim=0,
        ),
    )
    low_words_tail = nl.ndarray((tile_size, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=low_words_tail,
        src=packed_tail[:, 0:1],
        engine=nisa.engine.vector,
    )
    nisa.dma_copy(
        dst=result.ap(
            pattern=[[1, tile_size], [1, 1]],
            offset=0,
            scalar_offset=control_sbuf[0:1, 4:5],
            indirect_dim=0,
        ),
        src=low_words_tail,
    )

    def extra_body(_i):
        """Run one synchronized odd-length copy.

        Args:
            _i (int): Unused loop index; one program writes the center while the other
                rewrites its final element.
        """
        extra_low_word = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=extra_low_word,
            src=grouped_in.ap(
                pattern=[[2, 1], [1, 1]],
                offset=0,
                scalar_offset=control_sbuf[0:1, 6:7],
                indirect_dim=0,
            ),
        )
        nisa.dma_copy(
            dst=result.ap(
                pattern=[[1, 1], [1, 1]],
                offset=0,
                scalar_offset=control_sbuf[0:1, 7:8],
                indirect_dim=0,
            ),
            src=extra_low_word,
        )

    nl.fori_loop(0, extra_count_reg, extra_body)
    return result


@nki.jit
def packed_int64_dynamic_gather_small_kernel(grouped_in, control):
    """Gather fewer than 128 packed int64 low words element by element.

    Args:
        grouped_in (nl.NkiTensor): Interleaved int32 low and high words in shared HBM.
        control (nl.NkiTensor): Per-program element counts and packed/output offsets.

    Returns:
        nl.NkiTensor: Contiguous int32 tensor containing the gathered low words.

    Note:
        For odd lengths, an extra synchronized iteration writes the center on one program
        and rewrites the last element on the other.
    """
    prog_id = nl.program_id(0)
    result = nl.ndarray((grouped_in.shape[0] // 2,), dtype=nl.int32, buffer=nl.shared_hbm)

    control_sbuf = nl.ndarray((1, 6), dtype=nl.int32, buffer=nl.sbuf)
    packed_offset = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    output_offset = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=control_sbuf, src=control[prog_id : prog_id + 1, 0:6])
    nisa.tensor_copy(dst=packed_offset, src=control_sbuf[0:1, 1:2])
    nisa.tensor_copy(dst=output_offset, src=control_sbuf[0:1, 2:3])
    numel_reg = nisa.register_alloc()
    nisa.register_load(numel_reg, control_sbuf[0:1, 0:1])
    extra_count_reg = nisa.register_alloc()
    nisa.register_load(extra_count_reg, control_sbuf[0:1, 3:4])

    def body(_i):
        """Copy one packed low word into the contiguous output.

        Args:
            _i (int): Unused loop iteration index supplied by ``nl.fori_loop``.
        """
        low_word = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=low_word,
            src=grouped_in.ap(
                pattern=[[2, 1], [1, 1]],
                offset=0,
                scalar_offset=packed_offset,
                indirect_dim=0,
            ),
        )
        nisa.dma_copy(
            dst=result.ap(
                pattern=[[1, 1], [1, 1]],
                offset=0,
                scalar_offset=output_offset,
                indirect_dim=0,
            ),
            src=low_word,
        )
        nisa.tensor_scalar(packed_offset, packed_offset, nl.add, 2)
        nisa.tensor_scalar(output_offset, output_offset, nl.add, 1)

    nl.fori_loop(0, numel_reg, body)

    def extra_body(_i):
        """Run one synchronized odd-length copy.

        Args:
            _i (int): Unused loop index; one program writes the center while the other
                rewrites its final element.
        """
        extra_low_word = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=extra_low_word,
            src=grouped_in.ap(
                pattern=[[2, 1], [1, 1]],
                offset=0,
                scalar_offset=control_sbuf[0:1, 4:5],
                indirect_dim=0,
            ),
        )
        nisa.dma_copy(
            dst=result.ap(
                pattern=[[1, 1], [1, 1]],
                offset=0,
                scalar_offset=control_sbuf[0:1, 5:6],
                indirect_dim=0,
            ),
            src=extra_low_word,
        )

    nl.fori_loop(0, extra_count_reg, extra_body)
    return result


_wrapped_gather = wrap_nki(packed_int64_dynamic_gather_kernel)
_wrapped_small_gather = wrap_nki(packed_int64_dynamic_gather_small_kernel)


@nki_op("nki_kernels::packed_int64_dynamic_gather", mutates_args={})
def packed_int64_dynamic_gather(grouped_in: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """Launch the tiled packed-int64 low-word gather.

    Args:
        grouped_in (torch.Tensor): Interleaved int32 low and high words.
        control (torch.Tensor): Per-program tiled gather controls.

    Returns:
        torch.Tensor: Contiguous tensor of gathered low words.
    """
    return _wrapped_gather[(_LNC,)](grouped_in, control)


@nki_op("nki_kernels::packed_int64_dynamic_gather_small", mutates_args={})
def packed_int64_dynamic_gather_small(
    grouped_in: torch.Tensor, control: torch.Tensor
) -> torch.Tensor:
    """Launch the small packed-int64 low-word gather.

    Args:
        grouped_in (torch.Tensor): Interleaved int32 low and high words.
        control (torch.Tensor): Per-program elementwise gather controls.

    Returns:
        torch.Tensor: Contiguous tensor of gathered low words.
    """
    return _wrapped_small_gather[(_LNC,)](grouped_in, control)


@lru_cache(maxsize=2)
def _compiled_dynamic_packed_gather(small: bool) -> tuple[str, str]:
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    if small:
        graph_name = "dynamic_packed_int64_gather_small"
        grouped = torch.empty(2, dtype=torch.int32, device="meta")
        control = torch.zeros((_LNC, 6), dtype=torch.int32, device="meta")
        output = torch.empty(1, dtype=torch.int32, device="meta")
        wrapper = packed_int64_dynamic_gather_small
    else:
        graph_name = "dynamic_packed_int64_gather"
        grouped = torch.empty(2 * _TILE_SIZE, dtype=torch.int32, device="meta")
        control = torch.zeros((_LNC, 8), dtype=torch.int32, device="meta")
        output = torch.empty(_TILE_SIZE, dtype=torch.int32, device="meta")
        wrapper = packed_int64_dynamic_gather
    kernel = TorchMlirKernel(torch_fn=wrapper, op_name=graph_name)
    _, cache_key = _compile_or_load_neff(kernel, graph_name, (grouped, control), output)
    return graph_name, cache_key


_CONTROL_CACHE: dict[tuple[int, str, int], torch.Tensor] = {}


def _packed_gather_control(numel: int, device: torch.device) -> torch.Tensor:
    device_index = device.index if device.index is not None else -1
    key = (numel, device.type, device_index)
    control = _CONTROL_CACHE.get(key)
    if control is None:
        if _LNC == 1:
            local_numel = numel
            odd = 0
            starts = [0]
            extra_offsets = [0]
        else:
            local_numel = numel // 2
            odd = numel % 2
            starts = [0, local_numel + odd]
            extra_offsets = [local_numel, numel - 1]
        values = []
        for start, extra_offset in zip(starts, extra_offsets, strict=True):
            if numel < _TILE_SIZE:
                values.append(
                    [
                        local_numel,
                        2 * start,
                        start,
                        odd,
                        2 * extra_offset,
                        extra_offset,
                    ]
                )
            else:
                tile_size = _TILE_SIZE // _LNC
                tail_start = start + local_numel - tile_size
                values.append(
                    [
                        (local_numel - 1) // tile_size,
                        2 * start,
                        start,
                        2 * tail_start,
                        tail_start,
                        odd,
                        2 * extra_offset,
                        extra_offset,
                    ]
                )
        control = torch.tensor(values, dtype=torch.int32, device=device)
        _CONTROL_CACHE[key] = control
    return control


def try_run_dynamic_packed_int64_gather(grouped, out, numel: int) -> bool:
    """Queue the dynamic packed-int64 gather when the LNC is supported.

    Args:
        grouped (torch.Tensor): Interleaved int32 low and high words.
        out (torch.Tensor): Contiguous destination written by the queued compiled graph.
        numel (int): Number of low words to gather.

    Returns:
        bool: ``True`` after scheduling, or ``False`` when the logical Neuron core count is
            unsupported.

    Note:
        Execution is asynchronous unless ``NEURON_LAUNCH_BLOCKING=1``.
    """
    if _LNC not in _SUPPORTED_LNCS:
        return False

    import torch_neuronx._C as _C

    small = numel < _TILE_SIZE
    graph_name, cache_key = _compiled_dynamic_packed_gather(small)
    control = _packed_gather_control(numel, out.device)
    _C.execute_compiled_graph(
        graph_name,
        cache_key,
        [grouped, control],
        [out],
        False,
    )
    return True

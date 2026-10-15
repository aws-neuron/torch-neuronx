"""MoE-combine gather-reduce: out_buffer[t,:] += sum_{k<top_k} routed[inv_idx[t,k], :].

Conflict-free replacement for the many-to-one combine scatter_add (N = T*top_k
rows summed into T token rows). Each output row is produced by exactly one tile
(gather its top_k source rows, sum in an fp32 accumulator, add into the buffer,
write once) — no shared-target read-modify-write, so duplicate targets can't race.

LNC-sharded over the token (row) dim like add_kernel: program p owns row-tiles
p, p+P, ... (disjoint rows) so the non-idempotent per-token accumulate never races
across cores. top_k and D static; T dynamic via num_t_tiles (per-program count).
"""

import nki
import nki.isa as nisa
import nki.language as nl

_TILE_M = 128


@nki.jit()
def combine_gather_reduce_kernel(routed, inv_idx, out_buffer, num_t_tiles):
    """Accumulate routed expert rows into a token output buffer.

    Args:
        routed (tensor): Routed expert values with shape ``[N, D]``.
        inv_idx (tensor): Int32 source-row indices with shape ``[T, top_k]``.
        out_buffer (tensor): ``[T, D]`` destination modified in place.
        num_t_tiles (tensor): Int32 ``[1, 1]`` token-tile count per program.

    Returns:
        tensor: The updated ``out_buffer``.

    Note:
        Each token adds the sum of its routed rows using a float32 accumulator.
    """
    T, top_k = inv_idx.shape  # noqa: N806
    _N, D = routed.shape  # noqa: N806

    num_programs = nl.num_programs(axes=0)
    program_id = nl.program_id(axis=0)
    # Static per-program base offset in ELEMENTS (offset= is element-granular, like
    # add_kernel's program_row_offset = program_id * tile_size_m * hidden).
    prog_out_off = program_id * _TILE_M * D  # into out_buffer [T, D]
    prog_inv_off = program_id * _TILE_M * top_k  # into inv_idx  [T, top_k]

    _dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}.get(str(routed.dtype), 2)
    tile_d = min(D, 4096 // _dtype_bytes)
    d_starts, d_widths, _off = [], [], 0
    while _off < D:
        w = min(tile_d, D - _off)
        d_starts.append(_off)
        d_widths.append(w)
        _off += w

    nt_sb = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=nt_sb, src=num_t_tiles[0])
    nt_reg = nisa.register_alloc()
    nisa.register_load(nt_reg, nt_sb)

    # Per-iteration row-tile advance (in elements), added by 128*P each body call.
    out_step = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.memset(out_step, 0.0)
    inv_step = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.memset(inv_step, 0.0)

    def body(_i):
        """Process one dynamic token-tile iteration.

        Args:
            _i (int): Unused loop iteration index.
        """
        for seg in range(len(d_starts)):
            ds, dw = d_starts[seg], d_widths[seg]
            acc = nl.ndarray((_TILE_M, dw), nl.float32, nl.sbuf)  # fp32 accumulate
            nisa.memset(acc, 0.0)
            for k in range(top_k):  # static
                idx_tile = nl.ndarray((_TILE_M, 1), nl.int32, nl.sbuf)
                nisa.dma_copy(
                    dst=idx_tile[0:_TILE_M, 0:1],
                    src=inv_idx.ap(
                        pattern=[[top_k, _TILE_M], [1, 1]],
                        offset=prog_inv_off + k,
                        scalar_offset=inv_step,
                        indirect_dim=0,
                    ),
                )
                g = nl.ndarray((_TILE_M, dw), routed.dtype, nl.sbuf)
                nisa.dma_copy(
                    dst=g[0:_TILE_M, 0:dw],
                    src=routed.ap(
                        pattern=[[D, _TILE_M], [1, dw]],
                        offset=ds,
                        vector_offset=idx_tile,
                        indirect_dim=0,
                    ),
                )
                nisa.tensor_tensor(
                    dst=acc[0:_TILE_M, 0:dw],
                    data1=acc[0:_TILE_M, 0:dw],
                    data2=g[0:_TILE_M, 0:dw],
                    op=nl.add,
                )
            cur = nl.ndarray((_TILE_M, dw), out_buffer.dtype, nl.sbuf)
            nisa.dma_copy(
                dst=cur[0:_TILE_M, 0:dw],
                src=out_buffer.ap(
                    pattern=[[D, _TILE_M], [1, dw]],
                    offset=prog_out_off + ds,
                    scalar_offset=out_step,
                    indirect_dim=0,
                ),
            )
            res = nl.ndarray((_TILE_M, dw), out_buffer.dtype, nl.sbuf)
            nisa.tensor_tensor(
                dst=res[0:_TILE_M, 0:dw],
                data1=cur[0:_TILE_M, 0:dw],
                data2=acc[0:_TILE_M, 0:dw],
                op=nl.add,
            )
            nisa.dma_copy(
                dst=out_buffer.ap(
                    pattern=[[D, _TILE_M], [1, dw]],
                    offset=prog_out_off + ds,
                    scalar_offset=out_step,
                    indirect_dim=0,
                ),
                src=res[0:_TILE_M, 0:dw],
            )
        nisa.tensor_scalar(out_step, out_step, nl.add, _TILE_M * num_programs)
        nisa.tensor_scalar(inv_step, inv_step, nl.add, _TILE_M * num_programs)

    nl.fori_loop(0, nt_reg, body)
    return out_buffer

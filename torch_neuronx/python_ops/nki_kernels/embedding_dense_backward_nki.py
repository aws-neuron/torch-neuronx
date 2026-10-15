"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

NKI kernel for embedding_dense_backward (the backward of an embedding lookup),
a scatter-add of gradient rows into the [E, D] weight-gradient table:
    output[index[i], :] += src[i, :]   (duplicate indices summed)

Indices are pre-sorted by the caller. The kernel allocates and zero-fills the
[E, D] output itself and scatters in place, so the table is the kernel's I/O
tensor rather than a mid-graph scratchpad buffer; only small SBUF tiles are
resident. Accumulation is fp32 (matmul PSUM + carry); the bf16 output is written
once at the end.

Duplicates are handled in two parts:
  - within a 128-row tile: M[i, j] = (index[i] == index[j]); src_ded = M @ src
    gives every row its group sum, so duplicate rows write identical values
    (idempotent indirect-write race).
  - across tiles: a running fp32 carry of the previous tile's last group is fed
    into this tile's group 0 before the matmul. Sorted indices make each group
    contiguous, so the carry is the only cross-tile state.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from neuronxcc.private_nkl.utils.kernel_assert import kernel_assert
from neuronxcc.private_nkl.utils.kernel_helpers import div_ceil, get_program_sharding_info
from nki.isa import constants as nisa_constants


def _validate_inputs(index, src):
    kernel_assert(len(index.shape) == 1, f"index must be 1D, got shape {index.shape}")
    kernel_assert(len(src.shape) == 2, f"src must be 2D, got shape {src.shape}")
    kernel_assert(
        index.shape[0] == src.shape[0],
        f"index and src must agree on dim 0: index {index.shape[0]}, src {src.shape[0]}",
    )


@nki.jit
def embedding_dense_backward_nki(num_weights, index, src):
    """Scatter-add sorted embedding gradients into a dense weight gradient.

    Args:
        num_weights (int): Number of rows in the output embedding table.
        index (nl.ndarray): Sorted ``int32`` row indices with shape ``(N,)``.
        src (nl.ndarray): Source gradients with shape ``(N, D)``.

    Returns:
        nl.ndarray: Zero-filled dense result with shape ``(num_weights, D)`` and
            data type matching ``src``.

    Raises:
        AssertionError: If ``index`` is not one-dimensional, ``src`` is not
            two-dimensional, or their first dimensions differ.

    Note:
        Programs update disjoint output-column slices without cross-program barriers.
        Duplicate indices are reduced in ``float32`` per tile, with a carry and ordered
        overwrite across tiles, before narrowing and scattering results. Columns must
        divide evenly among programs; the caller uses one program otherwise.
    """
    n, d = src.shape
    e = num_weights
    _validate_inputs(index, src)

    _, num_shards, shard_id = get_program_sharding_info()

    # Output in src.dtype so it IS the graph return (no stablehlo.convert, which
    # would re-tag the [E,D] table as scratchpad). Accumulation stays fp32 below.
    out_dtype = src.dtype
    output = nl.ndarray((e, d), dtype=out_dtype, buffer=nl.shared_hbm)

    p = nl.tile_size.pmax  # 128 rows per tile
    f = nl.tile_size.psum_fmax  # PSUM bank free dim (512 gen2/3)

    # This shard's disjoint column slice of the [E, D] table. All D-dim work
    # (zero-fill, load, matmul, scatter, carry) is confined to [d_offset, d_offset+d_shard).
    d_shard = d // num_shards
    d_offset = shard_id * d_shard

    num_d_tiles = div_ceil(d_shard, f)
    num_k_tiles = div_ceil(n, p)
    num_e_tiles = div_ceil(e, p)

    # Zero this shard's columns (absent vocab rows must end up 0). Full shard-width fill,
    # one DMA per E-tile, written to the full output tensor at an absolute column offset
    # [d_offset, d_offset+d_shard). dge_mode=hwdge on the zero-fill is REQUIRED under
    # multi-shard: empirically the default DGE mode mismatches at the shard column boundary
    # and hwdge resolves it.
    zero_tile = nl.ndarray((p, d_shard), dtype=out_dtype, buffer=nl.sbuf)
    nisa.memset(dst=zero_tile, value=0.0)
    for et in nl.affine_range(num_e_tiles):
        e_valid = min(p, e - et * p)
        nisa.dma_copy(
            dst=output[et * p : et * p + e_valid, d_offset : d_offset + d_shard],
            src=zero_tile[0:e_valid, 0:d_shard],
            dge_mode=nisa_constants.dge_mode.hwdge,
        )

    # Cross-tile carry: running fp32 sum of the previous tile's last group + its
    # index (carry_idx = -1 sentinel; real indices are >= 0). Sized to this shard's
    # columns only.
    carry = nl.ndarray((1, d_shard), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=carry, value=0.0)
    carry_idx = nl.ndarray((1, 1), dtype=index.dtype, buffer=nl.sbuf)
    nisa.memset(dst=carry_idx, value=-1)

    # Sequential so each tile's carry-in sees the previous tile's carry-out.
    for kt in nl.sequential_range(num_k_tiles):
        k_valid = min(p, n - kt * p)
        k_off = kt * p

        idx_col = nl.ndarray((k_valid, 1), dtype=index.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx_col, src=index.ap(pattern=[[1, k_valid], [1, 1]], offset=k_off))

        # Equality matrix M[i, j] = (index[i] == index[j]) via two broadcast loads
        # (free-dim and partition-dim) then compare. fp32 to match the src matmul.
        idx_colf = nl.ndarray((k_valid, k_valid), dtype=index.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=idx_colf, src=index.ap(pattern=[[1, k_valid], [0, k_valid]], offset=k_off)
        )
        idx_rowf = nl.ndarray((k_valid, k_valid), dtype=index.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=idx_rowf, src=index.ap(pattern=[[0, k_valid], [1, k_valid]], offset=k_off)
        )
        m_tile = nl.ndarray((k_valid, k_valid), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=m_tile, data1=idx_colf, data2=idx_rowf, op=nl.equal)

        # carry-in fires iff this tile's group 0 continues the previous tile's last
        # group. Tile-level (not per d-chunk), so compute once.
        match = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=match, data1=carry_idx[0:1, 0:1], data2=idx_col[0:1, 0:1], op=nl.equal
        )

        # One-hot selector for the last row. carry-out needs src_ded[last] (which
        # sits on partition k_valid-1) on partition 0; sel^T @ src_ded does that move
        # via the PE (compute engines can't cross partitions).
        row_iota = nl.ndarray((k_valid, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=row_iota, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        last_row_const = nl.ndarray((k_valid, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=last_row_const, value=k_valid - 1)
        sel = nl.ndarray((k_valid, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sel, data1=row_iota, data2=last_row_const, op=nl.equal)

        for dt in nl.affine_range(num_d_tiles):
            d0 = d_offset + dt * f  # absolute column into the full [E, D] table
            d_valid = min(f, d_shard - dt * f)

            # Load this shard's src columns, upcast to fp32 (keeps matmul + carry fp32).
            src_tile = nl.ndarray((k_valid, d_valid), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=src_tile,
                src=src.ap(pattern=[[d, k_valid], [1, d_valid]], offset=d0 + k_off * d),
            )

            # carry-in: add (carry * match) into row 0; the matmul broadcasts it to
            # all of group 0.
            masked_carry = nl.ndarray((1, d_valid), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=masked_carry,
                data=carry[0:1, dt * f : dt * f + d_valid],
                op0=nl.multiply,
                operand0=match,
            )
            nisa.tensor_tensor(
                dst=src_tile[0:1, 0:d_valid],
                data1=src_tile[0:1, 0:d_valid],
                data2=masked_carry,
                op=nl.add,
            )

            # Group-sum: src_ded[i] = sum over rows sharing i's index. nc_matmul
            # computes stationary^T @ moving; M is symmetric so M^T @ src == M @ src.
            src_ded_psum = nl.ndarray((k_valid, d_valid), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=src_ded_psum, stationary=m_tile, moving=src_tile)
            src_ded = nl.ndarray((k_valid, d_valid), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=src_ded, src=src_ded_psum)

            # Write-once scatter (overwrite, not RMW). Two cases:
            #  - within a tile: duplicate rows share the matmul group-sum, so they write
            #    IDENTICAL values -> the indirect-write race among them is idempotent.
            #  - across tiles: a group straddling the boundary is written partial by this
            #    tile and full (partial + carry) by the next. These values DIFFER, so
            #    correctness relies on ORDERED writes -- the later tile must overwrite the
            #    earlier partial. The sequential k-loop provides that order (last writer wins).
            result = nl.ndarray((k_valid, d_valid), dtype=out_dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=result, src=src_ded)
            nisa.dma_copy(
                dst=output.ap(
                    pattern=[[d, k_valid], [1, d_valid]],
                    offset=d0,  # absolute column into the full [E, D] table (no view)
                    vector_offset=idx_col,
                    indirect_dim=0,
                ),
                src=result,
                oob_mode=nisa.oob_mode.error,
            )

            # carry-out: src_ded[last] (full last-group sum) moved to partition 0.
            carry_psum = nl.ndarray((1, d_valid), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=carry_psum, stationary=sel, moving=src_ded)
            nisa.tensor_copy(
                dst=carry[0:1, dt * f : dt * f + d_valid], src=carry_psum[0:1, 0:d_valid]
            )

        # carry_idx = this tile's last index; reloaded from HBM (it lives on the last
        # partition in SBUF, which can't be copied to partition 0).
        nisa.dma_copy(
            dst=carry_idx[0:1, 0:1],
            src=index.ap(pattern=[[1, 1], [1, 1]], offset=k_off + k_valid - 1),
        )

    return output

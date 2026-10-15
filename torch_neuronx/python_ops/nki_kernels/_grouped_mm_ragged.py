"""Ragged grouped_mm NKI kernels: arbitrary ``offs``, no padding, no host transfer.

  grouped_mm_ragged_2d_3d  a(R, D)  @ b(E, D, F) -> (R + 128, F)  forward, dgrad
  grouped_mm_ragged_2d_2d  a(D1, T) @ b(T, D2)  -> (E, D1, D2)    weight gradient

Each derives its per-expert schedule on device from ``offs``. Every access-pattern
count is a compile-time constant; only loop bounds and offsets are data-dependent,
so routing changes do not recompile.

The 2Dx3D kernel returns 128 spare rows so its last token tile can write past R
without a bounds fault; the wrapper slices them off.
"""
# ruff: noqa: N806
# nl.fori_loop traces its body immediately, so closing over the enclosing loop
# variable is safe here.
# ruff: noqa: B023

import nki
import nki.isa as nisa
import nki.language as nl

M_TILE = 128  # token tile = stationary free dim limit
DC = 128  # contraction chunk = partition dim limit
TOK = 128  # K2: token chunk = contraction tile
D1T = 128  # K2: D1 tile = stationary free dim limit
PART = 128  # SBUF partitions; K2 blocks its schedule by this


@nki.jit()
def grouped_mm_ragged_2d_3d(a, b, offs):
    """Multiply token rows by expert-specific matrices using ragged offsets.

    Args:
        a (nl.ndarray): Left operand with shape ``(R, D)``.
        b (nl.ndarray): Expert matrices with shape ``(E, D, F)``.
        offs (nl.ndarray): Cumulative expert end offsets with shape ``(E,)``.

    Returns:
        nl.ndarray: Result with shape ``(R + 128, F)`` and 128 spare tail rows.

    Raises:
        AssertionError: If ``F`` is not divisible by the number of programs.

    Note:
        Offsets need no padding or alignment, and accumulation uses ``float32``.
    """
    R, D = a.shape
    E, _d1b, F = b.shape
    # One program per logical NeuronCore, each owning a disjoint F slice.
    # Overlapping writes between programs race.
    num_progs = nl.num_programs(0)
    prog_id = nl.program_id(0)
    assert F % num_progs == 0, f"{F=} must be divisible by {num_progs=}"
    PF = F // num_progs
    # Contraction chunks. The last one is short when D is not a multiple of DC; its
    # extent is a compile-time constant, so nc_matmul just contracts over fewer
    # partitions. No padding, no runtime count.
    NDC = (D + DC - 1) // DC
    FT = min(PF, 512)
    NFT = (PF + FT - 1) // FT
    # PSUM accumulates fp32; only the store narrows to the input dtype.
    out = nl.ndarray((R + M_TILE, F), b.dtype, nl.shared_hbm)

    # Per-expert schedule along the free dim: a compute instruction reading one
    # partition must start at a multiple of 32, so values cannot sit on partition g.
    starts = nl.ndarray((1, E), nl.int32, nl.sbuf)
    nisa.memset(starts, 0)
    nisa.dma_copy(starts.ap([[E, 1], [1, E - 1]], 1), offs.ap([[E, 1], [1, E - 1]], 0))
    ends = nl.ndarray((1, E), nl.int32, nl.sbuf)
    nisa.dma_copy(ends, offs.ap([[E, 1], [1, E]]))
    ntiles = nl.ndarray((1, E), nl.int32, nl.sbuf)
    nisa.tensor_tensor(ntiles, ends, starts, nl.subtract)
    nisa.tensor_scalar(ntiles, ntiles, nl.add, M_TILE - 1)
    nisa.tensor_scalar(ntiles, ntiles, nl.right_shift, 7)
    last = nl.ndarray((1, E), nl.int32, nl.sbuf)
    nisa.tensor_scalar(last, ntiles, nl.subtract, 1)
    nisa.tensor_scalar(last, last, nl.left_shift, 7)
    nisa.tensor_tensor(last, last, starts, nl.add)
    nsafe = nl.ndarray((1, E), nl.int32, nl.sbuf)
    nisa.tensor_scalar(nsafe, last, nl.greater, R - M_TILE)
    nisa.tensor_tensor(nsafe, ntiles, nsafe, nl.subtract)
    # HBM-staged: a stride-0 broadcast is only legal from HBM.
    last_hbm = nl.ndarray((1, E), nl.int32, nl.private_hbm)
    nisa.dma_copy(last_hbm, last)

    lo_reg = nisa.register_alloc()
    hi_reg = nisa.register_alloc()

    for g in range(E):
        nisa.register_load(lo_reg, nsafe.ap([[E, 1], [1, 1]], g))
        nisa.register_load(hi_reg, ntiles.ap([[E, 1], [1, 1]], g))

        # Duplicated because the tracer forbids calling inner functions; the bodies
        # differ only in the A load.
        def safe_body(i, g=g):
            """Process an expert token tile that is fully within the input.

            Args:
                i (int): Expert-local tile index.
                g (int, optional): Expert index captured by the traced loop body.
                    Defaults to the enclosing expert index.
            """
            i_sb = nl.ndarray((1, 1), nl.int32, nl.sbuf)
            base_sb = nl.ndarray((1, 1), nl.int32, nl.sbuf)
            a_tile = nl.ndarray((M_TILE, D), a.dtype, nl.sbuf)
            aT = nl.ndarray((DC, NDC * M_TILE), a.dtype, nl.sbuf)
            narrow = nl.ndarray((M_TILE, FT), b.dtype, nl.sbuf)
            ps_t = nl.ndarray((DC, M_TILE), a.dtype, nl.psum)
            psum = nl.ndarray((M_TILE, FT), nl.float32, nl.psum)
            b_all = nl.ndarray((DC, NDC * PF), b.dtype, nl.sbuf)
            NFULL = D // DC
            if NFULL:
                nisa.dma_copy(
                    b_all.ap([[NDC * PF, DC], [PF, NFULL], [1, PF]]),
                    b.ap([[F, DC], [DC * F, NFULL], [1, PF]], g * D * F + prog_id * PF),
                )
            if NFULL < NDC:
                nisa.dma_copy(
                    b_all.ap([[NDC * PF, D - NFULL * DC], [1, PF]], NFULL * PF),
                    b.ap(
                        [[F, D - NFULL * DC], [1, PF]],
                        g * D * F + NFULL * DC * F + prog_id * PF,
                    ),
                )

            nisa.register_store(i_sb, i)
            nisa.tensor_scalar(base_sb, i_sb, nl.left_shift, 7)
            nisa.tensor_tensor(base_sb, base_sb, starts.ap([[E, 1], [1, 1]], g), nl.add)
            nisa.dma_copy(
                a_tile,
                a.ap([[D, M_TILE], [1, D]], 0, scalar_offset=base_sb, indirect_dim=0),
                dge_mode=nisa.dge_mode.hwdge,
            )
            for c in range(NDC):
                kc = min(DC, D - c * DC)
                nisa.nc_transpose(
                    ps_t.ap([[M_TILE, kc], [1, M_TILE]]),
                    a_tile.ap([[D, M_TILE], [1, kc]], c * DC),
                )
                nisa.tensor_copy(
                    aT.ap([[NDC * M_TILE, kc], [1, M_TILE]], c * M_TILE),
                    ps_t.ap([[M_TILE, kc], [1, M_TILE]]),
                )
            for j in range(NFT):
                fw = min(FT, PF - j * FT)  # PF, not F: the per-core slice is what exists
                psum_ap = psum.ap([[FT, M_TILE], [1, fw]])
                for c in range(NDC):
                    kc = min(DC, D - c * DC)
                    nisa.nc_matmul(
                        psum_ap,
                        aT.ap([[NDC * M_TILE, kc], [1, M_TILE]], c * M_TILE),
                        b_all.ap([[NDC * PF, kc], [1, fw]], c * PF + j * FT),
                        accumulate=(c > 0),
                    )
                nisa.tensor_copy(narrow.ap([[FT, M_TILE], [1, fw]]), psum_ap)
                nisa.dma_copy(
                    out.ap(
                        [[F, M_TILE], [1, fw]],
                        j * FT + prog_id * PF,
                        scalar_offset=base_sb,
                        indirect_dim=0,
                    ),
                    narrow.ap([[FT, M_TILE], [1, fw]]),
                    dge_mode=nisa.dge_mode.hwdge,
                )

        def tail_body(i, g=g):
            # the tail tile IS the expert's last tile, so its base is last[g]
            """Process an expert's final token tile with clamped input rows.

            Args:
                i (int): Unused expert-local tile index supplied by the traced loop.
                g (int, optional): Expert index captured by the traced loop body.
                    Defaults to the enclosing expert index.
            """
            base_sb = nl.ndarray((1, 1), nl.int32, nl.sbuf)
            base_bc = nl.ndarray((M_TILE, 1), nl.int32, nl.sbuf)
            iot = nl.ndarray((M_TILE, 1), nl.int32, nl.sbuf)
            idx = nl.ndarray((M_TILE, 1), nl.int32, nl.sbuf)
            a_tile = nl.ndarray((M_TILE, D), a.dtype, nl.sbuf)
            aT = nl.ndarray((DC, NDC * M_TILE), a.dtype, nl.sbuf)
            narrow = nl.ndarray((M_TILE, FT), b.dtype, nl.sbuf)
            ps_t = nl.ndarray((DC, M_TILE), a.dtype, nl.psum)
            psum = nl.ndarray((M_TILE, FT), nl.float32, nl.psum)
            b_all = nl.ndarray((DC, NDC * PF), b.dtype, nl.sbuf)
            NFULL = D // DC
            if NFULL:
                nisa.dma_copy(
                    b_all.ap([[NDC * PF, DC], [PF, NFULL], [1, PF]]),
                    b.ap([[F, DC], [DC * F, NFULL], [1, PF]], g * D * F + prog_id * PF),
                )
            if NFULL < NDC:
                nisa.dma_copy(
                    b_all.ap([[NDC * PF, D - NFULL * DC], [1, PF]], NFULL * PF),
                    b.ap(
                        [[F, D - NFULL * DC], [1, PF]],
                        g * D * F + NFULL * DC * F + prog_id * PF,
                    ),
                )

            nisa.tensor_copy(base_sb, last.ap([[E, 1], [1, 1]], g))
            # idx[r] = min(base + r, R - 1) keeps a static 128-row read in bounds.
            nisa.iota(iot, [[0, 1]], 0, channel_multiplier=1)
            nisa.dma_copy(base_bc, last_hbm.ap([[0, M_TILE], [1, 1]], g))
            nisa.tensor_tensor(idx, iot, base_bc, nl.add)
            nisa.tensor_scalar(idx, idx, nl.minimum, R - 1)
            # Clamp both ends: min() alone would let a low base read before the
            # tensor.
            nisa.tensor_scalar(idx, idx, nl.maximum, 0)
            nisa.dma_copy(
                a_tile,
                a.ap([[D, M_TILE], [1, D]], 0, vector_offset=idx, indirect_dim=0),
            )

            for c in range(NDC):
                kc = min(DC, D - c * DC)
                nisa.nc_transpose(
                    ps_t.ap([[M_TILE, kc], [1, M_TILE]]),
                    a_tile.ap([[D, M_TILE], [1, kc]], c * DC),
                )
                nisa.tensor_copy(
                    aT.ap([[NDC * M_TILE, kc], [1, M_TILE]], c * M_TILE),
                    ps_t.ap([[M_TILE, kc], [1, M_TILE]]),
                )
            for j in range(NFT):
                fw = min(FT, PF - j * FT)  # PF, not F: the per-core slice is what exists
                psum_ap = psum.ap([[FT, M_TILE], [1, fw]])
                for c in range(NDC):
                    kc = min(DC, D - c * DC)
                    nisa.nc_matmul(
                        psum_ap,
                        aT.ap([[NDC * M_TILE, kc], [1, M_TILE]], c * M_TILE),
                        b_all.ap([[NDC * PF, kc], [1, fw]], c * PF + j * FT),
                        accumulate=(c > 0),
                    )
                nisa.tensor_copy(narrow.ap([[FT, M_TILE], [1, fw]]), psum_ap)
                nisa.dma_copy(
                    out.ap(
                        [[F, M_TILE], [1, fw]],
                        j * FT + prog_id * PF,
                        scalar_offset=base_sb,
                        indirect_dim=0,
                    ),
                    narrow.ap([[FT, M_TILE], [1, fw]]),
                )

        nl.fori_loop(0, lo_reg, safe_body)
        nl.fori_loop(lo_reg, hi_reg, tail_body)
    return out


@nki.jit()
def grouped_mm_ragged_2d_2d(a, b, offs):
    """Accumulate ragged token products into per-expert weight gradients.

    Args:
        a (nl.ndarray): Left operand with shape ``(D1, T)``.
        b (nl.ndarray): Right operand with shape ``(T, D2)``.
        offs (nl.ndarray): Cumulative expert end offsets with shape ``(E,)``.

    Returns:
        nl.ndarray: Per-expert result with shape ``(E, D1, D2)``.

    Raises:
        AssertionError: If ``D2`` is not divisible by the number of programs.

    Note:
        Offsets need no padding or alignment, and accumulation uses ``float32``.
    """
    D1, T = a.shape
    _tb, D2 = b.shape
    (E,) = offs.shape
    C_MAX = (T + TOK - 1) // TOK
    W = 4
    # D1 tiles. The last is short when D1 % D1T != 0; its extent stays a constant.
    NDT = (D1 + D1T - 1) // D1T
    NFULL_D1 = D1 // D1T
    # One program per logical NeuronCore, each owning a disjoint D2 slice.
    num_progs = nl.num_programs(0)
    prog_id = nl.program_id(0)
    assert D2 % num_progs == 0, f"{D2=} must be divisible by {num_progs=}"
    PD2 = D2 // num_progs
    # Output width tiles: PD2 is the moving operand free dim, limited to 512.
    F2T = min(PD2, 512)
    NF2 = (PD2 + F2T - 1) // F2T
    # Accumulator stays fp32; only the store narrows to the input dtype.
    out = nl.ndarray((E, D1, D2), b.dtype, nl.shared_hbm)

    # counts >= 1 so chunk 0 always initialises the accumulator, even for an empty
    # expert. Free-dim layout for the partition-start-multiple-of-32 rule.
    ends_r = nl.ndarray((1, E), nl.int32, nl.sbuf)
    nisa.dma_copy(ends_r, offs.ap([[E, 1], [1, E]]))
    starts_r = nl.ndarray((1, E), nl.int32, nl.sbuf)
    nisa.memset(starts_r, 0)
    nisa.dma_copy(starts_r.ap([[E, 1], [1, E - 1]], 1), offs.ap([[E, 1], [1, E - 1]], 0))
    counts_sb = nl.ndarray((1, E), nl.int32, nl.sbuf)
    nisa.tensor_tensor(counts_sb, ends_r, starts_r, nl.subtract)
    nisa.tensor_scalar(counts_sb, counts_sb, nl.add, TOK - 1)
    nisa.tensor_scalar(counts_sb, counts_sb, nl.right_shift, 7)
    nisa.tensor_scalar(counts_sb, counts_sb, nl.maximum, 1)

    # Per-chunk window [base_c, base_c + TOK) and the rows of it this chunk owns,
    # [r_lo, r_hi). Derived with experts on the partition dim and chunks on the free
    # dim, in blocks of PART experts so E is not bounded by the partition count. The
    # table itself lives in HBM, which has no such limit.
    chunks = nl.ndarray((E, C_MAX, W), nl.int32, nl.private_hbm)
    for eb in range(0, E, PART):
        EB = min(PART, E - eb)

        ends = nl.ndarray((EB, 1), nl.int32, nl.sbuf)
        nisa.dma_copy(ends, offs.ap([[1, EB], [1, 1]], eb))
        starts = nl.ndarray((EB, 1), nl.int32, nl.sbuf)
        if eb == 0:
            nisa.memset(starts, 0)
            if EB > 1:
                nisa.dma_copy(
                    starts.ap([[1, EB - 1], [1, 1]], 1), offs.ap([[1, EB - 1], [1, 1]], 0)
                )
        else:
            nisa.dma_copy(starts, offs.ap([[1, EB], [1, 1]], eb - 1))

        base = nl.ndarray((EB, C_MAX), nl.int32, nl.sbuf)
        nisa.iota(base, [[1, C_MAX]], 0, channel_multiplier=0)  # base[g][c] = c
        nisa.tensor_scalar(base, base, nl.left_shift, 7)
        nisa.tensor_tensor(base, base, starts.ap([[1, EB], [0, C_MAX]]), nl.add)

        base_c = nl.ndarray((EB, C_MAX), nl.int32, nl.sbuf)
        nisa.tensor_scalar(base_c, base, nl.minimum, T - TOK)

        r_lo = nl.ndarray((EB, C_MAX), nl.int32, nl.sbuf)
        nisa.tensor_tensor(r_lo, base, base_c, nl.subtract)

        r_hi = nl.ndarray((EB, C_MAX), nl.int32, nl.sbuf)
        nisa.tensor_scalar(r_hi, base, nl.add, TOK)
        nisa.tensor_tensor(r_hi, r_hi, ends.ap([[1, EB], [0, C_MAX]]), nl.minimum)
        nisa.tensor_tensor(r_hi, r_hi, base_c, nl.subtract)

        cols = [base_c, r_lo, r_hi]
        for j in range(3):
            nisa.dma_copy(
                chunks.ap([[C_MAX * W, EB], [W, C_MAX]], eb * C_MAX * W + j),
                cols[j].ap([[C_MAX, EB], [1, C_MAX]]),
            )

    n_reg = nisa.register_alloc()

    for g in range(E):
        nisa.register_load(n_reg, counts_sb.ap([[E, 1], [1, 1]], g))

        # Per group, not hoisted: a buffer live across the E loops is rejected
        acc = nl.ndarray((D1T, NDT * PD2), nl.float32, nl.sbuf)
        nisa.memset(acc, 0.0)

        def body(c, g=g):
            """Accumulate one token chunk for an expert's weight gradient.

            Args:
                c (int): Expert-local token-chunk index.
                g (int, optional): Expert index captured by the traced loop body.
                    Defaults to the enclosing expert index.
            """
            c_sb = nl.ndarray((1, 1), nl.int32, nl.sbuf)
            base_sb = nl.ndarray((1, 1), nl.int32, nl.sbuf)
            lo_bc = nl.ndarray((TOK, 1), nl.int32, nl.sbuf)
            hi_bc = nl.ndarray((TOK, 1), nl.int32, nl.sbuf)
            iot = nl.ndarray((TOK, 1), nl.int32, nl.sbuf)
            ge = nl.ndarray((TOK, 1), nl.int32, nl.sbuf)
            lt = nl.ndarray((TOK, 1), nl.int32, nl.sbuf)
            # fp32: tensor_scalar requires a float32 operand
            mask = nl.ndarray((TOK, 1), nl.float32, nl.sbuf)
            b_tile = nl.ndarray((TOK, PD2), b.dtype, nl.sbuf)
            # All NDT D1 slices in one dma_copy; per-slice copies saturate DMA issue.
            a_all = nl.ndarray((D1T, NDT * TOK), a.dtype, nl.sbuf)
            aT = nl.ndarray((TOK, D1T), a.dtype, nl.sbuf)
            ps_t = nl.ndarray((TOK, D1T), a.dtype, nl.psum)
            ps = nl.ndarray((D1T, F2T), nl.float32, nl.psum)

            nisa.register_store(c_sb, c)
            base_row = g * C_MAX * W
            nisa.dma_copy(
                base_sb,
                chunks.ap([[W, 1], [1, 1]], base_row, scalar_offset=c_sb, indirect_dim=1),
            )
            # mask over rows r of the window: r in [r_lo, r_hi)
            nisa.iota(iot, [[0, 1]], 0, channel_multiplier=1)
            nisa.dma_copy(
                lo_bc,
                chunks.ap([[0, TOK], [1, 1]], base_row + 1, scalar_offset=c_sb, indirect_dim=1),
            )
            nisa.dma_copy(
                hi_bc,
                chunks.ap([[0, TOK], [1, 1]], base_row + 2, scalar_offset=c_sb, indirect_dim=1),
            )
            nisa.tensor_tensor(ge, iot, lo_bc, nl.greater_equal)
            nisa.tensor_tensor(lt, iot, hi_bc, nl.less)
            nisa.tensor_tensor(mask, ge, lt, nl.multiply)

            # b rows at the clamped base, masked so tokens outside this chunk
            # contribute nothing.
            nisa.dma_copy(
                b_tile,
                b.ap(
                    [[D2, TOK], [1, PD2]],
                    prog_id * PD2,
                    scalar_offset=base_sb,
                    indirect_dim=0,
                ),
                dge_mode=nisa.dge_mode.hwdge,
            )
            nisa.tensor_scalar(b_tile, b_tile, nl.multiply, mask)

            if NFULL_D1:
                nisa.dma_copy(
                    a_all.ap([[NDT * TOK, D1T], [TOK, NFULL_D1], [1, TOK]]),
                    a.ap(
                        [[T, D1T], [D1T * T, NFULL_D1], [1, TOK]],
                        0,
                        scalar_offset=base_sb,
                        indirect_dim=1,
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                )
            if NFULL_D1 < NDT:
                nisa.dma_copy(
                    a_all.ap([[NDT * TOK, D1 - NFULL_D1 * D1T], [1, TOK]], NFULL_D1 * TOK),
                    a.ap(
                        [[T, D1 - NFULL_D1 * D1T], [1, TOK]],
                        NFULL_D1 * D1T * T,
                        scalar_offset=base_sb,
                        indirect_dim=1,
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                )

            for i in range(NDT):
                dw = min(D1T, D1 - i * D1T)
                aT_ap = aT.ap([[D1T, TOK], [1, dw]])
                nisa.nc_transpose(
                    ps_t.ap([[D1T, TOK], [1, dw]]),
                    a_all.ap([[NDT * TOK, dw], [1, TOK]], i * TOK),
                )
                nisa.tensor_copy(aT_ap, ps_t.ap([[D1T, TOK], [1, dw]]))
                for j in range(NF2):
                    fw = min(F2T, PD2 - j * F2T)
                    ps_ap = ps.ap([[F2T, dw], [1, fw]])
                    nisa.nc_matmul(
                        ps_ap,
                        aT_ap,
                        b_tile.ap([[PD2, TOK], [1, fw]], j * F2T),
                        accumulate=False,
                    )
                    acc_i = acc.ap([[NDT * PD2, dw], [1, fw]], i * PD2 + j * F2T)
                    nisa.tensor_tensor(acc_i, acc_i, ps_ap, nl.add)

        nl.fori_loop(0, n_reg, body)

        narrow = nl.ndarray((D1T, F2T), b.dtype, nl.sbuf)
        for i in range(NDT):
            dw = min(D1T, D1 - i * D1T)
            for j in range(NF2):
                fw = min(F2T, PD2 - j * F2T)
                narrow_ap = narrow.ap([[F2T, dw], [1, fw]])
                nisa.tensor_copy(narrow_ap, acc.ap([[NDT * PD2, dw], [1, fw]], i * PD2 + j * F2T))
                nisa.dma_copy(
                    out.ap(
                        [[D2, dw], [1, fw]],
                        g * D1 * D2 + i * D1T * D2 + prog_id * PD2 + j * F2T,
                    ),
                    narrow_ap,
                )
    return out

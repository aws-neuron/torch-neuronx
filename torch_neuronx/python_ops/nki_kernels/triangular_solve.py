"""Numerically stable batched triangular solve for small matrices."""

import nki
import nki.isa as nisa
import nki.language as nl


def _broadcast_scalar_to_partitions(scalar, ones, n):
    out_psum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=out_psum, stationary=ones, moving=scalar)
    out = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out, src=out_psum)
    return out


@nki.jit
def triangular_solve_batched_nki(matrix, rhs, upper, lnc):
    """Solve batched unit-triangular systems by substitution.

    Args:
        matrix (tensor): Flattened ``[batch * n, n]`` matrix stack.
        rhs (tensor): Flattened ``[batch * n, k]`` right-hand sides.
        upper (bool): Run backward substitution instead of forward substitution.
        lnc (int): Logical Neuron core count used to interleave batches.

    Returns:
        tensor: Float32 solutions with shape ``[batch * n, k]`` in shared HBM.

    Raises:
        AssertionError: If row counts differ, the batch is empty, ``n > 128``, or the RHS width
            is outside ``(0, nl.tile_size.psum_fmax]``.

    Note:
        Runtime loops keep graph size constant. Callers tile wider right-hand sides and handle
        empty inputs.
    """
    matrix_rows, n = matrix.shape
    rhs_rows, k = rhs.shape
    batch = matrix_rows // n
    assert rhs_rows == matrix_rows
    assert batch > 0, "triangular_solve_batched_nki requires a nonempty batch"
    assert n <= 128, f"triangular_solve_batched_nki requires n <= 128, got {n}"
    assert 0 < k <= nl.tile_size.psum_fmax, (
        "triangular_solve_batched_nki requires " f"0 < k <= {nl.tile_size.psum_fmax}, got {k}"
    )

    result = nl.ndarray((matrix_rows, k), dtype=nl.float32, buffer=nl.shared_hbm)

    matrix_work = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.sbuf)
    rhs_work = nl.ndarray((n, k), dtype=nl.float32, buffer=nl.sbuf)
    solution = nl.ndarray((n, k), dtype=nl.float32, buffer=nl.sbuf)

    partition_iota = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=partition_iota, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    free_iota = nl.ndarray((1, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=free_iota, pattern=[[1, n]], offset=0)
    ones_1n = nl.ndarray((1, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_1n, value=1.0)
    ones_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_11, value=1.0)

    matrix_offset = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    rhs_offset = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=matrix_offset, value=0)
    nisa.memset(dst=rhs_offset, value=0)

    program_id = nl.program_id(0)
    # Every program handles an interleaved subset of the batch. Ceil division
    # intentionally permits batch % lnc != 0; the final program iterations use
    # oob_mode.skip for their reads and writes.
    matrices_per_program = (batch + lnc - 1) // lnc
    batch_loop_bound = nisa.register_alloc(x=matrices_per_program)

    def _solve_one(_):
        nisa.memset(dst=matrix_work, value=0.0)
        nisa.memset(dst=rhs_work, value=0.0)
        nisa.memset(dst=solution, value=0.0)
        nisa.dma_copy(
            dst=matrix_work,
            src=matrix.ap(
                pattern=[[n, n], [1, n]],
                offset=program_id * n * n,
                scalar_offset=matrix_offset,
                indirect_dim=0,
            ),
            oob_mode=nisa.oob_mode.skip,
        )
        nisa.dma_copy(
            dst=rhs_work,
            src=rhs.ap(
                pattern=[[k, n], [1, k]],
                offset=program_id * n * k,
                scalar_offset=rhs_offset,
                indirect_dim=0,
            ),
            oob_mode=nisa.oob_mode.skip,
        )

        row_index = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=row_index, value=float(n - 1) if upper else 0.0)
        row_loop_bound = nisa.register_alloc(x=n)

        def _solve_row(_):
            row_index_partitions = _broadcast_scalar_to_partitions(row_index, ones_1n, n)
            row_selector_partitions = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=row_selector_partitions,
                data1=partition_iota,
                data2=row_index_partitions,
                op=nl.equal,
            )
            row_selector_free = nl.ndarray((1, n), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=row_selector_free,
                data=free_iota,
                op0=nl.equal,
                operand0=row_index,
            )

            matrix_row_psum = nl.ndarray((1, n), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=matrix_row_psum,
                stationary=row_selector_partitions,
                moving=matrix_work,
            )
            matrix_row = nl.ndarray((1, n), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=matrix_row, src=matrix_row_psum)
            matrix_row_column_psum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=matrix_row_column_psum,
                stationary=matrix_row,
                moving=ones_11,
            )
            matrix_row_column = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=matrix_row_column, src=matrix_row_column_psum)

            rhs_row_psum = nl.ndarray((1, k), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=rhs_row_psum,
                stationary=row_selector_partitions,
                moving=rhs_work,
            )
            rhs_row = nl.ndarray((1, k), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=rhs_row, src=rhs_row_psum)

            product_psum = nl.ndarray((1, k), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=product_psum,
                stationary=matrix_row_column,
                moving=solution,
            )
            solved_row = nl.ndarray((1, k), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=solved_row, src=product_psum)
            nisa.tensor_tensor(
                dst=solved_row,
                data1=rhs_row,
                data2=solved_row,
                op=nl.subtract,
            )

            row_update_psum = nl.ndarray((n, k), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=row_update_psum,
                stationary=row_selector_free,
                moving=solved_row,
            )
            row_update = nl.ndarray((n, k), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=row_update, src=row_update_psum)
            nisa.tensor_tensor(
                dst=solution,
                data1=solution,
                data2=row_update,
                op=nl.add,
            )

            nisa.tensor_scalar(
                dst=row_index,
                data=row_index,
                op0=nl.add,
                operand0=-1.0 if upper else 1.0,
            )

        nl.fori_loop(0, row_loop_bound, _solve_row, step=1)

        nisa.dma_copy(
            dst=result.ap(
                pattern=[[k, n], [1, k]],
                offset=program_id * n * k,
                scalar_offset=rhs_offset,
                indirect_dim=0,
            ),
            src=solution,
            oob_mode=nisa.oob_mode.skip,
        )
        nisa.tensor_scalar(
            dst=matrix_offset,
            data=matrix_offset,
            op0=nl.add,
            operand0=lnc * n,
        )
        nisa.tensor_scalar(
            dst=rhs_offset,
            data=rhs_offset,
            op0=nl.add,
            operand0=lnc * n,
        )

    nl.fori_loop(0, batch_loop_bound, _solve_one, step=1)
    return result

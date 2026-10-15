"""Eager end-to-end tests for the native ``all_to_all_v`` collective.

These tests drive ``torch.distributed.all_to_all_single(output, input,
output_split_sizes, input_split_sizes)`` with UNEVEN splits, which routes to the
new NKI ``all_to_all_v`` path in ``NeuronBackend::alltoall_base`` by default
(proposal REV 2, §6).

Structure mirrors ``test_alltoall_base.py``: standalone ``run_*(rank, world_size,
kwargs)`` functions driven by ``self.distributed_tester.run_test(fn, **kwargs)``
inside a ``TestAllToAllV(BaseCollectiveOpTest)`` with a ``world_size`` property,
and ``@assert_raises(...)`` for error cases.

Most NKI-path tests set ``NEURON_USE_NKI_ALLTOALLV=1`` explicitly inside the
``run_*`` function (which executes in the spawned worker process). The
``_nki_alltoallv_flag`` context manager saves/restores the prior value so the
setting never leaks across tests; the default-on test removes the variable.
"""

import contextlib
import os

import pytest
import torch
import torch.distributed as dist

from tests.utils.neuron_test_utils import assert_raises

from .base_collective_op import BaseCollectiveOpTest


def test_globally_empty_group_skips_payload_collective(monkeypatch):
    """The capacity all-reduce synchronizes an empty group without zero-row a2a-v."""
    from torch_neuronx.distributed.ops import all_to_all_v

    observed = {"payload_launched": False}

    class FakeNkiLauncher:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                if self.kernel is all_to_all_v._all_reduce_max_hbm:
                    return torch.tensor([[0, 0]], dtype=torch.int32)

                observed["payload_launched"] = True
                return torch.empty((args[3], 1))

            return launch

    monkeypatch.setattr(all_to_all_v, "ReplicaGroup", lambda groups: groups)
    monkeypatch.setattr(all_to_all_v, "get_logical_neuron_cores", lambda: 2)
    monkeypatch.setattr(all_to_all_v, "wrap_nki", lambda kernel: FakeNkiLauncher(kernel))

    all_to_all_v.nki_all_to_all_v_single(
        output=torch.empty(0),
        input=torch.empty(0),
        output_split_sizes=[0, 0, 0, 0],
        input_split_sizes=[0, 0, 0, 0],
        replica_groups=[[0, 1, 2, 3]],
    )

    assert observed["payload_launched"] is False


def test_list_output_copies_packed_result_into_each_tensor(monkeypatch):
    """List-form output consumes explicit-displacement blocks from a packed prefix."""
    from torch_neuronx.distributed.ops import all_to_all_v

    observed = {}

    class FakeNkiLauncher:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                if self.kernel is all_to_all_v._all_reduce_max_hbm:
                    assert torch.equal(args[0], torch.tensor([[2, 4]], dtype=torch.int32))
                    return torch.tensor([[2, 4]], dtype=torch.int32)

                assert args[0].shape == (4, 2)
                assert torch.count_nonzero(args[0][2:]) == 0
                observed["metadata"] = args[1].cpu()
                observed["dst_rows"] = args[3]
                observed["recv_counts_known"] = kwargs["recv_counts_known"]
                observed["has_rdispls"] = kwargs["has_rdispls"]
                return torch.tensor(
                    [
                        [10, 11],
                        [30, 31],
                        [32, 33],
                        [40, 41],
                    ],
                    dtype=torch.float32,
                )

            return launch

    monkeypatch.setattr(all_to_all_v, "ReplicaGroup", lambda groups: groups)
    monkeypatch.setattr(all_to_all_v, "get_logical_neuron_cores", lambda: 2)
    monkeypatch.setattr(all_to_all_v, "wrap_nki", lambda kernel: FakeNkiLauncher(kernel))

    strided_storage = torch.full((2, 2), -1.0)
    outputs = [
        torch.empty((1, 2)),
        torch.empty((0, 2)),
        strided_storage.t(),
        torch.empty((1, 2)),
    ]
    returned = all_to_all_v.nki_all_to_all_v_single(
        output=outputs,
        input=torch.empty((2, 2)),
        output_split_sizes=[1, 0, 2, 1],
        input_split_sizes=[1, 0, 1, 0],
        replica_groups=[[0, 1, 2, 3]],
    )

    assert returned is outputs
    assert observed["dst_rows"] == 4
    assert observed["recv_counts_known"] is True
    assert observed["has_rdispls"] is True
    assert torch.equal(
        observed["metadata"],
        torch.tensor(
            [
                [2, 0, 2, 0],
                [0, 2, 2, 4],
                [2, 0, 4, 2],
                [0, 2, 2, 6],
            ],
            dtype=torch.uint32,
        ),
    )
    assert torch.equal(outputs[0], torch.tensor([[10, 11]], dtype=torch.float32))
    assert outputs[1].shape == (0, 2)
    assert torch.equal(outputs[2], torch.tensor([[30, 31], [32, 33]], dtype=torch.float32))
    assert torch.equal(strided_storage, torch.tensor([[30, 32], [31, 33]], dtype=torch.float32))
    assert torch.equal(outputs[3], torch.tensor([[40, 41]], dtype=torch.float32))


@contextlib.contextmanager
def _nki_alltoallv_flag(value):
    """Set or unset NEURON_USE_NKI_ALLTOALLV for the block, then restore it.

    Runs inside the spawned worker process; the C++ backend reads the flag
    synchronously when it dispatches ``alltoall_base`` (routing decision happens
    at enqueue time, not during async execution), so wrapping just the
    collective call is sufficient to select the NKI vs padding path.
    """
    old = os.environ.get("NEURON_USE_NKI_ALLTOALLV")
    if value is None:
        os.environ.pop("NEURON_USE_NKI_ALLTOALLV", None)
    else:
        os.environ["NEURON_USE_NKI_ALLTOALLV"] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("NEURON_USE_NKI_ALLTOALLV", None)
        else:
            os.environ["NEURON_USE_NKI_ALLTOALLV"] = old


def verify_alltoall_v_results(rank, output_tensor, output_split_sizes, group_ranks, world_size):
    """Variable-offset verifier for uneven all_to_all_v (proposal M7).

    The existing ``verify_alltoall_base_results`` assumes a UNIFORM ``chunk_size``
    and cannot be reused for uneven splits. This walks ``output_split_sizes`` to
    compute each source block's ``[start:end)`` row range in the (contiguous,
    rank-ordered) output and asserts the expected per-source value using the same
    encoding scheme as the base test:

        value received from src == src * 10 + rank_idx_in_group + 1

    Blocks with a 0 split size contribute no rows and are skipped. Works for both
    1D outputs (each row is a scalar) and 2D outputs ``[rows, H]`` (each row is a
    length-``H`` vector) because it slices dim 0 only.

    Args:
        rank: Current (global) process rank.
        output_tensor: Output tensor from all_to_all_single.
        output_split_sizes: Per-source ROW counts, indexed by position in the group.
        group_ranks: List of (global) ranks participating in the group.
        world_size: Total world size (for messaging / defaulting).
    """
    rank_idx_in_group = group_ranks.index(rank)

    start = 0
    for i, src_rank in enumerate(group_ranks):
        n = output_split_sizes[i]
        if n > 0:
            expected_value = src_rank * 10 + rank_idx_in_group + 1
            actual_block = output_tensor[start : start + n].cpu()
            expected_block = torch.full(
                actual_block.shape, expected_value, dtype=output_tensor.dtype
            )
            assert torch.allclose(actual_block, expected_block), (
                f"Rank {rank}: expected value {expected_value} from rank {src_rank} "
                f"at rows {start}:{start + n}, got {actual_block}."
            )
        start += n


def _build_uneven_alltoall_v_input(rank, group_ranks, send_rows_fn, H, dtype):  # noqa: N803
    """Build a contiguously-packed (by destination) input tensor and split sizes.

    ``send_rows_fn(src_pos, dst_pos) -> int`` returns the number of rows the rank
    at group position ``src_pos`` sends to the rank at group position ``dst_pos``.
    Each destination block is filled with the value ``rank * 10 + dst_pos + 1`` so
    the receiver can verify provenance with ``verify_alltoall_v_results``.

    Returns:
        (input_tensor, output_tensor, input_split_sizes, output_split_sizes)
        with tensors already moved to the neuron device.
    """
    group_size = len(group_ranks)
    my_pos = group_ranks.index(rank)

    input_split_sizes = [send_rows_fn(my_pos, dst_pos) for dst_pos in range(group_size)]
    output_split_sizes = [send_rows_fn(src_pos, my_pos) for src_pos in range(group_size)]

    total_rows = sum(input_split_sizes)
    in_shape = (total_rows,) if H == 1 else (total_rows, H)
    input_tensor = torch.zeros(in_shape, dtype=dtype)
    off = 0
    for dst_pos in range(group_size):
        n = input_split_sizes[dst_pos]
        if n > 0:
            input_tensor[off : off + n] = rank * 10 + dst_pos + 1
        off += n

    out_total = sum(output_split_sizes)
    out_shape = (out_total,) if H == 1 else (out_total, H)
    output_tensor = torch.zeros(out_shape, dtype=dtype)

    return (
        input_tensor.to("neuron"),
        output_tensor.to("neuron"),
        input_split_sizes,
        output_split_sizes,
    )


def _default_send_rows(src_pos, dst_pos):
    """Asymmetric, always non-zero send count -> genuinely uneven both sides."""
    return src_pos + dst_pos + 1


def _isolated_rank_send_rows(src_pos, dst_pos):
    """Rank zero sends/receives nothing while the remaining ranks exchange rows."""
    if src_pos == 0 or dst_pos == 0:
        return 0
    return src_pos + dst_pos


def run_uneven_split_test(rank, world_size, kwargs):
    """Uneven (asymmetric) non-zero split sizes; verify each received block.

    Supports 1D and 2D (``H`` rows) inputs and arbitrary dtypes via kwargs. Uses
    the NKI path since the send/recv splits are genuinely uneven.
    """
    dtype = kwargs.get("dtype", torch.float32)
    H = kwargs.get("H", 1)  # noqa: N806
    group_ranks = list(range(world_size))

    with _nki_alltoallv_flag(kwargs.get("use_nki", "1")):
        input_t, output_t, in_sizes, out_sizes = _build_uneven_alltoall_v_input(
            rank, group_ranks, _default_send_rows, H, dtype
        )
        dist.all_to_all_single(
            output_t,
            input_t,
            output_split_sizes=out_sizes,
            input_split_sizes=in_sizes,
        )

    verify_alltoall_v_results(rank, output_t, out_sizes, group_ranks, world_size)


def run_isolated_empty_rank_test(rank, world_size, kwargs):
    """An empty rank must still participate while its peers exchange data."""
    group_ranks = list(range(world_size))

    with _nki_alltoallv_flag("1"):
        input_t, output_t, in_sizes, out_sizes = _build_uneven_alltoall_v_input(
            rank, group_ranks, _isolated_rank_send_rows, 1, torch.float32
        )
        dist.all_to_all_single(
            output_t,
            input_t,
            output_split_sizes=out_sizes,
            input_split_sizes=in_sizes,
        )

    verify_alltoall_v_results(rank, output_t, out_sizes, group_ranks, world_size)


def run_permute_pattern_test(rank, world_size, kwargs):
    """Ring / permute_tensor pattern (equal non-zero chunks, mirrors
    run_uneven_split_sizes_alltoall_base_test in test_alltoall_base.py).

    Each rank sends its whole buffer to (rank+1)%WS and receives from
    (rank-1)%WS; all other splits are 0. This directly exercises 0-count columns.
    The ``use_nki`` kwarg selects the NKI path ("1") or the padding fallback
    ("0") -- the padding fallback is only correct for equal non-zero chunks, which
    this pattern satisfies, so it is the only pattern used for the flag-off test.
    """
    use_nki = kwargs.get("use_nki", "1")
    N = 4  # elements per rank  # noqa: N806

    dst = (rank + 1) % world_size
    src = (rank - 1) % world_size

    input_split_sizes = [0] * world_size
    input_split_sizes[dst] = N

    output_split_sizes = [0] * world_size
    output_split_sizes[src] = N

    input_tensor = torch.full((N,), float(rank + 1), dtype=torch.float32).to("neuron")
    output_tensor = torch.zeros(N, dtype=torch.float32).to("neuron")

    with _nki_alltoallv_flag(use_nki):
        dist.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
        )

    expected_value = float(src + 1)
    expected = torch.full((N,), expected_value, dtype=torch.float32)
    actual = output_tensor.cpu()
    assert torch.allclose(
        actual, expected
    ), f"Rank {rank}: expected {expected} (from rank {src}), got {actual}."


def run_even_split_delegates_test(rank, world_size, kwargs):
    """Equal split sizes must still take the fast alltoall path (even with the
    NKI flag ON) and produce correct results -- guards against regressing the
    even case, which returns before the UseNkiAllToAllV() branch."""
    chunk = 2
    group_ranks = list(range(world_size))

    with _nki_alltoallv_flag("1"):
        input_tensor = torch.zeros(world_size * chunk, dtype=torch.float32)
        for dst_rank in range(world_size):
            start_idx = dst_rank * chunk
            end_idx = (dst_rank + 1) * chunk
            input_tensor[start_idx:end_idx] = rank * 10 + dst_rank + 1
        input_tensor = input_tensor.to("neuron")
        output_tensor = torch.zeros_like(input_tensor)

        split_sizes = [chunk] * world_size
        dist.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=split_sizes,
            input_split_sizes=split_sizes,
        )

    verify_alltoall_v_results(rank, output_tensor, split_sizes, group_ranks, world_size)


def run_async_op_test(rank, world_size, kwargs):
    """Asynchronous uneven all_to_all_v: async_op=True + work.wait()."""
    group_ranks = list(range(world_size))

    with _nki_alltoallv_flag("1"):
        input_t, output_t, in_sizes, out_sizes = _build_uneven_alltoall_v_input(
            rank, group_ranks, _default_send_rows, 1, torch.float32
        )
        work = dist.all_to_all_single(
            output_t,
            input_t,
            output_split_sizes=out_sizes,
            input_split_sizes=in_sizes,
            async_op=True,
        )
        assert work is not None
        work.wait()

    verify_alltoall_v_results(rank, output_t, out_sizes, group_ranks, world_size)


def run_empty_tensor_test(rank, world_size, kwargs):
    """Implicit all-empty equal splits complete through the no-op fast path."""
    with _nki_alltoallv_flag("1"):
        input_tensor = torch.empty(0, dtype=torch.float32).to("neuron")
        output_tensor = torch.empty(0, dtype=torch.float32).to("neuron")

        dist.all_to_all_single(output_tensor, input_tensor)

    assert output_tensor.numel() == 0, "Output tensor should be empty"


def run_invalid_split_sizes_test(rank, world_size, kwargs):
    """Wrong-length split sizes must raise RuntimeError. The length check fires
    in normalize() before the NKI branch, so it raises with the flag ON too."""
    input_tensor = torch.ones(world_size * 2).to("neuron")
    output_tensor = torch.zeros_like(input_tensor)

    # Wrong length -- should raise
    bad_split_sizes = [2] * (world_size + 1)
    with _nki_alltoallv_flag("1"):
        dist.all_to_all_single(output_tensor, input_tensor, input_split_sizes=bad_split_sizes)


def run_list_form_uneven_test(rank, world_size, kwargs):
    """LIST-FORM all_to_all (dist.all_to_all(output_list, input_list)) with UNEQUAL
    per-rank chunk shapes -- the shape torchtitan's MoE token dispatcher produces.

    Historically the list form hard-required every chunk to share one shape and died
    with "all input tensors must have same shape". This exercises the new path that
    routes unequal (dim-0-varying) chunks through the native NKI all_to_all_v kernel.

    Chunk provenance uses the same encoding as verify_alltoall_v_results:
        value received from src == src * 10 + rank_idx_in_group + 1
    Row counts vary per (src, dst) via _default_send_rows so both send and receive
    lists are genuinely uneven. Supports 1D and 2D (H rows) via kwargs.
    """
    dtype = kwargs.get("dtype", torch.float32)
    H = kwargs.get("H", 1)  # noqa: N806
    my_pos = rank

    # Per-destination send row counts and per-source receive row counts.
    input_split_sizes = [_default_send_rows(my_pos, dst) for dst in range(world_size)]
    output_split_sizes = [_default_send_rows(src, my_pos) for src in range(world_size)]

    def _rows(n):
        return (n, H)

    def _test_shape(tensor, n):
        return tensor.reshape(n) if H == 1 else tensor

    input_tensor_list = []
    for dst in range(world_size):
        n = input_split_sizes[dst]
        val = rank * 10 + dst + 1
        tensor = torch.full(_rows(n), val, dtype=dtype)
        input_tensor_list.append(_test_shape(tensor, n).to("neuron"))

    output_tensor_list = []
    for src in range(world_size):
        n = output_split_sizes[src]
        tensor = torch.zeros(_rows(n), dtype=dtype)
        output_tensor_list.append(_test_shape(tensor, n).to("neuron"))

    with _nki_alltoallv_flag("1"):
        dist.all_to_all(output_tensor_list, input_tensor_list)

    # Each output_tensor_list[src] must hold what src sent to us.
    for src in range(world_size):
        n = output_split_sizes[src]
        if n == 0:
            continue
        expected_value = src * 10 + my_pos + 1
        expected = _test_shape(torch.full(_rows(n), expected_value, dtype=dtype), n)
        actual = output_tensor_list[src].cpu()
        assert torch.allclose(
            actual, expected
        ), f"Rank {rank}: expected {expected_value} from rank {src}, got {actual}."


def run_list_form_isolated_empty_rank_test(rank, world_size, kwargs):
    """List-form empty rank participates while the remaining ranks exchange data."""
    input_sizes = [_isolated_rank_send_rows(rank, dst) for dst in range(world_size)]
    output_sizes = [_isolated_rank_send_rows(src, rank) for src in range(world_size)]
    inputs = [
        torch.full((n,), rank * 10 + dst + 1, dtype=torch.float32).to("neuron")
        for dst, n in enumerate(input_sizes)
    ]
    outputs = [torch.zeros((n,), dtype=torch.float32).to("neuron") for n in output_sizes]

    with _nki_alltoallv_flag("1"):
        dist.all_to_all(outputs, inputs)

    for src, (output, n) in enumerate(zip(outputs, output_sizes, strict=True)):
        expected = torch.full((n,), src * 10 + rank + 1, dtype=torch.float32)
        assert torch.allclose(output.cpu(), expected)


def run_list_form_even_still_delegates_test(rank, world_size, kwargs):
    """LIST-FORM all_to_all with EQUAL chunk shapes must keep the existing even
    path (regression guard) and still produce correct results, even with the NKI
    flag ON -- equal chunks must NOT be routed to the a2a-v kernel."""
    with _nki_alltoallv_flag("1"):
        input_tensor_list = [
            torch.full((5,), rank * 10 + dst + 1, dtype=torch.float32).to("neuron")
            for dst in range(world_size)
        ]
        output_tensor_list = [
            torch.zeros(5, dtype=torch.float32).to("neuron") for _ in range(world_size)
        ]
        dist.all_to_all(output_tensor_list, input_tensor_list)

    for src in range(world_size):
        expected_value = src * 10 + rank + 1
        expected = torch.full((5,), expected_value, dtype=torch.float32)
        assert torch.allclose(output_tensor_list[src].cpu(), expected), (
            f"Rank {rank}: expected {expected_value} from rank {src}, "
            f"got {output_tensor_list[src].cpu()}."
        )


def run_partial_group_test(rank, world_size, kwargs):
    """Uneven all_to_all_v within a dist.new_group([...]) sub-group.

    Mirrors run_partial_group_alltoall_base_test: ranks outside the sub-group do
    not participate in the collective, and every rank hits dist.barrier() on the
    default group at the end so participating ranks don't tear down the shared
    compilation socket while peers still have in-flight compiles.
    """
    partial_group = kwargs.get("partial_group", [0, 1, 2, 3])

    group = dist.new_group(partial_group)

    if rank in partial_group:
        with _nki_alltoallv_flag("1"):
            input_t, output_t, in_sizes, out_sizes = _build_uneven_alltoall_v_input(
                rank, partial_group, _default_send_rows, 1, torch.float32
            )
            dist.all_to_all_single(
                output_t,
                input_t,
                output_split_sizes=out_sizes,
                input_split_sizes=in_sizes,
                group=group,
            )

        verify_alltoall_v_results(rank, output_t, out_sizes, partial_group, world_size)
    else:
        # Ranks not in partial_group don't participate.
        pass

    # Barrier on the default group: participating ranks can finish at different
    # times; the compilation service is single-owner by design, and the first
    # rank to exit tears down its shared socket, starving any in-flight compiles
    # on peers.
    dist.barrier()


class TestAllToAllV(BaseCollectiveOpTest):
    """Test cases for the native all_to_all_v path of
    torch.distributed.all_to_all_single (uneven splits)."""

    @property
    def world_size(self) -> int:
        return 4

    def test_uneven_split_basic(self):
        """Asymmetric non-zero split sizes across a 4-rank group (NKI path)."""
        self.distributed_tester.run_test(run_uneven_split_test)

    def test_default_uses_nki(self):
        """An unset flag defaults genuinely uneven splits to the NKI path."""
        self.distributed_tester.run_test(run_uneven_split_test, use_nki=None)

    def test_isolated_empty_rank(self):
        """A rank with zero sends/receives still participates in all_to_all_v."""
        self.distributed_tester.run_test(run_isolated_empty_rank_test)

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.bfloat16, torch.float16, torch.int32, torch.int64],
    )
    def test_dtypes(self, dtype):
        """Uneven all_to_all_v across supported data types."""
        self.distributed_tester.run_test(run_uneven_split_test, dtype=dtype)

    def test_2d_rows(self):
        """2D input [rows, H] with uneven row splits, H > 1."""
        self.distributed_tester.run_test(run_uneven_split_test, H=3)

    def test_permute_pattern(self):
        """Ring / permute_tensor pattern (0-count columns) via the NKI path."""
        self.distributed_tester.run_test(run_permute_pattern_test, use_nki="1")

    def test_flag_off_uses_padding(self):
        """Flag OFF -> padding fallback; restricted to the permute pattern
        (equal non-zero chunks), the only case the fallback handles correctly."""
        self.distributed_tester.run_test(run_permute_pattern_test, use_nki="0")

    def test_even_split_still_delegates(self):
        """Equal splits still take the fast alltoall path (no regression)."""
        self.distributed_tester.run_test(run_even_split_delegates_test)

    def test_list_form_uneven(self):
        """LIST-FORM dist.all_to_all with UNEQUAL chunk shapes routes to the NKI
        a2a-v path (no longer raises 'same shape'). This is the shape torchtitan's
        standard MoE token dispatcher produces."""
        self.distributed_tester.run_test(run_list_form_uneven_test)

    def test_list_form_isolated_empty_rank(self):
        """List-form rank with all-empty chunks still participates."""
        self.distributed_tester.run_test(run_list_form_isolated_empty_rank_test)

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.bfloat16, torch.int32],
    )
    def test_list_form_uneven_dtypes(self, dtype):
        """LIST-FORM uneven all_to_all across supported dtypes."""
        self.distributed_tester.run_test(run_list_form_uneven_test, dtype=dtype)

    def test_list_form_uneven_2d(self):
        """LIST-FORM uneven all_to_all with 2D chunks [rows, H], H > 1."""
        self.distributed_tester.run_test(run_list_form_uneven_test, H=3)

    def test_list_form_even_still_delegates(self):
        """LIST-FORM equal chunks keep the existing even path (regression guard)."""
        self.distributed_tester.run_test(run_list_form_even_still_delegates_test)

    def test_empty_tensor(self):
        """Empty input/output early-returns correctly."""
        self.distributed_tester.run_test(run_empty_tensor_test)

    def test_async_op(self):
        """Asynchronous uneven all_to_all_v with work.wait()."""
        self.distributed_tester.run_test(run_async_op_test)

    @assert_raises(RuntimeError, match=r".*split_sizes length.*must match group_size.*")
    def test_invalid_split_sizes_error(self):
        """Wrong-length split sizes raise RuntimeError."""
        self.distributed_tester.run_test(run_invalid_split_sizes_test)


@pytest.mark.neuron_cores(8)
class TestAllToAllVWorldSize8(BaseCollectiveOpTest):
    """Partial-group all_to_all_v in world size 8 (group size 4 is
    hardware-supported), following test_alltoall.py's TestAllToAllWorldSize8."""

    @property
    def world_size(self) -> int:
        return 8

    def test_partial_group(self):
        """Uneven all_to_all_v within a supported 4-rank sub-group [0,1,2,3]."""
        self.distributed_tester.run_test(
            run_partial_group_test,
            partial_group=[0, 1, 2, 3],
        )

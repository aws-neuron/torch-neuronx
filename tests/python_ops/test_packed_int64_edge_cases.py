"""Regression tests for packed-int64 edge cases fixed in the fuse_packed_int64 work.

Each test compares a Neuron result against CPU (shape, dtype, values). They guard:

  * 0-dim scalar int64 CPU->Neuron upload (cast_policy.copy_cpu_to_neuron used
    tmp.view(torch.int32), which raises on a 0-dim tensor) and the scalar packed
    physical layout (() -> (2,), not (1,2)).
  * Empty (size-0) int64 gather (packed_gather_nki invoked the NKI kernel with a
    0-sized shape -> "shape must be a tuple of positive integers").
  * Non-contiguous int64 on the Python-op path (convert_for_neuron / masked_fill
    bound only the low-word view -> NRT "input tensor too small").
"""

import pytest
import torch

import torch_neuronx


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestPackedInt64EdgeCases:
    @pytest.fixture
    def device(self):
        return torch.device("neuron")

    @pytest.mark.parametrize("value", [7, -5, 0, 123456])
    def test_scalar_int64_roundtrip(self, device, value):
        """0-dim int64 CPU->Neuron->CPU round-trip (scalar upload + layout)."""
        n = torch.tensor(value, dtype=torch.int64, device=device)
        assert n.device.type == "neuron"
        assert tuple(n.shape) == ()
        assert n.dtype == torch.int64
        assert n.cpu().item() == value

    def test_scalar_int64_arithmetic(self, device):
        """0-dim int64 arithmetic keeps scalar shape () and int64 dtype."""
        a = torch.tensor(7, dtype=torch.int64, device=device)
        b = torch.tensor(3, dtype=torch.int64, device=device)
        for name, neu, cpu in [
            ("add", a + 1, a.cpu() + 1),
            ("mod", a % 3, a.cpu() % 3),
            ("floordiv", a // 3, a.cpu() // 3),
            ("sub", a - b, a.cpu() - b.cpu()),
        ]:
            assert tuple(neu.shape) == tuple(cpu.shape) == (), name
            assert neu.dtype == cpu.dtype == torch.int64, name
            torch.testing.assert_close(neu.cpu(), cpu)

    def test_empty_int64_cat(self, device):
        """cat with a size-0 int64 operand (empty packed gather guard)."""
        empty = torch.empty(0, dtype=torch.int64, device=device)
        rest = torch.arange(3, dtype=torch.int64, device=device)
        neu = torch.cat([empty, rest])
        cpu = torch.cat([empty.cpu(), rest.cpu()])
        assert neu.dtype == cpu.dtype == torch.int64
        torch.testing.assert_close(neu.cpu(), cpu)

    def test_empty_int64_unary(self, device):
        """Op on a size-0 int64 slice must not crash the NKI gather."""
        x = torch.arange(6, dtype=torch.int64, device=device)[:0]  # size-0
        neu = x + 1
        cpu = x.cpu() + 1
        assert tuple(neu.shape) == tuple(cpu.shape) == (0,)
        assert neu.dtype == cpu.dtype == torch.int64
        torch.testing.assert_close(neu.cpu(), cpu)

    def test_noncontiguous_int64_masked_fill(self, device):
        """masked_fill_ on a non-contiguous int64 (the argwhere/log repro)."""
        big = torch.arange(2 * 64, dtype=torch.int64, device=device)
        nc = big[::2]  # non-contiguous, stride 2, shape (64,)
        big_cpu = torch.arange(2 * 64, dtype=torch.int64)
        nc_cpu = big_cpu[::2]

        nc.masked_fill_(nc > 20, 1)
        nc_cpu.masked_fill_(nc_cpu > 20, 1)

        assert nc.dtype == torch.int64
        torch.testing.assert_close(nc.cpu(), nc_cpu)

    def test_noncontiguous_int64_binary(self, device):
        """Binary op on a non-contiguous int64 operand."""
        big = torch.arange(2 * 32, dtype=torch.int64, device=device)
        nc = big[::2]
        big_cpu = torch.arange(2 * 32, dtype=torch.int64)
        nc_cpu = big_cpu[::2]

        neu = nc + 5
        cpu = nc_cpu + 5
        assert tuple(neu.shape) == tuple(cpu.shape) == (32,)
        assert neu.dtype == cpu.dtype == torch.int64
        torch.testing.assert_close(neu.cpu(), cpu)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestPackedInt64OutContract:
    """out= contracts for packed int64 outputs.

    Regression for the AllocateExecutionOutputs Step 2a bug: the user's out=
    was resized to the *physical* packed spec shape (scalar [] -> [2], tensor
    [N] -> [N, 2]) even when it was not used as the exec buffer, reallocating
    its storage and emitting a spurious resize warning that is undone by the
    copy-back. The fix binds the out='s zero-copy packed physical_view as the
    exec buffer instead, so:
      * no resize warning is emitted (empty-out= contract), and
      * the out='s storage is reused (data_ptr preserved).
    """

    @pytest.fixture
    def device(self):
        return torch.device("neuron")

    def _resize_warnings(self, rec):
        return [str(w.message) for w in rec if "was resized" in str(w.message)]

    @pytest.mark.parametrize(
        "shape",
        [(), (0,), (4,), (2, 3)],
    )
    def test_bitwise_or_int64_out_no_resize_warning(self, device, shape):
        """Packed-int64 out= must not raise a resize warning and must be correct."""
        import warnings

        a = torch.randint(0, 100, shape, dtype=torch.int64, device=device)
        b = torch.randint(0, 100, shape, dtype=torch.int64, device=device)
        out = torch.empty(shape, dtype=torch.int64, device=device)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            res = torch.bitwise_or(a, b, out=out)
        rw = self._resize_warnings(rec)
        assert rw == [], f"unexpected resize warning(s): {rw}"
        assert res.dtype == torch.int64
        assert tuple(res.shape) == tuple(shape)
        torch.testing.assert_close(res.cpu(), torch.bitwise_or(a.cpu(), b.cpu()))

    def test_bitwise_or_int64_out_storage_reused(self, device):
        """Correctly-shaped packed-int64 out= reuses its storage (data_ptr stable)."""
        a = torch.arange(1, 9, dtype=torch.int64, device=device)
        b = torch.arange(8, 16, dtype=torch.int64, device=device)
        out = torch.empty(8, dtype=torch.int64, device=device)
        ptr = out.data_ptr()
        torch.bitwise_or(a, b, out=out)
        assert out.data_ptr() == ptr, "packed-int64 out= storage was not reused"
        torch.testing.assert_close(out.cpu(), torch.bitwise_or(a.cpu(), b.cpu()))

    def test_nonzero_out_storage_reused(self, device):
        """nonzero(out=) with a correctly-shaped out= reuses storage (no realloc)."""
        import warnings

        torch.manual_seed(0)
        tensor = torch.randint(2, (4, 4), device=device)
        dst1 = tensor.nonzero(as_tuple=False)
        dst = torch.empty_like(dst1)
        ptr = dst.data_ptr()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            torch.nonzero(tensor, out=dst)
        rw = self._resize_warnings(rec)
        assert dst.data_ptr() == ptr, "nonzero out= storage was not reused"
        assert rw == [], f"unexpected resize warning(s): {rw}"
        torch.testing.assert_close(dst.cpu(), dst1.cpu())

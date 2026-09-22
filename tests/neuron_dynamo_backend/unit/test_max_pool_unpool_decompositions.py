"""End-to-end compile tests for max_pool2d and max_unpool3d.

- max_pool2d_with_indices (forward): index-correctness tests for the im2col
  decomposition, including padding, dilation, asymmetric kernels, and integer
  dtype promotion (when padding) validation.
- max_pool2d_with_indices_backward: the PyTorch composite default lowered to a
  graph that crashed neuronx-cc (InsertIOTransposes); we register an explicit
  scatter-add decomp that compiles cleanly.
- max_unpool3d: lowered via PyTorch's builtin decomposition (enabled in the table).
"""

import re

import pytest
import torch

_TOL = {torch.float32: 1e-4, torch.bfloat16: 1e-2}


def _close(a, b, dtype):
    tol = _TOL.get(dtype, 1e-4)
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestMaxPoolUnpoolDecompositionsE2E:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize(
        "shape,kernel,stride,padding",
        [
            ((1, 2, 8, 8), 2, 2, 0),  # non-overlapping
            ((2, 3, 7, 7), 3, 1, 0),  # overlapping windows
        ],
    )
    def test_max_pool2d_backward_e2e(self, shape, kernel, stride, padding, dtype):
        """Unpadded cases go through the full autograd pipeline (the forward
        decomp picks the argmax, the backward scatter-add decomp handles the
        gradient). Padded cases are covered by
        test_max_pool2d_padded_backward_e2e (full compiled forward+backward) and
        test_max_pool2d_backward_padded_fixed_indices (backward decomp in
        isolation, fed known-correct CPU indices with a varied gradient).
        """

        def fn(x):
            return torch.nn.functional.max_pool2d(x, kernel, stride, padding).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        torch.manual_seed(0)
        x_cpu = torch.randn(*shape, dtype=dtype, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)
        fn(x_cpu).backward()
        compiled_fn(x_neuron).backward()
        _close(x_neuron.grad.cpu(), x_cpu.grad, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_max_pool2d_backward_padded_fixed_indices(self, dtype):
        """Padded max_pool2d backward, tested directly with fixed indices.

        The indices come from the CPU forward, so both paths scatter to the same
        positions — this validates the backward decomp's handling of padded
        (unpadded-frame) indices without depending on cross-backend argmax
        tie-breaking in the forward.
        """
        torch.manual_seed(0)
        k, s, p, d = [2, 2], [2, 2], [1, 1], [1, 1]
        x = torch.randn(1, 2, 8, 8, dtype=torch.float32)
        _, idx = torch.ops.aten.max_pool2d_with_indices.default(x, k, s, p, d, False)
        grad_output = torch.randn(*idx.shape, dtype=torch.float32)

        def fn(grad_output, x, idx):
            return torch.ops.aten.max_pool2d_with_indices_backward.default(
                grad_output, x, k, s, p, d, False, idx
            )

        compiled_fn = torch.compile(fn, backend="neuron")
        ref = fn(grad_output, x, idx)
        out = compiled_fn(
            grad_output.to(dtype).to("neuron"), x.to(dtype).to("neuron"), idx.to("neuron")
        )
        _close(out.cpu().float(), ref.float() if dtype == torch.bfloat16 else ref, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_max_unpool3d_e2e(self, dtype):
        # Build pooling indices from a max_pool3d_with_indices, then unpool.
        torch.manual_seed(0)
        x = torch.randn(1, 2, 4, 4, 4, dtype=torch.float32)
        vals, idx = torch.ops.aten.max_pool3d_with_indices.default(
            x, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False
        )

        def fn(v, i):
            return torch.nn.functional.max_unpool3d(
                v, i, [2, 2, 2], [2, 2, 2], [0, 0, 0], [4, 4, 4]
            )

        compiled_fn = torch.compile(fn, backend="neuron")
        ref = fn(vals, idx)
        out = compiled_fn(vals.to(dtype).to("neuron"), idx.to("neuron"))
        _close(out.cpu().float(), ref.float() if dtype == torch.bfloat16 else ref, dtype)

    # --- Forward index correctness tests (padding/dilation regression) ---

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize(
        "shape,kernel,stride,padding,dilation",
        [
            # Each tuple is (shape, kernel, stride, padding, dilation).
            # dilation=1 means no dilation (adjacent kernel elements).
            # padding=0 (regression check, baseline that should still work)
            ((1, 2, 8, 8), 2, 2, 0, 1),
            # padding > 0: exercises the index formula fix
            ((2, 3, 8, 8), 2, 2, 1, 1),  # batched, multi-channel
            ((1, 1, 4, 4), 2, 1, 1, 1),  # small input, overlapping windows
            # dilation > 1
            ((1, 1, 8, 8), 2, 1, 0, 2),  # dilation without padding
            ((1, 1, 8, 8), 2, 1, 1, 2),  # dilation with padding
            # asymmetric padding and kernel
            ((1, 1, 8, 10), (3, 2), (2, 1), (1, 0), 1),  # padH != padW, kH != kW
        ],
    )
    def test_max_pool2d_forward_indices(self, shape, kernel, stride, padding, dilation, dtype):
        """Verify compiled forward returns correct max values AND indices.

        Compares both values and indices against CPU reference. Designed to
        catch index-shift or otherwise incorrect index computation across
        padding, dilation, and asymmetric padding/kernel configurations.
        """

        def fn(x):
            return torch.nn.functional.max_pool2d(
                x, kernel, stride, padding, dilation, return_indices=True
            )

        compiled_fn = torch.compile(fn, backend="neuron")
        torch.manual_seed(42)
        x_cpu = torch.randn(*shape, dtype=dtype)
        x_neuron = x_cpu.clone().to("neuron")

        ref_vals, ref_idx = fn(x_cpu)
        out_vals, out_idx = compiled_fn(x_neuron)

        assert torch.equal(out_vals.cpu(), ref_vals)
        assert torch.equal(out_idx.cpu(), ref_idx)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize(
        "shape,kernel,stride,padding,dilation",
        [
            ((2, 3, 8, 8), 2, 2, 1, 1),  # batched, multi-channel
        ],
    )
    def test_max_pool2d_padded_backward_e2e(self, shape, kernel, stride, padding, dilation, dtype):
        """End-to-end gradient check with padding > 0.

        Unlike test_max_pool2d_backward_padded_fixed_indices which feeds
        CPU-computed indices to isolate the backward, this test runs the full
        compiled forward+backward pipeline to verify gradients are correct when
        the compiled forward produces the indices.
        """

        def fn(x):
            return torch.nn.functional.max_pool2d(x, kernel, stride, padding, dilation).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        torch.manual_seed(42)
        x_cpu = torch.randn(*shape, dtype=dtype, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)
        fn(x_cpu).backward()
        compiled_fn(x_neuron).backward()
        assert torch.equal(x_neuron.grad.cpu(), x_cpu.grad)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_max_pool2d_all_negative_float_with_padding(self, dtype):
        """Regression guard: padding never wins the argmax for finite negative floats.

        The original bug filled padding cells with 0.0. For an all-negative
        input, a 0.0 padding cell would be greater than the real data and win
        the argmax, pointing into the padding region. With the -inf padding
        fix, padding never wins for finite inputs. Input is a 2x2 plane filled with -1.0.

        Note: ties can still occur if the input data itself contains -inf,
        since the padding fill is also -inf. That edge case is not supported,
        which is consistent with upstream PyTorch.
        """
        torch.manual_seed(42)
        x_cpu = torch.full((1, 1, 2, 2), -1.0, dtype=dtype)

        def fn(x):
            return torch.nn.functional.max_pool2d(x, 2, 1, 1, return_indices=True)

        compiled_fn = torch.compile(fn, backend="neuron")
        ref_vals, ref_idx = fn(x_cpu)
        out_vals, out_idx = compiled_fn(x_cpu.clone().to("neuron"))

        # All real cells equal -1.0, so each window's argmax picks the first
        # (smallest flat index) real cell. Padding cells must never win.
        expected_idx = torch.tensor([[[[0, 0, 1], [0, 0, 1], [2, 2, 3]]]], dtype=torch.int64)

        assert torch.equal(out_vals.cpu(), ref_vals)
        assert torch.equal(out_idx.cpu(), ref_idx)
        assert torch.equal(out_idx.cpu(), expected_idx)

    @pytest.mark.parametrize(
        "dtype",
        [torch.int8, torch.int16, torch.uint8],
    )
    def test_max_pool2d_int_promotion_with_padding(self, dtype):
        """Verify integer inputs are promoted for padding and cast back.

        Integer types need promotion to a wider int so the padding fill value
        (original_min - 1) fits. The output dtype must match the input dtype.
        """
        torch.manual_seed(42)
        x_cpu = torch.randint(0, 50, (1, 1, 6, 6), dtype=dtype)

        def fn(x):
            return torch.nn.functional.max_pool2d(x, 3, 2, 1, return_indices=True)

        compiled_fn = torch.compile(fn, backend="neuron")
        ref_vals, ref_idx = fn(x_cpu)
        out_vals, out_idx = compiled_fn(x_cpu.clone().to("neuron"))

        assert (
            out_vals.cpu().dtype == dtype
        ), f"Output dtype {out_vals.cpu().dtype} != input dtype {dtype}"
        assert torch.equal(out_vals.cpu(), ref_vals)
        assert torch.equal(out_idx.cpu(), ref_idx)

    @pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.uint8])
    def test_max_pool2d_all_int_min_with_padding(self, dtype):
        """Verify padding never wins the argmax when all data is the dtype min.

        Input is a 2x2 plane filled entirely with iinfo(dtype).min. With
        padding=1 and a 2x2 window, the fill value (original_min - 1) must
        stay strictly below the real data so a padding cell never wins the
        argmax, even though the real values are at the dtype minimum.
        """
        torch.manual_seed(42)
        info = torch.iinfo(dtype)
        x_cpu = torch.full((1, 1, 2, 2), info.min, dtype=dtype)

        def fn(x):
            return torch.nn.functional.max_pool2d(x, 2, 1, 1, return_indices=True)

        compiled_fn = torch.compile(fn, backend="neuron")
        ref_vals, ref_idx = fn(x_cpu)
        out_vals, out_idx = compiled_fn(x_cpu.clone().to("neuron"))

        # All real cells equal info.min, so each window's argmax picks the
        # first (smallest flat index) real cell. Padding cells (info.min - 1)
        # must never win. Hand-computed expected indices for the 3x3 output:
        expected_idx = torch.tensor([[[[0, 0, 1], [0, 0, 1], [2, 2, 3]]]], dtype=torch.int64)

        assert out_vals.cpu().dtype == dtype
        assert torch.equal(out_vals.cpu(), ref_vals)
        assert torch.equal(out_idx.cpu(), ref_idx)
        assert torch.equal(out_idx.cpu(), expected_idx)

    @pytest.mark.parametrize(
        "dtype",
        [torch.int8, torch.int16, torch.uint8, torch.int32],
    )
    def test_max_pool2d_int_without_padding(self, dtype):
        """Verify integer maxpool works without padding (no fill value needed).

        Without padding, no fill value is required, so no integer
        promotion is needed.
        """
        torch.manual_seed(42)
        x_cpu = torch.randint(0, 100, (1, 1, 8, 8), dtype=dtype)

        def fn(x):
            return torch.nn.functional.max_pool2d(x, 2, 2, 0, return_indices=True)

        compiled_fn = torch.compile(fn, backend="neuron")
        ref_vals, ref_idx = fn(x_cpu)
        out_vals, out_idx = compiled_fn(x_cpu.clone().to("neuron"))

        assert (
            out_vals.cpu().dtype == dtype
        ), f"Output dtype {out_vals.cpu().dtype} != input dtype {dtype}"
        assert torch.equal(out_vals.cpu(), ref_vals)
        assert torch.equal(out_idx.cpu(), ref_idx)

    def test_max_pool2d_int64_rejected(self):
        """int64 input is rejected early with a clear error."""
        torch.manual_seed(42)
        x_cpu = torch.randint(0, 100, (1, 1, 6, 6), dtype=torch.int64)

        def fn(x):
            return torch.nn.functional.max_pool2d(x, 3, 2, 0, return_indices=True)

        compiled_fn = torch.compile(fn, backend="neuron")
        msg = "max_pool2d: int64 (or wider) integer input is not supported on Neuron."
        with pytest.raises(Exception, match=re.escape(msg)):
            compiled_fn(x_cpu.clone().to("neuron"))

    def test_max_pool2d_int32_with_padding_rejected(self):
        """Due to compiler bug, int32 with padding is rejected early with a clear error."""
        torch.manual_seed(42)
        x_cpu = torch.randint(0, 100, (1, 1, 6, 6), dtype=torch.int32)

        def fn(x):
            return torch.nn.functional.max_pool2d(x, 3, 2, 1, return_indices=True)

        compiled_fn = torch.compile(fn, backend="neuron")
        msg = (
            "max_pool2d: int32 up to int64 integer input with padding > 0 "
            "or ceil_mode enabled is not currently supported on Neuron."
        )
        with pytest.raises(Exception, match=re.escape(msg)):
            compiled_fn(x_cpu.clone().to("neuron"))

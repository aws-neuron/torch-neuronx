import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)
from torch_neuronx.python_ops.kernel_cache import kernel_cache_size


def _get_compilation_cache_size():
    """Number of entries in the C++ CompilationCache.

    C++ dispatch (the default) routes every compilation through the C++
    CompilationCache regardless of sync/async mode, so this is the counter that
    reflects a NEFF compile for linspace.
    """
    return torch_neuronx._C._get_compilation_cache_stats()["total_entries"]


class TestLinspace:
    def test_linspace_basic(self):
        """Test linspace basic"""
        device = "neuron"

        with track_neuron_ops():
            input = torch.linspace(0, 10, 5, dtype=torch.float32, device=device)
            input_cpu = torch.linspace(0, 10, 5, dtype=torch.float32)
            assert input.device.type == device
            assert input.device.type != input_cpu.device.type
            assert_op_runs_on_neuron("aten::linspace")

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64],
    )
    def test_linspace_different_dtypes(self, dtype):
        """Test linspace with different dtypes"""
        device = "neuron"
        with track_neuron_ops():
            input = torch.linspace(0, 10, 5, dtype=dtype, device=device)
            input_cpu = torch.linspace(0, 10, 5, dtype=dtype)
            assert input.dtype == dtype
            assert input.device.type == device
            assert input_cpu.device.type == "cpu"
            torch.testing.assert_close(input.cpu(), input_cpu)
            assert_op_runs_on_neuron("aten::linspace")

    def test_linspace_descending(self):
        """Test linspace with end < start (descending)"""
        device = "neuron"
        with track_neuron_ops():
            input = torch.linspace(10, 0, 5, dtype=torch.float32, device=device)
            input_cpu = torch.linspace(10, 0, 5, dtype=torch.float32)

            assert input_cpu.device.type == "cpu"
            assert input.device.type == device
            torch.testing.assert_close(input.cpu(), input_cpu)
            assert_op_runs_on_neuron("aten::linspace")

    def test_linspace_float_bounds(self):
        """Test linspace with fractional bounds"""
        device = "neuron"
        with track_neuron_ops():
            input = torch.linspace(0, 2.5, 6, dtype=torch.float32, device=device)
            input_cpu = torch.linspace(0, 2.5, 6, dtype=torch.float32)

            assert input.device.type == device
            torch.testing.assert_close(input.cpu(), input_cpu)
            assert_op_runs_on_neuron("aten::linspace")

    def test_linspace_requires_grad(self):
        """Test linspace with requires_grad"""
        device = "neuron"
        with track_neuron_ops():
            input = torch.linspace(0, 10, 5, dtype=torch.float32, requires_grad=True, device=device)
            assert input.requires_grad
            assert input.device.type == device
            assert_op_runs_on_neuron("aten::linspace")

    def test_linspace_equal_bounds(self):
        """Test linspace with equal start and end values"""
        device = "neuron"
        with track_neuron_ops():
            input = torch.linspace(start=5, end=5, steps=10, dtype=torch.float32, device=device)
            input_cpu = torch.linspace(5, 5, 10, dtype=torch.float32)

            assert input.device.type == device
            assert input_cpu.device.type == "cpu"
            assert input.numel() == 10
            torch.testing.assert_close(input.cpu(), input_cpu)
            assert_op_runs_on_neuron("aten::linspace")

    @assert_raises(RuntimeError, match="number of steps must be non-negative")
    def test_linspace_negative_steps(self):
        """Test linspace with negative steps"""
        device = "neuron"
        torch.linspace(0, 10, -1, dtype=torch.float32, device=device)

    @assert_raises(RuntimeError, match="inferred dtype")
    def test_linspace_complex_endpoint(self):
        """A complex endpoint with a real dtype raises the dtype-inference error.

        Neuron has no complex support. The lowering must raise the exact
        "torch.linspace(): inferred dtype ..." error, so it takes the
        complex-endpoint branch in torch_linspace instead of computing a ramp.
        """
        device = "neuron"
        torch.linspace(1j, 2j, 5, dtype=torch.float32, device=device)

    def test_linspace_explicit_layout(self):
        """Test linspace with explicitly specified layouts"""
        device = "neuron"
        with track_neuron_ops():
            # Test with explicit strided layout
            input_strided = torch.linspace(0, 10, 5, device=device, layout=torch.strided)
            assert input_strided.layout == torch.strided

            # Test default matches strided
            input_default = torch.linspace(0, 10, 5, device=device)
            assert input_default.layout == torch.strided

            # Verify both produce same values
            torch.testing.assert_close(input_strided.cpu(), input_default.cpu())

            assert_op_runs_on_neuron("aten::linspace")

    def test_linspace_out_dtype_inherited(self):
        """Verify linspace inherits dtype from out tensor when dtype is not specified."""
        out_cpu = torch.empty(5, dtype=torch.int32)
        out_neuron = torch.empty(5, dtype=torch.int32, device="neuron")

        result_cpu = torch.linspace(0, 10, 5, out=out_cpu)
        result_neuron = torch.linspace(0, 10, 5, out=out_neuron)

        assert (
            result_neuron.dtype == result_cpu.dtype
        ), f"dtype mismatch: neuron={result_neuron.dtype}, cpu={result_cpu.dtype}"

    def test_linspace_base_overload(self):
        """Test aten::linspace base overload dispatches correctly."""
        device = "neuron"
        with track_neuron_ops():
            result = torch.ops.aten.linspace(0, 10, 5, device=device)
            expected = torch.linspace(0, 10, 5)
            assert result.device.type == device
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::linspace")

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
    def test_linspace_base_overload_dtypes(self, dtype):
        """Test aten::linspace base overload with various dtypes."""
        device = "neuron"
        with track_neuron_ops():
            result = torch.ops.aten.linspace(0, 8, 5, dtype=dtype, device=device)
            expected = torch.linspace(0, 8, 5, dtype=dtype)
            assert result.dtype == dtype
            torch.testing.assert_close(result.cpu(), expected)

    # ── linspace-specific edge cases (no arange analog) ──────────────

    def test_linspace_single_step(self):
        """steps=1 returns a single-element tensor equal to start."""
        device = "neuron"
        with track_neuron_ops():
            input = torch.linspace(3.14, 42.0, 1, dtype=torch.float32, device=device)
            input_cpu = torch.linspace(3.14, 42.0, 1, dtype=torch.float32)
            assert input.device.type == device
            torch.testing.assert_close(input.cpu(), input_cpu)
            assert_op_runs_on_neuron("aten::linspace")

    def test_linspace_zero_steps(self):
        """steps=0 returns an empty tensor."""
        device = "neuron"
        with track_neuron_ops():
            input = torch.linspace(0, 10, 0, dtype=torch.float32, device=device)
            input_cpu = torch.linspace(0, 10, 0, dtype=torch.float32)
            assert input.device.type == device
            assert input.numel() == 0
            torch.testing.assert_close(input.cpu(), input_cpu)
            assert_op_runs_on_neuron("aten::linspace")

    def test_linspace_out_variant(self):
        """The .out variant writes into a pre-allocated tensor."""
        device = "neuron"
        with track_neuron_ops():
            out = torch.empty(5, device=device)
            result = torch.linspace(0, 10, 5, out=out)
            assert result is out
            assert_op_runs_on_neuron("aten::linspace")
            expected = torch.linspace(0, 10, 5)
            torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.bfloat16, torch.int8, torch.int16, torch.int32, torch.int64]
    )
    def test_linspace_out_noncontiguous_narrow(self, dtype):
        """A strided out= view is scattered respecting its stride/offset.

        Mirrors the upstream test_tensor_creation_ops.py::test_linspace subtest
        `torch.linspace(0, 3, 4, out=x.narrow(1, 1, 2))`. The result must land in
        the view's two columns (giving ((0, 0, 1), (0, 2, 3))), NOT be written
        contiguously from the view's storage offset (which gave (0, 0, 1, 2, 3, 0)).
        """
        device = "neuron"
        x = torch.zeros(2, 3, device=device, dtype=dtype)
        torch.linspace(0, 3, 4, out=x.narrow(1, 1, 2), dtype=dtype)
        expected = torch.tensor(((0, 0, 1), (0, 2, 3)), dtype=dtype)
        torch.testing.assert_close(x.cpu(), expected, atol=0, rtol=0)

    def test_linspace_out_noncontiguous_permuted(self):
        """A permuted (transposed) out= view receives values in logical order.

        Mirrors the upstream subtest that writes into `res.permute(2, 0, 1)`.
        """
        device = "neuron"
        dtype = torch.float32
        res = torch.zeros(3, 3, 10, device=device, dtype=dtype).permute(2, 0, 1)
        torch.linspace(0, 90, 90, out=res)
        expected = torch.linspace(0, 90, 90, dtype=dtype)
        torch.testing.assert_close(res.flatten().cpu(), expected)

    def test_linspace_out_noncontiguous_preserves_view_shape(self):
        """out= keeps the view's shape and storage — no flattening resize."""
        device = "neuron"
        x = torch.zeros(2, 3, device=device, dtype=torch.float32)
        view = x.narrow(1, 1, 2)
        result = torch.linspace(0, 3, 4, out=view)
        # PyTorch preserves the strided view's logical shape rather than
        # flattening it to the [4] result shape.
        assert tuple(result.shape) == (2, 2), result.shape
        assert result.stride() == (3, 1), result.stride()

    def test_linspace_no_recompilation(self):
        """Calling linspace twice with the same args reuses the compiled kernel.

        Asserted against the kernel/compilation cache counters rather than wall
        clock, so the test can't be flaked by a slow host or a warm page cache.
        """
        device = "neuron"
        with track_neuron_ops():
            # Unique steps so cache entries from earlier tests can't satisfy the
            # lookup and hide a recompilation here.
            torch.linspace(0, 1, 97, device=device)
            torch.neuron.synchronize()
            after_first_kernel_cache_size = kernel_cache_size()
            after_first_compilation_cache_size = _get_compilation_cache_size()

            torch.linspace(0, 1, 97, device=device)
            torch.neuron.synchronize()
            after_second_kernel_cache_size = kernel_cache_size()
            after_second_compilation_cache_size = _get_compilation_cache_size()

            assert (
                after_second_kernel_cache_size == after_first_kernel_cache_size
            ), "Kernel cache size increased, indicating recompilation occurred"
            assert (
                after_second_compilation_cache_size == after_first_compilation_cache_size
            ), "Compilation cache size increased, indicating recompilation occurred"

            assert_op_runs_on_neuron("aten::linspace")

    def test_linspace_no_recompilation_on_new_bounds(self):
        """Different `start`/`end` with the same `steps` reuse the compiled NEFF.

        `start`/`end` are runtime 0-dim tensor inputs, not static, so they feed
        the ramp arithmetic without changing the graph. A distinct bound pair
        must therefore hit the cache, not compile a new NEFF. This pins the fix
        for the recompilation trap of baking the bounds in as constants.
        """
        device = "neuron"
        with track_neuron_ops():
            torch.linspace(0, 1, 131, device=device)
            torch.neuron.synchronize()
            after_first = _get_compilation_cache_size()

            torch.linspace(5, 20, 131, device=device)
            torch.neuron.synchronize()
            after_second = _get_compilation_cache_size()

            assert after_second == after_first, (
                "Compilation cache grew for a new bound pair; start/end must be "
                "runtime inputs, not baked-in constants"
            )
            assert_op_runs_on_neuron("aten::linspace")

    def test_linspace_recompiles_on_new_static_steps(self):
        """A different `steps` compiles a new NEFF.

        Only `steps` is static: it sets the output shape, so each distinct value
        must get its own NEFF. `start`/`end` are runtime inputs and must NOT
        force a recompile (see test_linspace_no_recompilation_on_new_bounds).
        Together these pin that the cache key is neither too coarse (wrong values
        from a reused NEFF) nor too fine (a recompile per bound value).
        """
        device = "neuron"
        with track_neuron_ops():
            torch.linspace(0, 1, 93, device=device)
            torch.neuron.synchronize()
            after_first_compilation_cache_size = _get_compilation_cache_size()

            torch.linspace(0, 1, 94, device=device)
            torch.neuron.synchronize()
            after_second_compilation_cache_size = _get_compilation_cache_size()

            assert after_second_compilation_cache_size > after_first_compilation_cache_size, (
                "Compilation cache size did not grow for a new `steps` value, "
                "indicating an incorrect NEFF was reused"
            )

            assert_op_runs_on_neuron("aten::linspace")

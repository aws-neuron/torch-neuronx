import os

import pytest
import torch

pytest.importorskip("jax")

import torch_neuronx


@pytest.fixture
def stubbed_compilation(monkeypatch):
    """Stub JAX->HLO and HLO->NEFF compilation and execution to be fast and observable.

    Returns a dict with counters for 'hlo_compiles', 'neff_compiles', 'executes'.
    """
    from torch_neuronx.kernels.base import BaseNeuronKernel
    from torch_neuronx.python_ops.compilation.hlo_compiler import HloCompiler
    from torch_neuronx.python_ops.jax.compilation.jax_compiler import JaxCompiler
    from torch_neuronx.python_ops.jax.kernel import JaxKernel

    calls = {"hlo_compiles": 0, "neff_compiles": 0, "executes": 0}

    def fake_compile_to_hlo(self, jax_fn, sample_inputs, kwargs=None):
        calls["hlo_compiles"] += 1

        # Return a minimal stub with the method as_serialized_hlo_module_proto
        class _Stub:
            def as_serialized_hlo_module_proto(self):
                return b"hlo"

        return _Stub()

    def fake_get_kept_input_indices(self):
        return list(range(len(getattr(self, "_last_sample_inputs", []))))

    def fake_compile_to_neff(self, hlo_protobuf, ir_type="XLA"):
        calls["neff_compiles"] += 1
        return b"neff"

    def fake_execute(self, neff_bytes, inputs, outputs, op_name, has_collectives=False):
        calls["executes"] += 1
        # Perform a minimal effect on outputs to simulate kernel execution.
        # Note: Avoid calling ops implemented using BaseNeuronKernel as the kernel's execute method
        # calls back into this fake_execute function, causing an infinite recursion.
        for t in outputs.values():
            t = torch.transpose(t, 0, 0)

    # Patch classes so every kernel instance uses the stubs
    monkeypatch.setattr(JaxCompiler, "compile_to_hlo", fake_compile_to_hlo, raising=True)
    monkeypatch.setattr(
        JaxCompiler, "get_kept_input_indices", fake_get_kept_input_indices, raising=True
    )
    monkeypatch.setattr(HloCompiler, "compile_to_neff", fake_compile_to_neff, raising=True)
    monkeypatch.setattr(BaseNeuronKernel, "execute_neff", fake_execute, raising=True)

    return calls


class TestKernelCacheKey:
    @pytest.mark.skipif(
        os.environ.get("TORCH_NEURONX_SYNC_MODE") == "0",
        reason="The stubbed compilations for NEFF is only relevant in the legacy execution mode",
    )
    def test_out_parameter_does_not_change_cache_key(self, stubbed_compilation):
        """Use a fresh JaxKernel instance directly to avoid prior caches.

        Verifies that changing only 'out' does not trigger recompilation.
        """
        from torch_neuronx.python_ops.jax.kernel import JaxKernel
        from torch_neuronx.python_ops.jax.ops.factory_ops import _aten_ones as jax_ones

        # No static_argnames to keep cache key insensitive to dtype presence
        kernel = JaxKernel(jax_ones, op_name="aten::ones", static_argnums=(0,), static_argnames=())

        device = "neuron"
        size = (3, 4)

        # First call: expect a compile
        y1 = kernel(size, device=device)
        assert tuple(y1.shape) == size

        hlo_compiles_1 = stubbed_compilation["hlo_compiles"]
        neff_compiles_1 = stubbed_compilation["neff_compiles"]

        # Second call: provide mismatched out; should reuse cache (no new compile)
        out = torch.empty((1,), dtype=torch.float32, device=device)
        # Do not pass device/dtype together with out; follow PyTorch semantics
        y2 = kernel(size, out=out)
        assert y2 is out
        assert tuple(y2.shape) == size

        assert stubbed_compilation["hlo_compiles"] == hlo_compiles_1
        assert stubbed_compilation["neff_compiles"] == neff_compiles_1

    @pytest.mark.skipif(
        os.environ.get("TORCH_NEURONX_SYNC_MODE") == "0",
        reason="The stubbed compilations for NEFF is only relevant in the legacy execution mode",
    )
    def test_static_dtype_changes_cache_key(self, stubbed_compilation):
        """Changing a static kwarg (dtype) should produce a different cache key and recompile."""
        from torch_neuronx.python_ops.jax.kernel import JaxKernel
        from torch_neuronx.python_ops.jax.ops.factory_ops import _aten_ones as jax_ones

        kernel = JaxKernel(
            jax_ones, op_name="aten::ones", static_argnums=(0,), static_argnames=("dtype",)
        )

        size = (2, 2)
        device = "neuron"

        kernel(size, dtype=torch.float32, device=device)
        hlo1 = stubbed_compilation["hlo_compiles"]
        neff1 = stubbed_compilation["neff_compiles"]

        kernel(size, dtype=torch.bfloat16, device=device)
        assert stubbed_compilation["hlo_compiles"] == hlo1 + 1
        assert stubbed_compilation["neff_compiles"] == neff1 + 1

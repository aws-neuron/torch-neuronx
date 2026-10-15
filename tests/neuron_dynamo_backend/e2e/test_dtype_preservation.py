"""E2E: 64-bit output dtypes are preserved through torch.compile(backend="neuron").

Trainium runs 32-bit, but the executor restores the caller-visible dtype from
cast_spec. Because the demotion happens at the MLIR level (after cast_spec is
snapshotted), an fp64 or complex128 model output must come back as fp64 /
complex128, not silently narrowed. Guards the BC contract for the complex128
demotion work.
"""

import torch


class TestOutputDtypePreservationE2E:
    def test_fp64_output_preserved(self):
        def fn(a):
            return a.to(torch.float64) * 2.0

        x = torch.randn(4, 8)
        with torch.no_grad():
            cpu_out = fn(x)
            neuron_out = torch.compile(fn, backend="neuron", fullgraph=True)(x.to("neuron"))

        assert neuron_out.dtype == torch.float64
        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=1e-2, atol=1e-2)

    def test_complex128_output_preserved(self):
        def fn(a):
            xc = torch.view_as_complex(a.to(torch.float64).reshape(a.shape[0], -1, 2))
            return xc * xc  # complex128 output

        x = torch.randn(4, 8)
        with torch.no_grad():
            cpu_out = fn(x)
            neuron_out = torch.compile(fn, backend="neuron", fullgraph=True)(x.to("neuron"))

        assert neuron_out.dtype == torch.complex128
        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=1e-2, atol=1e-2)

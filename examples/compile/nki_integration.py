"""
NKI Integration Example

Demonstrates how to write NKI kernels and integrate them with torch.compile.

Run: python nki_integration.py
"""

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import torch

from torch_neuronx import nki_op, wrap_nki


@wrap_nki
@nki.jit
def add_kernel(a_input, b_input):
    """NKI kernel for tensor addition."""
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
    ix = nl.arange(128)[:, None]
    iy = nl.arange(512)[None, :]
    a_tile = nl.load(a_input[ix, iy])
    b_tile = nl.load(b_input[ix, iy])
    nl.store(c_output[ix, iy], value=a_tile + b_tile)
    return c_output


@nki_op("example::nki_add", mutates_args={})
def nki_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return add_kernel(a, b)


@torch.compile(backend="neuron", fullgraph=True)
def compute_with_nki(a, b):
    c = a * 2
    d = nki_add(c, b)
    return d


def main():
    a = torch.rand(128, 512, device="neuron")
    b = torch.rand(128, 512, device="neuron")
    result = compute_with_nki(a, b)

    expected = a * 2 + b
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
    print("NKI kernel integration successful!")
    print(f"Result shape: {result.shape}, device: {result.device}")


if __name__ == "__main__":
    main()

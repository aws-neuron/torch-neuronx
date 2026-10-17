"""
NKI Integration Example

Demonstrates how to write NKI kernels and integrate them with torch.compile.

Run: python nki_integration.py
"""

import math

import nki
import nki.language as nl
import torch

from torch_neuronx import nki_op


@nki.jit
def add_kernel(a_input, b_input):
    """Add two input arrays with an NKI kernel.

    Args:
        a_input (nl.ndarray): First input array.
        b_input (nl.ndarray): Second input array.

    Returns:
        nl.ndarray: Element-wise sum stored in shared HBM.
    """
    sz_p, sz_f = a_input.shape
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)

    for p in range(math.ceil(sz_p / nl.tile_size.pmax)):
        p_start = p * nl.tile_size.pmax
        p_end = min(p_start + nl.tile_size.pmax, sz_p)
        i_p = slice(p_start, p_end)

        a_tile = nl.load(a_input[i_p, 0:sz_f])
        b_tile = nl.load(b_input[i_p, 0:sz_f])
        nl.store(c_output[i_p, 0:sz_f], value=nl.add(a_tile, b_tile))

    return c_output


@nki_op("example::nki_add", mutates_args={})
def nki_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Add two tensors with the NKI kernel.

    Args:
        a (torch.Tensor): First input tensor.
        b (torch.Tensor): Second input tensor.

    Returns:
        torch.Tensor: Element-wise sum produced by the NKI kernel.
    """
    return add_kernel(a, b)


@torch.compile(backend="neuron", fullgraph=True)
def compute_with_nki(a, b):
    """Combine compiled tensor arithmetic with an NKI custom operator.

    Args:
        a (torch.Tensor): Input tensor to multiply by two.
        b (torch.Tensor): Input tensor to add to the scaled value.

    Returns:
        torch.Tensor: Value of ``a * 2 + b`` on the Neuron device.
    """
    c = a * 2
    d = nki_add(c, b)
    return d


def main():
    """Run and validate the compiled NKI integration example."""
    a = torch.rand(128, 512, device="neuron")
    b = torch.rand(128, 512, device="neuron")
    result = compute_with_nki(a, b)

    expected = a * 2 + b
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
    print("NKI kernel integration successful!")
    print(f"Result shape: {result.shape}, device: {result.device}")


if __name__ == "__main__":
    main()

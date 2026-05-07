"""
Dtype Handling Example

Neuron hardware operates on 32-bit types internally. The backend automatically
handles 64-bit types and preserves dtypes in outputs.

Note: Using 32-bit dtypes is recommended to avoid precision loss.

Run: python dtype_handling.py
"""

import torch


@torch.compile(backend="neuron", dynamic=False)
def add_tensors(x, y):
    return x + y


def main():
    # int64 inputs produce int64 outputs
    x_int64 = torch.tensor([1, 2, 3], dtype=torch.int64, device="neuron")
    y_int64 = torch.tensor([4, 5, 6], dtype=torch.int64, device="neuron")
    result_int64 = add_tensors(x_int64, y_int64)

    assert result_int64.dtype == torch.int64
    print(f"int64 + int64 = {result_int64.dtype}")

    # float64 inputs produce float64 outputs
    x_f64 = torch.tensor([1.0, 2.0], dtype=torch.float64, device="neuron")
    y_f64 = torch.tensor([0.5, 0.5], dtype=torch.float64, device="neuron")
    result_f64 = add_tensors(x_f64, y_f64)

    assert result_f64.dtype == torch.float64
    print(f"float64 + float64 = {result_f64.dtype}")

    # float32 is native - recommended for best performance
    x_f32 = torch.randn(4, dtype=torch.float32, device="neuron")
    y_f32 = torch.randn(4, dtype=torch.float32, device="neuron")
    result_f32 = add_tensors(x_f32, y_f32)

    assert result_f32.dtype == torch.float32
    print(f"float32 + float32 = {result_f32.dtype} (native, recommended)")


if __name__ == "__main__":
    main()

# Getting started with `torch.compile` on neuron

This document contains quickstart examples for using `torch.compile` with the Neuron backend on AWS Trainium and Inferentia devices.

## Prerequisites

- AWS Trainium or Inferentia instance
- torch-neuronx installed
- neuronx-cc compiler installed
- torch-mlir (neuron) installed
- Neuron Runtime and Collectives installed

## Quick Start

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel().to("neuron")
compiled_model = torch.compile(model, backend="neuron")

with torch.inference_mode():
    x = torch.randn(2, 10, device="neuron")
    y = compiled_model(x)

assert y.device.type == "neuron"
assert y.shape == (2, 5)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {y.shape}")
```

## Examples

The following standalone examples demonstrate specific features:

**[Device placement](device_placement.py)**

Example demonstrates how the Neuron backend handles tensors from
different devices.

CPU tensors are automatically transferred to Neuron under `torch.compile`. Results are on neuron.

Prefer using tensors on neuron devices.


**[Dtype Handling](dtype_handling.py)**

Working with different data types. See [Known Issues](#known-issues) for dtype limitations.

**[NKI Integration](nki_integration.py)**

Write kernels using [NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html) and integrate with `torch.compile`.

**[Distributed](distributed.py)**

Multi-process execution with torchrun


## Debugging common issues.

**Dynamic shapes**

neuron backend does not currently support dynamic shapes. Use `dynamic=False` to get around dynamic shapes.

Common errors: `SymInt` within FX Graphs or `Unbounded dynamism is disabled`.

**Example:**
```python
@torch.compile(backend="neuron", dynamic=False)
def static_shape_fn(x):
    return x + 1
```

**Device mismatch**

Mixing CPU and neuron tensors can result in errors from dynamo during tracing.

Ensure model and inputs are on same device.

**Example:**
```python
# Avoid
model = Model()
compiled_model = torch.compile(model, backend="neuron")

# Prefer
model = Model().to(torch.neuron.current_device())
compiled_model = torch.compile(model, backend="neuron", dynamic=False)
```


## Known Issues

### Unsupported dtypes
- `float64` and `int64` are not natively supported by Neuron hardware
- The backend automatically casts these to 32-bit types internally and casts back on output
- This may result in precision loss for `float64` operations
- Recommendation: Use `float32` and `int32` dtypes when possible

## Environment Variables Reference

See [Environment Variables](../../docs/environment_variables.md) for the complete list of configuration options.

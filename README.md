# torch-neuronx

torch-neuronx is a PyTorch backend for AWS Neuron devices. It uses PyTorch's [PrivateUse1](https://docs.pytorch.org/tutorials/advanced/privateuseone.html) mechanism to add a 'neuron' device to PyTorch.

torch-neuronx uses Bazel to build. Please read BAZEL_BUILD.md for instructions and more information.

## Limitations

torch-neuronx is currently in development with the following limitations:

1. Compilation and execution latency may be significant.
2. The neuron backend only supports `mode="default"` for torch.compile. Other modes like "reduce-overhead", "max-autotune", or "max-autotune-no-cudagraphs" are not supported.

## Installation

Run [prerequisites.sh](prerequisites.sh)

```sh

## Install neuron compiler from neuron repo
uv pip install 'neuronx-cc==2.*' --extra-index-url https://pip.repos.neuron.amazonaws.com --prerelease=allow --index-strategy unsafe-best-match

# Clone the repository
git clone <repository-url>
cd torch-neuronx

#If permissions are needed, execute below command
chmod +x ./tools/*

# Perform the build (defaults to Python version in .python-version)
uv run ./tools/build

# Or specify a Python version explicitly
uv run ./tools/build --python 3.12

```

## Usage

```python
import torch

# Use neuron device
x_neuron = torch.randn(10, 5, device='neuron')
linear = torch.nn.Linear(5, 3).to('neuron')
output = linear(x_neuron)
```

### torch.compile Backend

torch-neuronx includes a Dynamo backend for `torch.compile`:


```python
import torch

# Models and input tensors must be in neuron device
model = YourModel().to('neuron')
input_neuron = input_cpu.to('neuron')

# Supported: default mode and static shapes
compiled_model = torch.compile(model, backend="neuron", dynamic=False)

# Not supported: other modes or dynamic shapes
# compiled_model = torch.compile(model, backend="neuron", mode="reduce-overhead", dynamic=True)

# Run inference
output = compiled_model(input_neuron)
```

**Note**: 64-bit datatypes (int64/float64) are casted into 32-bit types internally. This may have accuracy impact when the tensor's dynamic range exceeds 32 bits. Use `float32` and `int32` dtypes when possible.

#### Supported Features

- Full graph compilation support
- Model parallelism via DTensor or PyTorch functional collectives
- 95% Aten ops and 80% of collective ops are supported.
- NKI kernel integration is supported
- Neuron Compiler options are passed via environment variable `NEURON_CC_FLAGS`
- Supported data types are documented in [public Neuron documentations](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/data-types.html#data-types)

#### Limitations

- Caching implementation is partial (StableHLO → NEFF only)
- FlexAttention support is limited
- Only contiguous tensors are supported
- torch.compile parameter support:

| Parameter | Support Status | Notes |
|-----------|----------------|-------|
| `model` | Supported | Module/function to optimize |
| `fullgraph` | Supported | `True` for optimal performance |
| `dynamic` | `False` only | Dynamic shapes and symbolic integers are not supported. |
| `mode` | `'default'` only | Mode options not available |
| `options` | Unsupported | Backend options not configurable |
| `disable` | Supported | Can disable compilation for testing |

#### Torch.compile Environment Variables
See [Environment Variables](../../docs/environment_variables.md) for the complete list of configuration options.

- `NEURON_CC_FLAGS` - Flags passed to neuronx-cc compiler
  ```bash
  export NEURON_CC_FLAGS="--model-type=transformer"
  ```

- `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS` - Preserve compilation artifacts directory (default: False)
  ```bash
  export TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1
  ```

- `TORCH_NEURONX_DEBUG_DIR` - Directory for compilation artifacts (default: /tmp/neuron_backend_<random>)
  ```bash
  export TORCH_NEURONX_DEBUG_DIR="./my_artifacts"
  ```

- `TORCH_NEURONX_ENABLE_NKI_SDPA` - Enable NKI kernel for scaled_dot_product_attention when seqlen % 2048 == 0 (default: True)
  ```bash
  export TORCH_NEURONX_ENABLE_NKI_SDPA=0
  ```


### Using NKI kernels in torch code

To integrate a nki kernel with torch code, there are 2 decorators:

1. nki.jit - First we need to trace the kernel code using this api from [NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.jit.html)
2. nki_op - This is used to define a custom PyTorch operator with a NKI kernel implementation. You can define a backward pass of the `nki_op` to support PyTorch autograd.

Below is an example of using nki kernels:

```python
import nki
import nki.language as nl
import nki.isa as nisa

import torch
from torch_neuronx import nki_op

@nki.jit
def add_kernel(x1, x2, y):
    x1_sbuf = nl.ndarray(x1.shape, x1.dtype, nl.sbuf)
    x2_sbuf = nl.ndarray(x2.shape, x2.dtype, nl.sbuf)
    y_sbuf = nl.ndarray(y.shape, y.dtype, nl.sbuf)

    nisa.dma_copy(x1_sbuf, x1)
    nisa.dma_copy(x2_sbuf, x2)
    nisa.tensor_tensor(y_sbuf, x1_sbuf, x2_sbuf, nl.add)
    nisa.dma_copy(y, y_sbuf)
    return y

@nki_op("test::nki_add", mutates_args={"y"})
def nki_add(x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return add_kernel(x1, x2, y)


x1 = torch.randn(128, device="neuron")
x2 = torch.randn(128, device="neuron")
y = torch.empty(128, device="neuron")

nki_add(x1, x2, y)
```

#### Using NKI kernels with torch.compile
NKI kernels are compatible with torch.compile. Our backend integrates them into the compiled graph, avoiding the graph breaks and reducing the runtime overhead.

```python

@torch.compile
def compiled_add_op(x1, x2):
    out = torch.empty_like(x1)
    return nki_add(x1, x2, out)

# NKI kernel

x1 = torch.randn(128, device="neuron")
x2 = torch.randn(128, device="neuron")

y = compiled_add_op(x1,x2)
```

## Examples

### Llama

```sh
cd examples/llama
python llama.py
```

## Documentation

```bash
./tools/build --docs
# Output: build/docs/html/index.html
```

## Resources

- [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

# torch-neuronx

torch-neuronx is PyTorch backend for AWS Neuron devices. It uses PyTorch's [PrivateUse1](https://docs.pytorch.org/tutorials/advanced/privateuseone.html) mechanism to add 'neuron' device to PyTorch.

torch-neuronx uses Bazel to build. Please read BAZEL_BUILD.md for instructions and more information.

## Limitations

torch-neuronx is currently in development with the following limitations:

1. JAX installation is currently required as a dependency.
2. Compilation and execution latency may be significant.

## Installation

Run [prerequisites.sh](prerequisites.sh)

```sh
# Clone the repository
git clone <repository-url>
cd torch-neuronx

#If permission are needed, execute below command
chmod +x ./tools/*

# Install in editable mode for development
pip install -e .

# Perform the build
./tools/build

```

## Usage

```python
import torch

# Register 'neuron' device
import torch_neuronx

# Use neuron device
x_neuron = torch.randn(10, 5, device='neuron')
linear = torch.nn.Linear(5, 3).to('neuron')
output = linear(x_neuron)
```

### torch.compile Backend

torch-neuronx includes a Dynamo backend for `torch.compile`:

```python
import torch

# Compile
model = YourModel()
compiled_model = torch.compile(model, backend="neuron")

# Run inference
output = compiled_model(input_tensor)
```


### Using NKI kernels in torch code

To integrate a nki kernel with torch code, there are 3 decorators:

1. nki.jit - First we need to trace the kernel code using this api from [neuronxcc](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.jit.html)
2. wrap_nki - This is used to wrap the nki kernel calls
3. nki_op - Finally the nki kernel can be used inside a nki op and used like any other torch operator.

Below is an example of using nki kernels:

```python
from neuronxcc import nki

import torch
from torch_neuronx import nki_op, wrap_nki

@wrap_nki
@nki.jit
def add_kernel(x1, x2, y: nt.mutable_tensor):
    import neuronxcc.nki.language as nl

    x1_tile = nl.load(x1[0:128])
    x2_tile = nl.load(x2[0:128])
    y_tile = x1_tile + x2_tile
    nl.store(y[0:128], value=y_tile)
    return y

@nki_op("test::nki_add", mutates_args={"y"})
def nki_add(x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    add_kernel(x1, x2, y)


x1 = torch.randn(128, device="neuron")
x2 = torch.randn(128, device="neuron")
y = torch.empty(128, device="neuron")

nki_add(x1, x2, y)

```

## Examples

### Llama

```sh
cd examples/llama
python llama.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

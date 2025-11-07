# torch-neuron

torch-neuron is PyTorch backend for AWS Neuron devices. It uses PyTorch's [PrivateUse1](https://docs.pytorch.org/tutorials/advanced/privateuseone.html) mechanism to add 'neuron' device to PyTorch.

## Installation

Run [prerequisites.sh](prerequisites.sh)

```sh
# Clone the repository
git clone <repository-url>
cd torch-neuron

# Install
pip install .
```

## Usage

```python
import torch

# Register 'neuron' device
import torch_neuron

# Use neuron device
x_neuron = torch.randn(10, 5, device='neuron')
linear = torch.nn.Linear(5, 3).to('neuron')
output = linear(x_neuron)
```

## Examples

### Llama

```sh
cd examples/llama
python llama.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

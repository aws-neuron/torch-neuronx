# torch-neuronx (Beta)

torch-neuronx is the native PyTorch backend for AWS Trainium. It adds a `neuron` device to upstream PyTorch through the [PrivateUse1](https://docs.pytorch.org/tutorials/advanced/privateuseone.html) mechanism. You use the PyTorch APIs that you already know. Move a model to `neuron`, then train or run it.

- **Eager by default.** Operations run one at a time on the device, as they do on any other PyTorch device. There is no graph to trace before you see a result.
- **`torch.compile` when you want it.** The `neuron` backend for `torch.compile` compiles a model or a region of a model for better performance.
- **Distributed training.** `torch.distributed`, DTensor, and functional collectives run on Neuron devices, for data, tensor, and fully sharded parallelism.
- **Custom kernels.** Call [NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html) kernels from PyTorch code, in eager mode and inside `torch.compile`.

> **If you use the XLA-based torch-neuronx (PyTorch/XLA),** see [PyTorch support on Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/archive/frameworks/torch/index.html). This repository is the PyTorch-native backend.

## Installation

You need a Trn2 or Trn3 instance, and Python 3.12, 3.13, or 3.14.

**Neuron DLAMI or Deep Learning Container (recommended).** torch-neuronx is already installed. Activate the Python environment that comes with it.

**Other Linux instance.** Install the Neuron driver and runtime for your OS, and then install torch-neuronx:

<details>
<summary>Ubuntu 22.04, 24.04, or 26.04</summary>

```bash
. /etc/os-release
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo gpg --dearmor -o /usr/share/keyrings/neuron.gpg
echo "deb [signed-by=/usr/share/keyrings/neuron.gpg] https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" | sudo tee /etc/apt/sources.list.d/neuron.list
sudo apt-get update -y
sudo apt-get install -y linux-headers-$(uname -r) aws-neuronx-dkms aws-neuronx-collectives aws-neuronx-runtime-lib aws-neuronx-tools
export PATH=/opt/aws/neuron/bin:$PATH
```

</details>

<details>
<summary>Amazon Linux 2023</summary>

```bash
sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOT
[neuron]
name=Neuron YUM Repository
baseurl=https://yum.repos.neuron.amazonaws.com
enabled=1
metadata_expire=0
EOT
sudo rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
sudo dnf install -y "kernel-devel-uname-r = $(uname -r)" aws-neuronx-dkms aws-neuronx-collectives aws-neuronx-runtime-lib aws-neuronx-tools
export PATH=/opt/aws/neuron/bin:$PATH
```

</details>

<details>
<summary>EFA driver, for multi-node training</summary>

```bash
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
tar -xf aws-efa-installer-latest.tar.gz
cd aws-efa-installer && sudo bash efa_installer.sh --yes
```

</details>

```bash
python3 -m venv ~/neuron_venv && source ~/neuron_venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install "torch-neuronx==2.14.*" "neuronx-cc==2.*" "nki>=0.6" --extra-index-url https://pip.repos.neuron.amazonaws.com
```

Keep the version pins. Earlier `torch-neuronx` versions are the XLA-based product.

To check the installation, run `python -m torch_neuronx.collect_env`. The `device usable from torch` line must show `yes`.

To build from source, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick Start

After you install torch-neuronx, run this script on a Trainium instance. It runs a model in eager mode, and then compiles the same model with the `neuron` backend:

```python
import torch  # PyTorch loads torch-neuronx automatically.

model = torch.nn.Linear(16, 8).to("neuron").eval()
inputs = torch.randn(4, 16, device="neuron")

with torch.no_grad():
    eager_output = model(inputs)
    compiled_model = torch.compile(model, backend="neuron", dynamic=False)
    compiled_output = compiled_model(inputs)

print(tuple(eager_output.cpu().shape), tuple(compiled_output.cpu().shape))
```

The script prints `(4, 8) (4, 8)`. The first run compiles the operations and the graph, so it takes longer than later runs.

To run your own model, move the model and its inputs to the `neuron` device. For training, distributed training, and NKI kernels, see the [TorchNeuron tutorials](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/tutorials/index.html) and the [examples](#examples).

## Features

The [API compatibility guide](docs/torch_api_compatibility.md) lists each PyTorch API as supported, supported with CPU fallback, or not supported.

**`torch.compile`:**

- Full graph compilation
- Model parallelism through DTensor or PyTorch functional collectives
- NKI kernels inside a compiled graph
- The [data types](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/data-types.html#data-types) that Neuron supports

| `torch.compile` parameter | Support | Notes |
|---|---|---|
| `model` | Supported | Module or function to optimize |
| `fullgraph` | Supported | `True` gives the best performance |
| `dynamic` | `False` only | Dynamic shapes and symbolic integers are not supported |
| `mode` | `"default"` only | Other modes log a warning and run as `"default"` |
| `options` | Limited | The backend reads a small set of Neuron-specific keys. Inductor options have no effect. Pass compiler flags in `NEURON_CC_FLAGS` |
| `disable` | Supported | Turns off compilation, for testing |

**Experimental:** adaptive eager execution fuses eligible eager operations into one compiled unit. It is off by default. To try it, set `TORCH_NEURONX_ENABLE_CONCATENATION=1` before you start Python.

**Known limitations:**

- The first call of an operation or a compiled graph includes compilation time. This can be significant for a large model. The persistent cache reuses compiled NEFFs in later runs.
- `torch.compile` needs static shapes. Pass `dynamic=False`. Each new input shape compiles again. Pad or bucket the inputs to limit the number of shapes.
- `torch.compile` supports only contiguous tensors. The backend copies a non-contiguous input to a contiguous tensor on each call.
- FlexAttention support is limited.
- float64 runs as float32. This can change the result when a value needs 64-bit precision. int64 support depends on the operation. Use `float32` and `int32` when possible.
- TorchAO integration, for MXFP8 and other composite data types, is not available in this release.

## Using NKI kernels

Define the kernel with `nki.jit`. Then register the kernel as a PyTorch operator with `nki_op`, and call the operator like any other PyTorch operation:

```python
import nki
import nki.isa as nisa
import nki.language as nl
import torch
from torch_neuronx import nki_op

@nki.jit
def add_kernel(a, b):
    a_sbuf = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_sbuf, src=a)
    b_sbuf = nl.ndarray(b.shape, dtype=b.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_sbuf, src=b)
    c_sbuf = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=c_sbuf, data1=a_sbuf, data2=b_sbuf, op=nl.add)
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=c_sbuf)
    return out

@nki_op("example::add", mutates_args={})
def nki_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return add_kernel(a, b)

a = torch.rand(128, 128, device="neuron")
b = torch.rand(128, 128, device="neuron")
print(torch.allclose(nki_add(a, b).cpu(), (a + b).cpu()))
```

The script prints `True`. An `nki_op` also works inside `torch.compile`. The backend puts the kernel into the compiled graph, so the graph does not break. To train through the kernel, register a backward pass with [`torch.library.register_autograd`](https://docs.pytorch.org/docs/stable/library.html#torch.library.register_autograd). For in-place kernels and `wrap_nki`, see [How to work with NKI kernels](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/guides/how-to-work-with-nki.html).

## Examples

| Example | Description |
|---|---|
| [compile/QUICKSTART.md](examples/compile/QUICKSTART.md) | Getting started with `torch.compile` on Neuron |
| [llama](examples/llama) | Llama-style model: training step and text generation |
| [torch_compile/whisper](examples/torch_compile/whisper) | Whisper speech-to-text |
| [torch_compile/qwen2](examples/torch_compile/qwen2) | Qwen2 model |
| [fused_attention](examples/fused_attention) | Scaled dot-product and multi-head attention |
| [simple_moe](examples/simple_moe) | Mixture of Experts |
| [gpt2-train-loop](examples/gpt2-train-loop) | GPT-2 training loop |
| [simple_mlp_tpxfsdp](examples/simple_mlp_tpxfsdp) | Tensor parallel and FSDP |
| [multi_node_tests](examples/multi_node_tests) | Multi-node distributed training |
| [simple_asr](examples/simple_asr) | Conformer ASR training |

## Documentation

The full documentation for torch-neuronx is in the [AWS Neuron documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/index.html).

| Neuron documentation | What is inside |
|---|---|
| [Get started](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/get-started/quickstart-torch-neuron.html) | Install torch-neuronx and run a first model |
| [Tutorials](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/tutorials/index.html) | Training and distributed training |
| [Guides](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/guides/index.html) | Setup, NKI kernels, `torch.distributed`, memory profiling |
| [torch.compile](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/compile/index.html) | How `torch.compile` works on Neuron, and how to optimize with it |
| [Reference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/reference/index.html) | API compatibility, supported operators, environment variables |
| [Troubleshooting](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/troubleshooting.html) | Common errors and fixes |

**Reference in this repository:**

| File | What is inside |
|---|---|
| [docs/torch_api_compatibility.md](docs/torch_api_compatibility.md) | Support status of each PyTorch API |
| [docs/environment_variables.md](docs/environment_variables.md) | Environment variables for configuration and debugging |
| [docs/decompositions.md](docs/decompositions.md) | Operator decompositions |
| [docs/numa_optimization.md](docs/numa_optimization.md) | NUMA placement for better host performance |
| [docs/statistics/torch_neuronx_stats_readme.md](docs/statistics/torch_neuronx_stats_readme.md) | Runtime statistics and memory metrics |

**Contributing to torch-neuronx:**

| File | What is inside |
|---|---|
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute, build, and test |
| [BAZEL_BUILD.md](BAZEL_BUILD.md) | Build from source with Bazel |
| [AI_POLICY.md](AI_POLICY.md) | Rules for using AI tools in contributions |
| [AGENTS.md](AGENTS.md) | Rules for AI agents that work in this repository |
| [.external/README.md](.external/README.md) | Agent skills for AI coding tools |
| [SECURITY.md](SECURITY.md) | How to report a security issue |

## Version

The torch-neuronx version has the format `<PyTorch major.minor>.<torch-neuronx major.minor>.<build>`. For example, `2.14.3.0.4527` is torch-neuronx 3.0, build 4527, for PyTorch 2.14.

| torch-neuronx | PyTorch | Python | Neuron SDK | Instance Support | Status |
|---|---|---|---|---|---|
| 2.14.3.0.x (latest) | 2.14.0 | 3.12, 3.13, 3.14 | 2.33 | Trn2, Trn3 | Beta |

For the features and known gaps of this release, see [release-notes/2.33.0.md](release-notes/2.33.0.md).

## Issues

Report bugs or request features: [GitHub Issues](https://github.com/aws-neuron/torch-neuronx/issues). To report a security issue, follow [SECURITY.md](SECURITY.md).

## Code of Conduct

This project follows the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct). For more information, see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact <opensource-codeofconduct@amazon.com>.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

# Qwen3 Inference with torch.compile using TorchTitan

## Build torch-neuronx from source

See Installation section in [README.md](../../../README.md)

## Setup TorchTitan
```shell
cd ~
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
pip install -r requirements.txt
pip install transformers
```

## Download Qwen3 Model Assets
```shell
# Download model weights and tokenizer
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-8B --assets tokenizer safetensors config
```

## Inference with torch.compile

Create inference script `:

```python
import torch
import torch_neuronx
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.device_mesh import init_device_mesh

# Load Qwen3 model and tokenizer
model_name = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply tensor parallel based on NEURON_RT_NUM_CORES
import os
torch.distributed.init_process_group()
num_cores = torch.distributed.get_world_size()
tp_mesh = init_device_mesh("neuron", (num_cores,))
assert tp_mesh is not None, "Mesh initialization failed"
parallel_plan = {}
for name, module in model.named_modules():
    if "mlp.gate_proj" in name or "mlp.up_proj" in name:
        parallel_plan[name] = ColwiseParallel()
    elif "mlp.down_proj" in name:
        parallel_plan[name] = RowwiseParallel()
model = parallelize_module(model, tp_mesh, parallel_plan)
model = model.to('neuron')

# Compile model with neuron backend
compiled_model = torch.compile(model, backend="neuron")

# Prepare input
text = "Hello, how are you?"
input_ids = tokenizer.encode(text, return_tensors="pt").to('neuron')

# Run inference
with torch.no_grad():
    output = compiled_model.generate(input_ids, max_length=50, do_sample=False)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Response: {response}")
```

## Environment Setup and Inference
```shell
export NEURON_RT_NUM_CORES=32  # Adjust based on your instance
export NEURON_CC_FLAGS="--model-type=transformer"  # Trigger compiler heuristics better tuned for Transformer model architecture
torchrun --nproc-per-node=$NEURON_RT_NUM_CORES qwen3_inference.py
```

The first inference will trigger compilation which may take longer. Subsequent runs will use cached graphs for faster execution.

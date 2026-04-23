# BF16 Training: Avoiding dtype promotion pitfalls

When training in bf16 on Neuron using `model.to(torch.bfloat16)`, be aware
that PyTorch type promotion silently upcasts `bf16 * fp32 -> fp32`. This
causes `RuntimeError` when the fp32 result is passed to a bf16 layer.

## The Problem

```python
# Model in bf16
model = MyModel().to(torch.bfloat16).to("neuron")

# Scheduling math produces fp32
sigma = timestep.float() / 1000  # fp32

# Type promotion: bf16 * fp32 -> fp32!
noisy = (1 - sigma) * latents_bf16 + sigma * noise_bf16  # fp32 output!

# CRASH: fp32 input vs bf16 Conv3d weights
output = model(noisy)
```

## Solution 1: Explicit cast (recommended for training)

```python
noisy = ((1 - sigma) * latents + sigma * noise).to(latents.dtype)
output = model(noisy)  # works: bf16 -> bf16
```

## Solution 2: torch.autocast (recommended for inference)

```python
with torch.autocast(device_type="neuron", dtype=torch.bfloat16):
    output = model(noisy_fp32)  # auto-cast handles it
```

## Run the example

```bash
python bf16_training.py
```

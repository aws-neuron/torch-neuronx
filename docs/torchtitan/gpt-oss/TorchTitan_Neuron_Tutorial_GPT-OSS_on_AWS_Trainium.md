# **TorchTitan Neuron Tutorial: GPT-OSS on AWS Trainium**

This tutorial demonstrates how to train a GPT-OSS model using TorchTitan on AWS Trainium (Trn2) instances. The guide covers environment setup, configuration, and training execution.

## Configuration

This section outlines the model, training, and hardware configurations used in this tutorial.

### Model Configuration

We use a modified version of the standard GPT-OSS 20B configuration with the following adjustments to optimize for single-core Trainium training:

* `1` layer (instead of `24` layers)
* `8192` vocab size (instead of `201088`)

### Training Configuration

The training uses minimal batch settings suitable for debugging and initial testing:

* Batch size of `1`
* Sequence length of `2048`

### Hardware Configuration

This tutorial targets the following hardware setup:

* Trn2
* Single core (with LNC2)
* 24 GB HBM

## Environment Setup

Follow these steps to set up your environment. You'll need to configure both TorchNeuronx and TorchTitan.

### TorchNeuronx Setup

First, set up the TorchNeuronx environment with the necessary dependencies:

```shell
cd ~
git clone https://github.com/aws-neuron/torch-neuronx.git
cd torch-neuronx
git apply ~/torch-neuronx/docs/torchtitan/gpt-oss/TorchNeuronx.diff
./install_bazelisk.sh
pip install uv
uv pip install --system -e . --force-reinstall
./tools/build
```

#### Key Differences from TorchNeuronx Upstream

This setup uses modified versions of the upstream dependencies:

* **PyTorch version**: PyTorch Nightly (instead of PyTorch 2.8/2.9)
    * Additional compiler flag required: `--param=destructive-interference-size=64`

For a complete diff, see: [TorchNeuronx.diff](TorchNeuronx.diff)

### TorchTitan Setup

Next, clone and configure the TorchTitan repository with Neuron-specific modifications:

```shell
cd ~
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
git checkout 29eb9102d73bb6f8b8bb294ba9f5c816cad84adf
git apply ~/torch-neuronx/docs/torchtitan/gpt-oss/TorchTitan.diff
uv pip install --system -r requirements.txt
uv pip install --system -U tensorboard
```

#### Key Differences from TorchTitan Upstream

The modified TorchTitan branch includes both functional and performance optimizations:

**Functional changes:**

* Model configuration updated as described in the Configuration section
* CUDA assertion removed for `torch._grouped_mm` to enable Neuron compatibility
* Triton kernel for permuting indices replaced with CPU variant

**Performance optimizations:**

* FlexAttention replaced with SDPA (Scaled Dot-Product Attention)
    * Note: This temporarily skips sliding window attention and attention sinks
* RoPE (Rotary Position Embedding) replaced with optimized kernel implementation
* Gate/up expert weights reordered for better memory access patterns
* Slicing operations replaced with split for improved performance
* Explicit casting added before gather/scatter_add operations

For a complete diff, see: [TorchTitan.diff](TorchTitan.diff)

For a more detailed explanation of each of these changes, see [Appendix A](#appendix-a---torchtitan-changes-explained)

## Training GPT-OSS with TorchTitan

Once your environment is set up, you can start training using the following commands.

### Running the Training

Execute the training script with the debug configuration:

```shell
export TRAIN_FILE="torchtitan.train"
export CONFIG_FILE="torchtitan/experiments/gpt_oss/train_configs/debug_model.toml"
torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter 0 --role rank --tee 3 \
-m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE}
```

### Command Explanation

* `--nproc_per_node=1`: Uses a single process (matching our single-core configuration)
* `--rdzv_backend c10d`: Uses the c10d rendezvous backend for distributed coordination
* `--rdzv_endpoint="localhost:0"`: Sets the rendezvous endpoint for process coordination
* `--local-ranks-filter 0`: Filters output to show only rank 0
* `--role rank --tee 3`: Configures logging output
* `${CONFIG_FILE}`: Points to the debug model configuration file

### Expected Behavior

During training, you should ideally observe:

* Loss values decreasing over iterations (at least for the first few iterations)
* Memory usage staying within the 24 GB HBM limit (at least for the first few iterations)
* MFU around 2% once all recompiles are eliminated

## Issues/Limitations

* Requires code changes for enablement and performance reasons
* Temporarily skips certain functionality
    * Specifically, sliding window attention and attention sinks are removed
* Crashes when using 16 experts since `>=` isn’t accurate with certain data/shapes
* Crashes when using 32 experts due to OOM (out of memory)
* Has no thorough accuracy validation (besides loss decreasing)
* Requires recompiles due to changing shapes in the MoE section
* Crashes/inefficiences depending on the compiler version

## Next Steps

After successfully running this tutorial, consider the following.

### Using a Custom Tokenizer

Follow the instructions at https://github.com/pytorch/torchtitan/tree/main?tab=readme-ov-file#downloading-a-tokenizer to download a tokenizer:

```shell
# Get your HF token from https://huggingface.co/settings/tokens
# GPT-OSS tokenizer
python scripts/download_hf_assets.py --repo_id openai/gpt-oss-20b --assets tokenizer --hf_token=...
```

Then, add this tokenizer to the appropriate config file (e.g., `torchtitan/experiments/gpt_oss/train_configs/debug_model.toml`)

# Appendix A - TorchTitan Changes Explained

This appendix explains every line changed from the TorchTitan upstream, organized by category.

* * *


## 1. Configuration Changes

### File: `torchtitan/experiments/gpt_oss/__init__.py`

**Lines removed:**

```diff
-        dim=256,
-        n_layers=4,
```

**Lines added:**

```diff
+        n_layers=1,
```

**Purpose:** Model configuration update - reduced number of layers from 4 to 1 for debug model, removed explicit dim parameter (using default).

* * *

**Lines changed:**

```diff
-            num_experts=8,
+            num_experts=32,
```

**Purpose:** Model configuration update - increased number of MoE experts from 8 to 32.

* * *

**Lines added:**

```diff
+        vocab_size=8192,
+        # n_heads=64//4,
+        # n_kv_heads=8//4,
+        # moe_inter_dim=2880//4,
+        use_flex_attn=False,
```

**Purpose:** Model configuration update - set explicit vocab size to 8192, disabled FlexAttention (switching to SDPA), added commented-out options for reduced head counts.

* * *


### File: `torchtitan/experiments/gpt_oss/train_configs/debug_model.toml`

**Lines changed:**

```diff
-enable_profiling = false
+enable_profiling = true
```

**Purpose:** Configuration update - enabled profiling for performance analysis.

* * *

**Lines changed:**

```diff
-profile_freq = 10
+profile_freq = 5
```

**Purpose:** Configuration update - increased profiling frequency (profile every 5 steps instead of 10).

* * *

**Lines added:**

```diff
+profiler_active = 1
+profiler_warmup = 1
```

**Purpose:** Configuration update - set profiler to 1 warmup step and 1 active step for faster profiling cycles.

* * *

**Lines changed:**

```diff
-local_batch_size = 8
+local_batch_size = 1
```

**Purpose:** Configuration update - reduced batch size from 8 to 1 for debugging/testing.

* * *

**Lines added:**

```diff
+# seq_len = 4096
```

**Purpose:** Configuration update - added commented option for longer sequence length.

* * *

**Lines changed:**

```diff
-steps = 10
+steps = 200
```

**Purpose:** Configuration update - increased training steps from 10 to 50.

* * *


## 2. CUDA Assertion Removal for Neuron Compatibility

### File: `torchtitan/experiments/gpt_oss/model/args.py`

**Lines commented out:**

```diff
-        if self.moe_args.use_grouped_mm and not has_cuda_capability(9, 0):
-            logger.warning(
-                "Failed to use grouped mm, which is only supported on SM90 or later",
-            )
-            self.moe_args.use_grouped_mm = False
+        # if self.moe_args.use_grouped_mm and not has_cuda_capability(9, 0):
+        #     logger.warning(
+        #         "Failed to use grouped mm, which is only supported on SM90 or later",
+        #     )
+        #     self.moe_args.use_grouped_mm = False
```

**Purpose:** CUDA assertion removed - disabled CUDA capability check for torch._grouped_mm to enable Neuron compatibility (grouped_mm can work on Neuron devices).

* * *


## 3. Triton Kernel Replacement with CPU Variant

### File: `torchtitan/models/moe/kernels.py`

**Lines changed:**

```diff
-    # device = tokens_per_expert_group.device
+    device = tokens_per_expert_group.device
```

**Purpose:** Triton kernel replacement - re-enabled device tracking for CPU-based permutation.

* * *

**Lines changed:**

```diff
-    return permuted_indices
+    return permuted_indices.to(device)
```

**Purpose:** Triton kernel replacement - added explicit device transfer to return indices on correct device after CPU computation.

* * *


### File: `torchtitan/models/moe/utils.py`

**Lines added:**

```diff
+            use_cpu=True,
```

**Purpose:** Triton kernel replacement - forced CPU execution for permute indices generation (replacing Triton kernel).

* * *


## 4. FlexAttention Replaced with SDPA

### File: `torchtitan/experiments/gpt_oss/model/model.py`

**Lines added:**

```diff
+    ScaledDotProductAttentionWrapper,
```

**Purpose:** FlexAttention → SDPA - imported SDPA wrapper to replace FlexAttention.

* * *

**Lines changed:**

```diff
-        self.inner_attention = FlexAttentionWrapper()
+        self.inner_attention = ScaledDotProductAttentionWrapper()
```

**Purpose:** FlexAttention → SDPA - replaced FlexAttention with ScaledDotProductAttention for better compatibility.

* * *

**Lines commented out:**

```diff
-        assert isinstance(attention_masks, BlockMask), attention_masks
-        output, lse = self.inner_attention(
-            xq, xk, xv, block_mask=attention_masks, scale=None, return_lse=True
+        # assert isinstance(attention_masks, BlockMask), attention_masks
+        output = self.inner_attention(
+            xq, xk, xv
         )
```

**Purpose:** FlexAttention → SDPA - removed BlockMask assertion and LSE (log-sum-exp) return value, simplified to basic SDPA call (temporarily skips sliding window attention).

* * *

**Lines commented out:**

```diff
-        sink_scale = torch.sigmoid(lse - self.sinks.view(1, -1, 1)).unsqueeze(-1)
-        output = output * sink_scale.to(output.dtype)
+        # sink_scale = torch.sigmoid(lse - self.sinks.view(1, -1, 1)).unsqueeze(-1)
+        # output = output * sink_scale.to(output.dtype)
```

**Purpose:** FlexAttention → SDPA - disabled attention sink rescaling (temporarily skips attention sinks feature since LSE is not available from SDPA).

* * *

**Lines commented out:**

```diff
-        if self.use_sliding_attention:
-            layer_mask = attention_masks.get("sliding_window_mask", None)
-        else:
-            layer_mask = attention_masks.get("basic_mask", None)
-        assert layer_mask is not None
+        # if self.use_sliding_attention:
+        #     layer_mask = attention_masks.get("sliding_window_mask", None)
+        # else:
+        #     layer_mask = attention_masks.get("basic_mask", None)
+        # assert layer_mask is not None
```

**Purpose:** FlexAttention → SDPA - disabled sliding window mask extraction (temporarily skips sliding window attention).

* * *

**Lines changed:**

```diff
-        x = x + self.attention(self.attention_norm(x), rope_cache, layer_mask)
+        x = x + self.attention(self.attention_norm(x), rope_cache, None)
```

**Purpose:** FlexAttention → SDPA - pass None for attention masks instead of layer_mask.

* * *

**Lines changed:**

```diff
-        attention_masks: AttentionMasksType,
+        attention_masks: AttentionMasksType = None,
```

**Purpose:** FlexAttention → SDPA - made attention_masks optional parameter with default None.

* * *


### File: `torchtitan/models/attention.py`

**Lines changed:**

```diff
-        flex_attention, mode="max-autotune-no-cudagraphs"
+        flex_attention, mode="max-autotune-no-cudagraphs", backend='eager'
```

**Purpose:** FlexAttention → SDPA - added eager backend for FlexAttention compilation (for compatibility when FlexAttention is still used elsewhere).

* * *

**Lines changed:**

```diff
-_compiled_create_block_mask = torch.compile(create_block_mask)
+_compiled_create_block_mask = torch.compile(create_block_mask, backend='eager')
```

**Purpose:** FlexAttention → SDPA - added eager backend for block mask compilation.

* * *


## 5. RoPE Replaced with Optimized Kernel

### File: `torchtitan/experiments/gpt_oss/model/model.py`

**Lines added:**

```diff
+# RoPE nki kernel
+from torchtitan.tools.neuron_kernels import apply_rope_kernel
```

**Purpose:** RoPE optimization - imported NKI-accelerated RoPE kernel for Neuron devices.

* * *

**Lines changed:**

```diff
-        q, k = apply_rotary_emb(q, k, rope_cache)
+        # q, k = apply_rotary_emb(q, k, rope_cache)
+        q, k = apply_rope_kernel(apply_rotary_emb, q, k, rope_cache)
```

**Purpose:** RoPE optimization - replaced standard RoPE with optimized NKI kernel implementation (falls back to original if kernel unavailable).

* * *


### File: `torchtitan/models/qwen3/model/model.py`

**Lines added:**

```diff
+# RoPE nki kernel
+from torchtitan.tools.neuron_kernels import apply_rope_kernel
```

**Purpose:** RoPE optimization - imported NKI-accelerated RoPE kernel for Qwen3 model.

* * *

**Lines changed:**

```diff
-        xq, xk = apply_rotary_emb(xq, xk, rope_cache)
+        # xq, xk = apply_rotary_emb(xq, xk, rope_cache)
+        xq, xk = apply_rope_kernel(apply_rotary_emb, xq, xk, rope_cache)
```

**Purpose:** RoPE optimization - replaced standard RoPE with optimized NKI kernel in Qwen3 model.

* * *


### File: `torchtitan/tools/neuron_kernels.py` (NEW FILE)

**All 106 lines added:**

```diff
+import logging
+import os
+import torch
+from torch_neuronx import TorchNeuronNKIKernel
+from torch_neuronx.utils import get_logical_neuron_cores
+
+import neuronxcc.nki.language as nl
+
+from .logging import warn_once
+
+logger = logging.getLogger()
+
+
+# check for kernel in compiler package
+try:
+    from neuronxcc.nki._pre_prod_kernels.experimental.misc.apply_rope_nki import (
+        nki_apply_rotary_embedding_with_cache
+    )
+except ImportError as e:
+    nki_apply_rotary_embedding_with_cache = None
+
+
+def get_rope_grid_size(vnc_degree, seq_len):
+    grid_size = ()
+    if (
+        vnc_degree > 1
+        and seq_len >= vnc_degree
+        and seq_len % vnc_degree == 0
+        and seq_len > nl.tile_size.pmax
+    ):
+        grid_size = (nl.nc(vnc_degree),)
+
+    return grid_size
+
+
+class RoPEFunctionNKI(torch.autograd.Function):
+    """
+    NKI-accelerated autograd function for Rotary Position Embedding (RoPE).
+    """
+
+    @staticmethod
+    def forward(ctx, xq, xk, rope_cache):
+        ctx.save_for_backward(rope_cache)
+        xq_out = torch.empty(
+            xq.shape, dtype=xq.dtype, device=xq.device, requires_grad=xq.requires_grad
+        )
+        xk_out = torch.empty(
+            xk.shape, dtype=xk.dtype, device=xk.device, requires_grad=xk.requires_grad
+        )
+        seq_len = xq.shape[2]
+        vnc_degree = int(get_logical_neuron_cores())
+        grid = get_rope_grid_size(vnc_degree, seq_len)
+
+        kernel = TorchNeuronNKIKernel(
+            func=getattr(nki_apply_rotary_embedding_with_cache, "func", nki_apply_rotary_embedding_with_cache),
+            grid=grid,
+            kernel_return=True,
+        )
+        kernel(xq, xk, rope_cache, xq_out, xk_out, backward=False)
+        return xq_out, xk_out
+
+    @staticmethod
+    def backward(ctx, grad_xq, grad_xk):
+        rope_cache, = ctx.saved_tensors
+        grad_xq_out = torch.empty(
+            grad_xq.shape,
+            dtype=grad_xq.dtype,
+            device=grad_xq.device,
+            requires_grad=grad_xq.requires_grad,
+        )
+        grad_xk_out = torch.empty(
+            grad_xk.shape,
+            dtype=grad_xk.dtype,
+            device=grad_xk.device,
+            requires_grad=grad_xk.requires_grad,
+        )
+        seq_len = grad_xq.shape[2]
+        vnc_degree = int(get_logical_neuron_cores())
+        grid = get_rope_grid_size(vnc_degree, seq_len)
+
+        kernel = TorchNeuronNKIKernel(
+            func=getattr(nki_apply_rotary_embedding_with_cache, "func", nki_apply_rotary_embedding_with_cache),
+            grid=grid,
+            kernel_return=True,
+        )
+        kernel(grad_xq, grad_xk, rope_cache, grad_xq_out, grad_xk_out, backward=True)
+        return grad_xq_out, grad_xk_out, None, None
+
+
+def apply_rope_kernel(original_rope, xq, xk, rope_cache):
+    """
+    xq: b, s, h_q, d
+    xk: b, s, h_k, d
+    rope_cache: s, d*2
+    """
+    if nki_apply_rotary_embedding_with_cache is not None and not os.getenv('DISABLE_ROPE_KERNEL', False):
+        try:
+            # transpose to B, H, S, D
+            xq_out, xk_out = RoPEFunctionNKI.apply(xq.transpose(1, 2), xk.transpose(1, 2), rope_cache)
+            return xq_out.transpose(1, 2).contiguous(), xk_out.transpose(1, 2).contiguous()
+        except Exception as e:
+            warn_once(logger, "Failed to use NKI-accelerated RoPE kernel. Falling back to original RoPE kernel.")
+            warn_once(logger, f"failed with {e}")
+    else:
+        warn_once(logger, "NKI-accelerated RoPE kernel not available")
+    return original_rope(xq, xk, rope_cache)
```

**Purpose:** RoPE optimization - complete NKI kernel implementation for Neuron devices with:

* Grid size calculation for multi-core execution
* Custom autograd function with forward/backward passes
* Automatic fallback to original RoPE if kernel unavailable
* Tensor transposition for correct layout (B,S,H,D → B,H,S,D)

* * *


## 6. Gate/Up Expert Weights Reordering

### File: `torchtitan/experiments/gpt_oss/model/moe.py`

**Lines changed:**

```diff
-    x_glu, x_linear = x[..., ::2], x[..., 1::2]
+    dim = x.shape[-1] // 2
+    x_glu, x_linear = x.split(dim, -1)
```

**Purpose:** Performance optimization - replaced strided slicing (::2) with split operation for better memory access patterns in SwiGLU activation (gate and up weights now contiguous in memory).

* * *


## 7. Slicing Replaced with Split

### File: `torchtitan/models/moe/utils.py`

**Lines changed:**

```diff
-    out = out_unpermuted[:-1]
+    out = out_unpermuted.split(out_unpermuted.shape[0] - 1)[0]
```

**Purpose:** Performance optimization - replaced slicing with split operation for improved performance (split is more efficient for certain tensor operations).

* * *


## 8. Explicit Casting Before Gather/Scatter Operations

### File: `torchtitan/models/moe/moe.py`

**Lines added:**

```diff
+        token_indices_experts_sorted = token_indices_experts_sorted.to(torch.int32)
```

**Purpose:** Performance optimization - added explicit int32 casting before gather/scatter_add operations for better performance and compatibility.

* * *


## 9. Additional Infrastructure Changes

### File: `torchtitan/tools/profiling.py`

**Lines added:**

```diff
+            with_stack=True,
```

**Purpose:** Profiling enhancement - enabled stack trace recording in profiler for better debugging information.

* * *


### File: `torchtitan/tools/utils.py`

**Lines added:**

```diff
+import torch_neuronx
```

**Purpose:** Neuron support - imported torch_neuronx for Neuron device support.

* * *

**Lines added:**

```diff
+    elif "neuron" in device_name:
+        # we assume lnc2 here
+        systolic_array_size = 128 * 128  # 16,384 processing units
+        flops_per_cycle = 2  # BF16 operations
+        clock_rate_ghz = 2.4
+
+        # Calculate theoretical peak in FLOPS
+        theoretical_peak_flops = systolic_array_size * flops_per_cycle * clock_rate_ghz * 1e9 * 2 # lnc2
+        return theoretical_peak_flops
```

**Purpose:** Neuron support - added peak FLOPS calculation for Neuron devices (assumes LNC2 with 128×128 systolic array, 2.4 GHz clock, BF16 operations).

* * *


## Summary

Total changes across 11 files:

* **Configuration**: 13 changes (model params, training config, profiling)
* **CUDA Compatibility**: 1 change (removed assertion)
* **Triton → CPU**: 3 changes (kernel replacement)
* **FlexAttention → SDPA**: 11 changes (attention mechanism)
* **RoPE Optimization**: 4 changes + 1 new file (106 lines)
* **Memory Optimization**: 2 changes (split operations)
* **Type Casting**: 1 change (explicit int32)
* **Infrastructure**: 3 changes (profiling, Neuron support)

All changes work together to enable Neuron device compatibility while maintaining or improving performance.

# TorchTitan Neuron Tutorial: GPT-OSS on AWS Trainium

This tutorial demonstrates how to train a GPT-OSS model using TorchTitan on AWS Trainium (Trn2) instances. The guide covers environment setup, configuration, and training execution.

## Configuration

This section outlines the model, training, and hardware configurations used in this tutorial.

### Model Configurations

This tutorial provides two configurations:

**Debug Model (Single-Core):**
* `1` layer (instead of `24` layers)
* `8192` vocab size (instead of `201088`)
* `32` MoE experts

**Full 20B Model (Multi-Core):**
* `24` layers
* `201088` vocab size
* `32` MoE experts
* TP4 + FSDP16 parallelism

### Training Configuration

Both configurations use:

* Batch size of `1`
* Sequence length of `2048`

### Hardware Configuration

This tutorial targets the following hardware setup:

* trn2.48xlarge instance (single node)
* LNC2 configuration (2 physical cores combined into 1 logical core)
* Single-core for debug model
* 64 logical cores for full 20B model: 16 Trainium2 chips × 4 logical cores per chip (each chip has 8 physical cores, LNC2 combines 2 physical cores into 1 logical core)

## Environment Setup

Follow these steps to set up your environment. You'll need to configure both TorchNeuronx and TorchTitan.

### TorchNeuronx Setup

First, set up the TorchNeuronx environment with the necessary dependencies:

```shell
cd ~
git clone https://github.com/aws-neuron/torch-neuronx.git
cd torch-neuronx
./install_bazelisk.sh
pip install uv
uv pip install --system -e . --force-reinstall
./tools/build
```

### TorchTitan Setup

Next, clone and configure the TorchTitan repository with Neuron-specific modifications:

```shell
cd ~
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
git checkout 0a2107f984639e23a0e5b07fc278785345f03b73
git apply ~/torch-neuronx/docs/torchtitan/gpt-oss/TorchTitan.diff
uv pip install --system -r requirements.txt
```

#### Key Differences from TorchTitan Upstream

The modified TorchTitan branch includes both functional and performance optimizations:

**Functional changes:**

* Model configuration updated as described in the Configuration section
* CUDA assertion updated for `torch._grouped_mm` to enable Neuron compatibility
* Triton kernel for permuting indices replaced with CPU variant
* Truncated normal initialization replaced with standard normal (PyTorch bug workaround)

**Performance optimizations:**

* FlexAttention replaced with SDPA (Scaled Dot-Product Attention)
    * Note: This temporarily skips sliding window attention and attention sinks
* Slicing operations replaced with split for improved performance
* MoE token group alignment size increased (8 → 128)

For a complete diff, see: [TorchTitan.diff](TorchTitan.diff)

For a more detailed explanation of each of these changes, see [Appendix A](#appendix-a---torchtitan-changes-explained)

## Training GPT-OSS with TorchTitan

Once your environment is set up, you can start training using the following commands.

### Single-Core Training (Debug Model)

Run the debug model on a single Neuron core:

```shell
export TRAIN_FILE="torchtitan.train"
export CONFIG_FILE="torchtitan/models/gpt_oss/train_configs/debug_model.toml"
NEURON_LAUNCH_BLOCKING=1 \
torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter 0 --role rank --tee 3 \
-m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE}
```

### Multi-Core Training (Full 20B Model)

Run the full 20B model across 64 Neuron cores with TP4 + FSDP16:

```shell
export TRAIN_FILE="torchtitan.train"
export CONFIG_FILE="torchtitan/models/gpt_oss/train_configs/20b_tp4_fac.toml"
NEURON_LAUNCH_BLOCKING=1 \
torchrun --nproc_per_node=64 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter 0 --role rank --tee 3 \
-m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE}
```

### Environment Variables

* `NEURON_LAUNCH_BLOCKING=1`: Enable blocking/synchronous execution mode

### Command Explanation

* `--nproc_per_node`: Number of processes (1 for single-core, 64 for multi-core)
* `--rdzv_backend c10d`: Uses the c10d rendezvous backend for distributed coordination
* `--rdzv_endpoint="localhost:0"`: Sets the rendezvous endpoint for process coordination
* `--local-ranks-filter 0`: Filters output to show only rank 0
* `--role rank --tee 3`: Configures logging output
* `${CONFIG_FILE}`: Points to the model configuration file

### Expected Behavior

During training, you should observe:

* Loss values decreasing over iterations
* Compilation during initial steps (recompiles may occur due to MoE dynamic shapes)

## Issues/Limitations

* Requires code changes for enablement and performance reasons
* Temporarily skips certain functionality
    * Specifically, sliding window attention and attention sinks are removed
* Crashes when using 16 experts since `>=` isn’t accurate with certain data/shapes
* May crash when using 32 experts due to OOM (out of memory)
* Has no thorough accuracy validation (besides loss decreasing)
* Requires recompiles due to changing shapes in the MoE section
    * grouped_mm implementation removes majority of recompiles, but variable padding still causes some recompiles
* Lack of robustness of various features (MLIR, sync, and blocking)
* Crashes/inefficiencies depending on the compiler version

## Next Steps

### Using a Custom Tokenizer

Follow the instructions at https://github.com/pytorch/torchtitan/tree/main?tab=readme-ov-file#downloading-a-tokenizer to download a tokenizer:

```shell
# Get your HF token from https://huggingface.co/settings/tokens
# GPT-OSS tokenizer
python scripts/download_hf_assets.py --repo_id openai/gpt-oss-20b --assets tokenizer --hf_token=...
```

Then, update the `hf_assets_path` in the appropriate config file (e.g., `torchtitan/models/gpt_oss/train_configs/debug_model.toml`)

# Appendix A - TorchTitan Changes Explained

This appendix explains every line changed from the TorchTitan upstream, organized by category.

* * *

## 1. Configuration Changes

### File: `torchtitan/models/gpt_oss/__init__.py`

**Lines removed:**

```diff
-        dim=256,
-        n_layers=4,
```

**Lines added:**

```diff
+        n_layers=1,
```

**Purpose:** Model configuration update - reduced number of layers from 4 to 1 for debug model, removed explicit dim parameter (using default value of 2880).

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
```

**Purpose:** Model configuration update - set explicit vocab size to 8192.

* * *

### File: `torchtitan/models/gpt_oss/train_configs/debug_model.toml`

**Lines changed:**

```diff
-enable_profiling = false
+enable_profiling = true
```

**Purpose:** Configuration update - enabled profiling for performance analysis.

* * *

**Lines changed:**

```diff
-local_batch_size = 8
+local_batch_size = 1
```

**Purpose:** Configuration update - reduced batch size from 8 to 1 for debugging/testing.

* * *

### File: `torchtitan/models/gpt_oss/train_configs/20b_tp4_fac.toml` (NEW FILE)

**All 82 lines added:**

**Purpose:** New configuration file for training the full 20B GPT-OSS model with TP4 + FSDP16 parallelism. Key settings:
* `flavor = "20b"` - Full 20B model
* `tensor_parallel_degree = 4` - 4-way tensor parallelism
* `data_parallel_shard_degree = -1` - FSDP across remaining ranks (16)
* `local_batch_size = 1`, `seq_len = 2048`
* `activation_checkpoint.mode = "full"` - Full activation checkpointing for memory efficiency

* * *

## 2. FlexAttention Replaced with SDPA

### File: `torchtitan/models/gpt_oss/model/model.py`

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

**Purpose:** FlexAttention → SDPA - replaced FlexAttention with ScaledDotProductAttention for better Neuron compatibility.

* * *

**Lines changed:**

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

**Lines changed:**

```diff
-        sink_scale = torch.sigmoid(lse - self.sinks.view(1, -1, 1)).unsqueeze(-1)
-        output = output * sink_scale.to(output.dtype)
+        # sink_scale = torch.sigmoid(lse - self.sinks.view(1, -1, 1)).unsqueeze(-1)
+        # output = output * sink_scale.to(output.dtype)
```

**Purpose:** FlexAttention → SDPA - disabled attention sink rescaling (temporarily skips attention sinks feature since LSE is not available from SDPA).

* * *

**Lines changed:**

```diff
-        if self.use_sliding_attention:
-            # pyrefly: ignore [missing-attribute]
-            layer_mask = attention_masks.get("sliding_window_mask", None)
-        else:
-            # pyrefly: ignore [missing-attribute]
-            layer_mask = attention_masks.get("basic_mask", None)
-        assert layer_mask is not None
-
-        x = x + self.attention(self.attention_norm(x), rope_cache, layer_mask)
+        # if self.use_sliding_attention:
+        #     # pyrefly: ignore [missing-attribute]
+        #     layer_mask = attention_masks.get("sliding_window_mask", None)
+        # else:
+        #     # pyrefly: ignore [missing-attribute]
+        #     layer_mask = attention_masks.get("basic_mask", None)
+        # assert layer_mask is not None
+
+        x = x + self.attention(self.attention_norm(x), rope_cache, None)
```

**Purpose:** FlexAttention → SDPA - disabled sliding window mask extraction and pass None for attention masks (temporarily skips sliding window attention).

* * *

### File: `torchtitan/train.py`

**Lines changed:**

```diff
-            extra_kwargs["attention_masks"] = self.model_parts[0].get_attention_masks(
-                input_batch=inputs,
-                tokenizer=self.tokenizer,
-                extra_inputs=extra_inputs,
-            )
+            extra_kwargs["attention_masks"] = None
```

**Purpose:** FlexAttention → SDPA - disabled attention mask generation, set attention_masks to None since we're using SDPA instead of FlexAttention.

* * *

### File: `torchtitan/models/gpt_oss/infra/parallelize.py`

**Lines changed:**

```diff
-                output_layouts=(Shard(1), Shard(1)),
-                desired_output_layouts=(Shard(1), Shard(1)),
+                output_layouts=Shard(1),
+                desired_output_layouts=Shard(1),
```

**Purpose:** FlexAttention → SDPA - fixed output layouts for attention, changed from tuple to single Shard(1) since SDPA returns a single tensor instead of (output, lse) tuple.

* * *

## 3. Slicing Replaced with Split

### File: `torchtitan/models/gpt_oss/model/moe.py`

**Lines changed:**

```diff
-    x_glu, x_linear = x[..., ::2], x[..., 1::2]
+    x_glu, x_linear = x.split(x.shape[-1] // 2, -1)
```

**Purpose:** Performance optimization - replaced strided slicing (::2) with split operation for better memory access patterns in SwiGLU activation (gate and up weights now contiguous in memory).

* * *

### File: `torchtitan/models/moe/utils.py`

**Lines changed:**

```diff
-    out = out_unpermuted[:-1]
+    out = out_unpermuted.split(out_unpermuted.shape[0] - 1)[0]
```

**Purpose:** Performance optimization - replaced slicing with split operation for improved performance on Neuron.

* * *

## 4. MoE Token Group Alignment

### File: `torchtitan/models/moe/utils.py`

**Lines changed:**

```diff
-TOKEN_GROUP_ALIGN_SIZE_M = 8
-ValidTokenGroupAlignmentSize = Literal[8, 16, 32]
+TOKEN_GROUP_ALIGN_SIZE_M = 128
+ValidTokenGroupAlignmentSize = Literal[8, 16, 32, 128]
```

**Purpose:** Increased token group alignment size from 8 to 128 for better Neuron performance.

* * *

## 5. Triton Kernel Replaced with CPU

### File: `torchtitan/models/moe/utils.py`

**Lines changed:**

```diff
-            num_tokens_per_expert,
+            num_tokens_per_expert.cpu(),
             num_local_experts,
             ep_degree,
             padded_max_len,
             TOKEN_GROUP_ALIGN_SIZE_M,
+            use_cpu=True,
         )
+        permuted_indices = permuted_indices.neuron()
+        num_tokens_per_expert = num_tokens_per_expert.neuron()
```

**Purpose:** CPU-based permutation - moved permute indices generation to CPU (replacing Triton kernel), then transfer results back to Neuron device using `.neuron()`.

* * *

## 6. Neuron Device Support

### File: `torchtitan/models/gpt_oss/model/args.py`

**Lines changed:**

```diff
-from torchtitan.tools.utils import has_cuda_capability
+from torchtitan.tools.utils import has_cuda_capability, device_type
```

**Purpose:** Import device_type to check for Neuron devices.

* * *

**Lines changed:**

```diff
-        if self.moe_args.use_grouped_mm and not has_cuda_capability(9, 0):
+        if self.moe_args.use_grouped_mm and not has_cuda_capability(9, 0) and "neuron" not in device_type:
```

**Purpose:** Added Neuron device check - grouped_mm is supported on Neuron devices, so skip the CUDA capability warning when running on Neuron.

* * *

### File: `torchtitan/tools/utils.py`

**Lines added:**

```diff
+    elif "neuron" in device_name:
+        # Trainium2 LNC2: 128x128 systolic array, 2.4GHz, BF16
+        return 157e12
```

**Purpose:** Added peak FLOPS calculation for Neuron devices (157 TFLOPS for Trainium2 LNC2).

* * *

## 7. Truncated Normal Initialization Workaround

### File: `torchtitan/train.py`

**Lines added:**

```diff
+# Workaround for PyTorch bugs:
+# https://github.com/pytorch/pytorch/issues/145498
+# https://github.com/pytorch/pytorch/issues/155588
+import torch.nn as nn
+nn.init.trunc_normal_ = lambda tensor, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=None: nn.init.normal_(tensor, mean, std, generator)
```

**Purpose:** Workaround for PyTorch trunc_normal_ bugs - replaces truncated normal initialization with standard normal initialization to avoid issues on Neuron.

* * *

## Summary

Total changes across 10 files:

* **Configuration**: 3 changes + 1 new file (model params, training configs)
* **FlexAttention → SDPA**: 7 changes (attention mechanism, TP layout fix)
* **Slicing → Split**: 2 changes (SwiGLU, unpermute)
* **MoE Token Group Alignment**: 1 change
* **Triton → CPU**: 1 change (permutation kernel)
* **Neuron Device Support**: 2 changes (grouped_mm check, peak FLOPS)
* **Truncated Normal Workaround**: 1 change

All changes work together to enable Neuron device compatibility while maintaining or improving performance.

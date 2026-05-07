"""GPT-OSS-specific block and performance calculations."""

import torch
import torch.nn as nn
from transformers import AutoConfig, PretrainedConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts, GptOssMLP


class CustomGptOssMLP(GptOssMLP):
    def forward(self, hidden_states, **kwargs):
        routed_output, _ = super().forward(hidden_states)
        return routed_output


def create_gpt_oss_block(config_dict, device, dtype=None):
    """
    Create and configure a GptOssMLP block.

    Args:
        config_dict: dict of model config
        device: Device to place model on
        dtype: Data type for model weights (optional)

    Returns:
        Configured block on specified device with specified dtype
    """
    config = PretrainedConfig.from_dict(config_dict)
    block = CustomGptOssMLP(config)

    # Initialize model weights
    for name, param in block.named_parameters():
        if "weight" in name and param.data.dim() == 2:
            nn.init.kaiming_uniform_(param)

    block = block.to(device, dtype=dtype) if dtype is not None else block.to(device)
    block.eval()
    return block


def count_gpt_oss_mlp_flops(
    hidden_size, intermediate_size, num_local_experts, experts_per_token, seq_len, batch_size
):
    """
    Count FLOPs for a single forward pass of a GptOssMLP block.

    Model implementation reference: https://github.com/huggingface/transformers/blob/58e13b9f129bb0dccc3b51e5da22f45ef3ff0ae7/src/transformers/models/gpt_oss/modular_gpt_oss.py#L80

    Notes on what is counted:
    - Uses the common convention that 1 FMA = 2 FLOPs.
    - Counts the dominant matmul-based ops only (projections).
      Light-weight ops like norm, activations, elementwise, reshapes,
      and softmax are excluded as they contribute comparatively little FLOPs.
    - Backward pass FLOPs are NOT included here; this is forward-only.

    Components:
    - router: single matmul
      (B*S, hidden_size) * (num_local_experts, hidden_size)
    - gate, up projections: 2 linear layers of shape
      (B*S, hidden_size) x (num_local_experts, intermediate_size, hidden_size)
    - down projection: selected expert down proj
      (B*S, num_local_experts * intermediate_size) * (intermediate_size, hidden_size)

    Args:
        hidden_size: Embedding dimension
        intermediate_size: Feed-forward dimension
        num_local_experts: Number of experts for MLP
        experts_per_token: Number of experts activated per token (topK)
        seq_len: Sequence length
        batch_size: Batch size

    Returns:
        Total FLOPs for one forward pass (forward-only)
    """

    # Router FLOPs
    # router_logits = F.linear(hidden_states, self.weight, self.bias)
    # 2 * batch * seq_len * hidden * num_local_experts
    router_flops = 2 * batch_size * seq_len * hidden_size * num_local_experts

    # gate up proj
    # torch.bmm(hidden_states, self.gate_up_proj) + self.gate_up_proj_bias[..., None, :]
    # 2 * batch_size * seq_len * hidden_size * intermediate_size * num_local_experts
    gate_up_proj_flops = (
        2 * batch_size * seq_len * hidden_size * intermediate_size * num_local_experts
    )

    # gate up FLOPs
    # gate * up elementwise
    # gated_output = (up + 1) * glu
    # batch_size * seq_len * num_local_experts * intermediate_size
    gate_up_flops = batch_size * seq_len * num_local_experts * intermediate_size

    # down proj FLOPs
    # next_states = torch.bmm(((up + 1) * glu), self.down_proj)
    # 2 * batch_size * seq_len * num_local_experts * intermediate_size * hidden_size
    down_proj_flops = 2 * batch_size * seq_len * num_local_experts * intermediate_size * hidden_size

    total_flops = router_flops + gate_up_proj_flops + gate_up_flops + down_proj_flops

    return total_flops


def get_default_config():
    """Get default configuration for GptOssMLP blocks."""
    config = AutoConfig.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
    return config.to_dict()


def get_preset_config(preset_name):
    """Get preset configuration for specific GptOssMLP model sizes."""
    config = AutoConfig.from_pretrained(f"openai/{preset_name}", trust_remote_code=True)
    return config.to_dict()


def list_presets():
    """List available GptOssMLP configuration presets."""
    return ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]


def get_presets():
    """Return list of available presets for dynamic loading."""
    return list_presets()


def create_config(preset_name, batch_size, seq_len):
    """Create complete config from preset and runtime parameters.

    Args:
        preset_name: Name of the preset configuration
        batch_size: Batch size for benchmarking
        seq_len: Sequence length for benchmarking

    Returns:
        Dictionary with complete configuration
    """
    config = AutoConfig.from_pretrained(preset_name, trust_remote_code=True)
    config.batch_size = batch_size
    config.seq_len = seq_len
    return config.to_dict()


def create_block(config, device, dtype=None, tp_mesh=None, dp_mesh=None):
    """Create block from config dictionary.

    Args:
        config: Configuration dictionary
        device: Device to place model on
        dtype: Data type for model weights (optional)

    Returns:
        Configured block on specified device with specified dtype
    """
    return create_gpt_oss_block(config, device, dtype)


def count_flops(config):
    """Count FLOPs for given configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Total FLOPs for one forward pass
    """
    return count_gpt_oss_mlp_flops(
        config["hidden_size"],
        config["intermediate_size"],
        config["num_local_experts"],
        config["experts_per_token"],
        config["seq_len"],
        config["batch_size"],
    )


def run_block(block, input_tensor, **kwargs):
    """Execute GPT-OSS block with standard signature."""
    return block(input_tensor, **kwargs)

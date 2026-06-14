"""
LTX-2 Native PyTorch on Trainium (Beta 3)
==========================================
Single-file implementation of LTX-2 19B audio-video diffusion transformer
on Trainium using native PyTorch + torchrun. No NxDI/XLA dependencies.

Architecture:
  CPU: proj_in → time_embed → caption_projection → RoPE → mask → cross_attn conditioning
  NEURON: 48 × LTX2TransformerBlock + norm_out + proj_out + audio_norm_out + audio_proj_out
  CPU: unpack latents → denormalize → VAE decode

Launch:
  torchrun --nproc_per_node=4 --standalone neuron_ltx2_native.py

Requirements:
  - Trainium2 instance (trn2.48xlarge recommended for TP=4)
  - Beta 3 DLC with torch 2.11, torch_neuronx 2.11.x
  - diffusers >= 0.31.0 (with LTX2Pipeline support)
  - safetensors, accelerate

Environment:
  NEURON_RT_VIRTUAL_CORE_SIZE=2
  NEURON_FUSE_SOFTMAX=1
"""

import gc
import logging
import math
import os
import time
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. BMM-SDPA Replacement
# ─────────────────────────────────────────────────────────────────────────────

_sdpa_original = None
_sdpa_replaced = False


def install_bmm_sdpa():
    """Replace F.scaled_dot_product_attention with explicit BMM+softmax for Neuron.

    Stock SDPA miscomputes on Neuron device. This replacement:
    - CPU calls (query.device.type == "cpu") fall through to original SDPA
    - Neuron calls use explicit torch.bmm(Q, K^T) * scale + mask → softmax → bmm(probs, V)
    - Handles 4D→3D reshape transparently
    """
    global _sdpa_replaced, _sdpa_original

    if _sdpa_replaced:
        return _sdpa_original

    _sdpa_original = torch.nn.functional.scaled_dot_product_attention

    def neuron_sdpa(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        # CPU fallback — use original SDPA (for text encoder, preprocessing)
        if query.device.type == "cpu":
            return _sdpa_original(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )

        # ── BMM path for Neuron device ──────────────────────────────────
        d = query.shape[-1]
        if scale is None:
            scale = 1.0 / math.sqrt(d)

        orig_shape = None
        if len(query.shape) == 4:
            orig_shape = query.shape
            b, h, sq, d_head = query.shape
            query = query.reshape(b * h, sq, d_head)
            key = key.reshape(b * h, -1, d_head)
            value = value.reshape(b * h, -1, d_head)
            if attn_mask is not None:
                if attn_mask.ndim == 4:
                    attn_mask = attn_mask.reshape(
                        b * h, attn_mask.shape[-2], attn_mask.shape[-1]
                    )
                elif attn_mask.ndim == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                elif attn_mask.ndim == 3:
                    if attn_mask.shape[0] == orig_shape[0]:
                        # (B, 1, K) → expand to (B, H, 1, K) → reshape to (B*H, 1, K)
                        attn_mask = (
                            attn_mask.unsqueeze(1)
                            .expand(orig_shape[0], orig_shape[1], -1, -1)
                            .reshape(
                                orig_shape[0] * orig_shape[1],
                                attn_mask.shape[-2],
                                attn_mask.shape[-1],
                            )
                        )

        # Explicit BMM attention
        scores = torch.bmm(query, key.transpose(-1, -2)) * scale

        if attn_mask is not None:
            scores = scores + attn_mask

        probs = scores.softmax(dim=-1)
        out = torch.bmm(probs, value)

        if orig_shape is not None:
            out = out.reshape(orig_shape[0], orig_shape[1], -1, orig_shape[3])

        return out

    torch.nn.functional.scaled_dot_product_attention = neuron_sdpa
    _sdpa_replaced = True
    logger.info("Installed BMM-based SDPA replacement for Neuron")
    return _sdpa_original


# ─────────────────────────────────────────────────────────────────────────────
# 2. TP-aware RMSNorm (all-reduce for QK-norm across sharded heads)
# ─────────────────────────────────────────────────────────────────────────────


class TPRMSNorm(nn.Module):
    """RMSNorm with all-reduce for global variance computation across TP ranks.

    After ColwiseParallel sharding, each rank sees only inner_dim/N of the
    hidden dimension. Standard RMSNorm would compute incorrect statistics.
    This version all-reduces the sum-of-squares across ranks for correctness.
    """

    def __init__(self, local_dim: int, eps: float = 1e-5, world_size: int = 4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(local_dim, dtype=torch.bfloat16))
        self.eps = eps
        self.world_size = world_size
        self.local_dim = local_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_f32 = hidden_states.to(torch.float32)
        local_sum_sq = hidden_f32.pow(2).sum(dim=-1, keepdim=True)
        # All-reduce to get global sum of squares
        dist.all_reduce(local_sum_sq, op=dist.ReduceOp.SUM)
        global_dim = self.local_dim * self.world_size
        rms = torch.rsqrt(local_sum_sq / global_dim + self.eps)
        hidden_states = hidden_f32 * rms
        hidden_states = hidden_states.to(self.weight.dtype)
        return hidden_states * self.weight


# ─────────────────────────────────────────────────────────────────────────────
# 3. NeuronLTX2Backbone — the 48 blocks + output layers on Neuron
# ─────────────────────────────────────────────────────────────────────────────


class NeuronLTX2Backbone(nn.Module):
    """The LTX-2 transformer backbone running on Neuron device.

    Contains:
    - 48 LTX2TransformerBlocks (video+audio self-attn, cross-attn, a2v/v2a, FFN)
    - norm_out + proj_out + scale_shift_table (video output)
    - audio_norm_out + audio_proj_out + audio_scale_shift_table (audio output)

    All layers are TP-sharded and moved to torch.device("neuron").
    Takes 22 preprocessed tensor inputs (all preprocessing done on CPU).
    """

    def __init__(
        self,
        cpu_transformer,
        world_size: int = 4,
        rank: int = 0,
        device_mesh=None,
    ):
        """Extract backbone layers from CPU model, shard, and move to Neuron.

        Args:
            cpu_transformer: The LTX2VideoTransformer3DModel loaded on CPU
            world_size: Tensor parallel degree
            rank: This process's rank
            device_mesh: DeviceMesh for parallelize_module
        """
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        self.neuron_device = torch.device("neuron")

        # Extract layers from CPU model
        self.transformer_blocks = cpu_transformer.transformer_blocks
        self.norm_out = cpu_transformer.norm_out
        self.proj_out = cpu_transformer.proj_out
        self.scale_shift_table = cpu_transformer.scale_shift_table
        self.audio_norm_out = cpu_transformer.audio_norm_out
        self.audio_proj_out = cpu_transformer.audio_proj_out
        self.audio_scale_shift_table = cpu_transformer.audio_scale_shift_table

        # Apply TP sharding
        if world_size > 1 and device_mesh is not None:
            self._apply_tp_sharding(device_mesh)

        # Move everything to Neuron
        self.to(self.neuron_device)
        self.eval()
        logger.info(
            f"[Rank {rank}] Backbone on Neuron: {len(self.transformer_blocks)} blocks"
        )

    def _apply_tp_sharding(self, device_mesh):
        """Apply ColwiseParallel/RowwiseParallel TP plan to all attention + FFN layers.

        TP plan per block (48 blocks × 6 attention paths + 2 FFNs):
          Attention (attn1, attn2, audio_attn1, audio_attn2, audio_to_video_attn, video_to_audio_attn):
            to_q, to_k, to_v → ColwiseParallel (shard output dim)
            to_out.0 → RowwiseParallel (shard input dim)
          FFN (ff, audio_ff):
            ff.net.0.proj → ColwiseParallel
            ff.net.2 → RowwiseParallel
        """
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            RowwiseParallel,
            parallelize_module,
        )

        attn_names = [
            "attn1", "attn2",
            "audio_attn1", "audio_attn2",
            "audio_to_video_attn", "video_to_audio_attn",
        ]

        for i, block in enumerate(self.transformer_blocks):
            # Build the TP plan for this block
            plan = {}

            # Attention projections
            for attn_name in attn_names:
                attn = getattr(block, attn_name, None)
                if attn is None:
                    continue
                plan[f"{attn_name}.to_q"] = ColwiseParallel()
                plan[f"{attn_name}.to_k"] = ColwiseParallel()
                plan[f"{attn_name}.to_v"] = ColwiseParallel()
                plan[f"{attn_name}.to_out.0"] = RowwiseParallel()

            # FFN projections
            for ff_name in ["ff", "audio_ff"]:
                ff = getattr(block, ff_name, None)
                if ff is None:
                    continue
                plan[f"{ff_name}.net.0.proj"] = ColwiseParallel()
                plan[f"{ff_name}.net.2"] = RowwiseParallel()

            parallelize_module(block, device_mesh, plan)

            # Post-sharding patches per attention module
            for attn_name in attn_names:
                attn = getattr(block, attn_name, None)
                if attn is None:
                    continue

                # Patch attn.heads to heads/world_size
                # Block forward does query.unflatten(2, (attn.heads, -1))
                # With sharded inner_dim, we need fewer heads per rank
                if hasattr(attn, "heads") and attn.heads % self.world_size == 0:
                    attn.heads = attn.heads // self.world_size

                # Install TP-aware QK norm (all-reduce for rms_norm_across_heads)
                for norm_name in ["norm_q", "norm_k"]:
                    norm = getattr(attn, norm_name, None)
                    if norm is not None and hasattr(norm, "weight") and norm.weight is not None:
                        local_dim = norm.weight.shape[0] // self.world_size
                        tp_norm = TPRMSNorm(
                            local_dim=local_dim,
                            eps=getattr(norm, "eps", 1e-5),
                            world_size=self.world_size,
                        )
                        # Copy the local shard of the norm weight
                        start = self.rank * local_dim
                        end = start + local_dim
                        tp_norm.weight.data = norm.weight.data[start:end].clone().to(torch.bfloat16)
                        setattr(attn, norm_name, tp_norm)

            if i == 0 or i == 47:
                logger.info(f"  Block {i}: TP plan applied ({len(plan)} layers sharded)")

    def _slice_rope(self, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Slice RoPE embeddings to heads owned by this TP rank.

        In torchrun each process has its own Python rank (not a traced tensor),
        so we simply compute the slice indices as Python ints.

        Args:
            cos: (B, num_heads, seq, rope_dim) — full heads
            sin: (B, num_heads, seq, rope_dim) — full heads

        Returns:
            Sliced (cos, sin) with shape (B, num_heads/world_size, seq, rope_dim)
        """
        if self.world_size <= 1:
            return cos, sin

        num_heads = cos.shape[1]
        h_per_rank = num_heads // self.world_size
        start = self.rank * h_per_rank
        end = start + h_per_rank
        return cos[:, start:end], sin[:, start:end]

    def forward(
        self,
        hidden_states: torch.Tensor,           # 1. (B, video_seq, inner_dim)
        audio_hidden_states: torch.Tensor,     # 2. (B, audio_seq, audio_inner_dim)
        encoder_hidden_states: torch.Tensor,   # 3. (B, text_seq, inner_dim)
        audio_encoder_hidden_states: torch.Tensor,  # 4. (B, text_seq, audio_inner_dim)
        temb: torch.Tensor,                    # 5. (B, 1, 6*inner_dim)
        temb_audio: torch.Tensor,              # 6. (B, 1, 6*audio_inner_dim)
        embedded_timestep: torch.Tensor,       # 7. (B, 1, inner_dim)
        audio_embedded_timestep: torch.Tensor, # 8. (B, 1, audio_inner_dim)
        video_cross_attn_scale_shift: torch.Tensor,  # 9. (B, 1, 4*inner_dim)
        audio_cross_attn_scale_shift: torch.Tensor,  # 10. (B, 1, 4*audio_inner_dim)
        video_cross_attn_a2v_gate: torch.Tensor,     # 11. (B, 1, inner_dim)
        audio_cross_attn_v2a_gate: torch.Tensor,     # 12. (B, 1, audio_inner_dim)
        video_rotary_cos: torch.Tensor,        # 13. (B, num_heads, video_seq, rope_dim)
        video_rotary_sin: torch.Tensor,        # 14.
        audio_rotary_cos: torch.Tensor,        # 15. (B, audio_heads, audio_seq, rope_dim)
        audio_rotary_sin: torch.Tensor,        # 16.
        cross_video_rotary_cos: torch.Tensor,  # 17. (B, num_heads, video_seq, ca_rope_dim)
        cross_video_rotary_sin: torch.Tensor,  # 18.
        cross_audio_rotary_cos: torch.Tensor,  # 19. (B, audio_heads, audio_seq, ca_rope_dim)
        cross_audio_rotary_sin: torch.Tensor,  # 20.
        encoder_attention_mask: torch.Tensor,  # 21. (B, 1, text_seq) additive bias
        audio_encoder_attention_mask: torch.Tensor,  # 22. (B, 1, text_seq) additive bias
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run 48 transformer blocks + output projection on Neuron.

        Returns:
            (video_output, audio_output) tensors
        """
        # Slice RoPE to local TP rank's head range
        video_rotary_emb = self._slice_rope(video_rotary_cos, video_rotary_sin)
        audio_rotary_emb = self._slice_rope(audio_rotary_cos, audio_rotary_sin)
        ca_video_rotary_emb = self._slice_rope(cross_video_rotary_cos, cross_video_rotary_sin)
        ca_audio_rotary_emb = self._slice_rope(cross_audio_rotary_cos, cross_audio_rotary_sin)

        # Run 48 transformer blocks
        for block in self.transformer_blocks:
            hidden_states, audio_hidden_states = block(
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                temb=temb,
                temb_audio=temb_audio,
                temb_ca_scale_shift=video_cross_attn_scale_shift,
                temb_ca_audio_scale_shift=audio_cross_attn_scale_shift,
                temb_ca_gate=video_cross_attn_a2v_gate,
                temb_ca_audio_gate=audio_cross_attn_v2a_gate,
                video_rotary_emb=video_rotary_emb,
                audio_rotary_emb=audio_rotary_emb,
                ca_video_rotary_emb=ca_video_rotary_emb,
                ca_audio_rotary_emb=ca_audio_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                audio_encoder_attention_mask=audio_encoder_attention_mask,
            )

        # ── Video output projection ────────────────────────────────────
        # scale_shift_table shape: (2, inner_dim)
        scale_shift_values = (
            self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        video_output = self.proj_out(hidden_states)

        # ── Audio output projection ────────────────────────────────────
        audio_ssv = (
            self.audio_scale_shift_table[None, None]
            + audio_embedded_timestep[:, :, None]
        )
        audio_shift, audio_scale = audio_ssv[:, :, 0], audio_ssv[:, :, 1]
        audio_hidden_states = self.audio_norm_out(audio_hidden_states)
        audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
        audio_output = self.audio_proj_out(audio_hidden_states)

        return video_output, audio_output


# ─────────────────────────────────────────────────────────────────────────────
# 4. NeuronTransformerWrapper — drop-in for pipe.transformer
# ─────────────────────────────────────────────────────────────────────────────


class NeuronTransformerWrapper(nn.Module):
    """Drop-in replacement for LTX2VideoTransformer3DModel in the Diffusers pipeline.

    Keeps CPU copies of preprocessing modules:
      proj_in, audio_proj_in, time_embed, audio_time_embed,
      caption_projection, audio_caption_projection,
      rope, audio_rope, cross_attn_rope, cross_attn_audio_rope,
      av_cross_attn_video_scale_shift, av_cross_attn_video_a2v_gate,
      av_cross_attn_audio_scale_shift, av_cross_attn_audio_v2a_gate

    In forward():
      1. CPU preprocessing (proj_in, time_embed, caption_projection, RoPE, masks, cross-attn conditioning)
      2. Move 22 tensors to Neuron device
      3. Call NeuronLTX2Backbone
      4. Return (video_output, audio_output)
    """

    def __init__(
        self,
        backbone: NeuronLTX2Backbone,
        cpu_transformer,
        text_seq: int = 1024,
    ):
        """
        Args:
            backbone: NeuronLTX2Backbone (TP-sharded, on Neuron device)
            cpu_transformer: Original LTX2VideoTransformer3DModel (CPU)
            text_seq: Maximum text sequence length
        """
        super().__init__()
        self.backbone = backbone
        self.text_seq = text_seq
        self.neuron_device = torch.device("neuron")

        # Copy config and attributes the pipeline expects
        self.config = cpu_transformer.config
        self.dtype = cpu_transformer.dtype
        self.device = cpu_transformer.device

        # Proxy cache_context (pipeline calls self.transformer.cache_context)
        from contextlib import nullcontext
        self._nullcontext = nullcontext

        # Keep CPU preprocessing layers
        self.proj_in = cpu_transformer.proj_in
        self.audio_proj_in = cpu_transformer.audio_proj_in
        self.time_embed = cpu_transformer.time_embed
        self.audio_time_embed = cpu_transformer.audio_time_embed
        self.caption_projection = cpu_transformer.caption_projection
        self.audio_caption_projection = cpu_transformer.audio_caption_projection
        self.rope = cpu_transformer.rope
        self.audio_rope = cpu_transformer.audio_rope
        self.cross_attn_rope = cpu_transformer.cross_attn_rope
        self.cross_attn_audio_rope = cpu_transformer.cross_attn_audio_rope
        self.av_cross_attn_video_scale_shift = cpu_transformer.av_cross_attn_video_scale_shift
        self.av_cross_attn_video_a2v_gate = cpu_transformer.av_cross_attn_video_a2v_gate
        self.av_cross_attn_audio_scale_shift = cpu_transformer.av_cross_attn_audio_scale_shift
        self.av_cross_attn_audio_v2a_gate = cpu_transformer.av_cross_attn_audio_v2a_gate

        # Step-invariant cache
        self._step_cache = None
        self._step_cache_key = None

    def cache_context(self, context_name: str):
        """Proxy for pipeline's cache_context calls. Returns a no-op context manager."""
        return self._nullcontext()

    def _compute_step_invariant(
        self,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        encoder_attention_mask,
        audio_encoder_attention_mask,
        video_coords,
        audio_coords,
        batch_size,
        inner_dim,
        audio_inner_dim,
        dtype,
    ):
        """Compute and cache step-invariant preprocessing (caption proj, RoPE, masks).

        These depend only on the prompt and spatial layout, not the timestep,
        so they are identical across all denoising steps.
        """
        with torch.no_grad():
            # Caption projection (CPU)
            enc_hs = self.caption_projection(encoder_hidden_states)
            enc_hs = enc_hs.view(batch_size, -1, inner_dim)
            audio_enc_hs = self.audio_caption_projection(audio_encoder_hidden_states)
            audio_enc_hs = audio_enc_hs.view(batch_size, -1, audio_inner_dim)

            # RoPE (CPU) — compute from coords, cast to bf16
            video_rotary_emb = self.rope(video_coords, device="cpu")
            audio_rotary_emb = self.audio_rope(audio_coords, device="cpu")
            video_cross_rotary_emb = self.cross_attn_rope(
                video_coords[:, 0:1, :], device="cpu"
            )
            audio_cross_rotary_emb = self.cross_attn_audio_rope(
                audio_coords[:, 0:1, :], device="cpu"
            )

        # Cast RoPE from float32 to bfloat16 for Neuron
        video_rotary_emb = (video_rotary_emb[0].to(dtype), video_rotary_emb[1].to(dtype))
        audio_rotary_emb = (audio_rotary_emb[0].to(dtype), audio_rotary_emb[1].to(dtype))
        video_cross_rotary_emb = (video_cross_rotary_emb[0].to(dtype), video_cross_rotary_emb[1].to(dtype))
        audio_cross_rotary_emb = (audio_cross_rotary_emb[0].to(dtype), audio_cross_rotary_emb[1].to(dtype))

        # Attention masks — convert binary (B, text_seq) to additive bias (B, 1, text_seq)
        with torch.no_grad():
            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                enc_mask = (1 - encoder_attention_mask.to(dtype)) * -10000.0
                enc_mask = enc_mask.unsqueeze(1)  # (B, 1, text_seq)
            else:
                enc_mask = encoder_attention_mask

            if audio_encoder_attention_mask is not None and audio_encoder_attention_mask.ndim == 2:
                audio_enc_mask = (1 - audio_encoder_attention_mask.to(dtype)) * -10000.0
                audio_enc_mask = audio_enc_mask.unsqueeze(1)
            else:
                audio_enc_mask = audio_encoder_attention_mask

            # Default to zeros (attend to everything) if mask is None
            if enc_mask is None:
                enc_mask = torch.zeros(batch_size, 1, self.text_seq, dtype=dtype)
            if audio_enc_mask is None:
                audio_enc_mask = torch.zeros(batch_size, 1, self.text_seq, dtype=dtype)

        return (
            enc_hs,
            audio_enc_hs,
            video_rotary_emb,
            audio_rotary_emb,
            video_cross_rotary_emb,
            audio_cross_rotary_emb,
            enc_mask,
            audio_enc_mask,
        )

    def forward(
        self,
        hidden_states,
        audio_hidden_states=None,
        encoder_hidden_states=None,
        audio_encoder_hidden_states=None,
        timestep=None,
        encoder_attention_mask=None,
        audio_encoder_attention_mask=None,
        num_frames=None,
        height=None,
        width=None,
        fps=None,
        audio_num_frames=None,
        video_coords=None,
        audio_coords=None,
        return_dict=False,
        **kwargs,
    ):
        """Preprocess on CPU, run 48 blocks on Neuron, return results.

        Step-invariant computations (caption projection, RoPE, attention masks)
        are cached after the first call and reused for subsequent denoising steps.
        """
        batch_size = hidden_states.shape[0]
        dtype = torch.bfloat16

        with torch.no_grad():
            # ── 1. Project inputs (CPU) — step-varying ──────────────────
            hs = self.proj_in(hidden_states)
            ahs = self.audio_proj_in(audio_hidden_states)

            # ── 2. Time embeddings (CPU) — step-varying ─────────────────
            temb, embedded_ts = self.time_embed(
                timestep.flatten(), batch_size=batch_size, hidden_dtype=dtype
            )
            temb = temb.view(batch_size, -1, temb.size(-1))
            embedded_ts = embedded_ts.view(batch_size, -1, embedded_ts.size(-1))

            temb_audio, audio_embedded_ts = self.audio_time_embed(
                timestep.flatten(), batch_size=batch_size, hidden_dtype=dtype
            )
            temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))
            audio_embedded_ts = audio_embedded_ts.view(batch_size, -1, audio_embedded_ts.size(-1))

            # ── 3. Cross-attention conditioning (CPU) — step-varying ────
            ts_scale = (
                self.config.cross_attn_timestep_scale_multiplier
                / self.config.timestep_scale_multiplier
            )

            video_ca_ss, _ = self.av_cross_attn_video_scale_shift(
                timestep.flatten(), batch_size=batch_size, hidden_dtype=dtype
            )
            video_ca_gate, _ = self.av_cross_attn_video_a2v_gate(
                timestep.flatten() * ts_scale, batch_size=batch_size, hidden_dtype=dtype
            )
            video_ca_ss = video_ca_ss.view(batch_size, -1, video_ca_ss.shape[-1])
            video_ca_gate = video_ca_gate.view(batch_size, -1, video_ca_gate.shape[-1])

            audio_ca_ss, _ = self.av_cross_attn_audio_scale_shift(
                timestep.flatten(), batch_size=batch_size, hidden_dtype=dtype
            )
            audio_ca_v2a_gate, _ = self.av_cross_attn_audio_v2a_gate(
                timestep.flatten() * ts_scale, batch_size=batch_size, hidden_dtype=dtype
            )
            audio_ca_ss = audio_ca_ss.view(batch_size, -1, audio_ca_ss.shape[-1])
            audio_ca_v2a_gate = audio_ca_v2a_gate.view(batch_size, -1, audio_ca_v2a_gate.shape[-1])

        # ── 4-6. Step-invariant: caption projection, RoPE, masks (cached) ──
        cache_key = encoder_hidden_states.data_ptr()
        if self._step_cache is None or self._step_cache_key != cache_key:
            (
                enc_hs,
                audio_enc_hs,
                video_rotary_emb,
                audio_rotary_emb,
                video_cross_rotary_emb,
                audio_cross_rotary_emb,
                enc_mask,
                audio_enc_mask,
            ) = self._compute_step_invariant(
                encoder_hidden_states,
                audio_encoder_hidden_states,
                encoder_attention_mask,
                audio_encoder_attention_mask,
                video_coords,
                audio_coords,
                batch_size,
                hs.size(-1),
                ahs.size(-1),
                dtype,
            )
            self._step_cache = (
                enc_hs,
                audio_enc_hs,
                video_rotary_emb,
                audio_rotary_emb,
                video_cross_rotary_emb,
                audio_cross_rotary_emb,
                enc_mask,
                audio_enc_mask,
            )
            self._step_cache_key = cache_key
        else:
            (
                enc_hs,
                audio_enc_hs,
                video_rotary_emb,
                audio_rotary_emb,
                video_cross_rotary_emb,
                audio_cross_rotary_emb,
                enc_mask,
                audio_enc_mask,
            ) = self._step_cache

        # ── 7. Move all 22 tensors to Neuron and call backbone ──────────
        def _to_neuron(t):
            return t.to(self.neuron_device) if t.device.type != "neuron" else t

        video_output, audio_output = self.backbone(
            _to_neuron(hs),                       # 1
            _to_neuron(ahs),                      # 2
            _to_neuron(enc_hs),                   # 3
            _to_neuron(audio_enc_hs),             # 4
            _to_neuron(temb),                     # 5
            _to_neuron(temb_audio),               # 6
            _to_neuron(embedded_ts),              # 7
            _to_neuron(audio_embedded_ts),        # 8
            _to_neuron(video_ca_ss),              # 9
            _to_neuron(audio_ca_ss),              # 10
            _to_neuron(video_ca_gate),            # 11
            _to_neuron(audio_ca_v2a_gate),        # 12
            _to_neuron(video_rotary_emb[0]),      # 13 cos
            _to_neuron(video_rotary_emb[1]),      # 14 sin
            _to_neuron(audio_rotary_emb[0]),      # 15 cos
            _to_neuron(audio_rotary_emb[1]),      # 16 sin
            _to_neuron(video_cross_rotary_emb[0]),  # 17 cos
            _to_neuron(video_cross_rotary_emb[1]),  # 18 sin
            _to_neuron(audio_cross_rotary_emb[0]),  # 19 cos
            _to_neuron(audio_cross_rotary_emb[1]),  # 20 sin
            _to_neuron(enc_mask),                 # 21
            _to_neuron(audio_enc_mask),           # 22
        )

        # Move output back to CPU for pipeline post-processing
        video_output = video_output.cpu()
        audio_output = audio_output.cpu()

        return video_output, audio_output


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main — load pipeline, build backbone, generate
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """Entry point: load LTX-2, shard transformer on Neuron, run generation.

    Launch with:
      NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_FUSE_SOFTMAX=1 \\
      torchrun --nproc_per_node=4 --standalone neuron_ltx2_native.py
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ── Install BMM-SDPA globally ───────────────────────────────────────
    install_bmm_sdpa()

    # ── Initialize distributed ──────────────────────────────────────────
    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    logger.info(f"[Rank {rank}/{world_size}] Distributed initialized (backend=neuron)")

    # Set environment
    os.environ.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")
    os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")
    os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "0")

    # ── Load LTX2Pipeline on CPU ────────────────────────────────────────
    from diffusers import LTX2Pipeline

    logger.info(f"[Rank {rank}] Loading LTX2Pipeline from Lightricks/LTX-2...")
    t0 = time.time()
    pipe = LTX2Pipeline.from_pretrained(
        "Lightricks/LTX-2",
        torch_dtype=torch.bfloat16,
    )
    logger.info(f"[Rank {rank}] Pipeline loaded in {time.time() - t0:.1f}s")

    # ── Extract CPU transformer ─────────────────────────────────────────
    cpu_transformer = pipe.transformer
    cpu_transformer.eval()

    # ── Create device mesh for TP ───────────────────────────────────────
    from torch.distributed.device_mesh import init_device_mesh

    device_mesh = init_device_mesh("neuron", (world_size,))
    logger.info(f"[Rank {rank}] Device mesh created: {device_mesh}")

    # ── Build NeuronLTX2Backbone (TP-sharded, on Neuron) ────────────────
    logger.info(f"[Rank {rank}] Building Neuron backbone (TP={world_size})...")
    t0 = time.time()
    backbone = NeuronLTX2Backbone(
        cpu_transformer=cpu_transformer,
        world_size=world_size,
        rank=rank,
        device_mesh=device_mesh,
    )
    logger.info(f"[Rank {rank}] Backbone built in {time.time() - t0:.1f}s")

    # ── Wrap with NeuronTransformerWrapper ──────────────────────────────
    wrapper = NeuronTransformerWrapper(
        backbone=backbone,
        cpu_transformer=cpu_transformer,
        text_seq=1024,
    )

    # Free the heavy transformer blocks from CPU (already on Neuron)
    del cpu_transformer.transformer_blocks
    del cpu_transformer.norm_out, cpu_transformer.proj_out
    del cpu_transformer.audio_norm_out, cpu_transformer.audio_proj_out
    gc.collect()

    # ── Swap into pipeline ──────────────────────────────────────────────
    pipe.transformer = wrapper
    logger.info(f"[Rank {rank}] Transformer swapped to Neuron backbone")

    # ── Generate ────────────────────────────────────────────────────────
    prompt = (
        "A woman with long brown hair and light skin smiles at another woman "
        "with long blonde hair. The woman with brown hair wears a black jacket "
        "and has a small, barely noticeable parsing of her lips. Background is "
        "a dimly lit room with a faint, warm glow."
    )

    logger.info(f"[Rank {rank}] Starting generation...")
    t0 = time.time()

    # Set seed for reproducibility
    generator = torch.Generator(device="cpu").manual_seed(42)

    output = pipe(
        prompt=prompt,
        height=384,
        width=512,
        num_frames=25,
        num_inference_steps=8,
        guidance_scale=4.0,
        generator=generator,
    )
    gen_time = time.time() - t0
    logger.info(f"[Rank {rank}] Generation complete in {gen_time:.1f}s")

    # ── Save output (rank 0 only) ──────────────────────────────────────
    if rank == 0:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(output_dir, exist_ok=True)

        # Save video frames
        if hasattr(output, "frames") and output.frames is not None:
            frames = output.frames
            if hasattr(frames, "__len__") and len(frames) > 0:
                # Export as mp4 if export_to_video is available
                try:
                    from diffusers.utils import export_to_video
                    video_path = os.path.join(output_dir, "ltx2_output.mp4")
                    export_to_video(frames[0], video_path, fps=24)
                    logger.info(f"Video saved to {video_path}")
                except Exception as e:
                    logger.warning(f"Could not export video: {e}")
                    # Fall back to saving individual frames
                    for i, frame in enumerate(frames[0][:5]):
                        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
                        if hasattr(frame, "save"):
                            frame.save(frame_path)

        # Save benchmark info
        import json
        bench_info = {
            "model": "Lightricks/LTX-2",
            "height": 384,
            "width": 512,
            "num_frames": 25,
            "num_inference_steps": 8,
            "guidance_scale": 4.0,
            "tp_degree": world_size,
            "generation_time_s": round(gen_time, 2),
            "per_step_time_s": round(gen_time / 8, 2),
            "seed": 42,
        }
        bench_path = os.path.join(output_dir, "bench_results.json")
        with open(bench_path, "w") as f:
            json.dump(bench_info, f, indent=2)
        logger.info(f"Benchmark saved to {bench_path}")

    # ── Cleanup ─────────────────────────────────────────────────────────
    dist.barrier()
    dist.destroy_process_group()
    logger.info(f"[Rank {rank}] Done.")


if __name__ == "__main__":
    main()

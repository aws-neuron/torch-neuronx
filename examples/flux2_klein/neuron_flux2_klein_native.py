"""FLUX.2-klein 4B image-to-image — native PyTorch + Beta 3.

A subclass of `diffusers.Flux2KleinPipeline` that keeps the text
encoder + VAE on CPU and runs only the DiT on Neuron. Lifted from a
parallel vllm-omni implementation, with vllm-omni-specific scaffolding
removed:

  * No `od_config`, `weights_sources`, `setup_diffusion_pipeline_profiler`
  * No `forward(req)` — uses the parent's standard `__call__`
  * No `compile_transformer` with vllm-omni compiler args — we call
    `torch.compile(model, backend="neuron", dynamic=False)` directly
    on the wrapped DiT once it's on Neuron.

The Neuron-specific patches are unchanged because they're framework-
agnostic (they fix actual Neuron / diffusers boundary problems):

  1. Scheduler `set_timesteps` rebuilt CPU-side then moved + bf16 cast
  2. `Timesteps` modules' sinusoidal embedding forced to CPU compute
  3. Module-level `get_1d_rotary_pos_embed` swapped for a real-
     arithmetic version (no `torch.polar` → no complex64 in the FX
     graph; Neuron has no complex dtype).
  4. `Flux2PosEmbed` submodule swapped for a CPU+fp32+real version
     (NeuronFluxPosEmbed) so position embeddings are computed before
     the compiled subgraph runs.
  5. `encode_prompt` runs Qwen3 on CPU; embeddings get moved to Neuron.
  6. `_encode_vae_image` runs VAE encode on CPU.
  7. `prepare_latents` / `prepare_image_latents` force bf16 + CPU
     `torch.Generator`.
  8. `_NeuronTransformerWrapper` coerces every transformer-input tensor
     to the device + `.contiguous()` at the call boundary.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch
import torch.nn as nn

from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scheduler patch (CPU build + bf16 pre-cast + move)
# ---------------------------------------------------------------------------
def _make_neuron_scheduler(scheduler, target_dtype):
    """Patch `set_timesteps` to build CPU-side + cast + move."""
    import functools

    base_set = scheduler.set_timesteps

    @functools.wraps(base_set)
    def patched_set_timesteps(num_inference_steps=None, device=None,
                              sigmas=None, mu=None, **kwargs):
        target_device = device
        cpu_kwargs = dict(kwargs)
        if sigmas is not None:
            cpu_kwargs["sigmas"] = sigmas
        if mu is not None:
            cpu_kwargs["mu"] = mu
        cpu_kwargs["device"] = torch.device("cpu")
        result = (
            base_set(num_inference_steps=num_inference_steps, **cpu_kwargs)
            if num_inference_steps is not None
            else base_set(**cpu_kwargs)
        )
        if hasattr(scheduler, "timesteps") and scheduler.timesteps is not None:
            if target_dtype is not None:
                scheduler.timesteps = scheduler.timesteps.to(dtype=target_dtype)
            if target_device is not None:
                scheduler.timesteps = scheduler.timesteps.to(device=target_device)
        if hasattr(scheduler, "sigmas") and scheduler.sigmas is not None:
            if target_dtype is not None:
                scheduler.sigmas = scheduler.sigmas.to(dtype=target_dtype)
            if target_device is not None:
                scheduler.sigmas = scheduler.sigmas.to(device=target_device)
        return result

    scheduler.set_timesteps = patched_set_timesteps
    return scheduler


# ---------------------------------------------------------------------------
# Timesteps modules → CPU compute
# ---------------------------------------------------------------------------
def _patch_timesteps_to_cpu(transformer):
    """Force `Timesteps.forward()` (sin/cos table) on CPU."""
    from diffusers.models.embeddings import Timesteps
    targets = [(n, m) for n, m in transformer.named_modules() if isinstance(m, Timesteps)]
    for name, sub in targets:
        orig_forward = sub.forward

        def _make_wrapped(_orig):
            def _wrapped(timesteps):
                orig_device = timesteps.device
                ts_cpu = timesteps.to(device="cpu")
                with torch.no_grad():
                    out = _orig(ts_cpu)
                return out.to(device=orig_device)
            return _wrapped

        sub.forward = _make_wrapped(orig_forward)
        logger.info("[neuron-flux2] patched Timesteps at %s to run on CPU", name)

    # Class-level: even if Dynamo unwraps the instance patch.
    try:
        import diffusers.models.embeddings as _emb_mod
        _orig_gte = _emb_mod.get_timestep_embedding

        def _gte_cpu_arange(timesteps, embedding_dim, *args, **kwargs):
            orig_device = timesteps.device
            ts_cpu = timesteps.to(device="cpu")
            with torch.no_grad():
                out = _orig_gte(ts_cpu, embedding_dim, *args, **kwargs)
            return out.to(device=orig_device)

        if not getattr(_emb_mod, "_neuron_flux2_patched", False):
            _emb_mod.get_timestep_embedding = _gte_cpu_arange
            _emb_mod._neuron_flux2_patched = True
    except Exception as _e:
        logger.debug("[neuron-flux2] class-level patch skipped: %s", _e)


# ---------------------------------------------------------------------------
# Real-arithmetic 1D RoPE (no torch.polar, no complex64)
# ---------------------------------------------------------------------------
def _patch_get_1d_rotary_pos_embed_real():
    try:
        import diffusers.models.embeddings as _emb_mod
    except Exception as _e:
        logger.debug("[neuron-flux2] diffusers.embeddings import skipped: %s", _e)
        return
    if getattr(_emb_mod, "_neuron_flux2_g1d_real_patched", False):
        return

    class _ComplexShim:
        __slots__ = ("real", "imag")
        def __init__(self, real, imag):
            self.real = real
            self.imag = imag

    def _real_get_1d_rotary_pos_embed(
        dim, pos, theta=10000.0, use_real=False,
        linear_factor=1.0, ntk_factor=1.0,
        repeat_interleave_real=True, freqs_dtype=torch.float32,
    ):
        if freqs_dtype is None or freqs_dtype == torch.float64:
            freqs_dtype = torch.float32

        orig_device = pos.device if torch.is_tensor(pos) else None
        pos_cpu = pos.detach().to(device="cpu") if torch.is_tensor(pos) else pos

        theta = theta * ntk_factor
        freqs = (
            1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)) / linear_factor
        )

        if torch.is_tensor(pos_cpu):
            freqs = torch.outer(pos_cpu.double(), freqs)
        else:
            freqs = torch.outer(torch.tensor(pos_cpu, dtype=torch.float64), freqs)

        cos = freqs.cos()
        sin = freqs.sin()

        if use_real:
            if repeat_interleave_real:
                cos_full = cos.repeat_interleave(2, dim=-1).to(freqs_dtype)
                sin_full = sin.repeat_interleave(2, dim=-1).to(freqs_dtype)
            else:
                cos_full = torch.cat([cos, cos], dim=-1).to(freqs_dtype)
                sin_full = torch.cat([sin, sin], dim=-1).to(freqs_dtype)
            if orig_device is not None:
                cos_full = cos_full.to(device=orig_device)
                sin_full = sin_full.to(device=orig_device)
            return cos_full, sin_full

        cos_half = cos.to(freqs_dtype)
        sin_half = sin.to(freqs_dtype)
        if orig_device is not None:
            cos_half = cos_half.to(device=orig_device)
            sin_half = sin_half.to(device=orig_device)
        return _ComplexShim(cos_half, sin_half)

    _emb_mod.get_1d_rotary_pos_embed = _real_get_1d_rotary_pos_embed
    _emb_mod._neuron_flux2_g1d_real_patched = True
    logger.info("[neuron-flux2] module-level get_1d_rotary_pos_embed patched (real-arith, no torch.polar)")


# ---------------------------------------------------------------------------
# Flux2PosEmbed → CPU + fp32 + real RoPE
# ---------------------------------------------------------------------------
def _patch_pos_embed_to_cpu(transformer):
    pe = getattr(transformer, "pos_embed", None)
    if pe is None or not hasattr(pe, "forward"):
        return

    pe_class = type(pe)
    if getattr(pe_class, "_neuron_flux2_patched", False):
        pass

    from diffusers.models.embeddings import get_1d_rotary_pos_embed

    class _NeuronFluxPosEmbed(nn.Module):
        def __init__(self, axes_dim, theta):
            super().__init__()
            self.axes_dim = list(axes_dim)
            self.theta = theta

        def forward(self, ids):
            orig_device = ids.device
            ids_cpu = ids.to(device="cpu")
            cos_out = []
            sin_out = []
            pos = ids_cpu.float()
            with torch.no_grad():
                for i in range(len(self.axes_dim)):
                    # Match the upstream Flux2PosEmbed.forward signature
                    # exactly: use_real=True, repeat_interleave_real=True
                    # produces full-dim cos / sin tensors. The CPU+fp32
                    # compute keeps torch.polar / complex64 out of the
                    # FX graph (Neuron has no complex dtype) but emits
                    # the same shapes the rest of the model expects.
                    cos, sin = get_1d_rotary_pos_embed(
                        self.axes_dim[i], pos[..., i], theta=self.theta,
                        repeat_interleave_real=True,
                        use_real=True,
                        freqs_dtype=torch.float32,
                    )
                    cos_out.append(cos.contiguous())
                    sin_out.append(sin.contiguous())
            freqs_cos = torch.cat(cos_out, dim=-1).to(device=orig_device)
            freqs_sin = torch.cat(sin_out, dim=-1).to(device=orig_device)
            return freqs_cos, freqs_sin

    new_pe = _NeuronFluxPosEmbed(pe.axes_dim, pe.theta)
    transformer.pos_embed = new_pe
    if hasattr(transformer, "rope_prepare") and hasattr(transformer.rope_prepare, "pos_embed"):
        transformer.rope_prepare.pos_embed = new_pe
    pe_class.forward = _NeuronFluxPosEmbed.forward
    pe_class._neuron_flux2_patched = True
    logger.info("[neuron-flux2] swapped Flux2PosEmbed for CPU+fp32+real RoPE")


# ---------------------------------------------------------------------------
# Wrapper around the DiT — coerces inputs at the boundary
# ---------------------------------------------------------------------------
class _NeuronTransformerWrapper(nn.Module):
    """Sits OUTSIDE the torch.compile boundary. Moves inputs +
    .contiguous() before the inner DiT runs.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self._target_device = None

    @property
    def config(self):
        return self.inner.config

    @property
    def dtype(self):
        if hasattr(self.inner, "dtype"):
            return self.inner.dtype
        for p in self.inner.parameters():
            return p.dtype
        return torch.bfloat16

    def _resolve_target_device(self):
        if self._target_device is not None:
            return self._target_device
        for p in self.inner.parameters():
            self._target_device = p.device
            return self._target_device
        return torch.device("cpu")

    def forward(self, *args, **kwargs):
        target = self._resolve_target_device()
        for key in (
            "hidden_states", "timestep", "guidance",
            "encoder_hidden_states", "txt_ids", "img_ids",
        ):
            v = kwargs.get(key)
            if v is not None and torch.is_tensor(v) and v.device != target:
                kwargs[key] = v.contiguous().to(device=target)
        # Args may also have unmoved tensors
        new_args = []
        for a in args:
            if torch.is_tensor(a) and a.device != target:
                new_args.append(a.contiguous().to(device=target))
            else:
                new_args.append(a)
        return self.inner(*new_args, **kwargs)

    def to(self, *args, **kwargs):
        self._target_device = None
        self.inner.to(*args, **kwargs)
        return self

    def __getattr__(self, name):
        # Forward arbitrary attribute lookups to inner (e.g.
        # `cache_context`, `enable_*`, custom diffusers utilities)
        # while preserving nn.Module's own dispatch.
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self.__dict__.get("_modules", {}).get("inner")
            if inner is None:
                raise
            return getattr(inner, name)


# ---------------------------------------------------------------------------
# The pipeline subclass
# ---------------------------------------------------------------------------
class NeuronFlux2KleinPipeline(Flux2KleinPipeline):
    """Native-PyTorch Beta 3 FLUX.2-klein with Neuron-resident DiT only.

    Construct with
        pipe = NeuronFlux2KleinPipeline.from_pretrained(...)
    then call `apply_neuron_patches()` to install all the boundary
    fixes, then move the transformer to Neuron via
        pipe.transformer.to(torch.device("neuron"))
    Encoders + VAE stay on CPU.

    Use the standard `__call__` for inference.
    """

    def apply_neuron_patches(self, neuron_device: torch.device, dtype=torch.bfloat16):
        """Apply the 8 Neuron patches in the right order. Idempotent."""
        # 1. Scheduler — CPU build then move + bf16 cast
        self.scheduler = _make_neuron_scheduler(self.scheduler, dtype)
        # 2. Timesteps modules (sin/cos table off device)
        _patch_timesteps_to_cpu(self.transformer)
        # 3. Module-level real-arithmetic RoPE (avoid torch.polar / complex64)
        _patch_get_1d_rotary_pos_embed_real()
        # 4. Flux2PosEmbed → CPU+fp32+real
        _patch_pos_embed_to_cpu(self.transformer)
        # 5+8. Wrap the DiT (will keep params; .to(device) moves them)
        self.transformer = _NeuronTransformerWrapper(self.transformer)
        self._neuron_device = neuron_device

        # 9. Patch VAE.decode to coerce its (Neuron-resident) latents
        # input to CPU. The VAE itself stays on CPU.
        _vae = self.vae
        _orig_decode = _vae.decode

        import functools as _ft

        @_ft.wraps(_orig_decode)
        def _patched_decode(z, *args, **kwargs):
            if torch.is_tensor(z) and z.device.type != "cpu":
                z = z.to(device="cpu")
            return _orig_decode(z, *args, **kwargs)

        _vae.decode = _patched_decode
        return self

    # ---- Selective device move ----
    def to(self, *args, **kwargs):
        """Move ONLY the (wrapped) transformer; encoders + VAE stay on CPU."""
        if self.transformer is not None:
            self.transformer.to(*args, **kwargs)
        return self

    @property
    def device(self):
        # Pretend the pipeline lives on the Neuron device so
        # `_execution_device` and downstream code that depends on
        # `pipeline.device` see Neuron, not CPU.
        return getattr(self, "_neuron_device", torch.device("cpu"))

    @property
    def _execution_device(self):  # noqa: D401
        return getattr(self, "_neuron_device", torch.device("cpu"))

    # ---- encode_prompt — Qwen3 stays on CPU; embeddings get moved ----
    def encode_prompt(
        self,
        prompt,
        device=None,
        num_images_per_prompt: int = 1,
        prompt_embeds=None,
        max_sequence_length: int = 512,
        text_encoder_out_layers=(9, 18, 27),
    ):
        target_device = device or self._execution_device

        if prompt is None:
            prompt = ""
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            # Force CPU — encoder lives on CPU.
            prompt_embeds = self._get_qwen3_prompt_embeds(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                prompt=prompt,
                device=torch.device("cpu"),
                max_sequence_length=max_sequence_length,
                hidden_states_layers=text_encoder_out_layers,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1,
        )

        text_ids = self._prepare_text_ids(prompt_embeds)

        # Move embeddings to target device (Neuron) — text_encoder
        # itself stays on CPU, so its weights aren't disturbed.
        encoder_dtype = getattr(self.text_encoder, "dtype", torch.bfloat16)
        prompt_embeds = prompt_embeds.to(dtype=encoder_dtype)
        prompt_embeds = prompt_embeds.to(device=target_device)
        text_ids = text_ids.to(device=target_device)
        return prompt_embeds, text_ids

    # ---- VAE encode on CPU ----
    def _encode_vae_image(self, image, generator):
        if image.ndim != 4:
            raise ValueError(f"Expected image dims 4, got {image.ndim}.")
        image_cpu = image.to(device="cpu")
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
            retrieve_latents,
        )
        image_latents = retrieve_latents(
            self.vae.encode(image_cpu), generator=generator, sample_mode="argmax",
        )
        image_latents = self._patchify_latents(image_latents)
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
            image_latents.device, image_latents.dtype,
        )
        latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps,
        )
        image_latents = (image_latents - latents_bn_mean) / latents_bn_std
        return image_latents

    # ---- prepare_latents — force bf16 + CPU generator ----
    def prepare_latents(self, batch_size, num_latents_channels, height, width,
                        dtype, device, generator, latents=None):
        transformer_dtype = (
            self.transformer.dtype
            if (self.transformer is not None and hasattr(self.transformer, "dtype"))
            else torch.bfloat16
        )
        dtype = transformer_dtype
        if isinstance(generator, torch.Generator) and generator.device.type != "cpu":
            generator = torch.Generator(device="cpu").manual_seed(generator.initial_seed())
        return super().prepare_latents(
            batch_size, num_latents_channels, height, width,
            dtype, device, generator, latents,
        )

    def prepare_image_latents(self, images, batch_size, generator, device, dtype):
        transformer_dtype = (
            self.transformer.dtype
            if (self.transformer is not None and hasattr(self.transformer, "dtype"))
            else torch.bfloat16
        )
        dtype = transformer_dtype
        if isinstance(generator, torch.Generator) and generator.device.type != "cpu":
            generator = torch.Generator(device="cpu").manual_seed(generator.initial_seed())
        image_latents, image_latent_ids = super().prepare_image_latents(
            images, batch_size, generator, device, dtype,
        )
        # Both results must live on the target device for the downstream
        # `torch.cat([latents, image_latents], dim=1)` in the denoising
        # loop. The parent moves `image_latent_ids` (final `.to(device)`)
        # but leaves `image_latents` on CPU because we forced the VAE
        # encode there.
        target_device = self._execution_device
        image_latents = image_latents.to(device=target_device)
        image_latent_ids = image_latent_ids.to(device=target_device)
        return image_latents, image_latent_ids

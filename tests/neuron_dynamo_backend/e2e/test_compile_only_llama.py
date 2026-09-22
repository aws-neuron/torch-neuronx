"""E2E compile-only tests: Llama-3.2 architecture (prefill + decode/KV-aliasing).

Compiles a reduced-layer but otherwise architecture-faithful Llama-3.2-1B
(GQA 32/8 heads, 128k vocab, rope_theta=500k) to NEFF with NO Neuron device,
asserting what the tiny-config HF sweep does not:

- Prefill: placeholder output SPEC PARITY vs eager (shape/dtype/device) — a
  compile-only backend must return exactly what AOTAutograd's runtime wrapper
  expects, not merely "some artifact".
- Decode: a KV-cache graph with in-place ``index_copy_`` updates — the input
  mutations must surface as ``io_spec.aliases`` (the mhlo.input_output_alias
  handoff a caller-owned runtime binds buffers with). The bundled HF configs
  all run ``use_cache=False``, so this is the only coverage of the mutation
  aliasing path through the public artifact.

Follows the conventions of test_compile_only_hf_e2e.py (device-free, gated on
neuronxcc, compile_only() context).
"""

import pytest
import torch
import torch.nn as nn

from tests.neuron_dynamo_backend.compile_only_utils import (
    COMPILE_ONLY_TARGET,
    requires_compiler,
)

transformers = pytest.importorskip("transformers", reason="transformers not installed")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

BATCH, SEQ = 1, 128


def _llama32_1b_config(num_layers: int = 2) -> LlamaConfig:
    """Llama-3.2-1B geometry, layer count reduced for compile speed.

    Head geometry, GQA ratio, vocab, and rope are the real 1B values so the
    attention/rope lowering matches production graphs; only depth is reduced.
    """
    cfg = LlamaConfig(
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=num_layers,  # 1B ships 16
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        rope_theta=500000.0,
        max_position_embeddings=131072,
        tie_word_embeddings=True,
    )
    cfg._attn_implementation = "eager"  # CPU flash-SDPA op has no Neuron lowering
    cfg.use_cache = False  # prefill graph; decode covered separately below
    return cfg


def _unwrap_forwards(model):
    """Strip transformers' decorators (try/finally + contextvars) Dynamo can't trace."""
    patched = set()
    for module in model.modules():
        cls = type(module)
        if cls in patched:
            continue
        fwd = cls.__dict__.get("forward")
        if fwd is None:
            continue
        original = fwd
        while hasattr(original, "__wrapped__"):
            original = original.__wrapped__
        if original is not fwd:
            cls.forward = original
            patched.add(cls)


@requires_compiler
class TestCompileOnlyLlamaPrefill:
    """Llama-3.2-1B prefill forward, device-free, with output-spec parity."""

    def test_prefill_compiles_with_output_parity(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        import torch_neuronx
        from torch_neuronx.compiler import CompileOptions, compile_only

        cfg = _llama32_1b_config()
        model = LlamaForCausalLM(cfg).eval()
        _unwrap_forwards(model)
        input_ids = torch.randint(0, cfg.vocab_size, (BATCH, SEQ))

        with torch.no_grad():
            eager_logits = model(input_ids=input_ids).logits

        torch._dynamo.reset()
        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            m = model.to("neuron")
            ids = input_ids.to("neuron")
            with torch.no_grad():
                out = torch.compile(m, backend="neuron")(input_ids=ids)
            logits = out.logits

            artifact = getattr(logits, "_neff_artifact", None)
            assert (
                artifact is not None and artifact.cache_key
            ), "expected a NEFF handle on the compile-only prefill output"
            assert not torch_neuronx._C._is_neuron_runtime_initialized()

            # Placeholder output spec parity vs eager.
            assert tuple(logits.shape) == tuple(eager_logits.shape)
            assert logits.dtype == eager_logits.dtype
            assert logits.device.type == "neuron"

            torch.neuron.synchronize()
            import torch_neuronx

            neff_info = torch_neuronx._C.get_neff_info(artifact.cache_key)
            neff = neff_info[1] if neff_info else None
            assert neff and len(neff) > 0
            assert artifact.io_spec.inputs and artifact.io_spec.outputs


class _DecodeAttention(nn.Module):
    """Decode-shaped attention block with in-place KV-cache updates.

    Mirrors the production decode pattern (Llama-3.2-1B geometry: GQA 32/8,
    head_dim 64): cache buffers mutated via ``index_copy_`` — input mutations
    that AOTAutograd functionalizes and the Neuron FX passes must surface as
    ``io_spec.aliases`` so a runtime binds the outputs onto the input buffers.
    """

    def __init__(self, n_heads=32, n_kv_heads=8, head_dim=64, max_seq=256, hidden=2048):
        super().__init__()
        self.n_heads, self.n_kv, self.hd = n_heads, n_kv_heads, head_dim
        self.q = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.k = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.v = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.o = nn.Linear(n_heads * head_dim, hidden, bias=False)
        self.register_buffer("k_cache", torch.zeros(1, n_kv_heads, max_seq, head_dim))
        self.register_buffer("v_cache", torch.zeros(1, n_kv_heads, max_seq, head_dim))

    def forward(self, x, pos):
        batch, seq, _ = x.shape  # decode: seq == 1
        q = self.q(x).view(batch, seq, self.n_heads, self.hd).transpose(1, 2)
        k = self.k(x).view(batch, seq, self.n_kv, self.hd).transpose(1, 2)
        v = self.v(x).view(batch, seq, self.n_kv, self.hd).transpose(1, 2)
        self.k_cache.index_copy_(2, pos, k)
        self.v_cache.index_copy_(2, pos, v)
        rep = self.n_heads // self.n_kv
        keys = self.k_cache.repeat_interleave(rep, dim=1)
        vals = self.v_cache.repeat_interleave(rep, dim=1)
        attn = torch.softmax(q @ keys.transpose(-1, -2) / (self.hd**0.5), dim=-1)
        out = (attn @ vals).transpose(1, 2).reshape(batch, seq, -1)
        return self.o(out)


@requires_compiler
class TestCompileOnlyLlamaDecode:
    """Decode graph with in-place KV mutation: the io_spec.aliases handoff."""

    def test_decode_kv_mutations_surface_as_aliases(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        from torch_neuronx.compiler import CompileOptions, compile_only

        torch._dynamo.reset()
        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            m = _DecodeAttention().to("neuron")
            x = torch.randn(1, 1, 2048, device="neuron")
            pos = torch.tensor([5], device="neuron")
            with torch.no_grad():
                out = torch.compile(m, backend="neuron", fullgraph=True)(x, pos)

            artifact = getattr(out, "_neff_artifact", None)
            assert artifact is not None and artifact.cache_key
            assert tuple(out.shape) == (1, 1, 2048)

            # Both cache mutations must be donated via input_output_alias.
            aliases = artifact.io_spec.aliases
            assert len(aliases) >= 2, (
                f"expected k_cache+v_cache mutation aliases, got "
                f"{[(a.input_index, a.output_index) for a in aliases]}"
            )

            torch.neuron.synchronize()
            import torch_neuronx

            neff_info = torch_neuronx._C.get_neff_info(artifact.cache_key)
            neff = neff_info[1] if neff_info else None
            assert neff and len(neff) > 0

"""E2E compile-only tests over small HuggingFace CausalLM configs.

Compiles real HF architectures to NEFF on a host with NO Neuron device, via
``torch.compile(backend="neuron")``, under BOTH the ``compile_only()`` context
manager and the ``TORCH_NEURONX_COMPILE_ONLY`` env var (subprocess).

Config-driven, matching tests/huggingface convention: models are built from the
bundled JSON configs with ``AutoModel.from_config`` and reduced layers — no
weight download, no hub access.
"""

import os

import pytest
import torch

from tests.neuron_dynamo_backend.compile_only_utils import (
    COMPILE_ONLY_LOGICAL_NC,
    COMPILE_ONLY_TARGET,
    count_artifacts,
    parse_result_line,
    requires_compiler,
    run_in_subprocess,
)

transformers = pytest.importorskip("transformers", reason="transformers not installed")

from transformers import AutoConfig, AutoModel  # noqa: E402

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "huggingface",
    "configs",
    "huggingface",
    "CausalLM",
)

MODELS = [
    "openai-community_gpt2",
    "bigscience_bloomz-560m",
    "fxmarty_really-tiny-falcon-testing",
    "onnx-internal-testing_tiny-random-ArceeForCausalLM",
]

NUM_LAYERS = 2
SEQ_LEN = 16


def _config_path(model_id: str) -> str:
    return os.path.join(CONFIG_DIR, f"{model_id}_config.json")


def _reduce_layers(config, num_layers: int):
    for attr in ("num_hidden_layers", "num_layers", "n_layer", "num_encoder_layers"):
        if hasattr(config, attr):
            setattr(config, attr, num_layers)


def _build_model(config_path: str):
    """Build a small HF model from config, with the workarounds needed for
    fullgraph Dynamo tracing: eager attention (the CPU flash-SDPA op has no
    Neuron lowering), no KV cache (DynamicCache's deepcopy breaks Dynamo), and
    unwrapped forward decorators (try/finally + contextvars can't be traced)."""
    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    config._attn_implementation = "eager"  # avoid CPU flash-SDPA op that won't lower
    config.use_cache = False  # DynamicCache uses deepcopy, breaks Dynamo
    _reduce_layers(config, NUM_LAYERS)
    model = AutoModel.from_config(config, trust_remote_code=True)

    # Unwrap decorators that use contextvars (untraceable by Dynamo).
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
    return model, config


def _make_inputs(config, batch=1, seq_len=SEQ_LEN):
    vocab = getattr(config, "vocab_size", getattr(config, "n_vocab", 32000))
    return {
        "input_ids": torch.randint(0, vocab, (batch, seq_len)),
        "attention_mask": torch.ones(batch, seq_len, dtype=torch.long),
    }


@requires_compiler
class TestCompileOnlyHFContext:
    """backend='neuron' under compile_only() over bundled HF configs, device-free."""

    @pytest.mark.parametrize("model_id", MODELS)
    def test_forward_compiles(self, model_id, tmp_path, monkeypatch):
        config_path = _config_path(model_id)
        if not os.path.exists(config_path):
            pytest.skip(f"bundled config missing: {config_path}")

        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        monkeypatch.setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "1")
        import torch_neuronx
        from torch_neuronx.compiler import CompileOptions, compile_only

        model, config = _build_model(config_path)
        inputs = _make_inputs(config)

        torch._dynamo.reset()
        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            model = model.to("neuron")
            inputs = {k: v.to("neuron") for k, v in inputs.items()}
            with torch.no_grad():
                out = torch.compile(model, backend="neuron")(**inputs)

        assert not torch_neuronx._C._is_neuron_runtime_initialized()
        artifact = getattr(_first_tensor(out), "_neff_artifact", None)
        assert (
            artifact is not None and artifact.cache_key
        ), f"{model_id}: expected a NEFF handle from compile-only forward"
        num_stablehlo, _ = count_artifacts(str(tmp_path))
        assert num_stablehlo >= 1, f"{model_id}: no StableHLO produced"

    @pytest.mark.parametrize("model_id", MODELS[:2])
    def test_backward_compiles(self, model_id, tmp_path, monkeypatch):
        """Training path: fwd+bwd both lower to NEFFs device-free."""
        config_path = _config_path(model_id)
        if not os.path.exists(config_path):
            pytest.skip(f"bundled config missing: {config_path}")

        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        monkeypatch.setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "1")
        from torch_neuronx.compiler import CompileOptions, compile_only

        model, config = _build_model(config_path)
        inputs = _make_inputs(config)

        torch._dynamo.reset()
        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            model = model.to("neuron").train()
            inputs = {k: v.to("neuron") for k, v in inputs.items()}
            out = torch.compile(model, backend="neuron")(**inputs)
            loss = _first_tensor(out).sum()
            loss.backward()  # triggers the backward compiler → second NEFF

        num_stablehlo, _ = count_artifacts(str(tmp_path))
        assert num_stablehlo >= 2, f"{model_id}: expected fwd+bwd StableHLO, got {num_stablehlo}"


@requires_compiler
class TestCompileOnlyHFEnvVar:
    """Same compile, driven by TORCH_NEURONX_COMPILE_ONLY=1 in a fresh process."""

    @pytest.mark.parametrize("model_id", MODELS[:2])
    def test_forward_compiles_env(self, model_id, tmp_path):
        config_path = _config_path(model_id)
        if not os.path.exists(config_path):
            pytest.skip(f"bundled config missing: {config_path}")

        body = f"""
        import os, torch
        import torch_neuronx
        from transformers import AutoConfig, AutoModel

        assert not torch_neuronx._C._is_neuron_runtime_initialized()

        config = AutoConfig.from_pretrained({config_path!r}, trust_remote_code=True)
        config._attn_implementation = "eager"
        config.use_cache = False
        for attr in ("num_hidden_layers", "num_layers", "n_layer"):
            if hasattr(config, attr):
                setattr(config, attr, {NUM_LAYERS})
        model = AutoModel.from_config(config, trust_remote_code=True)

        patched = set()
        for m in model.modules():
            cls = type(m)
            if cls in patched:
                continue
            fwd = cls.__dict__.get("forward")
            if fwd is None:
                continue
            orig = fwd
            while hasattr(orig, "__wrapped__"):
                orig = orig.__wrapped__
            if orig is not fwd:
                cls.forward = orig
                patched.add(cls)

        vocab = getattr(config, "vocab_size", 32000)
        inputs = {{
            "input_ids": torch.randint(0, vocab, (1, {SEQ_LEN})).to("neuron"),
            "attention_mask": torch.ones(1, {SEQ_LEN}, dtype=torch.long).to("neuron"),
        }}
        model = model.to("neuron")
        with torch.no_grad():
            out = torch.compile(model, backend="neuron")(**inputs)

        def first_tensor(o):
            if isinstance(o, torch.Tensor):
                return o
            if hasattr(o, "last_hidden_state"):
                return o.last_hidden_state
            if hasattr(o, "logits"):
                return o.logits
            if isinstance(o, (tuple, list)):
                return o[0]
            for v in getattr(o, "__dict__", {{}}).values():
                if isinstance(v, torch.Tensor):
                    return v
            raise AssertionError("no tensor in output")

        art = getattr(first_tensor(out), "_neff_artifact", None)
        assert art is not None and art.cache_key
        print("RESULT:" + art.cache_key)
        """
        proc = run_in_subprocess(
            body,
            env_overrides={
                "TORCH_NEURONX_COMPILE_ONLY": "1",
                "NEURON_PLATFORM_TARGET_OVERRIDE": COMPILE_ONLY_TARGET,
                "NEURON_LOGICAL_NC_CONFIG": COMPILE_ONLY_LOGICAL_NC,
                "TORCH_NEURONX_DEBUG_DIR": str(tmp_path),
            },
        )
        assert parse_result_line(proc.stdout), f"{model_id}: no cache_key from env-var compile"


def _first_tensor(out):
    """Pull the primary tensor from an HF model output (ModelOutput/tuple/tensor)."""
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits
    if isinstance(out, tuple | list):
        return out[0]
    for v in getattr(out, "__dict__", {}).values():
        if isinstance(v, torch.Tensor):
            return v
    raise AssertionError(f"no tensor found in output of type {type(out)}")

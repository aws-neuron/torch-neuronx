import pytest
import torch

import torch_neuronx.neuron_dynamo_backend.decompositions as decompositions
from torch_neuronx.neuron_dynamo_backend.decompositions import (
    _can_use_nki_flash_attention,
    _can_use_nki_flash_attention_backward,
    _reduce_gqa_gradients,
)


@pytest.fixture(autouse=True)
def enable_nki_sdpa(monkeypatch):
    monkeypatch.setenv("TORCH_NEURONX_ENABLE_NKI_SDPA", "1")
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    monkeypatch.setattr(decompositions, "_NKI_ATTENTION_SUPPORTS_V_HEAD_DIM", True)


def make_qkv(
    *,
    batch=1,
    q_heads=2,
    kv_heads=2,
    q_len=512,
    kv_len=512,
    qk_dim=128,
    v_dim=128,
    dtype=torch.bfloat16,
):
    query = torch.empty(batch, q_heads, q_len, qk_dim, dtype=dtype, device="meta")
    key = torch.empty(batch, kv_heads, kv_len, qk_dim, dtype=dtype, device="meta")
    value = torch.empty(batch, kv_heads, kv_len, v_dim, dtype=dtype, device="meta")
    return query, key, value


@pytest.mark.parametrize("seq_len", [128, 512])
@pytest.mark.parametrize(
    "dtype,expected",
    [(torch.float16, False), (torch.bfloat16, True), (torch.float32, True)],
)
def test_nki_flash_attention_dtype_selection(dtype, expected, seq_len):
    qkv = make_qkv(dtype=dtype, q_len=seq_len, kv_len=seq_len)
    assert _can_use_nki_flash_attention(*qkv) == expected
    assert _can_use_nki_flash_attention(*qkv, requires_lse=False) == expected
    assert _can_use_nki_flash_attention_backward(*qkv, is_causal=True) == expected


@pytest.mark.parametrize(
    "shape",
    [
        {},
        {"q_heads": 24, "kv_heads": 24, "qk_dim": 192, "v_dim": 128},
        {"q_heads": 4, "kv_heads": 2, "qk_dim": 192, "v_dim": 128},
        {"q_len": 640, "kv_len": 577, "qk_dim": 192, "v_dim": 128},
        {"q_len": 512, "kv_len": 1024, "qk_dim": 192, "v_dim": 128},
        {"qk_dim": 512, "v_dim": 512},
        {"q_heads": 512, "kv_heads": 512, "q_len": 128, "kv_len": 128},
    ],
)
def test_nki_flash_attention_forward_accepts_supported_boundaries(shape):
    assert _can_use_nki_flash_attention(*make_qkv(**shape))


@pytest.mark.parametrize(
    "shape",
    [
        {"qk_dim": 513},
        {"v_dim": 513},
        {"qk_dim": 128, "v_dim": 256},
        {"q_heads": 513, "kv_heads": 513, "q_len": 128, "kv_len": 128},
        {"q_heads": 3, "kv_heads": 2},
    ],
)
def test_nki_flash_attention_forward_rejects_unsupported_boundaries(shape):
    assert not _can_use_nki_flash_attention(*make_qkv(**shape))


def test_nki_flash_attention_forward_lse_sequence_boundary():
    qkv = make_qkv(q_len=129)
    assert not _can_use_nki_flash_attention(*qkv)
    assert _can_use_nki_flash_attention(*qkv, requires_lse=False)


def test_nki_flash_attention_forward_rejects_unsupported_options():
    qkv = make_qkv()
    assert not _can_use_nki_flash_attention(*qkv, attn_bias=torch.empty(1, device="meta"))
    assert not _can_use_nki_flash_attention(*qkv, dropout_p=0.1)
    assert not _can_use_nki_flash_attention(*qkv, return_debug_mask=True)


def test_nki_flash_attention_rejects_asymmetric_dims_without_nkl_support(monkeypatch):
    monkeypatch.setattr(decompositions, "_NKI_ATTENTION_SUPPORTS_V_HEAD_DIM", False)
    asymmetric = make_qkv(qk_dim=192, v_dim=128)
    assert not _can_use_nki_flash_attention(*asymmetric)
    assert not _can_use_nki_flash_attention_backward(*asymmetric)


@pytest.mark.parametrize(
    "shape",
    [
        {},
        {"q_heads": 24, "kv_heads": 24, "qk_dim": 192, "v_dim": 128},
        {"q_heads": 4, "kv_heads": 2, "qk_dim": 192, "v_dim": 128},
        {"qk_dim": 512, "v_dim": 512},
        {"q_heads": 1, "kv_heads": 1, "qk_dim": 640},
        {"q_heads": 1, "kv_heads": 1, "qk_dim": 1024},
        {"q_heads": 1, "kv_heads": 1, "qk_dim": 2048},
        {"q_len": 128, "kv_len": 128},
        {"q_len": 256, "kv_len": 256},
    ],
)
def test_nki_flash_attention_backward_accepts_supported_boundaries(shape):
    assert _can_use_nki_flash_attention_backward(*make_qkv(**shape), is_causal=True)


@pytest.mark.parametrize(
    "shape",
    [
        {"qk_dim": 129},
        {"qk_dim": 260},
        {"qk_dim": 192, "v_dim": 260},
        {"q_heads": 1, "kv_heads": 1, "qk_dim": 2176},
        {"q_len": 384, "kv_len": 384},
        {"q_len": 512, "kv_len": 513},
        {"q_len": 129},
    ],
)
def test_nki_flash_attention_backward_rejects_unsupported_boundaries(shape):
    assert not _can_use_nki_flash_attention_backward(*make_qkv(**shape))


def test_nki_flash_attention_backward_causal_requires_square_sequence():
    qkv = make_qkv(q_len=512, kv_len=1024)
    assert _can_use_nki_flash_attention_backward(*qkv)
    assert not _can_use_nki_flash_attention_backward(*qkv, is_causal=True)


def test_nki_flash_attention_backward_rejects_zero_scale():
    assert not _can_use_nki_flash_attention_backward(*make_qkv(), scale=0.0)


def test_reduce_gqa_gradients_supports_asymmetric_head_dimensions():
    batch, kv_heads, q_heads, seq_len = 1, 2, 4, 3
    qk_dim, v_dim = 5, 7
    repeat_factor = q_heads // kv_heads
    grad_k = torch.arange(batch * q_heads * seq_len * qk_dim, dtype=torch.float32).reshape(
        batch * q_heads, seq_len, qk_dim
    )
    grad_v = torch.arange(batch * q_heads * seq_len * v_dim, dtype=torch.float32).reshape(
        batch * q_heads, seq_len, v_dim
    )

    reduced_k, reduced_v = _reduce_gqa_gradients(
        grad_k,
        grad_v,
        True,
        batch,
        kv_heads,
        q_heads,
        seq_len,
        qk_dim,
        v_dim,
    )

    expected_k = grad_k.view(batch, kv_heads, repeat_factor, seq_len, qk_dim).sum(dim=2)
    expected_v = grad_v.view(batch, kv_heads, repeat_factor, seq_len, v_dim).sum(dim=2)
    torch.testing.assert_close(reduced_k, expected_k)
    torch.testing.assert_close(reduced_v, expected_v)

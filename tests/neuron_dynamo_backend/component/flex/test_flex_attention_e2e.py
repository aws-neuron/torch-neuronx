"""End-to-end tests for flex_attention on Neuron.

| id                          | mask_mod                | score_mod        | fwd     | bwd     |
|-----------------------------|-------------------------|------------------|---------|---------|
| causal_fp32                 | causal                  | None             | PASS    | PASS    |
| causal_bf16                 | causal                  | None             | PASS    | PASS    |
| sliding_window_fp32         | sliding_window_mod      | None             | PASS    | PASS    |
| sliding_window_bf16         | sliding_window_mod      | soft_cap         | PASS    | PASS    |
| sliding_window_causal_fp32  | sliding_window_causal   | soft_cap         | PASS    | PASS    |
| sliding_window_causal_bf16  | sliding_window_causal   | None             | PASS    | PASS    |
| noop_fp32                   | noop_mask               | None             | PASS    | PASS    |
| head_bias_bf16              | noop_mask               | head_bias_score  | PASS    | PASS    |
| rel_fp32                    | causal                  | rel_score        | PASS    | PASS    |
| alibi_bf16                  | causal                  | alibi_score      | PASS    | PASS    |
| prefix_lm_fp32              | prefix_lm_mask_512      | None             | PASS    | PASS    |
| softcap_fp32                | causal                  | soft_cap         | PASS    | PASS    |
| softcap_exact_fp32          | causal                  | soft_cap         | PASS    | PASS    |
| dilated_sliding_window_bf16 | dilated_sliding_window  | None             | PASS    | PASS    |
| batchify_fp32               | batchify_causal_mask    | None             | PASS    | PASS    |
| sta_2d_bf16                 | sta_2d_mask             | None             | PASS    | PASS    |
| svg_spatial_fp32            | svg_spatial_mask        | None             | PASS    | PASS    |
| svg_temporal_bf16           | svg_temporal_mask       | None             | PASS    | PASS    |
| natten_bf16                 | natten_mask             | None             | PASS    | PASS    |
| sta_3d_bf16                 | sta_3d_mask             | None             | PASS    | PASS    |
| document_mask_bf16          | document_mask           | None             | PASS    | PASS    |
"""

import logging
import math
import os
from unittest.mock import patch

import pytest
import torch
from nkilib.core.attention.attention_cte_torch import attention_cte_torch_ref
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import (
    AuxRequest,
    and_masks,
    create_block_mask,
    create_mask,
    flex_attention,
)

from tests.utils.neuron_test_utils import max_all_close

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True, scope="module")
def enable_nki_free_var():
    """Enable NKI free-variable lowering for the suite.

    Masks/scores that capture tensors (e.g. document_mask) only lower correctly
    when TORCH_NEURONX_ENABLE_NKI_FREE_VAR is set; it is off by default.
    """
    prev = os.environ.get("TORCH_NEURONX_ENABLE_NKI_FREE_VAR")
    os.environ["TORCH_NEURONX_ENABLE_NKI_FREE_VAR"] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TORCH_NEURONX_ENABLE_NKI_FREE_VAR", None)
        else:
            os.environ["TORCH_NEURONX_ENABLE_NKI_FREE_VAR"] = prev


@pytest.fixture(autouse=True)
def disable_flex_attention_device_check():
    with patch("torch.nn.attention.flex_attention._validate_device", lambda *args, **kwargs: None):
        yield


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def sliding_window_mod(
    b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
) -> torch.Tensor:
    return q_idx - kv_idx < 128


def alibi_score(score, b, h, q_idx, kv_idx):
    m = 2 ** (-8)
    bias = m * (kv_idx - q_idx)
    return score + bias


softcap = 20


def soft_cap(score, b, h, q_idx, kv_idx):
    score = torch.tanh(score / softcap)
    return score


sliding_window_causal = and_masks(causal, sliding_window_mod)


def noop_mask(b, h, q_idx, kv_idx):
    return torch.ones_like(q_idx, dtype=torch.bool)


def head_bias_score(score, b, h, q_idx, kv_idx):
    return score + 2 * h


def rel_score(score, b, h, q_idx, kv_idx):
    return score + (q_idx - kv_idx)


def make_prefix_lm_mask(prefix_length):
    def prefix_mask(b, h, q_idx, kv_idx):
        return (kv_idx < prefix_length) | (q_idx >= kv_idx)

    return prefix_mask


def make_dilated_sliding_window_mask(window_size, dilation):
    def dilated_sliding_window(b, h, q_idx, kv_idx):
        diff = torch.abs(q_idx - kv_idx)
        return (diff <= window_size) & ((diff % dilation) == 0)

    return dilated_sliding_window


def make_batchify_mask(inner_mask_mod, batchify_size):
    def batched_mask_mod(b, h, q_idx, kv_idx):
        q_batch = q_idx // batchify_size
        kv_batch = kv_idx // batchify_size
        same_batch = q_batch == kv_batch
        inner = inner_mask_mod(b, h, q_idx % batchify_size, kv_idx % batchify_size)
        return same_batch & inner

    return batched_mask_mod


def make_sta_2d_mask(canvas_hw, kernel_hw, tile_hw):
    canvas_h, canvas_w = canvas_hw
    kernel_h, kernel_w = kernel_hw
    tile_h, tile_w = tile_hw
    tile_numel = tile_h * tile_w
    canvas_tile_h, canvas_tile_w = canvas_h // tile_h, canvas_w // tile_w
    kernel_tile_h, kernel_tile_w = kernel_h // tile_h, kernel_w // tile_w
    vision_seq_len = canvas_h * canvas_w

    def get_h_w_idx_tiled(idx):
        tile_id = idx // tile_numel
        return tile_id // canvas_tile_w, tile_id % canvas_tile_w

    def get_border(kernel_size):
        return kernel_size // 2, kernel_size // 2 + (kernel_size % 2 - 1)

    def sta_mask_mod_2d(b, h, q_idx, kv_idx):
        q_tile_h, q_tile_w = get_h_w_idx_tiled(q_idx)
        kv_tile_h, kv_tile_w = get_h_w_idx_tiled(kv_idx)
        left_border_h, right_border_h = get_border(kernel_tile_h)
        left_border_w, right_border_w = get_border(kernel_tile_w)
        kernel_center_h = q_tile_h.clamp(left_border_h, (canvas_tile_h - 1) - right_border_h)
        kernel_center_w = q_tile_w.clamp(left_border_w, (canvas_tile_w - 1) - right_border_w)
        h_mask = (kv_tile_h >= kernel_center_h - left_border_h) & (
            kv_tile_h <= kernel_center_h + right_border_h
        )
        w_mask = (kv_tile_w >= kernel_center_w - left_border_w) & (
            kv_tile_w <= kernel_center_w + right_border_w
        )
        vision_mask = (q_idx < vision_seq_len) & (kv_idx < vision_seq_len)
        return vision_mask & h_mask & w_mask

    return sta_mask_mod_2d


def make_svg_spatial_mask(prompt_length, num_frames, token_per_frame, width=2, round_width=128):
    mask_width = (width * token_per_frame) // round_width * round_width

    def spatial_mask_mod(b, h, q_idx, kv_idx):
        first_row_mask = q_idx < prompt_length
        first_column_mask = kv_idx < prompt_length
        spatial_head_mask = torch.abs(q_idx - kv_idx) < mask_width
        return first_column_mask | first_row_mask | spatial_head_mask

    return spatial_mask_mod


def make_svg_temporal_mask(prompt_length, num_frames, token_per_frame, width=2):
    mask_width = (width * token_per_frame) // num_frames

    def temporal_mask_mod(b, h, q_idx, kv_idx):
        first_row_mask = q_idx < prompt_length
        first_column_mask = kv_idx < prompt_length
        q_token_id = (q_idx - prompt_length) % token_per_frame
        kv_token_id = (kv_idx - prompt_length) % token_per_frame
        temporal_head_mask = torch.abs(q_token_id - kv_token_id) < mask_width
        return first_column_mask | first_row_mask | temporal_head_mask

    return temporal_mask_mod


def make_doc_mask(lengths, device="neuron"):
    # Captures tensors → exercises mask_mod_other_buffers path.
    offsets = torch.cumsum(torch.tensor([0, *lengths], dtype=torch.int32, device=device), dim=-1)
    counts = offsets[1:] - offsets[:-1]
    document_id = torch.repeat_interleave(
        torch.arange(len(counts), dtype=torch.int32, device=device), counts
    )

    def doc_mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_id[q_idx] == document_id[kv_idx]
        is_causal = q_idx >= kv_idx
        return same_doc & is_causal

    return doc_mask_mod


def make_natten_mask(canvas_w, canvas_h, kernel_w, kernel_h):
    def natten_mask_mod(b, h, q_idx, kv_idx):
        q_x, q_y = q_idx // canvas_w, q_idx % canvas_w
        kv_x, kv_y = kv_idx // canvas_w, kv_idx % canvas_w
        kernel_center_x = q_x.clamp(kernel_w // 2, (canvas_w - 1) - kernel_w // 2)
        kernel_center_y = q_y.clamp(kernel_h // 2, (canvas_h - 1) - kernel_h // 2)
        hori_mask = (kernel_center_x - kv_x).abs() <= kernel_w // 2
        vert_mask = (kernel_center_y - kv_y).abs() <= kernel_h // 2
        return hori_mask & vert_mask

    return natten_mask_mod


def make_sta_3d_mask(canvas_twh, kernel_twh, tile_twh):
    canvas_t, canvas_h, canvas_w = canvas_twh
    kernel_t, kernel_h, kernel_w = kernel_twh
    tile_t, tile_h, tile_w = tile_twh
    tile_numel = tile_t * tile_h * tile_w
    canvas_tile_t = canvas_t // tile_t
    canvas_tile_h = canvas_h // tile_h
    canvas_tile_w = canvas_w // tile_w
    kernel_tile_t = kernel_t // tile_t
    kernel_tile_h = kernel_h // tile_h
    kernel_tile_w = kernel_w // tile_w
    vision_seq_len = canvas_t * canvas_h * canvas_w

    def get_t_h_w_idx_tiled(idx):
        tile_id = idx // tile_numel
        tile_t_idx = tile_id // (canvas_tile_h * canvas_tile_w)
        tile_h_idx = (tile_id % (canvas_tile_h * canvas_tile_w)) // canvas_tile_w
        tile_w_idx = tile_id % canvas_tile_w
        return tile_t_idx, tile_h_idx, tile_w_idx

    def get_border(kernel_size):
        return kernel_size // 2, kernel_size // 2 + (kernel_size % 2 - 1)

    def sta_mask_mod_3d(b, h, q_idx, kv_idx):
        q_tile_t, q_tile_h, q_tile_w = get_t_h_w_idx_tiled(q_idx)
        kv_tile_t, kv_tile_h, kv_tile_w = get_t_h_w_idx_tiled(kv_idx)
        left_border_t, right_border_t = get_border(kernel_tile_t)
        left_border_h, right_border_h = get_border(kernel_tile_h)
        left_border_w, right_border_w = get_border(kernel_tile_w)
        kernel_center_t = q_tile_t.clamp(left_border_t, (canvas_tile_t - 1) - right_border_t)
        kernel_center_h = q_tile_h.clamp(left_border_h, (canvas_tile_h - 1) - right_border_h)
        kernel_center_w = q_tile_w.clamp(left_border_w, (canvas_tile_w - 1) - right_border_w)
        t_mask = (kv_tile_t >= kernel_center_t - left_border_t) & (
            kv_tile_t <= kernel_center_t + right_border_t
        )
        h_mask = (kv_tile_h >= kernel_center_h - left_border_h) & (
            kv_tile_h <= kernel_center_h + right_border_h
        )
        w_mask = (kv_tile_w >= kernel_center_w - left_border_w) & (
            kv_tile_w <= kernel_center_w + right_border_w
        )
        vision_mask = (q_idx < vision_seq_len) & (kv_idx < vision_seq_len)
        return vision_mask & t_mask & h_mask & w_mask

    return sta_mask_mod_3d


prefix_lm_mask_512 = make_prefix_lm_mask(prefix_length=512)
dilated_sliding_window_mask = make_dilated_sliding_window_mask(window_size=128, dilation=2)
batchify_causal_mask = make_batchify_mask(causal, batchify_size=512)
sta_2d_mask = make_sta_2d_mask(canvas_hw=(16, 128), kernel_hw=(4, 16), tile_hw=(2, 16))
svg_spatial_mask = make_svg_spatial_mask(
    prompt_length=128, num_frames=15, token_per_frame=128, width=2, round_width=128
)
svg_temporal_mask = make_svg_temporal_mask(
    prompt_length=128, num_frames=15, token_per_frame=128, width=2
)
natten_mask = make_natten_mask(canvas_w=32, canvas_h=16, kernel_w=3, kernel_h=3)
sta_3d_mask = make_sta_3d_mask(canvas_twh=(8, 8, 8), kernel_twh=(4, 4, 4), tile_twh=(2, 2, 2))


# Capture-using masks/score_mod are built lazily per-device so the captured
# tensors land on the same device as q_idx/kv_idx during HOP tracing.
_document_mask_mod_by_device = {}


def document_mask(b, h, q_idx, kv_idx):
    dev = q_idx.device
    fn = _document_mask_mod_by_device.get(dev)
    if fn is None:
        fn = make_doc_mask([256, 256], device=dev)
        _document_mask_mod_by_device[dev] = fn
    return fn(b, h, q_idx, kv_idx)


def case(
    *,
    mask_mod,
    dtype,
    id,
    score_mod=None,
    b=2,
    h=4,
    seq_len=2048,
    s=2048,
    e=64,
    k_tile_size=256,
    marks=(),
):
    return pytest.param(
        b, h, seq_len, s, e, score_mod, mask_mod, k_tile_size, dtype, id=id, marks=marks
    )


class TestFlexAttentionE2E:
    """End-to-end tests for flex attention with user-defined score and mask functions."""

    @pytest.mark.parametrize(
        "b,h,seq_len,s,e,score_mod,mask_mod,k_tile_size,dtype",
        [
            case(
                b=4,
                h=3,
                seq_len=1024,
                s=512,
                e=128,
                k_tile_size=128,
                mask_mod=causal,
                dtype=torch.float32,
                id="causal_fp32",
            ),
            case(
                seq_len=4096, s=4096, e=128, mask_mod=causal, dtype=torch.bfloat16, id="causal_bf16"
            ),
            case(
                h=1,
                seq_len=512,
                s=4096,
                k_tile_size=512,
                mask_mod=sliding_window_mod,
                dtype=torch.float32,
                id="sliding_window_fp32",
            ),
            case(
                score_mod=soft_cap,
                mask_mod=sliding_window_mod,
                dtype=torch.bfloat16,
                id="sliding_window_bf16",
            ),
            case(
                k_tile_size=128,
                score_mod=soft_cap,
                mask_mod=sliding_window_causal,
                dtype=torch.float32,
                id="sliding_window_causal_fp32",
            ),
            case(
                b=1,
                h=2,
                seq_len=4096,
                s=8192,
                k_tile_size=512,
                mask_mod=sliding_window_causal,
                dtype=torch.bfloat16,
                id="sliding_window_causal_bf16",
            ),
            case(mask_mod=noop_mask, dtype=torch.float32, id="noop_fp32"),
            case(
                seq_len=1024,
                s=1024,
                k_tile_size=128,
                score_mod=head_bias_score,
                mask_mod=noop_mask,
                dtype=torch.bfloat16,
                id="head_bias_bf16",
            ),
            case(
                seq_len=512,
                s=512,
                k_tile_size=128,
                score_mod=rel_score,
                mask_mod=causal,
                dtype=torch.float32,
                id="rel_fp32",
            ),
            case(
                seq_len=1024,
                s=1024,
                k_tile_size=128,
                score_mod=alibi_score,
                mask_mod=causal,
                dtype=torch.bfloat16,
                id="alibi_bf16",
            ),
            case(h=2, mask_mod=prefix_lm_mask_512, dtype=torch.float32, id="prefix_lm_fp32"),
            case(score_mod=soft_cap, mask_mod=causal, dtype=torch.float32, id="softcap_fp32"),
            case(
                seq_len=1024,
                s=1024,
                k_tile_size=128,
                score_mod=soft_cap,
                mask_mod=causal,
                dtype=torch.float32,
                id="softcap_exact_fp32",
            ),
            case(
                h=2,
                mask_mod=dilated_sliding_window_mask,
                dtype=torch.bfloat16,
                id="dilated_sliding_window_bf16",
            ),
            case(h=2, mask_mod=batchify_causal_mask, dtype=torch.float32, id="batchify_fp32"),
            case(b=1, h=2, mask_mod=sta_2d_mask, dtype=torch.bfloat16, id="sta_2d_bf16"),
            case(b=1, h=2, mask_mod=svg_spatial_mask, dtype=torch.float32, id="svg_spatial_fp32"),
            case(
                b=1, h=2, mask_mod=svg_temporal_mask, dtype=torch.bfloat16, id="svg_temporal_bf16"
            ),
            case(
                seq_len=512,
                s=512,
                k_tile_size=128,
                mask_mod=natten_mask,
                dtype=torch.bfloat16,
                id="natten_bf16",
            ),
            case(
                seq_len=512,
                s=512,
                k_tile_size=128,
                mask_mod=sta_3d_mask,
                dtype=torch.bfloat16,
                id="sta_3d_bf16",
            ),
            case(
                seq_len=512,
                s=512,
                k_tile_size=128,
                mask_mod=document_mask,
                dtype=torch.bfloat16,
                id="document_mask_bf16",
            ),
        ],
    )
    def test_flex_attention_forward(
        self, b, h, seq_len, s, e, score_mod, mask_mod, k_tile_size, dtype
    ):
        """Test flex attention forward pass end-to-end across score_mod / mask_mod variants."""
        torch.manual_seed(42)
        query = torch.randn(b, h, seq_len, e, device="cpu", dtype=dtype)
        key = torch.randn(b, h, s, e, device="cpu", dtype=dtype)
        value = torch.randn(b, h, s, e, device="cpu", dtype=dtype)

        block_mask = create_block_mask(
            mask_mod,
            B=b,
            H=h,
            Q_LEN=seq_len,
            KV_LEN=s,
            BLOCK_SIZE=(128, k_tile_size),
            device="neuron",
        )

        cpu_mask = create_block_mask(
            mask_mod,
            B=b,
            H=h,
            Q_LEN=seq_len,
            KV_LEN=s,
            BLOCK_SIZE=(128, k_tile_size),
            device="cpu",
        )
        cpu_output, cpu_aux = flex_attention(
            query,
            key,
            value,
            score_mod=score_mod,
            block_mask=cpu_mask,
            return_aux=AuxRequest(lse=True),
        )
        cpu_lse = cpu_aux.lse

        @torch.compile(backend="neuron")
        def flex_attn_neuron(q, k, v, mask):
            output, aux = flex_attention(
                q,
                k,
                v,
                score_mod=score_mod,
                block_mask=mask,
                return_aux=AuxRequest(lse=True),
            )
            return output, aux.lse

        with torch.no_grad():
            neuron_output, neuron_lse = flex_attn_neuron(
                query.to("neuron"), key.to("neuron"), value.to("neuron"), block_mask
            )

        assert max_all_close(
            neuron_output.cpu().float().numpy(),
            cpu_output.float().numpy(),
            rtol=2e-2,
            atol=1e-3,
        )

        # CPU LSE uses base 2, while Neuron LSE uses base e and is negated.
        neuron_lse_converted = neuron_lse.cpu() / (-math.log(2))
        assert max_all_close(
            neuron_lse_converted.float().numpy(),
            cpu_lse.float().numpy(),
            rtol=2e-2,
            atol=1e-3,
        )

    @pytest.mark.parametrize(
        "b,h,seq_len,s,e,score_mod,mask_mod,k_tile_size,dtype",
        [
            case(
                b=4,
                h=3,
                e=128,
                k_tile_size=512,
                mask_mod=causal,
                dtype=torch.float32,
                id="causal_fp32",
            ),
            case(k_tile_size=512, mask_mod=causal, dtype=torch.bfloat16, id="causal_bf16"),
            case(
                seq_len=4096,
                s=4096,
                k_tile_size=128,
                mask_mod=sliding_window_mod,
                dtype=torch.float32,
                id="sliding_window_fp32",
            ),
            case(
                k_tile_size=512,
                score_mod=soft_cap,
                mask_mod=sliding_window_mod,
                dtype=torch.bfloat16,
                id="sliding_window_bf16",
            ),
            case(
                k_tile_size=512,
                score_mod=soft_cap,
                mask_mod=sliding_window_causal,
                dtype=torch.float32,
                id="sliding_window_causal_fp32",
            ),
            case(
                seq_len=4096,
                s=8192,
                mask_mod=sliding_window_causal,
                dtype=torch.bfloat16,
                id="sliding_window_causal_bf16",
            ),
            case(mask_mod=noop_mask, dtype=torch.float32, id="noop_fp32"),
            case(
                seq_len=1024,
                s=1024,
                k_tile_size=128,
                score_mod=head_bias_score,
                mask_mod=noop_mask,
                dtype=torch.bfloat16,
                id="head_bias_bf16",
            ),
            case(
                seq_len=512,
                s=512,
                k_tile_size=128,
                score_mod=rel_score,
                mask_mod=causal,
                dtype=torch.float32,
                id="rel_fp32",
            ),
            case(
                seq_len=1024,
                s=1024,
                k_tile_size=128,
                score_mod=alibi_score,
                mask_mod=causal,
                dtype=torch.bfloat16,
                id="alibi_bf16",
            ),
            case(h=2, mask_mod=prefix_lm_mask_512, dtype=torch.float32, id="prefix_lm_fp32"),
            case(score_mod=soft_cap, mask_mod=causal, dtype=torch.float32, id="softcap_fp32"),
            case(
                seq_len=1024,
                s=1024,
                k_tile_size=128,
                score_mod=soft_cap,
                mask_mod=causal,
                dtype=torch.float32,
                id="softcap_exact_fp32",
            ),
            case(
                h=2,
                mask_mod=dilated_sliding_window_mask,
                dtype=torch.bfloat16,
                id="dilated_sliding_window_bf16",
            ),
            case(h=2, mask_mod=batchify_causal_mask, dtype=torch.float32, id="batchify_fp32"),
            case(b=1, h=2, mask_mod=sta_2d_mask, dtype=torch.bfloat16, id="sta_2d_bf16"),
            case(b=1, h=2, mask_mod=svg_spatial_mask, dtype=torch.float32, id="svg_spatial_fp32"),
            case(
                b=1, h=2, mask_mod=svg_temporal_mask, dtype=torch.bfloat16, id="svg_temporal_bf16"
            ),
            case(
                seq_len=512,
                s=512,
                k_tile_size=128,
                mask_mod=natten_mask,
                dtype=torch.bfloat16,
                id="natten_bf16",
            ),
            case(
                seq_len=512,
                s=512,
                k_tile_size=128,
                mask_mod=sta_3d_mask,
                dtype=torch.bfloat16,
                id="sta_3d_bf16",
            ),
            case(
                seq_len=512,
                s=512,
                k_tile_size=128,
                mask_mod=document_mask,
                dtype=torch.bfloat16,
                id="document_mask_bf16",
            ),
        ],
    )
    def test_flex_attention_backward(
        self, b, h, seq_len, s, e, score_mod, mask_mod, k_tile_size, dtype
    ):
        """Test flex attention backward pass (grad_q, grad_k, grad_v) end-to-end."""
        torch.manual_seed(42)
        query_cpu = torch.rand(b, h, seq_len, e, device="cpu", dtype=dtype, requires_grad=True)
        key_cpu = torch.rand(b, h, s, e, device="cpu", dtype=dtype, requires_grad=True)
        value_cpu = torch.rand(b, h, s, e, device="cpu", dtype=dtype, requires_grad=True)

        block_mask = create_block_mask(
            mask_mod,
            B=b,
            H=h,
            Q_LEN=seq_len,
            KV_LEN=s,
            BLOCK_SIZE=(128, k_tile_size),
            device="neuron",
        )
        cpu_mask = create_block_mask(
            mask_mod,
            B=b,
            H=h,
            Q_LEN=seq_len,
            KV_LEN=s,
            BLOCK_SIZE=(128, k_tile_size),
            device="cpu",
        )
        cpu_out = flex_attention(
            query_cpu,
            key_cpu,
            value_cpu,
            score_mod=score_mod,
            block_mask=cpu_mask,
        )
        grad_output = torch.ones_like(cpu_out)
        cpu_out.backward(grad_output)
        cpu_grad_q = query_cpu.grad.clone()
        cpu_grad_k = key_cpu.grad.clone()
        cpu_grad_v = value_cpu.grad.clone()

        query_neuron = query_cpu.detach().to("neuron").requires_grad_(True)
        key_neuron = key_cpu.detach().to("neuron").requires_grad_(True)
        value_neuron = value_cpu.detach().to("neuron").requires_grad_(True)

        @torch.compile(backend="neuron")
        def flex_attn_neuron(q, k, v, mask):
            neuron_out = flex_attention(
                q,
                k,
                v,
                score_mod=score_mod,
                block_mask=mask,
            )
            return neuron_out

        neuron_out = flex_attn_neuron(
            query_neuron,
            key_neuron,
            value_neuron,
            block_mask,
        )
        neuron_out.backward(torch.ones_like(neuron_out))

        for neuron_grad, cpu_grad in [
            (query_neuron.grad.cpu(), cpu_grad_q),
            (key_neuron.grad.cpu(), cpu_grad_k),
            (value_neuron.grad.cpu(), cpu_grad_v),
        ]:
            assert max_all_close(
                neuron_grad.float().numpy(),
                cpu_grad.float().numpy(),
                rtol=2e-2,
                atol=1e-2,
            )

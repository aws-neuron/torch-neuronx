"""Regression tests that flex_attention keeps reaching the NKI kernel.

The ATen fallback returns the same forward numbers, so a numbers-only test stays
green through a lowering regression. It does not return the same gradients: the
pass legalizes the forward and the backward separately, so a forward-only
fallback pairs an NKI backward with an ATen forward. Assert the lowering.
"""

import contextlib
from unittest.mock import patch

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from torch_neuronx.neuron_dynamo_backend.fx.passes.flex_attention_nki_lowering import (
    FlexAttentionNKILowering,
)

pytestmark = pytest.mark.usefixtures("disable_flex_attention_device_check")

_BATCH = 1
_HEADS = 4
_HEAD_DIM = 64
_PAD = 64


@contextlib.contextmanager
def nki_lowering(method="_legalize_flex_forward"):
    """Record what the named NKI lowering returned for each flex_attention node."""
    results = []
    original = getattr(FlexAttentionNKILowering, method)

    def record(self, graph, gm, node):
        lowered = original(self, graph, gm, node)
        results.append(lowered)
        return lowered

    with patch.object(FlexAttentionNKILowering, method, record):
        yield results


# assert_close is per-element, so near-zero elements fall back to atol alone. The
# kernel matmuls in bfloat16 whatever the input dtype, putting the error near 8e-3.
_FWD_TOL = {"rtol": 2e-2, "atol": 1e-2}
_GRAD_TOL = {"rtol": 2e-2, "atol": 2e-2}


def assert_lowered_to_nki(results, what="forward"):
    assert results, f"the NKI {what} lowering never ran"
    assert all(results), f"NKI {what} lowering fell back to the ATen path: {results}"


def causal(b, h, q_idx, kv_idx):
    return kv_idx <= q_idx


def make_causal_left_padding(valid):
    """Causal, and the first _PAD key positions are padding. Captures a bool tensor."""

    def mask_mod(b, h, q_idx, kv_idx):
        return (kv_idx <= q_idx) & valid[b, kv_idx]

    return mask_mod


def _reference(query, key, value, mask):
    """Dense attention in float32. flex_attention on CPU has no backward."""
    scores = query.float() @ key.float().transpose(-1, -2) / (query.shape[-1] ** 0.5)
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1).nan_to_num() @ value.float()


def _dense_mask(mask_mod, q_len, kv_len, device, q_start=0):
    """Evaluate mask_mod over the whole index grid for batch 0, head 0."""
    zero = torch.zeros((), dtype=torch.long, device=device)
    q_idx = torch.arange(q_start, q_start + q_len, device=device).view(q_len, 1)
    kv_idx = torch.arange(kv_len, device=device).view(1, kv_len)
    return mask_mod(zero, zero, q_idx, kv_idx)


def _reference_in_q_blocks(query, key, value, mask_mod, q_block=1024):
    """_reference over blocks of query rows. A whole float32 score matrix is too large."""
    kv_len = key.shape[-2]
    blocks = []
    for q_start in range(0, query.shape[-2], q_block):
        q_rows = query[:, :, q_start : q_start + q_block]
        mask = _dense_mask(mask_mod, q_rows.shape[-2], kv_len, query.device, q_start)
        blocks.append(_reference(q_rows, key, value, mask))
    return torch.cat(blocks, dim=-2)


def _block_mask(mask_mod, seq_len, kv_block, device, heads=_HEADS):
    return create_block_mask(
        mask_mod,
        B=_BATCH,
        H=heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        BLOCK_SIZE=(128, kv_block),
        device=device,
    )


class TestNKILoweringRegressions:
    @pytest.mark.parametrize("kv_block", [128, 512])
    @pytest.mark.parametrize("seq_len", [512, 640, 768])
    def test_seq_len_not_multiple_of_large_tile_reaches_nki(self, seq_len, kv_block):
        """The forward must reach NKI whatever the seq_len to large-tile ratio.

        A large tile holds 4 * kv_block keys. (512, 128) divides evenly and is the
        control; the rest leave a partial last large tile.
        """
        torch.manual_seed(42)
        query = torch.randn(_BATCH, _HEADS, seq_len, _HEAD_DIM)
        key = torch.randn(_BATCH, _HEADS, seq_len, _HEAD_DIM)
        value = torch.randn(_BATCH, _HEADS, seq_len, _HEAD_DIM)

        @torch.compile(backend="neuron")
        def flex_attn_neuron(q, k, v, mask):
            return flex_attention(q, k, v, block_mask=mask)

        with nki_lowering() as lowered, torch.no_grad():
            neuron_output = flex_attn_neuron(
                query.to("neuron"),
                key.to("neuron"),
                value.to("neuron"),
                _block_mask(causal, seq_len, kv_block, "neuron"),
            )

        assert_lowered_to_nki(lowered)
        expected = _reference(query, key, value, _dense_mask(causal, seq_len, seq_len, "cpu"))
        torch.testing.assert_close(neuron_output.cpu().float(), expected, **_FWD_TOL)

    def test_multi_section_seq_len_reaches_nki(self):
        """The forward must reach NKI when the kernel splits the keys into sections.

        Above _FLASH_ATTENTION_THRESHOLD the kernel cuts the keys into 8192-element
        sections. The short final section still iterates a whole section of k-tiles. The
        QK loop therefore needs the seqlen_k bound, not only the tile-index bound. One
        head keeps the dense block mask and the reference small.
        """
        seq_len = 12288
        heads = 1
        torch.manual_seed(42)
        query = torch.randn(_BATCH, heads, seq_len, _HEAD_DIM)
        key = torch.randn(_BATCH, heads, seq_len, _HEAD_DIM)
        value = torch.randn(_BATCH, heads, seq_len, _HEAD_DIM)

        @torch.compile(backend="neuron")
        def flex_attn_neuron(q, k, v, mask):
            return flex_attention(q, k, v, block_mask=mask)

        with nki_lowering() as lowered, torch.no_grad():
            neuron_output = flex_attn_neuron(
                query.to("neuron"),
                key.to("neuron"),
                value.to("neuron"),
                _block_mask(causal, seq_len, 512, "neuron", heads),
            )

        assert_lowered_to_nki(lowered)
        expected = _reference_in_q_blocks(query, key, value, causal)
        torch.testing.assert_close(neuron_output.cpu().float(), expected, **_FWD_TOL)

    def test_seq_len_not_multiple_of_large_tile_backward_matches_reference(self):
        """Gradients must be right at a seq_len the forward used to bail on."""
        seq_len = 640
        torch.manual_seed(42)
        inputs = [
            torch.randn(_BATCH, _HEADS, seq_len, _HEAD_DIM, requires_grad=True) for _ in range(3)
        ]
        grad_out = torch.randn(_BATCH, _HEADS, seq_len, _HEAD_DIM)

        expected = _reference(*inputs, _dense_mask(causal, seq_len, seq_len, "cpu"))
        expected.backward(grad_out)

        neuron_inputs = [t.detach().clone().to("neuron").requires_grad_(True) for t in inputs]

        @torch.compile(backend="neuron")
        def flex_attn_neuron(q, k, v, mask):
            return flex_attention(q, k, v, block_mask=mask)

        with (
            nki_lowering() as lowered_fwd,
            nki_lowering("_legalize_flex_backward") as lowered_bwd,
        ):
            neuron_output = flex_attn_neuron(
                *neuron_inputs, _block_mask(causal, seq_len, 128, "neuron")
            )
            neuron_output.backward(grad_out.to("neuron"))

        assert_lowered_to_nki(lowered_fwd)
        assert_lowered_to_nki(lowered_bwd, "backward")
        torch.testing.assert_close(
            neuron_output.detach().cpu().float(), expected.detach(), **_FWD_TOL
        )
        for neuron_t, cpu_t, name in zip(neuron_inputs, inputs, "qkv", strict=True):
            torch.testing.assert_close(
                neuron_t.grad.cpu().float(),
                cpu_t.grad.float(),
                **_GRAD_TOL,
                msg=lambda m, name=name: f"grad {name} mismatch\n{m}",
            )

    @pytest.mark.parametrize("seq_len", [512, 640])
    def test_bool_capturing_mask_reaches_nki(self, seq_len, enable_nki_free_var):
        """A mask that captures a bool tensor must reach NKI.

        A capture renders the runtime-mask branch, so 640 covers its tile-count overrun.
        """
        torch.manual_seed(42)
        query = torch.randn(_BATCH, _HEADS, seq_len, _HEAD_DIM)
        key = torch.randn(_BATCH, _HEADS, seq_len, _HEAD_DIM)
        value = torch.randn(_BATCH, _HEADS, seq_len, _HEAD_DIM)

        def valid_for(device):
            valid = torch.ones(_BATCH, seq_len, dtype=torch.bool, device=device)
            valid[:, :_PAD] = False
            return valid

        @torch.compile(backend="neuron")
        def flex_attn_neuron(q, k, v, mask):
            return flex_attention(q, k, v, block_mask=mask)

        with nki_lowering() as lowered, torch.no_grad():
            neuron_output = flex_attn_neuron(
                query.to("neuron"),
                key.to("neuron"),
                value.to("neuron"),
                _block_mask(make_causal_left_padding(valid_for("neuron")), seq_len, 128, "neuron"),
            )

        assert_lowered_to_nki(lowered)

        neuron_output = neuron_output.cpu().float()
        expected = _reference(
            query,
            key,
            value,
            _dense_mask(make_causal_left_padding(valid_for("cpu")), seq_len, seq_len, "cpu"),
        )
        # Rows below _PAD attend to nothing, so only their finiteness is defined.
        assert torch.isfinite(neuron_output[:, :, :_PAD, :]).all()
        torch.testing.assert_close(
            neuron_output[:, :, _PAD:, :], expected[:, :, _PAD:, :], **_FWD_TOL
        )

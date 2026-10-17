"""Device tests for the log-sum-exp gradient of flex_attention.

A loss that reads `aux.lse` makes argument 6 of the backward HOP a real node. That
argument renders the `dlse` operand into the NKI backward kernel, and it makes the
decomposition fold the log-sum-exp gradient into the softmax row term. A test that reads
the log-sum-exp without a backward compiles neither path, so it keeps both unexercised.

Both lowerings must agree with the reference, so every case runs on both. The compile
budget selects the lowering, and the pass probe asserts which one ran.
"""

import pytest
import torch
from torch.nn.attention.flex_attention import AuxRequest, create_block_mask, flex_attention

from tests.neuron_dynamo_backend.component.flex.test_nki_lowering_regressions import (
    assert_lowered_to_nki,
    nki_lowering,
)

pytestmark = pytest.mark.usefixtures("disable_flex_attention_device_check")

_BATCH = 1
_HEADS = 2
_SEQ_LEN = 256
_HEAD_DIM = 64
_KV_BLOCK = 128
_COMPILE_BUDGET_ENV = "TORCH_NEURONX_FLEX_NKI_COMPILE_BUDGET_S"

# The kernel matmuls in bfloat16 whatever the input dtype, so the error sits near 8e-3.
_TOL = {"rtol": 2e-2, "atol": 2e-2}


def causal(b, h, q_idx, kv_idx):
    return kv_idx <= q_idx


@pytest.fixture
def lowering(request, monkeypatch):
    """Select the lowering. A budget of 0 turns the budget off; 0.001 s forces the bail."""
    monkeypatch.setenv(_COMPILE_BUDGET_ENV, "0" if request.param == "nki" else "0.001")
    return request.param


both_lowerings = pytest.mark.parametrize("lowering", ["nki", "decomposition"], indirect=True)


def _grad_or_zeros(tensor):
    """A tensor outside the loss keeps grad None. Compare it as a zero gradient."""
    return tensor.grad if tensor.grad is not None else torch.zeros_like(tensor)


def _reference(query, key, value, grad_out, lse_weight, use_output):
    """Dense attention in float64. flex_attention has no CPU backward.

    `aux.lse` is the natural-log log-sum-exp, because the public API scales the HOP's
    log2 value by ln(2). The reference therefore uses torch.logsumexp directly.
    """
    q, k, v = (t.double().detach().requires_grad_(True) for t in (query, key, value))
    scale = query.shape[-1] ** -0.5
    scores = q @ k.transpose(-1, -2) * scale
    rows = torch.arange(query.shape[-2]).view(-1, 1)
    cols = torch.arange(key.shape[-2]).view(1, -1)
    scores = scores.masked_fill(~causal(None, None, rows, cols), float("-inf"))

    out = torch.softmax(scores, dim=-1) @ v
    lse = torch.logsumexp(scores, dim=-1)

    loss = (lse * lse_weight.double()).sum()
    if use_output:
        loss = loss + (out * grad_out.double()).sum()
    loss.backward()
    return out, lse, tuple(_grad_or_zeros(t) for t in (q, k, v))


class TestFlexAttentionLogSumExpGradient:
    @both_lowerings
    @pytest.mark.parametrize("use_output", [True, False], ids=["out_and_lse", "lse_only"])
    def test_lse_in_the_loss_matches_reference(self, lowering, use_output):
        """A loss that reads the log-sum-exp must give the reference Q, K and V gradients.

        `lse_only` drops the attention output from the loss. That is the case a dropped
        log-sum-exp gradient breaks completely: every Q and K gradient comes out zero.
        """
        torch.manual_seed(42)
        query, key, value = (torch.randn(_BATCH, _HEADS, _SEQ_LEN, _HEAD_DIM) for _ in range(3))
        grad_out = torch.randn(_BATCH, _HEADS, _SEQ_LEN, _HEAD_DIM)
        lse_weight = torch.randn(_BATCH, _HEADS, _SEQ_LEN)

        expected_out, expected_lse, expected_grads = _reference(
            query, key, value, grad_out, lse_weight, use_output
        )

        neuron_inputs = [
            t.detach().clone().to("neuron").requires_grad_(True) for t in (query, key, value)
        ]

        @torch.compile(backend="neuron")
        def loss_fn(q, k, v, mask, dy, weight):
            out, aux = flex_attention(q, k, v, block_mask=mask, return_aux=AuxRequest(lse=True))
            loss = (aux.lse * weight).sum()
            if use_output:
                loss = loss + (out * dy).sum()
            return loss, out, aux.lse

        block_mask = create_block_mask(
            causal,
            B=_BATCH,
            H=_HEADS,
            Q_LEN=_SEQ_LEN,
            KV_LEN=_SEQ_LEN,
            BLOCK_SIZE=(128, _KV_BLOCK),
            device="neuron",
        )

        with (
            nki_lowering() as lowered_fwd,
            nki_lowering("_legalize_flex_backward") as lowered_bwd,
        ):
            loss, neuron_out, neuron_lse = loss_fn(
                *neuron_inputs,
                block_mask,
                grad_out.to("neuron"),
                lse_weight.to("neuron"),
            )
            loss.backward()

        if lowering == "nki":
            assert_lowered_to_nki(lowered_fwd)
            assert_lowered_to_nki(lowered_bwd, "backward")
        else:
            assert not lowered_fwd and not lowered_bwd, (
                f"the compile budget must stop the whole NKI pass, got "
                f"forward={lowered_fwd} backward={lowered_bwd}"
            )

        torch.testing.assert_close(
            neuron_out.detach().cpu().double(), expected_out.detach(), **_TOL
        )
        torch.testing.assert_close(
            neuron_lse.detach().cpu().double(), expected_lse.detach(), **_TOL
        )

        grads = [_grad_or_zeros(t).cpu().double() for t in neuron_inputs]
        for grad, expected, name in zip(grads, expected_grads, "qkv", strict=True):
            torch.testing.assert_close(
                grad,
                expected,
                **_TOL,
                msg=lambda m, name=name: f"grad {name} mismatch\n{m}",
            )

        if not use_output:
            # A dropped log-sum-exp gradient zeroes exactly these two.
            assert grads[0].abs().max() > 0, "grad q is all zeros"
            assert grads[1].abs().max() > 0, "grad k is all zeros"

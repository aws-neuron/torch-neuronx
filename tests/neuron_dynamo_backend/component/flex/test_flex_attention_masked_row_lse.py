"""Device tests for the log-sum-exp of a Q row that attends to nothing.

The forward NKI kernel marks an empty row with a sentinel, and it uses two of them:

- `dead_rows_in_live_tile` keeps live rows in the same 128-row Q group. Every score in the
  empty row is _FLOAT32_MIN, and the kernel holds the negated row max, so the kernel
  log-sum-exp is +FLT_MAX.
- `dead_q_group` empties a whole 128-row Q group. Static tile elimination then sets the
  reciprocal to 0 and the running max to _FLOAT32_MIN, so the kernel log-sum-exp is -inf.

The pass converts the kernel value to the HOP scale, and converts back before it gives the
log-sum-exp to the backward kernel as an exp bias. An unbounded multiply overflows on both
sentinels, and the forward saves its result, so the infinity crosses the autograd boundary
and every gradient becomes nan.

A sliding window never reaches this state, because every row attends to its own key. A
mask that empties a row is what reaches it.
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
_Q_GROUP = 128
_COMPILE_BUDGET_ENV = "TORCH_NEURONX_FLEX_NKI_COMPILE_BUDGET_S"

_TOL = {"rtol": 2e-2, "atol": 2e-2}


def dead_rows_in_live_tile(b, h, q_idx, kv_idx):
    """Causal, and the last 64 rows of the second Q group attend to nothing."""
    return (kv_idx <= q_idx) & (q_idx < 192)


def dead_q_group(b, h, q_idx, kv_idx):
    """Causal, and the whole second 128-row Q group attends to nothing."""
    return (kv_idx <= q_idx) & (q_idx < _Q_GROUP)


@pytest.fixture
def lowering(request, monkeypatch):
    """Select the lowering. A budget of 0 turns the budget off; 0.001 s forces the bail."""
    monkeypatch.setenv(_COMPILE_BUDGET_ENV, "0" if request.param == "nki" else "0.001")
    return request.param


def _reference(query, key, value, grad_out, lse_weight, mask_mod, live_rows):
    """Dense attention in float64 over the live Q rows only.

    An empty Q row contributes nothing to the loss, so its own gradient is zero and it adds
    nothing to the K and V gradients. A reference over the live rows alone therefore holds,
    and it stays free of the nan that softmax gives an all-masked row.
    """
    q, k, v = (t.double().detach().requires_grad_(True) for t in (query, key, value))
    scale = query.shape[-1] ** -0.5
    rows = torch.arange(live_rows).view(-1, 1)
    cols = torch.arange(key.shape[-2]).view(1, -1)

    scores = q[:, :, :live_rows] @ k.transpose(-1, -2) * scale
    scores = scores.masked_fill(~mask_mod(None, None, rows, cols), float("-inf"))

    out = torch.softmax(scores, dim=-1) @ v
    lse = torch.logsumexp(scores, dim=-1)
    loss = (out * grad_out[:, :, :live_rows].double()).sum() + (lse * lse_weight.double()).sum()
    loss.backward()
    return lse, (q.grad, k.grad, v.grad)


@pytest.mark.parametrize("lowering", ["nki", "decomposition"], indirect=True)
@pytest.mark.parametrize(
    "mask_mod,live_rows",
    [
        pytest.param(dead_rows_in_live_tile, 192, id="dead_rows_in_live_tile"),
        pytest.param(dead_q_group, _Q_GROUP, id="dead_q_group"),
    ],
)
def test_fully_masked_row_lse(lowering, mask_mod, live_rows):
    """An empty Q row must not poison a gradient.

    The loss reads only the live rows, so the empty rows add nothing to it. Their sentinel
    still reaches the backward kernel as an exp bias, and the K and V gradients sum over
    every Q row, so one bad row shows up in all of them.
    """
    torch.manual_seed(42)
    query, key, value = (torch.randn(_BATCH, _HEADS, _SEQ_LEN, _HEAD_DIM) for _ in range(3))
    grad_out = torch.randn(_BATCH, _HEADS, _SEQ_LEN, _HEAD_DIM)
    lse_weight = torch.randn(_BATCH, _HEADS, live_rows)

    expected_lse, expected_grads = _reference(
        query, key, value, grad_out, lse_weight, mask_mod, live_rows
    )

    neuron_inputs = [
        t.detach().clone().to("neuron").requires_grad_(True) for t in (query, key, value)
    ]

    @torch.compile(backend="neuron")
    def loss_fn(q, k, v, mask, dy, weight):
        out, aux = flex_attention(q, k, v, block_mask=mask, return_aux=AuxRequest(lse=True))
        live_out = (out[:, :, :live_rows] * dy[:, :, :live_rows]).sum()
        return live_out + (aux.lse[:, :, :live_rows] * weight).sum(), aux.lse

    block_mask = create_block_mask(
        mask_mod,
        B=_BATCH,
        H=_HEADS,
        Q_LEN=_SEQ_LEN,
        KV_LEN=_SEQ_LEN,
        BLOCK_SIZE=(_Q_GROUP, _KV_BLOCK),
        device="neuron",
    )

    with (
        nki_lowering() as lowered_fwd,
        nki_lowering("_legalize_flex_backward") as lowered_bwd,
    ):
        loss, neuron_lse = loss_fn(
            *neuron_inputs, block_mask, grad_out.to("neuron"), lse_weight.to("neuron")
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

    grads = [t.grad.cpu().double() for t in neuron_inputs]
    for grad, name in zip(grads, "qkv", strict=True):
        assert torch.isfinite(grad).all(), (
            f"grad {name} is not finite: {int(torch.isnan(grad).sum())} nan and "
            f"{int(torch.isinf(grad).sum())} inf of {grad.numel()}"
        )

    lse = neuron_lse.detach().cpu().double()
    dead = lse[:, :, live_rows:]
    # Neither lowering reports upstream's -inf for an empty row: both carry a large finite
    # value instead. What matters is that it stays finite, because the backward turns it into
    # an exp bias and an infinite one makes every gradient nan.
    assert torch.isfinite(
        dead
    ).all(), f"an empty Q row must stay finite, got {sorted(set(dead.flatten().tolist()))[:4]}"
    torch.testing.assert_close(lse[:, :, :live_rows], expected_lse.detach(), **_TOL)

    for grad, expected, name in zip(grads, expected_grads, "qkv", strict=True):
        live = slice(None, live_rows) if name == "q" else slice(None)
        torch.testing.assert_close(
            grad[:, :, live],
            expected[:, :, live],
            **_TOL,
            msg=lambda m, name=name: f"grad {name} mismatch\n{m}",
        )
    assert grads[0][:, :, live_rows:].abs().max() == 0, "an empty Q row must get a zero gradient"

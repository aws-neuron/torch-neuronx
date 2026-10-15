"""Unit tests for the head dimension support of FlexAttentionNKILowering.

The rejected shapes return before the pass renders a kernel, so these tests need no device.
"""

import logging
from unittest.mock import patch

import pytest

from tests.neuron_dynamo_backend.unit.fx.passes.test_flex_attention_nki_compile_cost import (
    _capture_flex_backward_graph,
    _capture_flex_graph,
    _count_flex_nodes,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.flex_attention_nki_lowering import (
    FlexAttentionNKILowering,
    _get_fwd_section_len,
    _nki_supports_head_dims,
)


@pytest.fixture(autouse=True)
def disable_flex_attention_device_check():
    with patch("torch.nn.attention.flex_attention._validate_device", lambda *args, **kwargs: None):
        yield


@pytest.mark.parametrize(
    "d_head_qk,d_head_v,forward,backward",
    [
        (16, 16, True, True),
        (128, 128, True, True),
        (192, 128, True, True),
        (128, 192, True, True),
        (160, 160, True, True),
        (256, 256, True, True),
        (384, 384, True, True),
        (512, 512, True, True),
        (129, 129, True, False),
        (320, 320, True, False),
        (192, 129, True, False),
        (513, 513, False, False),
        (640, 640, False, False),
        (128, 513, False, False),
    ],
)
def test_supports_head_dims(d_head_qk, d_head_v, forward, backward):
    assert _nki_supports_head_dims(d_head_qk, d_head_v, backward=False) == forward
    assert _nki_supports_head_dims(d_head_qk, d_head_v, backward=True) == backward


@pytest.mark.parametrize(
    "seqlen_kv,d_head_qk,d_head_v,expected",
    [
        (8192, 128, 128, 8192),
        (12288, 128, 128, 8192),
        (4224, 192, 192, 4224),
        (6144, 192, 192, 4096),
        (6144, 384, 128, 2048),
        (6144, 128, 384, 2048),
        (6144, 512, 512, 2048),
    ],
)
def test_fwd_section_len(seqlen_kv, d_head_qk, d_head_v, expected):
    assert _get_fwd_section_len(seqlen_kv, d_head_qk, d_head_v) == expected


@pytest.mark.parametrize(
    "capture,name,d_head_qk,d_head_v",
    [
        (_capture_flex_graph, "flex_attention", 640, 640),
        (_capture_flex_graph, "flex_attention", 128, 513),
        (_capture_flex_backward_graph, "flex_attention_backward", 320, 320),
        (_capture_flex_backward_graph, "flex_attention_backward", 192, 129),
    ],
    ids=["fwd-640", "fwd-v513", "bwd-320", "bwd-v129"],
)
def test_unsupported_head_dims_keep_the_operator(capture, name, d_head_qk, d_head_v, caplog):
    gm = capture(d_head_qk=d_head_qk, d_head_v=d_head_v)
    with caplog.at_level(logging.WARNING):
        result = FlexAttentionNKILowering()(gm)

    assert not result.modified
    assert _count_flex_nodes(gm, name) == 1
    assert "Unsupported head dimensions" in caplog.text

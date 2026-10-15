from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from torch_neuronx.neuron_dynamo_backend.fx.passes import complex_decomposition


@pytest.mark.parametrize("available", [False, True])
def test_maybe_decompose_complex_default_matches_torch_version(monkeypatch, available):
    """Use complex decomposition by default only with PyTorch 2.13 or later."""
    monkeypatch.delenv("TORCH_NEURONX_COMPLEX_DECOMPOSE", raising=False)
    monkeypatch.setattr(complex_decomposition, "_COMPLEX_DECOMP_AVAILABLE", available)
    gm = object()
    decomposed = object()

    with patch.object(
        complex_decomposition, "_decompose_complex", return_value=decomposed
    ) as decompose:
        result = complex_decomposition.maybe_decompose_complex(gm, [])

    if available:
        assert result is decomposed
        decompose.assert_called_once_with(gm, [], path_name="torch.compile")
    else:
        assert result is gm
        decompose.assert_not_called()


@pytest.mark.skipif(
    not complex_decomposition._COMPLEX_DECOMP_AVAILABLE,
    reason="Complex decomposition requires PyTorch 2.13 or later.",
)
def test_complex_abs_decomposes_hypot():
    """Decompose the hypot operation that complex abs produces."""
    # Complex abs produces hypot(real, imag). The Neuron table must decompose it.
    graph = fx.Graph()
    arg = graph.placeholder("arg")
    result = graph.call_function(torch.ops.aten.abs.default, (arg,))
    graph.output(result)
    gm = fx.GraphModule(torch.nn.Module(), graph)
    complex_input = torch.tensor([3 + 4j, 5 + 12j, 0j], dtype=torch.complex64)

    decomposed = complex_decomposition.eager_decompose_complex(gm, [complex_input])

    targets = {node.target for node in decomposed.graph.nodes if node.op == "call_function"}
    assert torch.ops.aten.hypot.default not in targets
    torch.testing.assert_close(decomposed(complex_input), torch.abs(complex_input))


@pytest.mark.skipif(
    not complex_decomposition._COMPLEX_DECOMP_AVAILABLE,
    reason="Complex decomposition requires PyTorch 2.13 or later.",
)
@pytest.mark.parametrize("mode_active", [False, True])
def test_complex_retrace_with_fake_inputs_and_scalar_factory(monkeypatch, mode_active):
    """Keep scalar factories fake during the complex graph retrace."""
    monkeypatch.setenv("TORCH_NEURONX_COMPLEX_DECOMPOSE", "1")
    monkeypatch.setenv("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", "1")

    def fn(counts, value):
        lower = torch.full((), 1, dtype=torch.int32, device=counts.device)
        return torch.maximum(counts, lower), value * value

    with FakeTensorMode() as fake_mode:
        fake_inputs = (
            torch.empty(64, dtype=torch.int32),
            torch.empty(2, dtype=torch.complex64),
        )
        gm = make_fx(fn)(*fake_inputs)

    with fake_mode if mode_active else nullcontext():
        decomposed = complex_decomposition.maybe_decompose_complex(gm, fake_inputs)

    assert decomposed is not gm
    inputs = (
        torch.arange(-32, 32, dtype=torch.int32),
        torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex64),
    )
    torch.testing.assert_close(decomposed(*inputs), fn(*inputs))


@pytest.mark.skipif(
    not complex_decomposition._COMPLEX_DECOMP_AVAILABLE,
    reason="Complex decomposition requires PyTorch 2.13 or later.",
)
@pytest.mark.parametrize("field", ["fwd_nn_module_stack", "call_hierarchy"])
def test_complex_retrace_preserves_hierarchy(field):
    """Retain each source operation's layer on real and decomposed operations."""

    def fn(x, z):
        return torch.sin(x) + torch.cos(x), z * z

    inputs = (torch.ones(3), torch.ones(2, dtype=torch.complex64))
    gm = make_fx(fn)(*inputs)
    expected = {}
    for index, node in enumerate(gm.graph.nodes):
        if node.op == "call_function":
            hierarchy = {f"layer{index}": (f"layers.{index}", "example.Block")}
            node.meta[field] = hierarchy
            expected[node.name] = hierarchy

    decomposed = complex_decomposition.eager_decompose_complex(gm, list(inputs))

    assert decomposed is not gm
    seen = set()
    for node in decomposed.graph.nodes:
        for source in node.meta.get("from_node", []):
            if source.graph_id == id(gm.graph) and source.name in expected:
                assert node.meta.get(field) == expected[source.name]
                seen.add(source.name)
    assert seen == expected.keys()
    torch.testing.assert_close(decomposed(*inputs), fn(*inputs))


@pytest.mark.skipif(
    not complex_decomposition._COMPLEX_DECOMP_AVAILABLE,
    reason="Complex decomposition requires PyTorch 2.13 or later.",
)
@pytest.mark.parametrize("capture_mask", [False, True])
def test_complex_rope_with_flex_attention_forward_backward(monkeypatch, capture_mask):
    """Keep FlexAttention and its gradients when complex RoPE is decomposed."""
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd
    from torch._higher_order_ops.flex_attention import (
        flex_attention,
        flex_attention_backward,
    )
    from torch.nn.attention.flex_attention import create_block_mask

    monkeypatch.setenv("TORCH_NEURONX_COMPLEX_DECOMPOSE", "1")
    monkeypatch.setenv("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", "1")
    complex_decomposition.apply_complex_decomposition_patches()
    torch.manual_seed(42)
    seq, dim = 32, 16
    documents = torch.arange(seq) // 16

    def mask_fn(b, h, q, k):
        causal = q >= k
        return causal & (documents[q] == documents[k]) if capture_mask else causal

    block_mask = create_block_mask(mask_fn, 1, None, seq, seq, device="cpu")

    def rotate(value, phases):
        pairs = torch.view_as_complex(value.reshape(*value.shape[:-1], -1, 2))
        return torch.view_as_real(pairs * phases).flatten(-2)

    def model(q, k, v, phases):
        def score_mod(score, b, h, q_index, k_index):
            return score

        return flex_attention(
            rotate(q, phases),
            rotate(k, phases),
            v,
            score_mod,
            block_mask.as_tuple(),
            dim**-0.5,
            {"OUTPUT_LOGSUMEXP": True, "OUTPUT_MAX": False},
        )[0]

    inputs = [torch.randn(1, heads, seq, dim, requires_grad=True) for heads in (4, 2, 2)]
    angles = torch.randn(seq, dim // 2)
    phases = torch.polar(torch.ones_like(angles), angles)
    dense_mask = torch.arange(seq)[:, None] >= torch.arange(seq)[None, :]
    if capture_mask:
        dense_mask &= documents[:, None] == documents[None, :]
    expected = torch.nn.functional.scaled_dot_product_attention(
        rotate(inputs[0], phases),
        rotate(inputs[1], phases),
        inputs[2],
        attn_mask=dense_mask,
        enable_gqa=True,
    )
    expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
    seen_hops = set()

    def compiler(gm, flat_args):
        decomposed = complex_decomposition.maybe_decompose_complex(gm, flat_args)
        assert decomposed is not gm
        for node in decomposed.graph.nodes:
            if node.target in (flex_attention, flex_attention_backward):
                seen_hops.add(node.target)
            if node.target == torch.ops.aten.mul.Tensor:
                assert not node.meta["val"].is_complex()
        return make_boxed_func(decomposed.forward)

    compiled = torch.compile(
        model,
        backend=aot_autograd(fw_compiler=compiler, bw_compiler=compiler),
        fullgraph=True,
    )
    actual = compiled(*inputs, phases)
    actual_grads = torch.autograd.grad(actual.square().sum(), inputs)
    assert seen_hops == {flex_attention, flex_attention_backward}
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-5, rtol=2e-5)

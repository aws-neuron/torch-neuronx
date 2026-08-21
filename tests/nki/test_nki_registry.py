"""Tests for NKIRegistry kernel registration and lookup."""

from unittest.mock import MagicMock

import pytest
import torch
from nki.collectives import ReplicaGroup
from nki.framework.torch_native import TorchNativeKernel
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

import torch_neuronx.nki_hop as nki_hop
from torch_neuronx.nki_hop import NKIHOPCaller, NKIRegistry, register_nki_constant_args


@pytest.fixture
def registry():
    registry = NKIRegistry()
    registry.reset_table()
    return registry


@pytest.fixture
def isolated_kernel_registry(monkeypatch):
    registry = NKIRegistry()
    registry.reset_table()
    monkeypatch.setattr(nki_hop, "kernel_registry", registry)
    return registry


def _make_mock_kernel():
    return MagicMock(spec=TorchNativeKernel)


class TestNKIRegistryAddKernel:
    def test_add_kernel_returns_index(self, registry):
        kernel = _make_mock_kernel()
        idx = registry.add_kernel(kernel, ["x", "y"])
        assert idx == 0

    def test_add_multiple_kernels_returns_sequential_indices(self, registry):
        k1 = _make_mock_kernel()
        k2 = _make_mock_kernel()
        assert registry.add_kernel(k1, ["x"]) == 0
        assert registry.add_kernel(k2, ["x"]) == 1

    def test_add_same_kernel_twice_returns_same_index(self, registry):
        kernel = _make_mock_kernel()
        idx1 = registry.add_kernel(kernel, ["x"])
        idx2 = registry.add_kernel(kernel, ["x"])
        assert idx1 == idx2
        assert len(registry.id_to_kernel) == 1


class TestNKIRegistryGetKernel:
    def test_get_kernel_returns_registered_kernel(self, registry):
        kernel = _make_mock_kernel()
        idx = registry.add_kernel(kernel, ["a", "b"])
        assert registry.get_kernel(idx) is kernel

    def test_get_kernel_invalid_index_raises(self, registry):
        with pytest.raises(AssertionError):
            registry.get_kernel(999)

    def test_get_kernel_arg_names(self, registry):
        kernel = _make_mock_kernel()
        idx = registry.add_kernel(kernel, ["a", "b", "c"])
        assert registry.get_kernel_arg_names(idx) == ["a", "b", "c"]

    def test_get_kernel_default_args(self, registry):
        kernel = _make_mock_kernel()
        defaults = {"scale": 1.0}
        idx = registry.add_kernel(kernel, ["x", "scale"], defaults)
        assert registry.get_kernel_default_args(idx) == {"scale": 1.0}

    def test_get_kernel_default_args_none_becomes_empty(self, registry):
        kernel = _make_mock_kernel()
        idx = registry.add_kernel(kernel, ["x"], None)
        assert registry.get_kernel_default_args(idx) == {}


class TestNKIRegistryResetTable:
    def test_reset_clears_all_state(self, registry):
        kernel = _make_mock_kernel()
        registry.add_kernel(kernel, ["x"])
        registry.add_constant_args({"c": 1}, "key1")

        registry.reset_table()

        assert registry.id_to_kernel == {}
        assert registry.kernel_to_id == {}
        assert registry.kernel_arg_names == {}
        assert registry.kernel_default_args == {}
        assert registry.constant_args == {}


class TestNKIRegistryConstantArgs:
    def test_add_and_get_constant_args(self, registry):
        registry.add_constant_args({"scale": 2.0}, "hash1")
        assert registry.get_constant_args("hash1") == {"scale": 2.0}

    def test_get_constant_args_missing_key_raises(self, registry):
        with pytest.raises(AssertionError):
            registry.get_constant_args("nonexistent")

    def test_replica_group_key_is_value_based(self, isolated_kernel_registry):
        first = ReplicaGroup([[0, 1], [2, 3]])
        second = ReplicaGroup([[0, 1], [2, 3]])

        first_key = register_nki_constant_args(("replica_group",), ("replica_group",), first)
        second_key = register_nki_constant_args(("replica_group",), ("replica_group",), second)

        assert first_key == second_key


@pytest.mark.parametrize(
    "mode_type",
    [nki_hop._CachingTorchDispatchMode, nki_hop._CachedTorchDispatchMode],
)
def test_nki_hop_delegates_checkpoint_modes(mode_type):
    handler = nki_hop.nki_kernel_wrapper.python_key_table[mode_type]
    mode = MagicMock(spec=mode_type)
    mode.__torch_dispatch__.return_value = "checkpoint-result"

    result = handler(mode, marker="value")

    assert result == "checkpoint-result"
    mode.__torch_dispatch__.assert_called_once_with(
        nki_hop.nki_kernel_wrapper,
        [],
        (),
        {"marker": "value"},
    )


def _call_test_nki_hop(x, *, backend_config=""):
    return nki_hop.nki_kernel_wrapper(
        kernel_idx=0,
        grid=(1,),
        backend_config=backend_config,
        operand_output_aliases={},
        args=(x,),
        arg_names=("x",),
        constant_args_key=-1,
    )


@pytest.fixture
def cpu_nki_hop():
    wrapper = nki_hop.nki_kernel_wrapper
    dispatch_key = torch._C.DispatchKey.CompositeExplicitAutograd
    original_impl = wrapper.py_kernels[dispatch_key]
    calls = []

    def impl(*, args, backend_config, **_kwargs):
        calls.append(args[0].detach().clone())
        if backend_config == "tuple":
            return args[0] * 2, args[0] + 1
        return args[0] * 2

    wrapper.py_kernels[dispatch_key] = impl
    wrapper._dispatch_cache.clear()
    try:
        yield calls
    finally:
        wrapper.py_kernels[dispatch_key] = original_impl
        wrapper._dispatch_cache.clear()


@pytest.mark.parametrize(
    ("policy", "expected_calls"),
    [
        (CheckpointPolicy.MUST_SAVE, 1),
        (CheckpointPolicy.PREFER_RECOMPUTE, 2),
    ],
)
def test_nki_hop_checkpoint_modes_cache_or_recompute(cpu_nki_hop, policy, expected_calls):
    def policy_fn(_ctx, _op, *_args, **_kwargs):
        return policy

    caching_mode, cached_mode = create_selective_checkpoint_contexts(policy_fn)
    forward_input = torch.ones(4)
    replay_input = torch.full((4,), 3.0)
    with caching_mode:
        forward = _call_test_nki_hop(forward_input)

    with cached_mode:
        replay = _call_test_nki_hop(replay_input)

    assert len(cpu_nki_hop) == expected_calls
    expected = forward if policy == CheckpointPolicy.MUST_SAVE else torch.full((4,), 6.0)
    torch.testing.assert_close(replay, expected)


def test_selective_checkpoint_contexts_replay_nki_hop_outputs_in_order(cpu_nki_hop):
    caching_mode, cached_mode = create_selective_checkpoint_contexts([nki_hop.nki_kernel_wrapper])
    forward_inputs = (torch.ones(4), torch.full((4,), 2.0))
    replay_inputs = (torch.full((4,), 10.0), torch.full((4,), 20.0))

    with caching_mode:
        forward_outputs = tuple(_call_test_nki_hop(x) for x in forward_inputs)
    with cached_mode:
        replay_outputs = tuple(_call_test_nki_hop(x) for x in replay_inputs)

    assert len(cpu_nki_hop) == len(forward_inputs)
    for actual, expected in zip(replay_outputs, forward_outputs, strict=True):
        torch.testing.assert_close(actual, expected)


def test_selective_checkpoint_contexts_replay_tuple_output(cpu_nki_hop):
    caching_mode, cached_mode = create_selective_checkpoint_contexts([nki_hop.nki_kernel_wrapper])
    with caching_mode:
        forward = _call_test_nki_hop(torch.ones(4), backend_config="tuple")
    with cached_mode:
        replay = _call_test_nki_hop(torch.full((4,), 10.0), backend_config="tuple")

    assert len(cpu_nki_hop) == 1
    assert isinstance(replay, tuple)
    for actual, expected in zip(replay, forward, strict=True):
        torch.testing.assert_close(actual, expected)


def test_nki_hop_runs_through_checkpoint_forward_and_backward(cpu_nki_hop):
    def context_fn():
        return create_selective_checkpoint_contexts([nki_hop.nki_kernel_wrapper])

    def fn(x):
        cached_activation = _call_test_nki_hop(x).detach()
        return x * cached_activation

    x = torch.tensor([0.25, -0.5, 1.0], requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)

    actual = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)
    expected = reference_x * (reference_x * 2).detach()
    actual.sum().backward()
    expected.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, reference_x.grad)
    assert len(cpu_nki_hop) == 1


class TestNKIHOPCallerReplicaGroup:
    def test_registers_replica_group_before_hop(self, monkeypatch, isolated_kernel_registry):
        hop = MagicMock(return_value="result")
        monkeypatch.setattr(nki_hop, "nki_kernel_wrapper", hop)

        caller = NKIHOPCaller(
            kernel_idx=3,
            grid=(2,),
            kernel_arg_names=["x", "scale", "replica_group"],
            kernel_default_args={},
        )
        tensor = torch.ones(2)
        replica_group = ReplicaGroup([[0, 1], [2, 3]])

        assert caller(tensor, 2.0, replica_group) == "result"

        call = hop.call_args.kwargs
        assert call["args"] == (tensor,)
        assert call["arg_names"] == ("x",)
        assert call["constant_args_key"] != -1
        assert isolated_kernel_registry.get_constant_args(call["constant_args_key"]) == {
            "scale": 2.0,
            "replica_group": replica_group,
        }

    def test_non_replica_group_call_keeps_existing_hop_path(self, monkeypatch):
        hop = MagicMock(return_value="result")
        monkeypatch.setattr(nki_hop, "nki_kernel_wrapper", hop)

        caller = NKIHOPCaller(
            kernel_idx=3,
            grid=(2,),
            kernel_arg_names=["x", "scale"],
            kernel_default_args={},
        )
        tensor = torch.ones(2)

        assert caller(tensor, 2.0) == "result"

        call = hop.call_args.kwargs
        assert call["args"] == (tensor, 2.0)
        assert call["arg_names"] == ["x", "scale"]
        assert call["constant_args_key"] == -1

    def test_dynamo_captures_replica_group_as_constant(self, monkeypatch, isolated_kernel_registry):
        def fake_hop(**kwargs):
            return kwargs["args"][0]

        fake_hop = torch._dynamo.allow_in_graph(fake_hop)
        monkeypatch.setattr(nki_hop, "nki_kernel_wrapper", fake_hop)

        caller = NKIHOPCaller(
            kernel_idx=3,
            grid=(2,),
            kernel_arg_names=["x", "scale", "replica_group"],
            kernel_default_args={},
        )
        graphs = []

        def backend(graph, _example_inputs):
            graphs.append(graph)
            return graph.forward

        def fn(x):
            return caller(x, 2.0, ReplicaGroup([[0, 1], [2, 3]]))

        result = torch.compile(fn, backend=backend, fullgraph=True)(torch.ones(2))

        assert torch.equal(result, torch.ones(2))
        assert len(graphs) == 1
        hop_node = next(
            node
            for node in graphs[0].graph.nodes
            if node.op == "call_function" and node.target is fake_hop
        )
        assert hop_node.kwargs["arg_names"] == ("x",)
        assert hop_node.kwargs["constant_args_key"] != -1
        assert all(
            not isinstance(value, ReplicaGroup)
            for value in torch.utils._pytree.tree_leaves(hop_node.kwargs)
        )
        constants = isolated_kernel_registry.get_constant_args(hop_node.kwargs["constant_args_key"])
        assert constants["replica_group"] == ReplicaGroup([[0, 1], [2, 3]])


class TestReplicaGroupDynamoGuards:
    def test_guards_nested_values(self):
        graphs = []

        def backend(graph, _example_inputs):
            graphs.append(graph)
            return graph.forward

        def fn(x, replica_group):
            return x + replica_group._value[0][-1]

        torch._dynamo.reset()
        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        tensor = torch.ones(2)
        replica_group = ReplicaGroup([[0, 1, 2, 3]])

        try:
            assert torch.equal(compiled(tensor, replica_group), tensor + 3)
            assert len(graphs) == 1

            # The same object and an equivalent object both satisfy the guards.
            assert torch.equal(compiled(tensor, replica_group), tensor + 3)
            assert len(graphs) == 1
            equivalent_group = ReplicaGroup([[0, 1, 2, 3]])
            assert torch.equal(compiled(tensor, equivalent_group), tensor + 3)
            assert len(graphs) == 1

            # Different values invalidate the element equality guards.
            different_group = ReplicaGroup([[0, 1, 2, 4]])
            assert torch.equal(compiled(tensor, different_group), tensor + 4)
            assert len(graphs) == 2

            # Check in-place mutation against a fresh specialization so Dynamo's
            # recompilation generalization from the previous case cannot mask it.
            torch._dynamo.reset()
            graphs.clear()
            compiled = torch.compile(fn, backend=backend, fullgraph=True)
            replica_group = ReplicaGroup([[0, 1, 2, 3]])
            assert torch.equal(compiled(tensor, replica_group), tensor + 3)
            assert len(graphs) == 1

            replica_group._value[0][-1] = 5
            assert torch.equal(compiled(tensor, replica_group), tensor + 5)
            assert len(graphs) == 2
        finally:
            torch._dynamo.reset()

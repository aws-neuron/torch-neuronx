"""
Tests for io_mutation.py.

Organised into:
  1. Parametrized unit tests for every helper (pure function, all edge-cases).
  2. Parametrized integration tests that verify the full pipeline behaviour using
     custom-constructed PassMutations (no production FX passes):
       - apply_input_mutations ↔ apply_output_mutations roundtrip
       - get_alias_output_buffer_map and get_mutation_copy_back_inputs
"""

import gc
import threading
import weakref

import pytest
import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch._subclasses.functional_tensor import FunctionalTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.overrides import TorchFunctionMode
from torch.utils import _python_dispatch

from torch_neuronx.neuron_dynamo_backend import backend
from torch_neuronx.neuron_dynamo_backend.fx import io_mutation
from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
    IOMutation,
    MutationKind,
    PassMutations,
    _shift_index_through_mutation,
    _translate_indices,
    apply_input_mutations,
    apply_output_mutations,
    get_alias_output_buffer_map,
    get_mutation_copy_back_inputs,
    rebase_pre_aot_pass_mutations,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def inp_remove(indices, name="P"):
    return IOMutation(kind=MutationKind.REMOVE, indices=indices, pass_name=name)


def inp_add(indices, key="none", params=None, name="P"):
    return IOMutation(
        kind=MutationKind.ADD,
        indices=indices,
        transform_key=key,
        transform_params=params or {},
        pass_name=name,
    )


def out_remove(indices, key="none", params=None, name="P"):
    return IOMutation(
        kind=MutationKind.REMOVE,
        indices=indices,
        transform_key=key,
        transform_params=params or {},
        pass_name=name,
    )


def out_add(indices, key="none", params=None, name="P"):
    return IOMutation(
        kind=MutationKind.ADD,
        indices=indices,
        transform_key=key,
        transform_params=params or {},
        pass_name=name,
    )


def alias(out_indices, in_indices, is_mutation=False, name="P"):
    return IOMutation(
        kind=MutationKind.ALIAS,
        indices=out_indices,
        alias_input_indices=in_indices,
        is_mutation=is_mutation,
        pass_name=name,
    )


def pm(name, in_muts=None, out_muts=None):
    return PassMutations(
        pass_name=name, input_mutations=in_muts or [], output_mutations=out_muts or []
    )


# A transform that injects a sentinel string value, used to test ADD output transforms
SENTINEL = "injected"


def _make_sentinel_pass(indices):
    """Custom PassMutations: input ADD that injects SENTINEL strings."""
    from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import TRANSFORM_REGISTRY

    TRANSFORM_REGISTRY["sentinel"] = lambda idx, dev, params, orig: SENTINEL
    return pm("SentinelAdd", in_muts=[inp_add(indices, key="sentinel")])


# ===========================================================================
# 1. _shift_index_through_mutation
# ===========================================================================


@pytest.mark.parametrize(
    "idx,mutation,forward,expected",
    [
        # ALIAS is always no-op
        (3, alias([0], [0]), True, 3),
        (3, alias([0], [0]), False, 3),
        # None propagates
        (None, inp_remove([1]), True, None),
        (None, inp_remove([1]), False, None),
        # ADD forward: shift up when add_pos <= idx
        (3, inp_add([1]), True, 4),  # 1 <= 3: +1
        (3, inp_add([3]), True, 4),  # 3 <= 3: +1
        (3, inp_add([4]), True, 3),  # 4 > 3: no shift
        (3, inp_add([1, 2]), True, 5),  # two adds <= 3
        (0, inp_add([0]), True, 1),  # add at 0
        # REMOVE forward: None if removed; else shift down
        (2, inp_remove([2]), True, None),
        (3, inp_remove([1]), True, 2),
        (3, inp_remove([4]), True, 3),
        (3, inp_remove([0, 2]), True, 1),
        (0, inp_remove([1]), True, 0),
        # ADD backward: undo (shift down; None if idx was the added slot)
        (2, inp_add([2]), False, None),
        (3, inp_add([1]), False, 2),
        (3, inp_add([4]), False, 3),
        (0, inp_add([1]), False, 0),
        # REMOVE backward: undo (shift up)
        (2, inp_remove([2]), False, 3),
        (2, inp_remove([3]), False, 2),
        (1, inp_remove([0, 1]), False, 3),
    ],
)
def test_shift_index(idx, mutation, forward, expected):
    assert _shift_index_through_mutation(idx, mutation, forward=forward) == expected


# ===========================================================================
# 2. _translate_indices
# ===========================================================================


@pytest.mark.parametrize(
    "in_idx,out_idx,passes,forward,exp_in,exp_out",
    [
        ([1, 3], [0, 2], [], True, [1, 3], [0, 2]),
        ([1, 3], [0, 2], [], False, [1, 3], [0, 2]),
        ([0, 2], [0], [pm("P", in_muts=[inp_add([1])])], True, [0, 3], [0]),
        ([0, 2, 1], [], [pm("P", in_muts=[inp_remove([1])])], True, [0, 1, None], []),
        ([], [1, 3], [pm("P", out_muts=[out_add([2])])], True, [], [1, 4]),
        ([], [0, 2, 1], [pm("P", out_muts=[out_remove([1])])], True, [], [0, 1, None]),
        (
            [3],
            [3],
            [pm("P", in_muts=[alias([0], [0])], out_muts=[alias([1], [1])])],
            True,
            [3],
            [3],
        ),
        (
            [2],
            [],
            [pm("P1", in_muts=[inp_remove([2])]), pm("P2", in_muts=[inp_add([0])])],
            True,
            [None],
            [],
        ),
        ([0, 1], [], [pm("P", in_muts=[inp_remove([1])])], False, [0, 2], []),
    ],
)
def test_translate_indices(in_idx, out_idx, passes, forward, exp_in, exp_out):
    got_in, got_out = _translate_indices(in_idx, out_idx, passes, forward=forward)
    assert got_in == exp_in
    assert got_out == exp_out


# ===========================================================================
# 3. apply_input_mutations
# ===========================================================================


@pytest.mark.parametrize(
    "inputs,passes,expected",
    [
        ([1, 2, 3], [], [1, 2, 3]),
        (["a", "b", "c"], [pm("P", in_muts=[inp_remove([1])])], ["a", "c"]),
        (["a", "b", "c", "d"], [pm("P", in_muts=[inp_remove([0, 2])])], ["b", "d"]),
        ([1, 2, 3], [pm("P", in_muts=[alias([0], [0])])], [1, 2, 3]),
    ],
)
def test_apply_input_mutations_simple(inputs, passes, expected):
    assert apply_input_mutations(list(inputs), passes, "cpu") == expected


def test_apply_input_mutations_add_shard():
    from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import TRANSFORM_REGISTRY

    passes = [pm("P", in_muts=[inp_add([2], "shard_index", {"shard_index": 5})])]
    result = apply_input_mutations(["a", "b"], passes, "cpu")
    assert result[0] == "a" and result[1] == "b" and result[2].item() == 5
    assert apply_input_mutations(["a", "b"], passes, "cpu")[2] is not result[2]


@pytest.fixture(autouse=True)
def empty_shard_index_cache():
    # Each test starts with an empty cache. An entry left by an earlier test can stand in for the
    # one a test makes, and hide a missing mode check.
    io_mutation._shard_index_on.cache_clear()
    yield
    io_mutation._shard_index_on.cache_clear()


def inject(inputs, passes, device):
    # The shard index as the executor injects it: with the cached transform.
    return apply_input_mutations(inputs, passes, device, io_mutation._EXECUTOR_INPUT_TRANSFORMS)


@pytest.fixture
def cache_on_cpu(monkeypatch):
    # Real execution reuses the tensor only on Neuron. These cache tests run on CPU.
    monkeypatch.setattr(
        io_mutation, "_SHARD_INDEX_CACHE_DEVICE_TYPES", frozenset({"neuron", "cpu"})
    )


@pytest.mark.usefixtures("cache_on_cpu")
@pytest.mark.parametrize(
    "make",
    [
        lambda: io_mutation.TRANSFORM_REGISTRY["shard_index"](0, "cpu", {"shard_index": 5}, None),
        lambda: apply_input_mutations(
            ["a"], [pm("P", in_muts=[inp_add([1], "shard_index", {"shard_index": 5})])], "cpu"
        )[1],
    ],
    ids=["registry", "apply_input_mutations"],
)
def test_shard_index_transform_new_tensor_each_call(make):
    # Only the executor reuses the tensor. A caller of the transform that changes the tensor it gets
    # must not change the next call's: 5, changed to 17, then 5 again.
    first = make()
    first.fill_(17)
    again = make()
    assert again is not first and again.item() == 5
    assert io_mutation._shard_index_on.cache_info().currsize == 0


@pytest.mark.usefixtures("cache_on_cpu")
def test_cpu_fallback_does_not_use_shard_index_cache():
    def vocab_mask(tokens, shard_index):
        return (tokens >= shard_index * 4) & (tokens < shard_index * 4 + 4)

    passes = [pm("P", in_muts=[inp_add([1], "shard_index", {"shard_index": 1})])]
    fallback = backend._make_fallback_fn(torch.fx.symbolic_trace(vocab_mask), passes)
    for _ in range(2):
        assert fallback([torch.tensor([1, 5, 9])])[0].tolist() == [False, True, False]
    assert io_mutation._shard_index_on.cache_info().currsize == 0


def _seventeen_for_five(index, device, params, original_inputs):
    # A replacement TRANSFORM_REGISTRY["shard_index"]: 17 for a request for 5.
    shard_index = 17 if params["shard_index"] == 5 else params["shard_index"]
    return torch.tensor(shard_index, dtype=torch.int64, device=device)


@pytest.mark.parametrize("cached_types", [{"neuron"}, {"neuron", "cpu"}], ids=["neuron", "cpu"])
def test_executor_transform_honors_registry_replacement(monkeypatch, cached_types):
    # A replaced TRANSFORM_REGISTRY["shard_index"] applies to the executor's injection too, so the
    # executor and the CPU fallback compute the same mask. With the original factory back, the
    # executor reuses its tensor again.
    monkeypatch.setattr(io_mutation, "_SHARD_INDEX_CACHE_DEVICE_TYPES", frozenset(cached_types))

    def vocab_mask(tokens, shard_index):
        return (tokens >= shard_index * 4) & (tokens < shard_index * 4 + 4)

    passes = [pm("P", in_muts=[inp_add([1], "shard_index", {"shard_index": 5})])]
    tokens = torch.tensor([1, 5, 9, 20, 69, 70])
    original = io_mutation.TRANSFORM_REGISTRY["shard_index"]
    warm = inject([tokens], passes, "cpu")[1]
    size = io_mutation._shard_index_on.cache_info().currsize
    monkeypatch.setitem(io_mutation.TRANSFORM_REGISTRY, "shard_index", _seventeen_for_five)
    injected = inject([tokens], passes, "cpu")
    fallback = backend._make_fallback_fn(torch.fx.symbolic_trace(vocab_mask), passes)
    assert injected[1].item() == 17
    expected = [False, False, False, False, True, True]
    assert vocab_mask(*injected).tolist() == fallback([tokens])[0].tolist() == expected
    assert io_mutation._shard_index_on.cache_info().currsize == size
    monkeypatch.setitem(io_mutation.TRANSFORM_REGISTRY, "shard_index", original)
    again = inject([tokens], passes, "cpu")[1]
    assert again.item() == 5
    assert (again is warm) == ("cpu" in cached_types)


class _FalsyTransform:
    # A transform whose truth value is false. The transforms argument selects it by key.
    def __init__(self, value):
        self.value = value

    def __call__(self, index, device, params, original_inputs):
        return torch.tensor(self.value, dtype=torch.int64, device=device)

    def __bool__(self):
        return False


@pytest.mark.parametrize("key", ["shard_index", "custom"], ids=["registry_key", "new_key"])
def test_apply_input_mutations_transforms_selected_by_key(key):
    passes = [pm("P", in_muts=[inp_add([1], key, {"shard_index": 5})])]
    assert apply_input_mutations(["a"], passes, "cpu", {key: _FalsyTransform(42)})[1].item() == 42


def _pre_dispatch_functional_trace(fn, x):
    # Pre-dispatch functionalization on real tensors. FunctionalTensorMode(pre_dispatch=True)
    # and make_fx's pre-dispatch tracing sit on the pre-dispatch stack, so the ordinary dispatch
    # stack stays empty.
    with enable_python_dispatcher(), FunctionalTensorMode(pre_dispatch=True):
        return make_fx(fn, pre_dispatch=True)(x)


def _fake_mode(fn, x):
    # FakeTensorMode alone: on the dispatch mode stack, with no torch function mode.
    with FakeTensorMode() as mode:
        return fn(mode.from_tensor(x))


def _pre_dispatch_mode(fn, x):
    # FunctionalTensorMode(pre_dispatch=True) alone: the PreDispatch key, with no torch function
    # mode (make_fx's pre-dispatch tracing adds one).
    with enable_python_dispatcher(), FunctionalTensorMode(pre_dispatch=True):
        return fn(x)


def _native_functionalization(fn, x):
    # Functionalization without a mode: the Functionalize key in this thread's include set.
    torch._enable_functionalization(reapply_views=True)
    try:
        return fn(torch._to_functional_tensor(x))
    finally:
        torch._disable_functionalization()


class _WrapNewTensors(TorchFunctionMode):
    # Wraps each tensor that torch.tensor makes. The dispatch mode stack and keys stay empty.
    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        return torch._to_functional_tensor(out) if func is torch.tensor else out


def _torch_function_mode(fn, x):
    with _WrapNewTensors():
        return fn(x)


class _CpuDeviceToMeta(TorchFunctionMode):
    # Turns torch.device("cpu") into the meta device. torch.tensor's own device argument is not
    # intercepted, so a call that keeps the original argument still makes its tensor on the CPU.
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.device and args and args[0] == "cpu":
            return torch.device("meta")
        return func(*args, **(kwargs or {}))


def _device_constructor_mode(fn, x):
    # A mode that intercepts torch.device(): the mask under the mode must still be right.
    with _CpuDeviceToMeta():
        out = fn(x)
    assert out[0].device == x.device and out[0].tolist() == [False, True, False]
    return out


def _dynamo_full_graph(fn, x):
    # Dynamo full-graph capture with a backend that runs the captured graph as it is.
    torch._dynamo.reset()
    try:
        out = torch.compile(fn, fullgraph=True, backend=lambda gm, example_inputs: gm.forward)(x)
    finally:
        torch._dynamo.reset()
    assert out[0].tolist() == [False, True, False]
    return out


@pytest.mark.parametrize("warm", [False, True], ids=["empty", "warm"])
@pytest.mark.parametrize("cached_types", [{"neuron"}, {"neuron", "cpu"}], ids=["neuron", "cpu"])
@pytest.mark.parametrize(
    "trace",
    [
        lambda fn, x: make_fx(fn, tracing_mode="fake")(x),
        lambda fn, x: torch.func.functionalize(fn)(x),
        _pre_dispatch_functional_trace,
        _fake_mode,
        _pre_dispatch_mode,
        _native_functionalization,
        _torch_function_mode,
        _device_constructor_mode,
        _dynamo_full_graph,
    ],
    ids=[
        "fake",
        "functionalize",
        "pre_dispatch",
        "fake_mode",
        "pre_dispatch_mode",
        "native_functionalization",
        "torch_function_mode",
        "device_constructor_mode",
        "dynamo_full_graph",
    ],
)
def test_apply_input_mutations_shard_index_trace_then_run(monkeypatch, cached_types, trace, warm):
    # Trace the graph with the shard index injected as the executor injects it, then run it with
    # real tokens. The trace must not read or fill the cache, and the real run must not get the
    # tensor made while tracing. With CPU in the cached types, only the mode checks prevent it.
    # "warm" fills the cache with an eager call first: a trace that reads the cache then gets the
    # real tensor, so only cache_info() shows it.
    monkeypatch.setattr(
        io_mutation, "_SHARD_INDEX_CACHE_DEVICE_TYPES", frozenset(cached_types), raising=False
    )

    def vocab_mask(tokens, shard_index):
        return (tokens >= shard_index * 4) & (tokens < shard_index * 4 + 4)

    passes = [pm("P", in_muts=[inp_add([1], "shard_index", {"shard_index": 1})])]
    gm = torch.fx.symbolic_trace(vocab_mask)

    def run(inputs):
        return (gm(*inject(list(inputs), passes, "cpu")),)

    if warm:
        assert run([torch.tensor([1, 5, 9])])[0].tolist() == [False, True, False]
    before = io_mutation._shard_index_on.cache_info()
    trace(lambda tokens: run([tokens]), torch.tensor([1, 5, 9]))
    assert io_mutation._shard_index_on.cache_info() == before
    assert run([torch.tensor([1, 5, 9])])[0].tolist() == [False, True, False]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "neuron",
            marks=pytest.mark.skipif(
                not torch.neuron.is_available(), reason="Neuron not available"
            ),
        ),
    ],
)
@pytest.mark.usefixtures("cache_on_cpu")
def test_apply_input_mutations_shard_index_fake_mode_in_other_thread(monkeypatch, device):
    # Two threads enter FakeTensorMode and the first exits. The first then calls outside any mode
    # while the second is still in its mode. Neither call may get the other's kind of tensor.
    # The exits leave torch's process-wide mode flags set; the test puts them back.
    for name in ("_is_in_torch_dispatch_mode", "_is_in_non_infra_torch_dispatch_mode"):
        monkeypatch.setattr(_python_dispatch, name, getattr(_python_dispatch, name))
    if device == "neuron":
        device = f"neuron:{torch.neuron.current_device()}"
    passes = [pm("P", in_muts=[inp_add([2], "shard_index", {"shard_index": 2})])]
    steps = {name: threading.Event() for name in ("a_in", "b_in", "a_out", "b_got", "a_got")}
    results, errors = {}, []

    def wait(name):
        assert steps[name].wait(30), f"timed out waiting for {name}"

    def first():
        with FakeTensorMode():
            steps["a_in"].set()
            wait("b_in")
        steps["a_out"].set()
        wait("b_got")
        results["first"] = inject(["a", "b"], passes, device)[2]
        steps["a_got"].set()

    def second():
        wait("a_in")
        with FakeTensorMode():
            steps["b_in"].set()
            wait("a_out")
            results["second"] = inject(["a", "b"], passes, device)[2]
            steps["b_got"].set()
            wait("a_got")

    def run(fn):
        try:
            fn()
        except BaseException as e:
            errors.append(e)
            for step in steps.values():
                step.set()

    threads = [threading.Thread(target=run, args=(fn,)) for fn in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not errors, errors
    assert not any(thread.is_alive() for thread in threads)
    assert isinstance(results["second"], FakeTensor)
    real = results["first"]
    assert not isinstance(real, FakeTensor)
    assert real.device == torch.device(device) and real.item() == 2
    assert inject(["a", "b"], passes, device)[2] is real


@pytest.mark.usefixtures("cache_on_cpu")
def test_apply_input_mutations_shard_index_reused():
    passes = [pm("P", in_muts=[inp_add([2], "shard_index", {"shard_index": 3})])]
    first = inject(["a", "b"], passes, "cpu")[2]
    second = inject(["a", "b"], passes, "cpu")[2]
    assert first is second and first.dtype == torch.int64 and first.item() == 3
    other = [pm("P", in_muts=[inp_add([2], "shard_index", {"shard_index": 4})])]
    assert inject(["a", "b"], other, "cpu")[2].item() == 4


@pytest.mark.usefixtures("cache_on_cpu")
def test_apply_input_mutations_shard_index_per_stream(monkeypatch):
    passes = [pm("P", in_muts=[inp_add([2], "shard_index", {"shard_index": 6})])]
    monkeypatch.setattr(io_mutation, "_current_stream_id", lambda device: 1)
    first = inject(["a", "b"], passes, "cpu")[2]
    monkeypatch.setattr(io_mutation, "_current_stream_id", lambda device: 2)
    second = inject(["a", "b"], passes, "cpu")[2]
    assert first is not second and second.item() == 6
    assert inject(["a", "b"], passes, "cpu")[2] is second


@pytest.mark.usefixtures("cache_on_cpu")
def test_apply_input_mutations_shard_index_stream_churn(monkeypatch):
    # Each new stream gets a new id. Discarded streams must not keep their tensors alive.
    passes = [pm("P", in_muts=[inp_add([2], "shard_index", {"shard_index": 8})])]
    refs = []
    for stream_id in range(1000):
        monkeypatch.setattr(io_mutation, "_current_stream_id", lambda device, s=stream_id: s)
        refs.append(weakref.ref(inject(["a", "b"], passes, "cpu")[2]))
    gc.collect()
    size = io_mutation._SHARD_INDEX_CACHE_SIZE
    assert io_mutation._shard_index_on.cache_info().currsize <= size
    assert sum(r() is not None for r in refs) <= size
    assert refs[-1]() is inject(["a", "b"], passes, "cpu")[2]


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
def test_apply_input_mutations_shard_index_two_neuron_streams():
    # The first upload is asynchronous on the current stream. Each stream must get its own tensor.
    # With TORCH_NEURONX_ENABLE_STREAMS=0, both streams are the default stream and share one.
    passes = [pm("P", in_muts=[inp_add([2], "shard_index", {"shard_index": 7})])]
    device = f"neuron:{torch.neuron.current_device()}"
    streams = [torch.neuron.Stream(), torch.neuron.Stream()]
    tensors = []
    for stream in streams:
        with torch.neuron.stream(stream):
            tensors.append(inject(["a", "b"], passes, device)[2])
            assert inject(["a", "b"], passes, device)[2] is tensors[-1]
    same_stream = streams[0].stream_id == streams[1].stream_id
    assert (tensors[0] is tensors[1]) == same_stream
    for stream, tensor in zip(streams, tensors, strict=True):
        with torch.neuron.stream(stream):
            assert tensor.device.type == "neuron" and tensor.dtype == torch.int64
            assert tensor.item() == 7


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
def test_apply_input_mutations_shard_index_compiled_eviction():
    # Eviction stress: later keys evict a cached tensor while a compiled read of it is still queued.
    # The read is on a created stream, behind a backlog. The keys that evict it are on the default
    # stream. All created streams share one runtime stream (kStreamPoolSize is 1). The default
    # stream is a different runtime stream. With TORCH_NEURONX_ENABLE_STREAMS=0, both are the same.
    device = f"neuron:{torch.neuron.current_device()}"
    matmul = torch.compile(lambda a: a @ a, backend="neuron", fullgraph=True, dynamic=False)
    add = torch.compile(lambda x, i: x + i, backend="neuron", fullgraph=True, dynamic=False)
    x = torch.arange(4, dtype=torch.float32, device=device)
    busy = torch.full((2048, 2048), 1 / 2048, device=device)
    add(x, torch.tensor(0, device=device))
    matmul(busy)
    torch.neuron.synchronize()

    def read(k):
        passes = [pm("P", in_muts=[inp_add([2], "shard_index", {"shard_index": k})])]
        return add(x, inject(["a", "b"], passes, device)[2])

    # The check reads an event 32 matmuls ahead of the read. An event after the read can still
    # be pending when the read has run. A stream has no hold that the host releases: a wait on
    # an event not yet recorded does not wait.
    created, ahead_of_read = torch.neuron.Stream(), torch.neuron.Event()
    with torch.neuron.stream(created):
        out = busy
        for i in range(96):
            if i == 64:
                ahead_of_read.record()
            out = matmul(out)
        outputs = [read(0)]
    # Key 0 is the oldest of these keys, so the last of them evicts it.
    outputs += [read(k) for k in range(1, io_mutation._SHARD_INDEX_CACHE_SIZE + 1)]
    queued_at_eviction = not ahead_of_read.query()
    torch.neuron.synchronize()
    for k, output in enumerate(outputs):
        assert torch.equal(output.cpu(), torch.arange(4, dtype=torch.float32) + k)
    if not queued_at_eviction:
        pytest.skip("The backlog drained before the eviction. The outputs are correct.")


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
def test_apply_input_mutations_shard_index_device_without_index():
    # A device without an index is the current device. It must not get a key of its own.
    passes = [pm("P", in_muts=[inp_add([2], "shard_index", {"shard_index": 9})])]
    device = f"neuron:{torch.neuron.current_device()}"
    tensor = inject(["a", "b"], passes, device)[2]
    assert inject(["a", "b"], passes, "neuron")[2] is tensor


def _run_shard_index_executor(shard_index, calls=2):
    # Run an Executor that injects ``shard_index`` with the graph call patched out. Return the
    # device and the injected input of each call.
    from unittest.mock import patch

    from torch_neuronx.neuron_dynamo_backend.executor import Executor
    from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import FunctionIO, TensorSpec

    passes = [
        pm(
            "ShardIndexInjection",
            in_muts=[inp_add([1], "shard_index", {"shard_index": shard_index})],
        )
    ]
    io = FunctionIO(
        inputs=[
            TensorSpec(dtype=torch.float32, shape=(4,)),
            TensorSpec(dtype=torch.int32, shape=()),
        ],
        outputs=[TensorSpec(dtype=torch.float32, shape=(4,))],
    )
    executor = Executor(
        "g", "k", io, [TensorSpec(dtype=torch.float32, shape=(4,))], pass_mutations=passes
    )
    device = f"neuron:{torch.neuron.current_device()}"
    x = torch.zeros(4, device=device)
    seen = []
    with patch(
        "torch_neuronx._C.execute_compiled_graph",
        lambda name, key, inputs, outputs, collectives: seen.append(inputs[1]),
    ):
        for _ in range(calls):
            executor(x)
    return device, seen


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
def test_executor_reuses_injected_shard_index():
    # The executor injects the shard index with the cached transform. Two calls with the same
    # device, stream and shard index pass the graph the same tensor.
    device, seen = _run_shard_index_executor(3)
    cached = io_mutation._shard_index_on(device, io_mutation._current_stream_id(device), 3)
    assert [t.data_ptr() for t in seen] == [cached.data_ptr()] * 2


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
def test_executor_uses_replaced_shard_index_transform(monkeypatch):
    # With TRANSFORM_REGISTRY["shard_index"] replaced, the executor passes the graph the
    # replacement's value, as without the cache, and leaves the cache alone.
    monkeypatch.setitem(io_mutation.TRANSFORM_REGISTRY, "shard_index", _seventeen_for_five)
    _, seen = _run_shard_index_executor(5)
    # The graph gets the int64 scalar as a one-element tensor viewed as two int32 words.
    assert [int(t.cpu().view(torch.int64)) for t in seen] == [17, 17]
    assert io_mutation._shard_index_on.cache_info().currsize == 0


def test_apply_input_mutations_multi_pass():
    passes = [
        pm("P1", in_muts=[inp_remove([1])]),
        pm("P2", in_muts=[inp_add([2], "shard_index", {"shard_index": 9})]),
    ]
    result = apply_input_mutations(["a", "b", "c"], passes, "cpu")
    assert result[0] == "a" and result[1] == "c" and result[2].item() == 9


# ===========================================================================
# 4. apply_output_mutations
# ===========================================================================


@pytest.mark.parametrize(
    "outputs,passes,original_inputs,expected",
    [
        ([1, 2, 3], [], None, [1, 2, 3]),
        (
            ["a", "c"],
            [pm("P", out_muts=[out_remove([1, 3], "none")])],
            None,
            ["a", None, "c", None],
        ),
        (["copy", "real"], [pm("P", out_muts=[out_add([0])])], None, ["real"]),
        (["a", "b", "c"], [pm("P", out_muts=[alias([1], [0])])], None, ["a", "b", "c"]),
        # Multi-pass reverse: P1=ADD filter, P2=REMOVE re-insert
        # Undo P2 first (re-insert Nones at [1,3]): ["copy",None,"r0",None,"r1"]
        # Then undo P1 (filter out [0]): [None,"r0",None,"r1"]
        (
            ["copy", "r0", "r1"],
            [pm("P1", out_muts=[out_add([0])]), pm("P2", out_muts=[out_remove([1, 3], "none")])],
            None,
            [None, "r0", None, "r1"],
        ),
        # input_passthrough
        (
            ["real"],
            [pm("P", out_muts=[out_remove([1], "input_passthrough", {"index_to_input": {1: 0}})])],
            ["restored"],
            ["real", "restored"],
        ),
    ],
)
def test_apply_output_mutations(outputs, passes, original_inputs, expected):
    result = apply_output_mutations(list(outputs), passes, "cpu", original_inputs)
    assert result == expected


# ===========================================================================
# 5. get_alias_output_buffer_map
# ===========================================================================


@pytest.mark.parametrize(
    "passes,expected",
    [
        ([], {}),
        ([pm("P", out_muts=[out_remove([1], "none")])], {}),
        ([pm("P", out_muts=[alias([2], [1])])], {2: 1}),
        ([pm("P", out_muts=[alias([0], [0], is_mutation=True)])], {0: 0}),
        ([pm("P", out_muts=[alias([0, 2], [1, 3])])], {0: 1, 2: 3}),
        # Subsequent input REMOVE shifts neff_in down
        ([pm("A", out_muts=[alias([1], [2])]), pm("N", in_muts=[inp_remove([0])])], {1: 1}),
        # Subsequent input ADD shifts neff_in up
        (
            [
                pm("A", out_muts=[alias([0], [1])]),
                pm("N", in_muts=[inp_add([0], "shard_index", {"shard_index": 0})]),
            ],
            {0: 2},
        ),
        # Subsequent output ADD shifts neff_out up
        ([pm("A", out_muts=[alias([1], [0])]), pm("N", out_muts=[out_add([0])])], {2: 0}),
        # Subsequent output REMOVE shifts neff_out down
        (
            [pm("A", out_muts=[alias([2], [0])]), pm("N", out_muts=[out_remove([0], "none")])],
            {1: 0},
        ),
        # Alias output eliminated by subsequent pass → not in result
        ([pm("A", out_muts=[alias([1], [0])]), pm("N", out_muts=[out_remove([1], "none")])], {}),
        # Alias input eliminated by subsequent pass → not in result
        ([pm("A", out_muts=[alias([0], [1])]), pm("N", in_muts=[inp_remove([1])])], {}),
        # Alias in middle pass
        (
            [
                pm("P1", in_muts=[inp_remove([0])]),
                pm("P2", out_muts=[alias([1], [1])]),
                pm("P3", in_muts=[inp_add([2], "shard_index", {"shard_index": 0})]),
            ],
            {1: 1},
        ),
    ],
)
def test_get_alias_output_buffer_map(passes, expected):
    assert get_alias_output_buffer_map(passes) == expected


def test_duplicate_alias_output_first_wins():
    """When two passes emit an alias for the same NEFF output index, the first one wins."""
    passes = [
        pm("PassA", out_muts=[alias([0], [1])]),
        pm("PassB", out_muts=[alias([0], [2])]),
    ]
    result = get_alias_output_buffer_map(passes)
    assert result == {0: 1}, f"First alias should win, got {result}"


def test_duplicate_alias_within_same_pass_first_wins():
    """Two ALIAS entries in the same pass targeting the same output: first wins."""
    passes = [
        pm("Pass", out_muts=[alias([0], [1], is_mutation=True), alias([0], [3])]),
    ]
    result = get_alias_output_buffer_map(passes)
    assert result == {0: 1}, f"First alias should win, got {result}"


# ===========================================================================
# 6. get_mutation_copy_back_inputs
# ===========================================================================


@pytest.mark.parametrize(
    "passes,expected",
    [
        ([], {}),
        ([pm("P", out_muts=[alias([1], [0], is_mutation=False)])], {}),
        ([pm("P", out_muts=[alias([0], [0], is_mutation=True)])], {0: 0}),
        # Predecessor REMOVE shifts caller_in up
        (
            [
                pm("Pre", in_muts=[inp_remove([0])]),
                pm("A", out_muts=[alias([0], [1], is_mutation=True)]),
            ],
            {0: 2},
        ),
        # Predecessor ADD shifts caller_in down
        (
            [
                pm("Pre", in_muts=[inp_add([0], "shard_index", {"shard_index": 0})]),
                pm("A", out_muts=[alias([0], [2], is_mutation=True)]),
            ],
            {0: 1},
        ),
        # Successor output ADD shifts neff_out up
        (
            [
                pm("A", out_muts=[alias([1], [0], is_mutation=True)]),
                pm("S", out_muts=[out_add([0])]),
            ],
            {2: 0},
        ),
        # Successor output REMOVE eliminates neff_out
        (
            [
                pm("A", out_muts=[alias([0], [0], is_mutation=True)]),
                pm("S", out_muts=[out_remove([0], "none")]),
            ],
            {},
        ),
        # Predecessor ADD on the exact slot → caller_in None → not in result
        (
            [
                pm("Pre", in_muts=[inp_add([1], "shard_index", {"shard_index": 0})]),
                pm("A", out_muts=[alias([0], [1], is_mutation=True)]),
            ],
            {},
        ),
        ([pm("P", out_muts=[alias([0, 1], [0, 1], is_mutation=True)])], {0: 0, 1: 1}),
        (
            [
                pm("P1", in_muts=[inp_remove([0])]),
                pm("P2", out_muts=[alias([0], [0], is_mutation=True)]),
                pm("P3", in_muts=[inp_add([2], "shard_index", {"shard_index": 0})]),
            ],
            {0: 1},
        ),
    ],
)
def test_get_mutation_copy_back_inputs(passes, expected):
    assert get_mutation_copy_back_inputs(passes) == expected


def test_duplicate_copy_back_output_first_wins():
    """When two passes emit a mutation alias for the same NEFF output, the first wins."""
    passes = [
        pm("PassA", out_muts=[alias([0], [1], is_mutation=True)]),
        pm("PassB", out_muts=[alias([0], [2], is_mutation=True)]),
    ]
    result = get_mutation_copy_back_inputs(passes)
    assert result == {0: 1}, f"First copy-back should win, got {result}"


def test_duplicate_copy_back_within_same_pass_first_wins():
    """Two mutation ALIAS entries in the same pass for the same output: first wins."""
    passes = [
        pm("Pass", out_muts=[alias([0], [1], is_mutation=True), alias([0], [3], is_mutation=True)]),
    ]
    result = get_mutation_copy_back_inputs(passes)
    assert result == {0: 1}, f"First copy-back should win, got {result}"


# ===========================================================================
# 7. Full pipeline roundtrip using custom PassMutations
#
# Each case defines a sequence of custom passes that manipulate inputs and
# outputs, then verifies:
#   (a) apply_input_mutations produces the correct NEFF inputs
#   (b) apply_output_mutations restores the correct caller outputs
# ===========================================================================


@pytest.mark.parametrize(
    "case",
    [
        {
            "desc": "input remove + output re-insert (None)",
            "original_inputs": ["a", "b", "c"],
            "neff_outputs": ["r0", "r1"],
            "passes": [
                pm("PassA", in_muts=[inp_remove([1])]),  # removes "b"
                pm("PassB", out_muts=[out_remove([2], "none")]),  # re-inserts None at pos 2
            ],
            "exp_neff_inputs": ["a", "c"],
            "exp_caller_outputs": ["r0", "r1", None],
        },
        {
            "desc": "output add filter (copy_ pattern)",
            "original_inputs": ["inp"],
            "neff_outputs": ["copy_result", "real_out"],
            "passes": [
                pm("PassA", out_muts=[out_add([0])]),  # copy_ result added at 0 → filter at exec
            ],
            "exp_neff_inputs": ["inp"],
            "exp_caller_outputs": ["real_out"],
        },
        {
            "desc": "multi-pass: input remove + output add + output re-insert",
            "original_inputs": ["x", "y", "z"],
            "neff_outputs": ["copy_res", "out0", "out1"],
            # Pass1: remove "y" (input[1])
            # Pass2: copy_ added output at [0] → filter at exec
            # Pass3: None removed from output at [3] → re-insert at exec
            "passes": [
                pm("PassA", in_muts=[inp_remove([1])]),
                pm("PassB", out_muts=[out_add([0])]),
                pm("PassC", out_muts=[out_remove([3], "none")]),
            ],
            "exp_neff_inputs": ["x", "z"],
            "exp_caller_outputs": ["out0", "out1", None],
        },
        {
            "desc": "input add (shard) + no output change",
            "original_inputs": ["a", "b"],
            "neff_outputs": ["result"],
            "passes": [
                pm("PassA", in_muts=[inp_add([2], "shard_index", {"shard_index": 4})]),
            ],
            "exp_neff_inputs_check": lambda r: r[0] == "a" and r[1] == "b" and r[2].item() == 4,
            "exp_caller_outputs": ["result"],
        },
        {
            "desc": "output passthrough re-inserted from original_inputs",
            "original_inputs": ["orig"],
            "neff_outputs": ["out"],
            "passes": [
                pm(
                    "PassA",
                    out_muts=[out_remove([1], "input_passthrough", {"index_to_input": {1: 0}})],
                ),
            ],
            "exp_neff_inputs": ["orig"],
            "exp_caller_outputs": ["out", "orig"],
        },
        {
            "desc": "alias mutations are no-ops in apply_input/output",
            "original_inputs": ["x", "y"],
            "neff_outputs": ["r"],
            "passes": [
                pm("PassA", in_muts=[alias([0], [0])], out_muts=[alias([0], [0])]),
            ],
            "exp_neff_inputs": ["x", "y"],
            "exp_caller_outputs": ["r"],
        },
    ],
)
def test_full_pipeline_roundtrip(case):
    passes = case["passes"]
    neff_inputs = apply_input_mutations(list(case["original_inputs"]), passes, "cpu")

    if "exp_neff_inputs" in case:
        assert neff_inputs == case["exp_neff_inputs"]
    if "exp_neff_inputs_check" in case:
        assert case["exp_neff_inputs_check"](neff_inputs)

    caller_outputs = apply_output_mutations(
        list(case["neff_outputs"]), passes, "cpu", list(case["original_inputs"])
    )
    assert caller_outputs == case["exp_caller_outputs"]


# ===========================================================================
# 8. Alias maps integration: verify buffer_map and copy_back together
# ===========================================================================


@pytest.mark.parametrize(
    "passes,exp_buffer_map,exp_copy_back",
    [
        ([], {}, {}),
        # Passthrough only
        ([pm("P", out_muts=[alias([2], [1], is_mutation=False)])], {2: 1}, {}),
        # Mutation only
        ([pm("P", out_muts=[alias([0], [0], is_mutation=True)])], {0: 0}, {0: 0}),
        # Both in same pass
        (
            [
                pm(
                    "P",
                    out_muts=[
                        alias([0, 2], [0, 1], is_mutation=True),
                        alias([3], [2], is_mutation=False),
                    ],
                )
            ],
            {0: 0, 2: 1, 3: 2},
            {0: 0, 2: 1},
        ),
        # Preceding input removal: buffer_map neff_in unchanged (no predecessors), copy_back shifted
        (
            [
                pm("Pre", in_muts=[inp_remove([0])]),
                pm("Alias", out_muts=[alias([1], [1], is_mutation=True)]),
            ],
            {1: 1},
            {1: 2},
        ),
        # Subsequent input add shifts neff_in for buffer_map but not copy_back (no predecessors)
        (
            [
                pm("Alias", out_muts=[alias([0], [0], is_mutation=False)]),
                pm("Shard", in_muts=[inp_add([0], "shard_index", {"shard_index": 0})]),
            ],
            {0: 1},
            {},
        ),
    ],
)
def test_alias_maps_combined(passes, exp_buffer_map, exp_copy_back):
    assert get_alias_output_buffer_map(passes) == exp_buffer_map
    assert get_mutation_copy_back_inputs(passes) == exp_copy_back


# ===========================================================================
# 9. rebase_pre_aot_pass_mutations
# ===========================================================================


@pytest.mark.parametrize(
    "pre_aot_pm,passes_before,exp_buffer_map",
    [
        # None input → None output
        (None, [], None),
        # No passes before: indices unchanged
        (
            pm("AliasingAnalysis", out_muts=[alias([1], [0], is_mutation=False)]),
            [],
            {1: 0},
        ),
        # Preceding input REMOVE shifts alias input index forward
        # passes_before has REMOVE [0] on input → input[0] shifts to None but alias is input[1]
        # input[1] after REMOVE [0] → still 0 (shifts down by 1 since removal at 0 < 1)
        (
            pm("AliasingAnalysis", out_muts=[alias([1], [1], is_mutation=False)]),
            [pm("Flex", in_muts=[inp_remove([0])])],
            {1: 0},  # neff_in: input[1] - 1 removal before = 0
        ),
        # Preceding output ADD shifts alias output index forward
        # passes_before has ADD [0] on output → output[1] shifts up to 2
        (
            pm("AliasingAnalysis", out_muts=[alias([1], [0], is_mutation=False)]),
            [pm("Flex", out_muts=[out_add([0])])],
            {2: 0},  # neff_out: output[1] + 1 insertion at 0 = 2
        ),
        # Preceding input ADD shifts alias input index forward
        # passes_before ADD [0] on input → input[1] shifts up to 2
        (
            pm("AliasingAnalysis", out_muts=[alias([0], [1], is_mutation=False)]),
            [pm("Flex", in_muts=[inp_add([0], "shard_index", {"shard_index": 0})])],
            {0: 2},  # neff_in: input[1] + 1 insertion at 0 = 2
        ),
        # Preceding output REMOVE eliminates alias output → not in result
        (
            pm("AliasingAnalysis", out_muts=[alias([0], [0], is_mutation=False)]),
            [pm("Flex", out_muts=[out_remove([0], "none")])],
            {},  # output[0] removed → None → dropped
        ),
        # Mutation alias (is_mutation=True) also rebased
        (
            pm("AliasingAnalysis", out_muts=[alias([0], [1], is_mutation=True)]),
            [pm("Flex", in_muts=[inp_remove([0])])],
            {0: 0},  # input[1] - 1 removal = 0
        ),
        # Multiple aliases in same PassMutations, mixed directions
        (
            pm(
                "AliasingAnalysis",
                out_muts=[
                    alias([0, 2], [0, 1], is_mutation=False),
                ],
            ),
            [pm("Flex", in_muts=[inp_remove([0])])],
            # input[0] removed → None (dropped); input[1] - 1 = 0
            {2: 0},
        ),
        # Two alias mutations in separate IOMutations
        (
            pm(
                "AliasingAnalysis",
                out_muts=[
                    alias([0], [0], is_mutation=True),
                    alias([2], [1], is_mutation=False),
                ],
            ),
            [],
            {0: 0, 2: 1},
        ),
    ],
)
def test_rebase_pre_aot_pass_mutations(pre_aot_pm, passes_before, exp_buffer_map):
    rebased = rebase_pre_aot_pass_mutations(pre_aot_pm, passes_before)
    if exp_buffer_map is None:
        assert rebased is None
    else:
        assert rebased is not None
        rebased_list = [rebased]
        assert get_alias_output_buffer_map(rebased_list) == exp_buffer_map

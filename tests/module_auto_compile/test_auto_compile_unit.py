"""Unit tests for the module auto-compile state machine (no Neuron hardware needed)."""

import copy
import functools
import gc
import io
import multiprocessing
import os
import pickle
import queue
import threading
import types
import weakref

import pytest
import torch
import torch.nn as nn
import torch.nn.modules.module as module_mod
from torch.distributed.fsdp import FSDPModule, FullyShardedDataParallel

import torch_neuronx
import torch_neuronx._module_auto_compile_integration as integration
import torch_neuronx.module_auto_compile as mac
from torch_neuronx.module_auto_compile import (
    _OBSERVATION_CALL_BUDGET,
    _PROMOTION_INVOCATION_THRESHOLD,
    _TERMINAL_REASON_COMPILED_ANCESTOR,
    _TERMINAL_REASON_COMPILER_OWNED_GRAPH,
    _TERMINAL_REASON_COVERAGE_RESTRICTED,
    _TERMINAL_REASON_EXPLICIT_COMPILE,
    _TERMINAL_REASON_PREEXISTING_COMPILED_CALL,
    _TERMINAL_STATES,
    ModuleAutoCompileState,
    _bootstrap_forward_pre_hook,
    _classify_and_transition,
    _disable_feature,
    _env_flag,
    _get_or_create_record,
    _inside_autograd_validation_context,
    _inside_checkpoint_context,
    _inside_non_reentrant_checkpoint_recompute,
    _inside_sac_forward,
    _inside_sac_recompute,
    _install_bootstrap_hooks,
    _internal_module_compile_scope,
    _mark_user_owned,
    _module_is_leaf,
    _observe_module_forward,
    _promote_module,
    _remove_bootstrap_hooks,
    _reset_feature_disabled,
    _reset_state,
    module_auto_compile_enabled,
)


def _run_from_fake_gradcheck_frame(fn, *args):
    namespace = {"fn": fn, "args": args}
    code = compile(
        "def _runner():\n    return fn(*args)\n",
        "/fake/site-packages/torch/autograd/gradcheck.py",
        "exec",
    )
    exec(code, namespace)
    return namespace["_runner"]()


class _ClassicFSDPStub(FullyShardedDataParallel):
    def __new__(cls):
        return object.__new__(cls)

    def __init__(self):
        nn.Module.__init__(self)


class _ComposableFSDPStub(FSDPModule, nn.Module):
    def __new__(cls):
        return object.__new__(cls)

    def __init__(self):
        nn.Module.__init__(self)


class _ComposableFSDPLeafStub(_ComposableFSDPStub):
    def __init__(self):
        super().__init__()
        self._is_fsdp_managed_module = True
        self._fsdp_use_orig_params = True

    def forward(self, value):
        return value + 1


def _record_forward_pre_hook(module, *_args):
    module.events.append("pre")


def _record_forward_hook(module, *_args):
    module.events.append("post")


@pytest.fixture(autouse=True)
def clear_registry():
    """Reset AutoCompile state and its kill switch between tests."""
    _reset_feature_disabled()
    _reset_state()
    yield
    _reset_state()
    _reset_feature_disabled()


def _reported_callable(callable_, *, backward=False, forward_nodes=3):
    from torch_neuronx.neuron_dynamo_backend.backend import _AutoCompileBackendAttempt

    attempt = _AutoCompileBackendAttempt()
    attempt.record_executor("forward", forward_nodes, "forward-key")
    if backward:
        attempt.record_executor("backward", 2, "backward-key")

    @functools.wraps(callable_)
    def reported(*args, **kwargs):
        return callable_(*args, **kwargs)

    setattr(reported, mac._BACKEND_ATTEMPT_ATTR, attempt)
    return reported, attempt


def _reported_module_call(module):
    return _reported_callable(module._call_impl)[0]


def _bootstrap_record(module):
    record, _ = _get_or_create_record(module)
    record.state = ModuleAutoCompileState.BOOTSTRAP_PENDING
    _install_bootstrap_hooks(module, record)
    return record


def _ready_record(module):
    record, _ = _get_or_create_record(module)
    record.state = ModuleAutoCompileState.PROMOTION_READY
    record.pending_promotion = True
    return record


class TestModuleAutoCompileEnabled:
    def test_disabled_by_default(self, monkeypatch):
        # Default is opt-in: feature is off unless TORCH_NEURONX_MODULE_AUTO_COMPILE=1.
        monkeypatch.delenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", raising=False)
        assert not module_auto_compile_enabled()

    def test_enabled_by_env(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "1")
        assert module_auto_compile_enabled()

    def test_late_env_activation_requires_explicit_install(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "0")
        torch_neuronx._uninstall_module_auto_compile_patches()
        try:
            monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "1")
            assert not module_auto_compile_enabled()

            torch_neuronx._install_module_auto_compile_patches()
            assert module_auto_compile_enabled()
        finally:
            monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "1")
            torch_neuronx._install_module_auto_compile_patches()

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "0")
        assert not _env_flag("TORCH_NEURONX_MODULE_AUTO_COMPILE", default=True)

    def test_safe_fork_child_keeps_feature_enabled(self, monkeypatch):
        if "fork" not in multiprocessing.get_all_start_methods():
            pytest.skip("fork start method is unavailable")
        monkeypatch.setattr(torch_neuronx, "_initialized", False)
        monkeypatch.setattr(torch_neuronx._C, "_is_neuron_runtime_initialized", lambda: False)
        context = multiprocessing.get_context("fork")
        reader, writer = context.Pipe(duplex=False)

        def inspect_child():
            writer.send(
                (
                    mac.module_auto_compile_enabled(),
                    mac._FEATURE_HARD_DISABLED,
                    torch_neuronx._is_in_bad_fork,
                    len(mac._MODULE_REGISTRY),
                )
            )
            writer.close()

        process = context.Process(target=inspect_child)
        process.start()
        writer.close()
        result = reader.recv()
        process.join(timeout=10)

        assert not process.is_alive()
        assert process.exitcode == 0
        assert result == (True, False, False, 0)

    def test_child_reset_failure_disables_feature_and_clears_candidate(self, monkeypatch):
        if "fork" not in multiprocessing.get_all_start_methods():
            pytest.skip("fork start method is unavailable")
        monkeypatch.setattr(torch_neuronx, "_initialized", False)
        monkeypatch.setattr(torch_neuronx._C, "_is_neuron_runtime_initialized", lambda: False)

        def fail_reset(*, hard_disable):
            del hard_disable
            raise RuntimeError("injected child reset failure")

        monkeypatch.setattr(mac, "_after_fork", fail_reset)
        module = nn.ReLU()
        _bootstrap_record(module)
        context = multiprocessing.get_context("fork")
        reader, writer = context.Pipe(duplex=False)

        def inspect_child():
            patches_restored = all(
                getattr(owner, attribute) is original
                for owner, attribute, original, _replacement in integration._MODULE_PATCHES
            )
            writer.send(
                (
                    mac.module_auto_compile_enabled(),
                    mac._FEATURE_HARD_DISABLED,
                    len(mac._MODULE_REGISTRY),
                    module._compiled_call_impl is None,
                    integration._module_auto_compile_patches_active,
                    patches_restored,
                    not integration._coerce_tangent_patch_owned,
                )
            )
            writer.close()

        process = context.Process(target=inspect_child)
        process.start()
        writer.close()
        result = reader.recv()
        process.join(timeout=10)

        assert not process.is_alive()
        assert process.exitcode == 0
        assert result == (False, True, 0, True, False, True, True)


class TestModuleIsLeaf:
    def test_linear_is_leaf(self):
        assert _module_is_leaf(nn.Linear(10, 10))

    def test_layernorm_is_leaf(self):
        assert _module_is_leaf(nn.LayerNorm(10))

    def test_sequential_not_leaf(self):
        model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        assert not _module_is_leaf(model)

    def test_custom_module_with_children_not_leaf(self):
        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.child = nn.Linear(10, 10)

        assert not _module_is_leaf(Parent())

    def test_overridden_children_does_not_affect_leaf_detection(self):
        class HostileLeaf(nn.Module):
            def children(self):
                raise RuntimeError("children override must not run")

        assert _module_is_leaf(HostileLeaf())

    def test_custom_leaf_is_leaf(self):
        class MyNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(10))

            def forward(self, x):
                return x * self.weight

        assert _module_is_leaf(MyNorm())

    def test_checkpoint_wrapper_keeps_leaf_compile_boundary(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
        )

        body = nn.Sequential(nn.Linear(8, 8), nn.SiLU())
        region = checkpoint_wrapper(body)

        assert not mac._module_is_compile_region(region)
        assert not mac._module_is_compile_region(body)
        assert mac._module_is_compile_region(body[0])
        assert mac._module_is_compile_region(body[1])


class TestPromotionSupported:
    @pytest.mark.parametrize(
        "wrapper_type",
        (
            pytest.param(_ClassicFSDPStub, id="classic-fsdp"),
            pytest.param(_ComposableFSDPStub, id="composable-fsdp"),
        ),
    )
    def test_fsdp_ancestor_does_not_statically_block_leaf(self, wrapper_type):
        leaf = nn.LayerNorm(10)
        wrapper = wrapper_type()
        record, _ = _get_or_create_record(leaf)
        _classify_and_transition(leaf, record, (leaf, wrapper))

        assert record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING
        assert record.terminal_reason is None

    def test_transformer_descendants_stay_hook_free_to_preserve_fused_fastpath(self):
        layer = nn.TransformerEncoderLayer(
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            batch_first=True,
        )

        mac.observe_module_tree_after_to(
            layer,
            target_device=torch.device("neuron"),
        )

        for leaf in (layer.linear1, layer.linear2, layer.norm1, layer.norm2):
            record = mac._MODULE_REGISTRY[leaf]
            assert record.state == ModuleAutoCompileState.BLOCKED
            assert record.terminal_reason == mac._TERMINAL_REASON_HOOK_SENSITIVE_ANCESTOR
            assert not leaf._forward_pre_hooks
            assert not leaf._forward_hooks

    def test_dynamo_disabled_user_hook_blocks_by_capability(self):
        def hook_impl(_module, _args):
            return None

        def disabled_hook(*args, **kwargs):
            return torch._dynamo.disable(hook_impl, recursive=True)(*args, **kwargs)

        leaf = nn.ReLU()
        leaf.register_forward_pre_hook(disabled_hook)
        _observe_module_forward(leaf)

        record = mac._MODULE_REGISTRY[leaf]
        assert record.state == ModuleAutoCompileState.BLOCKED
        assert record.terminal_reason == "compiler_capability_unavailable"

    @pytest.mark.parametrize("hook_kind", ("pre", "post"))
    def test_backward_hook_blocks_by_capability(self, hook_kind):
        leaf = nn.ReLU()
        if hook_kind == "pre":
            leaf.register_full_backward_pre_hook(lambda _module, grad_output: grad_output)
        else:
            leaf.register_full_backward_hook(lambda _module, grad_input, _grad_output: grad_input)

        _observe_module_forward(leaf)

        record = mac._MODULE_REGISTRY[leaf]
        assert record.state == ModuleAutoCompileState.BLOCKED
        assert record.terminal_reason == "compiler_capability_unavailable"

    @pytest.mark.parametrize("use_orig_params", (None, False))
    def test_fsdp_managed_leaf_does_not_depend_on_original_parameter_mode(
        self,
        use_orig_params,
    ):
        leaf = nn.Linear(8, 8)
        leaf._is_fsdp_managed_module = True
        if use_orig_params is not None:
            leaf._fsdp_use_orig_params = use_orig_params

        _observe_module_forward(leaf)

        record = mac._MODULE_REGISTRY[leaf]
        assert record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING
        assert mac._requires_forward_only_compile(leaf)

    def test_fsdp_managed_leaf_with_original_params_is_supported(self):
        leaf = nn.Linear(8, 8)
        leaf._is_fsdp_managed_module = True
        leaf._fsdp_use_orig_params = True

        _observe_module_forward(leaf)

        assert mac._MODULE_REGISTRY[leaf].state == ModuleAutoCompileState.BOOTSTRAP_PENDING
        assert mac._requires_forward_only_compile(leaf)

    @pytest.mark.parametrize("protocol", ("post-accumulate-grad", "all-gather-extension"))
    @pytest.mark.parametrize("parameter_owner", ("module", "composable-state"))
    def test_fsdp_parameter_lifecycle_protocol_uses_safe_compile_boundary(
        self,
        monkeypatch,
        protocol,
        parameter_owner,
    ):
        class AllGatherParameter(torch.Tensor):
            @staticmethod
            def __new__(cls, value):
                return torch.Tensor._make_subclass(cls, value, value.requires_grad)

            def fsdp_pre_all_gather(self, *_args, **_kwargs):
                raise AssertionError("capability detection must not execute the protocol")

        class LocalTensorWrapper(torch.Tensor):
            @staticmethod
            def __new__(cls, value, local_tensor):
                tensor = torch.Tensor._make_subclass(cls, value, value.requires_grad)
                tensor._local_tensor = local_tensor
                return tensor

        leaf = nn.Linear(8, 8)
        if parameter_owner == "module":
            leaf._is_fsdp_managed_module = True
            leaf._fsdp_use_orig_params = True
        if protocol == "post-accumulate-grad":
            protocol_parameter = nn.Parameter(leaf.weight.detach())
            protocol_parameter.register_post_accumulate_grad_hook(lambda _parameter: None)
        else:
            inner = AllGatherParameter(leaf.weight.detach())
            protocol_parameter = LocalTensorWrapper(leaf.weight.detach(), inner)

        if parameter_owner == "module":
            leaf._parameters["weight"] = protocol_parameter
        else:
            owner = nn.Module()
            fsdp_parameter = types.SimpleNamespace(
                _module_info=types.SimpleNamespace(
                    module=leaf,
                    shared_modules=(),
                ),
                sharded_param=protocol_parameter,
                _sharded_post_forward_param=None,
            )
            state = types.SimpleNamespace(
                _fsdp_param_groups=(types.SimpleNamespace(fsdp_params=(fsdp_parameter,)),)
            )
            monkeypatch.setattr(mac, "_recover_active_module_stack", lambda: (leaf, owner))
            monkeypatch.setattr(
                mac,
                "_get_composable_fsdp_state",
                lambda module: state if module is owner else None,
            )

        _observe_module_forward(leaf)

        record = mac._MODULE_REGISTRY[leaf]
        assert record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING
        assert mac._requires_forward_only_compile(leaf)

    def test_dtensor_parameter_only_blocks_embedding(self, monkeypatch):
        monkeypatch.setattr(torch.distributed.tensor, "DTensor", nn.Parameter)

        assert not mac._compiler_can_capture_module(nn.Embedding(16, 8))
        assert mac._compiler_can_capture_module(nn.Linear(8, 8))

    def test_cumulative_batch_norm_training_blocks_by_capability(self):
        leaf = nn.BatchNorm1d(8, momentum=None, track_running_stats=True)
        leaf.train()

        _observe_module_forward(leaf)

        record = mac._MODULE_REGISTRY[leaf]
        assert record.state == ModuleAutoCompileState.BLOCKED
        assert record.terminal_reason == "compiler_capability_unavailable"

    @pytest.mark.parametrize(
        ("kwargs", "training"),
        (
            pytest.param({}, True, id="default-momentum"),
            pytest.param({"momentum": None}, False, id="eval"),
            pytest.param(
                {"momentum": None, "track_running_stats": False},
                True,
                id="no-running-stats",
            ),
        ),
    )
    def test_other_batch_norm_modes_are_supported(self, kwargs, training):
        leaf = nn.BatchNorm1d(8, **kwargs)
        leaf.train(training)

        _observe_module_forward(leaf)

        assert mac._MODULE_REGISTRY[leaf].state == ModuleAutoCompileState.BOOTSTRAP_PENDING


class TestRecordCreation:
    def test_get_or_create_new(self):
        mod = nn.LayerNorm(10)
        record, is_new = _get_or_create_record(mod)
        assert is_new
        assert record.class_name == "LayerNorm"
        assert record.state == ModuleAutoCompileState.COLD

    def test_get_existing(self):
        mod = nn.LayerNorm(10)
        record1, is_new1 = _get_or_create_record(mod)
        record2, is_new2 = _get_or_create_record(mod)
        assert is_new1
        assert not is_new2
        assert record1 is record2

    def test_equal_modules_are_registered_by_identity(self):
        class EqualLeaf(nn.Module):
            def __eq__(self, other):
                return isinstance(other, EqualLeaf)

            def __hash__(self):
                return 1

        first = EqualLeaf()
        second = EqualLeaf()
        first_record, _ = _get_or_create_record(first)
        second_record, _ = _get_or_create_record(second)

        assert first_record is not second_record
        assert mac._MODULE_REGISTRY[first] is first_record
        assert mac._MODULE_REGISTRY[second] is second_record

    def test_concurrent_get_or_create_returns_one_record(self, monkeypatch):
        first_get_entered = threading.Event()
        second_get_entered = threading.Event()
        second_thread_started = threading.Event()
        release_first_get = threading.Event()

        class BlockingRegistry(weakref.WeakKeyDictionary):
            def __init__(self):
                super().__init__()
                self.get_calls = 0
                self.get_calls_lock = threading.Lock()

            def get(self, key, default=None):
                with self.get_calls_lock:
                    self.get_calls += 1
                    call_number = self.get_calls
                if call_number == 1:
                    first_get_entered.set()
                    if not release_first_get.wait(timeout=5):
                        raise TimeoutError("first registry read was not released")
                else:
                    second_get_entered.set()
                return super().get(key, default)

        registry = BlockingRegistry()
        monkeypatch.setattr(mac, "_MODULE_REGISTRY", registry)

        mod = nn.LayerNorm(10)
        results = []
        errors = []

        def get_record(started=None):
            if started is not None:
                started.set()
            try:
                results.append(_get_or_create_record(mod))
            except Exception as exc:
                errors.append(exc)

        first_thread = threading.Thread(target=get_record)
        second_thread = threading.Thread(target=get_record, args=(second_thread_started,))
        first_thread.start()
        second_was_started = False
        try:
            assert first_get_entered.wait(timeout=5)
            second_thread.start()
            second_was_started = True
            assert second_thread_started.wait(timeout=5)
            assert not second_get_entered.wait(timeout=0.2)
        finally:
            release_first_get.set()
            first_thread.join(timeout=10)
            if second_was_started:
                second_thread.join(timeout=10)

        assert not first_thread.is_alive()
        if second_was_started:
            assert not second_thread.is_alive()
        assert errors == []
        assert len(results) == 2
        assert results[0][0] is results[1][0]
        assert sum(is_new for _, is_new in results) == 1
        assert second_get_entered.is_set()

    def test_user_owned_module_is_garbage_collectable(self):
        """A terminal-state module must not be pinned by its registry record
        (regression: the removed expected_compiled_callable field held a bound
        method, defeating the WeakKeyDictionary)."""
        mod = nn.LayerNorm(10)
        record, _ = _get_or_create_record(mod)
        mod._compiled_call_impl = mod._call_impl  # simulate a user compile
        _mark_user_owned(mod, record, reason="test")
        assert record.state == ModuleAutoCompileState.USER_OWNED

        ref = weakref.ref(mod)
        del mod
        gc.collect()
        assert (
            ref() is None
        ), "user-owned module was not collected — its record still strong-refs it"


class TestBootstrapHooks:
    def test_hooks_installed(self):
        mod = nn.LayerNorm(10)
        record = _bootstrap_record(mod)
        assert record.bootstrap_forward_pre_handle is not None
        assert record.bootstrap_compiled_call_installed
        assert mac._bootstrap_compiled_call_is_owned(mod)
        assert "_call_impl" not in mod.__dict__
        assert not mod._forward_hooks

    def test_hooks_removed(self):
        mod = nn.LayerNorm(10)
        record = _bootstrap_record(mod)
        _remove_bootstrap_hooks(mod, record)
        assert record.bootstrap_forward_pre_handle is None
        assert not record.bootstrap_compiled_call_installed
        assert mod._compiled_call_impl is None
        assert "_call_impl" not in mod.__dict__

    def test_success_inside_outer_except_is_counted_as_success(self, monkeypatch):
        monkeypatch.setattr(
            "torch_neuronx.module_auto_compile._tensor_is_neuron",
            lambda v: isinstance(v, torch.Tensor),
        )
        mod = nn.LayerNorm(10)
        record = _bootstrap_record(mod)

        try:
            raise RuntimeError("outer")
        except RuntimeError:
            mod(torch.randn(2, 10))

        assert record.successful_invocation_count == 1

    def test_failed_forward_is_not_counted_as_success(self, monkeypatch):
        class FailingLeaf(nn.Module):
            def forward(self, x):
                _ = x
                raise RuntimeError("forward failed")

        monkeypatch.setattr(
            "torch_neuronx.module_auto_compile._tensor_is_neuron",
            lambda v: isinstance(v, torch.Tensor),
        )
        mod = FailingLeaf()
        record = _bootstrap_record(mod)

        with pytest.raises(RuntimeError, match="forward failed"):
            mod(torch.randn(2, 10))

        assert record.successful_invocation_count == 0

    def test_keyboard_interrupt_cleanup_allows_later_promotion(self, monkeypatch):
        class InterruptOnce(nn.Module):
            def __init__(self):
                super().__init__()
                self.interrupt = True

            def forward(self, value):
                if self.interrupt:
                    self.interrupt = False
                    raise KeyboardInterrupt
                return value

        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda value: isinstance(value, torch.Tensor))
        monkeypatch.setattr(mac, "_compile_module_call_impl", _reported_module_call)
        module = InterruptOnce()
        record = _bootstrap_record(module)
        record.successful_invocation_count = _PROMOTION_INVOCATION_THRESHOLD - 1

        with pytest.raises(KeyboardInterrupt):
            module(torch.ones(1))

        assert record.active_bootstrap_invocations == 0
        assert mac._latest_bootstrap_invocation(module) is None
        module(torch.ones(1))
        assert record.state == ModuleAutoCompileState.PROMOTION_READY

        module(torch.ones(1))
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL

        module(torch.ones(1))
        assert record.state == ModuleAutoCompileState.COMPILED

    def test_keyboard_interrupt_during_compile_setup_restores_promotion(self, monkeypatch):
        module = nn.ReLU()
        record = _ready_record(module)
        attempts = 0
        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *_args: True)

        def interrupt_once(target):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise KeyboardInterrupt
            return _reported_module_call(target)

        monkeypatch.setattr(mac, "_compile_module_call_impl", interrupt_once)

        with pytest.raises(KeyboardInterrupt):
            _promote_module(module, record)

        assert record.state == ModuleAutoCompileState.PROMOTION_READY
        assert record.pending_promotion
        assert record.bootstrap_compiled_call_installed
        assert mac._bootstrap_compiled_call_is_owned(module)

        _promote_module(module, record)
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
        module(torch.ones(1))
        assert record.state == ModuleAutoCompileState.COMPILED

    def test_recursive_forward_counts_only_top_level_invocation(self, monkeypatch):
        class RecursiveLeaf(nn.Module):
            def forward(self, x, depth):
                if depth:
                    return self(x, depth - 1)
                return x

        monkeypatch.setattr(
            "torch_neuronx.module_auto_compile._tensor_is_neuron",
            lambda v: isinstance(v, torch.Tensor),
        )
        mod = RecursiveLeaf()
        record = _bootstrap_record(mod)

        x = torch.randn(2, 10)
        assert mod(x, 1) is x

        assert record.successful_invocation_count == 1

    def test_ineligible_recursive_call_does_not_consume_outer_invocation(self, monkeypatch):
        class RecursiveLeaf(nn.Module):
            def forward(self, value, *, inner=False):
                if inner:
                    return value
                self(torch.empty(0), inner=True)
                raise RuntimeError("outer failed")

        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda value: isinstance(value, torch.Tensor))
        module = RecursiveLeaf()
        record = _bootstrap_record(module)

        with pytest.raises(RuntimeError, match="outer failed"):
            module(torch.ones(1))

        assert record.active_bootstrap_invocations == 0
        assert record.successful_invocation_count == 0
        assert record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING
        assert record.pending_promotion is False

    def test_concurrent_forwards_are_counted_per_thread(self, monkeypatch):
        barrier = threading.Barrier(2)

        class ConcurrentLeaf(nn.Module):
            def forward(self, x):
                barrier.wait(timeout=5)
                return x

        monkeypatch.setattr(
            "torch_neuronx.module_auto_compile._tensor_is_neuron",
            lambda v: isinstance(v, torch.Tensor),
        )
        mod = ConcurrentLeaf()
        record = _bootstrap_record(mod)
        errors = []

        def run_forward():
            try:
                mod(torch.randn(2, 10))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run_forward) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert all(not thread.is_alive() for thread in threads)
        assert errors == []
        assert record.successful_invocation_count == 2
        assert record.active_bootstrap_invocations == 0

    def test_cross_thread_explicit_compile_does_not_pin_active_module(self, monkeypatch):
        entered = threading.Event()
        release_forward = threading.Event()
        forward_complete = threading.Event()
        keep_worker_alive = threading.Event()
        modules = queue.Queue()
        errors = []

        class BlockingLeaf(nn.Module):
            def forward(self, value):
                entered.set()
                assert release_forward.wait(timeout=5)
                return value

        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda value: isinstance(value, torch.Tensor))
        module = BlockingLeaf()
        _bootstrap_record(module)
        modules.put(module)

        def run_forward():
            target = modules.get()
            try:
                target(torch.ones(1))
            except Exception as error:
                errors.append(error)
            finally:
                del target
                forward_complete.set()
                keep_worker_alive.wait(timeout=5)

        worker = threading.Thread(target=run_forward)
        worker.start()
        try:
            assert entered.wait(timeout=5)
            mac.record_explicit_module_compile(module)
            release_forward.set()
            assert forward_complete.wait(timeout=5)

            module_ref = weakref.ref(module)
            del module
            gc.collect()
            assert module_ref() is None
        finally:
            release_forward.set()
            keep_worker_alive.set()
            worker.join(timeout=10)

        assert not worker.is_alive()
        assert errors == []

    def test_promotion_installed_at_success_threshold(self, monkeypatch):
        monkeypatch.setattr(
            "torch_neuronx.module_auto_compile._tensor_is_neuron",
            lambda v: isinstance(v, torch.Tensor),
        )
        mod = nn.LayerNorm(10)
        record = _bootstrap_record(mod)

        x = torch.randn(2, 10)
        with torch.no_grad():
            for _ in range(_PROMOTION_INVOCATION_THRESHOLD - 1):
                mod(x)
            assert record.pending_promotion is False
            mod(x)
        assert record.successful_invocation_count == _PROMOTION_INVOCATION_THRESHOLD
        assert record.pending_promotion is True
        assert record.state == ModuleAutoCompileState.PROMOTION_READY

        with torch.no_grad():
            mod(x)
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL

    def test_failed_calls_do_not_advance_promotion(self, monkeypatch):
        class RaiseAfterWork(nn.Module):
            def __init__(self):
                super().__init__()
                self.should_raise = True

            def forward(self, value):
                value = value.relu()
                if self.should_raise:
                    raise RuntimeError("after work")
                return value

        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda value: isinstance(value, torch.Tensor))
        monkeypatch.setattr(mac, "_compile_module_call_impl", _reported_module_call)
        module = RaiseAfterWork()
        record = _bootstrap_record(module)
        value = torch.randn(2)

        with pytest.raises(RuntimeError, match="after work"):
            module(value)
        assert record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING
        assert record.pending_promotion is False

        with pytest.raises(RuntimeError, match="after work"):
            module(value)
        assert record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING

        module.should_raise = False
        for _ in range(_PROMOTION_INVOCATION_THRESHOLD):
            module(value)
        assert record.state == ModuleAutoCompileState.PROMOTION_READY

        module(value)
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL

        module(value)
        assert record.state == ModuleAutoCompileState.COMPILED

    def test_ready_module_waits_for_all_bootstrap_invocations(self, monkeypatch):
        first_entered = threading.Event()
        release_first = threading.Event()
        calls = 0
        calls_lock = threading.Lock()

        class ConcurrentLeaf(nn.Module):
            def forward(self, value):
                nonlocal calls
                with calls_lock:
                    calls += 1
                    call_number = calls
                if call_number == 1:
                    first_entered.set()
                    assert release_first.wait(timeout=5)
                return value

        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda value: isinstance(value, torch.Tensor))
        monkeypatch.setattr(mac, "_compile_module_call_impl", _reported_module_call)
        module = ConcurrentLeaf()
        record = _bootstrap_record(module)
        record.successful_invocation_count = _PROMOTION_INVOCATION_THRESHOLD - 1

        errors = []

        def run_blocked():
            try:
                module(torch.randn(2))
            except Exception as error:
                errors.append(error)

        blocked = threading.Thread(target=run_blocked)
        blocked.start()
        assert first_entered.wait(timeout=5)
        assert record.active_bootstrap_invocations == 1

        module(torch.randn(2))
        assert record.state == ModuleAutoCompileState.PROMOTION_READY
        assert record.active_bootstrap_invocations == 1

        module(torch.randn(2))
        assert mac._bootstrap_compiled_call_is_owned(module)
        assert record.active_bootstrap_invocations == 1

        release_first.set()
        blocked.join(timeout=5)
        assert not blocked.is_alive()
        assert errors == []
        assert record.active_bootstrap_invocations == 0
        assert record.state == ModuleAutoCompileState.PROMOTION_READY

        module(torch.randn(2))
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL

        module(torch.randn(2))
        assert record.state == ModuleAutoCompileState.COMPILED

    def test_module_with_hooks_is_picklable(self):
        """Modules carrying bootstrap hooks must stay picklable — regression for
        nested-closure hooks that broke pickle/deepcopy/state_dict (test_save_load)."""
        mod = nn.LayerNorm(10)
        _bootstrap_record(mod)

        # These all pickle the module (and thus its hook dicts) internally.
        pickle.loads(pickle.dumps(mod))
        clone = copy.deepcopy(mod)
        _ = mod.state_dict()

        # The deepcopied module is not in the registry, so its hooks resolve
        # record=None and are inert (no crash on forward).
        clone(torch.randn(2, 10))

    @pytest.mark.parametrize("create_graph", (False, True))
    def test_eager_higher_order_autograd_terminalizes_candidate(self, monkeypatch, create_graph):
        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda value: isinstance(value, torch.Tensor))
        mod = nn.Linear(4, 4)
        record = _bootstrap_record(mod)
        value = torch.randn(2, 4, requires_grad=True)

        output = mod(value)
        torch.autograd.grad(output.sum(), value, create_graph=create_graph)

        if create_graph:
            assert record.state == ModuleAutoCompileState.BLOCKED
            assert record.terminal_reason == mac._TERMINAL_REASON_HIGHER_ORDER_AUTOGRAD
            assert mod._compiled_call_impl is None
        else:
            assert record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING
            assert record.pending_backward_count == 0
            assert mac._bootstrap_compiled_call_is_owned(mod)

    def test_compiled_higher_order_autograd_uses_compatibility_backward(
        self,
        monkeypatch,
    ):
        class SinLeaf(nn.Module):
            def forward(self, value):
                return value.sin()

        from torch_neuronx.neuron_dynamo_backend.backend import _AutoCompileBackendAttempt

        attempt = _AutoCompileBackendAttempt()

        original_compile = torch.compile

        def compile_with_aot_eager(target, **_kwargs):
            compiled = original_compile(
                target,
                backend="aot_eager",
                fullgraph=True,
                dynamic=False,
            )

            def reported(value):
                if not attempt.has_real_forward_executor():
                    attempt.record_executor("forward", 3, "forward-key")
                output = compiled(value)

                def report_backward(gradient):
                    if torch.is_grad_enabled():
                        attempt.record_compatibility("backward", "higher_order_autograd")
                    elif not attempt.has_real_backward_executor():
                        attempt.record_executor("backward", 2, "backward-key")
                    return gradient

                output.register_hook(report_backward)
                return output

            setattr(reported, mac._BACKEND_ATTEMPT_ATTR, attempt)
            return reported

        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda value: isinstance(value, torch.Tensor))
        monkeypatch.setattr(mac, "_compile_isolated_callable", compile_with_aot_eager)
        module = SinLeaf()
        record = _ready_record(module)
        _promote_module(module, record)

        first_value = torch.randn(2, 4, requires_grad=True)
        first_output = module(first_value)
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
        assert record.pending_backward_count == 1
        torch.autograd.grad(first_output.sum(), first_value)
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
        assert record.pending_backward_count == 0

        value = torch.randn(2, 4, requires_grad=True)
        output = module(value)
        assert record.state == ModuleAutoCompileState.COMPILED
        gradient = torch.autograd.grad(output.sum(), value, create_graph=True)[0]
        second_gradient = torch.autograd.grad(gradient.sum(), value)[0]

        assert record.state == ModuleAutoCompileState.COMPILED
        assert record.terminal_reason is None
        assert module._compiled_call_impl is not None
        assert record.compile_attempt.compatibility_reasons == [
            ("backward", "higher_order_autograd")
        ]
        torch.testing.assert_close(second_gradient, -value.sin())

        next_value = torch.randn(2, 4, requires_grad=True)
        next_gradient = torch.autograd.grad(
            module(next_value).sum(),
            next_value,
            create_graph=True,
        )[0]
        torch.testing.assert_close(
            torch.autograd.grad(next_gradient.sum(), next_value)[0],
            -next_value.sin(),
        )

    def test_retained_graph_restores_eager_and_allows_second_backward(self, monkeypatch):
        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda value: isinstance(value, torch.Tensor))

        def fail_keep_graph_probe():
            raise RuntimeError("injected binary skew")

        monkeypatch.setattr(
            torch_neuronx._C,
            "_current_graph_task_keep_graph",
            fail_keep_graph_probe,
            raising=False,
        )
        module = nn.Linear(4, 4)
        record = _bootstrap_record(module)
        value = torch.randn(2, 4, requires_grad=True)
        loss = module(value).square().sum()

        first_gradient = torch.autograd.grad(loss, value, retain_graph=True)[0]
        assert record.state == ModuleAutoCompileState.BLOCKED
        assert record.terminal_reason == mac._TERMINAL_REASON_HIGHER_ORDER_AUTOGRAD
        assert module._compiled_call_impl is None

        second_gradient = torch.autograd.grad(loss, value)[0]
        torch.testing.assert_close(second_gradient, first_gradient)

    def test_copied_hooks_are_deduplicated_when_clone_is_discovered(self):
        mod = nn.LayerNorm(10)
        _bootstrap_record(mod)
        clone = copy.deepcopy(mod)

        mac.observe_module_tree_after_to(
            clone,
            target_device=torch.device("neuron"),
        )

        clone_record = mac._MODULE_REGISTRY[clone]
        assert len(clone._forward_pre_hooks) == 1
        assert len(clone._forward_hooks) == 0
        assert clone_record.bootstrap_compiled_call_installed
        assert mac._bootstrap_compiled_call_is_owned(clone)
        assert "_call_impl" not in clone.__dict__
        assert (
            clone._forward_pre_hooks[clone_record.bootstrap_forward_pre_handle.id]
            is mac._bootstrap_forward_pre_hook
        )

    def test_shallow_copy_cleanup_preserves_source_bootstrap(self, monkeypatch):
        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda value: isinstance(value, torch.Tensor))
        source = nn.LayerNorm(10)
        source_record = _bootstrap_record(source)
        copied = copy.copy(source)
        assert copied._forward_pre_hooks is source._forward_pre_hooks

        copied(torch.randn(2, 10))

        assert not copied._forward_pre_hooks
        assert copied._compiled_call_impl is None
        assert source._forward_pre_hooks
        assert mac._bootstrap_compiled_call_is_owned(source)
        source_record.successful_invocation_count = _PROMOTION_INVOCATION_THRESHOLD - 1
        with torch.no_grad():
            source(torch.randn(2, 10))
        assert source_record.successful_invocation_count == _PROMOTION_INVOCATION_THRESHOLD
        assert source_record.state == ModuleAutoCompileState.PROMOTION_READY
        assert source_record.pending_promotion

    def test_blocking_copy_scrubs_inherited_bootstrap_state(self):
        source = nn.LayerNorm(10)
        _bootstrap_record(source)
        copied = copy.deepcopy(source)
        record, _ = _get_or_create_record(copied)

        mac._mark_blocked(
            copied,
            record,
            reason="test",
            clear_compiled_call_impl=False,
        )

        assert not copied._forward_pre_hooks
        assert copied._compiled_call_impl is None
        assert "_call_impl" not in copied.__dict__

    def test_hook_inert_for_unregistered_module(self):
        """Hooks resolve the record via the registry, not a captured value.

        A module whose record was evicted (or that was never registered, e.g.
        an unpickled copy) must run its forward without error and without
        touching any record.
        """
        mod = nn.LayerNorm(10)
        record = _bootstrap_record(mod)
        del mac._MODULE_REGISTRY[mod]

        # No registry entry -> hooks are inert, forward still works.
        mod(torch.randn(2, 10))
        assert record.successful_invocation_count == 0
        assert not mod._forward_pre_hooks
        assert mod._compiled_call_impl is None
        assert "_call_impl" not in mod.__dict__

    def test_forward_registered_hook_failure_is_not_counted_as_success(self, monkeypatch):
        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda v: isinstance(v, torch.Tensor))

        class RegisteringLeaf(nn.Module):
            def __init__(self):
                super().__init__()
                self.registered = False

            def forward(self, value):
                if not self.registered:
                    self.registered = True
                    self.register_forward_hook(fail_after_forward)
                return value

        def fail_after_forward(_module, _args, _output):
            raise RuntimeError("later hook failed")

        mod = RegisteringLeaf()
        record = _bootstrap_record(mod)

        with pytest.raises(RuntimeError, match="later hook failed"):
            mod(torch.randn(2, 10))

        assert record.successful_invocation_count == 0
        assert record.pending_promotion is False

    def test_later_user_pre_hook_can_make_call_eligible(self, monkeypatch):
        mod = nn.Identity()
        record = _bootstrap_record(mod)
        auto_hook_id = record.bootstrap_forward_pre_handle.id

        def make_eligible(_module, args):
            return (args[0] + 1,)

        user_handle = mod.register_forward_pre_hook(make_eligible)
        monkeypatch.setattr(
            mac,
            "_call_contains_neuron_tensor",
            lambda args, _kwargs: bool(args[0].item()),
        )

        mod(torch.tensor(0))
        assert list(mod._forward_pre_hooks) == [user_handle.id, auto_hook_id]
        assert record.successful_invocation_count == 0

        record.successful_invocation_count = _PROMOTION_INVOCATION_THRESHOLD - 1
        mod(torch.tensor(0))
        assert record.successful_invocation_count == _PROMOTION_INVOCATION_THRESHOLD
        assert record.state == ModuleAutoCompileState.PROMOTION_READY
        assert record.pending_promotion

    def test_module_that_gains_children_before_first_forward_is_blocked(
        self,
        monkeypatch,
    ):
        class InitiallyEmptyModule(nn.Module):
            def forward(self, x):
                return self.child(x)

        monkeypatch.setattr(
            "torch_neuronx.module_auto_compile._tensor_is_neuron",
            lambda value: isinstance(value, torch.Tensor),
        )
        mod = InitiallyEmptyModule()
        _observe_module_forward(mod)
        record = mac._MODULE_REGISTRY[mod]
        assert record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING

        mod.child = nn.ReLU()
        output = mod(torch.tensor(-1.0))

        torch.testing.assert_close(output, torch.tensor(0.0))
        assert record.state == ModuleAutoCompileState.BLOCKED
        assert record.terminal_reason == _TERMINAL_REASON_COVERAGE_RESTRICTED
        assert record.successful_invocation_count == 0
        assert mod._compiled_call_impl is None

    def test_bootstrap_hooks_allow_jit_trace_and_save(self):
        mod = nn.LayerNorm(10)
        _bootstrap_record(mod)

        traced = torch.jit.trace(mod, (torch.randn(2, 10),))
        torch.jit.save(traced, io.BytesIO())

    def test_bootstrap_hooks_tolerate_lost_with_kwargs_metadata(self, monkeypatch):
        """Composable wrappers can copy hooks without their kwargs metadata."""

        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *args: True)
        mod = nn.LayerNorm(10)
        record = _bootstrap_record(mod)

        mod._forward_pre_hooks_with_kwargs.pop(record.bootstrap_forward_pre_handle.id, None)

        mod(torch.randn(2, 10))

        assert record.successful_invocation_count == 1

    @pytest.mark.parametrize("reason", ("empty", "functorch"))
    def test_ineligible_call_defers_pending_promotion(self, monkeypatch, reason):
        class Leaf(nn.Module):
            def forward(self, payload):
                return payload["batch"][0]

        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda v: isinstance(v, torch.Tensor))
        if reason == "functorch":
            monkeypatch.setattr(mac, "_functorch_transform_active", lambda: True)

        leaf = Leaf()
        record = _ready_record(leaf)
        _install_bootstrap_hooks(leaf, record)
        payload = {"batch": [torch.empty(0) if reason == "empty" else torch.randn(1)]}

        leaf(payload)

        assert record.pending_promotion is True
        assert record.state == ModuleAutoCompileState.PROMOTION_READY
        assert record.successful_invocation_count == 0

    def test_bootstrap_budget_exhaustion_restores_eager_without_error(self, monkeypatch):
        monkeypatch.setattr(
            mac,
            "_tensor_is_neuron",
            lambda value: isinstance(value, torch.Tensor),
        )
        module = nn.ReLU()
        record = _bootstrap_record(module)
        record.unresolved_output_budget = 1
        value = torch.randn(2, 3)

        torch.testing.assert_close(module(value), value.relu())
        assert record.state == ModuleAutoCompileState.BLOCKED
        assert record.terminal_reason == mac._TERMINAL_REASON_OBSERVATION_BUDGET
        assert module._compiled_call_impl is None


class TestPromotionBoundary:
    def test_promotion_installed_after_successful_threshold_call(self, monkeypatch):
        """Install only after the qualifying call and all of its hooks return."""

        monkeypatch.setattr(
            mac,
            "_tensor_is_neuron",
            lambda value: isinstance(value, torch.Tensor),
        )
        monkeypatch.setattr(
            mac,
            "_compile_module_call_impl",
            _reported_module_call,
        )
        mod = nn.LayerNorm(10)
        record = _bootstrap_record(mod)

        x = torch.randn(2, 10)
        with torch.no_grad():
            for _ in range(_PROMOTION_INVOCATION_THRESHOLD):
                mod(x)
        assert record.pending_promotion is True
        assert record.state == ModuleAutoCompileState.PROMOTION_READY

        with torch.no_grad():
            mod(x)
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
        assert record.pending_promotion is False

        with torch.no_grad():
            mod(x)
        assert record.state == ModuleAutoCompileState.COMPILED
        assert record.pending_promotion is False
        assert not mac._bootstrap_compiled_call_is_owned(mod)
        assert mod._compiled_call_impl is not None

    def test_training_promotion_waits_for_pending_backward(self, monkeypatch):
        compiled_calls = []

        def compile_call(module):
            def compiled(*args, **kwargs):
                compiled_calls.append(True)
                output = module.forward(*args, **kwargs)

                def report_backward(gradient):
                    if not attempt.has_real_backward_executor():
                        attempt.record_executor("backward", 2, "backward-key")
                    return gradient

                output.register_hook(report_backward)
                return output

            reported, attempt_value = _reported_callable(compiled)
            nonlocal attempt
            attempt = attempt_value
            return reported

        attempt = None
        monkeypatch.setattr(
            mac,
            "_tensor_is_neuron",
            lambda value: isinstance(value, torch.Tensor),
        )
        monkeypatch.setattr(mac, "_compile_module_call_impl", compile_call)
        mod = nn.ReLU()
        record = _bootstrap_record(mod)
        record.successful_invocation_count = _PROMOTION_INVOCATION_THRESHOLD - 2
        value = torch.randn(2, 10, requires_grad=True)

        first = mod(value)
        second = mod(value)

        assert record.state == ModuleAutoCompileState.PROMOTION_READY
        assert record.pending_backward_count == 2
        assert compiled_calls == []

        (first + second).sum().backward()
        assert record.pending_backward_count == 0

        transition_output = mod(torch.randn(2, 10, requires_grad=True))
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
        assert record.pending_backward_count == 1
        assert compiled_calls == []

        transition_output.sum().backward()
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
        assert record.pending_backward_count == 0

        probation_output = mod(torch.randn(2, 10, requires_grad=True))
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
        assert record.pending_backward_count == 1
        assert compiled_calls == [True]

        probation_output.sum().backward()
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
        assert record.pending_backward_count == 0

        mod(torch.randn(2, 10, requires_grad=True))
        assert record.state == ModuleAutoCompileState.COMPILED
        assert compiled_calls == [True, True]

    def test_compiled_training_waits_for_complete_backward(self, monkeypatch):
        from torch_neuronx.neuron_dynamo_backend.backend import _AutoCompileBackendAttempt

        observed_states = []

        class BackwardProbe(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.clone()

            @staticmethod
            def backward(ctx, gradient):
                observed_states.append((record.state, record.pending_backward_count))
                return gradient

        class Leaf(nn.Module):
            def forward(self, value):
                return BackwardProbe.apply(value)

        attempt = _AutoCompileBackendAttempt()
        attempt.record_executor("forward", 3, "forward-key")
        leaf = Leaf()
        record = _ready_record(leaf)

        def compiled(value):
            return leaf.forward(value)

        setattr(compiled, mac._BACKEND_ATTEMPT_ATTR, attempt)
        monkeypatch.setattr(mac, "_compile_module_call_impl", lambda _module: compiled)
        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *_args: True)

        _promote_module(leaf, record)
        output = leaf(torch.ones(1, requires_grad=True))

        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
        assert record.pending_backward_count == 1

        attempt.record_executor("backward", 2, "backward-key")
        output.sum().backward()

        assert observed_states == [(ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL, 1)]
        assert record.pending_backward_count == 0
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL

        leaf(torch.ones(1, requires_grad=True))
        assert record.state == ModuleAutoCompileState.COMPILED

    def test_overlapping_compiled_forwards_keep_one_execution_mode(self, monkeypatch):
        from torch_neuronx.neuron_dynamo_backend.backend import _AutoCompileBackendAttempt

        calls = []
        attempt = _AutoCompileBackendAttempt()
        attempt.record_executor("forward", 3, "forward-key")

        class Leaf(nn.Module):
            def forward(self, value):
                calls.append("eager")
                return value + 1

        leaf = Leaf()
        record = _ready_record(leaf)

        def compiled(value):
            calls.append("compiled")
            output = value + 1

            def report_backward(gradient):
                if not attempt.has_real_backward_executor():
                    attempt.record_executor("backward", 2, "backward-key")
                return gradient

            output.register_hook(report_backward)
            return output

        setattr(compiled, mac._BACKEND_ATTEMPT_ATTR, attempt)
        monkeypatch.setattr(mac, "_compile_module_call_impl", lambda _module: compiled)
        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *_args: True)

        _promote_module(leaf, record)
        first = leaf(torch.ones(1, requires_grad=True))

        assert calls == ["compiled"]
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
        assert record.pending_backward_count == 1

        second = leaf(torch.ones(1, requires_grad=True))

        assert calls == ["compiled", "compiled"]
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
        assert record.pending_backward_count == 2
        torch.testing.assert_close(first, second)

        (first + second).sum().backward()
        assert record.pending_backward_count == 0
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL

        leaf(torch.ones(1, requires_grad=True))
        assert record.state == ModuleAutoCompileState.COMPILED

    def test_separate_output_backwards_hold_training_probation(self, monkeypatch):
        from torch_neuronx.neuron_dynamo_backend.backend import _AutoCompileBackendAttempt

        attempt = _AutoCompileBackendAttempt()
        attempt.record_executor("forward", 4, "forward-key")

        class Leaf(nn.Module):
            def forward(self, value):
                return value * 2, value * 3

        leaf = Leaf()
        record = _ready_record(leaf)

        def compiled(value):
            outputs = leaf.forward(value)

            def report_backward(gradient):
                if not attempt.has_real_backward_executor():
                    attempt.record_executor("backward", 3, "backward-key")
                return gradient

            for output in outputs:
                output.register_hook(report_backward)
            return outputs

        setattr(compiled, mac._BACKEND_ATTEMPT_ATTR, attempt)
        monkeypatch.setattr(mac, "_compile_module_call_impl", lambda _module: compiled)
        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *_args: True)

        _promote_module(leaf, record)
        first, second = leaf(torch.ones(1, requires_grad=True))
        assert record.pending_backward_count == 2

        first.sum().backward()
        assert record.pending_backward_count == 1
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL

        second.sum().backward()
        assert record.pending_backward_count == 0
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL

        leaf(torch.ones(1, requires_grad=True))
        assert record.state == ModuleAutoCompileState.COMPILED

    def test_dropped_outputs_exhaust_probation_budget_and_restore_eager(self, monkeypatch):
        from torch_neuronx.neuron_dynamo_backend.backend import _AutoCompileBackendAttempt

        calls = []
        attempt = _AutoCompileBackendAttempt()
        attempt.record_executor("forward", 3, "forward-key")

        class Leaf(nn.Module):
            def forward(self, value):
                calls.append("eager")
                return value + 1

        leaf = Leaf()
        record = _ready_record(leaf)

        def compiled(value):
            calls.append("compiled")
            return value + 1

        setattr(compiled, mac._BACKEND_ATTEMPT_ATTR, attempt)
        monkeypatch.setattr(mac, "_compile_module_call_impl", lambda _module: compiled)
        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *_args: True)

        _promote_module(leaf, record)
        leaf(torch.ones(1, requires_grad=True))
        for _ in range(_OBSERVATION_CALL_BUDGET):
            leaf(torch.ones(1, requires_grad=True))

        assert record.state == ModuleAutoCompileState.BLOCKED
        assert record.terminal_reason == mac._TERMINAL_REASON_OBSERVATION_BUDGET
        assert leaf._compiled_call_impl is None
        compiled_call_count = calls.count("compiled")

        leaf(torch.ones(1, requires_grad=True))
        assert calls[-1] == "eager"
        assert calls.count("compiled") == compiled_call_count

    def test_backend_fallback_restores_terminal_eager_path(self, monkeypatch):
        from torch_neuronx.neuron_dynamo_backend.backend import _AutoCompileBackendAttempt

        calls = []
        attempt = _AutoCompileBackendAttempt()

        class Leaf(nn.Module):
            def forward(self, value):
                calls.append("eager")
                return value + 1

        leaf = Leaf()
        record = _ready_record(leaf)

        def compiled(value):
            calls.append("fallback")
            attempt.record_fallback("forward", "stablehlo_conversion")
            return value + 1

        setattr(compiled, mac._BACKEND_ATTEMPT_ATTR, attempt)
        monkeypatch.setattr(mac, "_compile_module_call_impl", lambda _module: compiled)
        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *_args: True)

        _promote_module(leaf, record)
        torch.testing.assert_close(leaf(torch.ones(1)), torch.full((1,), 2.0))

        assert calls == ["fallback"]
        assert record.state == ModuleAutoCompileState.BLOCKED
        assert record.terminal_reason == mac._TERMINAL_REASON_BACKEND_FALLBACK
        assert leaf._compiled_call_impl is None

        torch.testing.assert_close(leaf(torch.ones(1)), torch.full((1,), 2.0))
        assert calls == ["fallback", "eager"]

    def test_checkpoint_backend_fallback_drains_before_eager(self, monkeypatch):
        from torch.utils.checkpoint import checkpoint

        from torch_neuronx.neuron_dynamo_backend.backend import _AutoCompileBackendAttempt

        calls = []
        attempt = _AutoCompileBackendAttempt()

        class Leaf(nn.Module):
            def forward(self, value):
                calls.append("eager")
                return value.sin().cos()

        leaf = Leaf()
        record = _ready_record(leaf)

        def compiled(value):
            calls.append("fallback")
            if not attempt.failed():
                attempt.record_fallback("forward", "stablehlo_conversion")
            return value.sin().cos()

        setattr(compiled, mac._BACKEND_ATTEMPT_ATTR, attempt)
        monkeypatch.setattr(mac, "_compile_module_call_impl", lambda _module: compiled)
        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *_args: True)

        _promote_module(leaf, record)
        value = torch.ones(1, requires_grad=True)
        output = checkpoint(leaf, value, use_reentrant=False)

        assert calls == ["fallback"]
        assert record.state == ModuleAutoCompileState.DRAINING_TO_EAGER
        assert leaf._compiled_call_impl is not None

        output.sum().backward()

        assert calls == ["fallback", "fallback"]
        assert record.state == ModuleAutoCompileState.BLOCKED
        assert record.terminal_reason == mac._TERMINAL_REASON_BACKEND_FALLBACK
        assert leaf._compiled_call_impl is None

        leaf(torch.ones(1))
        assert calls == ["fallback", "fallback", "eager"]

    def test_terminal_sac_fallback_preserves_compiled_recompute(self, monkeypatch):
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            checkpoint,
            create_selective_checkpoint_contexts,
        )

        from torch_neuronx.neuron_dynamo_backend.backend import (
            _AUTOCOMPILE_SAC_RECOMPUTE_POLICY,
            _AutoCompileBackendAttempt,
        )

        calls = []
        trigger_fallback = False
        captured_context = False
        attempt = _AutoCompileBackendAttempt()
        attempt.record_executor("forward", 3, "forward-key")
        attempt.record_executor("backward", 2, "backward-key")

        class Leaf(nn.Module):
            def forward(self, value):
                calls.append(("eager", None))
                return value.sin().cos()

        saved_ops = {torch.ops.aten.sin.default}

        def context_fn():
            def policy(_context, op, *_args, **_kwargs):
                if op in saved_ops:
                    return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.PREFER_RECOMPUTE

            return create_selective_checkpoint_contexts(policy)

        def compiled(value):
            nonlocal captured_context
            signature = attempt.active_autocompile_context_signature()
            calls.append(("compiled", signature))
            if not captured_context:
                pending_policy, pending_signature = attempt.take_pending_autocompile_context()
                assert pending_policy is not None
                assert pending_signature == signature
                captured_context = True
            if (
                trigger_fallback
                and signature != (2, _AUTOCOMPILE_SAC_RECOMPUTE_POLICY)
                and not attempt.failed()
            ):
                attempt.record_fallback("forward", "runtime_execution")
            return value.sin().cos()

        setattr(compiled, mac._BACKEND_ATTEMPT_ATTR, attempt)
        monkeypatch.setattr(mac, "_compile_module_call_impl", lambda _module: compiled)
        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *_args: True)

        leaf = Leaf()
        record = _ready_record(leaf)
        _promote_module(leaf, record)

        first_value = torch.randn(4, requires_grad=True)
        checkpoint(
            leaf,
            first_value,
            use_reentrant=False,
            context_fn=context_fn,
        ).sum().backward()

        second_value = torch.randn(4, requires_grad=True)
        checkpoint(
            leaf,
            second_value,
            use_reentrant=False,
            context_fn=context_fn,
        ).sum().backward()
        assert record.state == ModuleAutoCompileState.COMPILED
        wrapper = leaf._compiled_call_impl
        assert wrapper is not None

        trigger_fallback = True
        calls_before_fallback = len(calls)
        third_value = torch.randn(4, requires_grad=True)
        output = checkpoint(
            leaf,
            third_value,
            use_reentrant=False,
            context_fn=context_fn,
        )

        assert record.state == ModuleAutoCompileState.DRAINING_TO_EAGER
        assert leaf._compiled_call_impl is wrapper
        output.sum().backward()

        fallback_calls = calls[calls_before_fallback:]
        assert [kind for kind, _signature in fallback_calls] == ["compiled", "compiled"]
        assert fallback_calls[-1][1] == (2, _AUTOCOMPILE_SAC_RECOMPUTE_POLICY)
        assert record.state == ModuleAutoCompileState.BLOCKED
        assert leaf._compiled_call_impl is None
        assert third_value.grad is not None

    @pytest.mark.parametrize(
        "distributed_context",
        ("classic_fsdp", "composable_fsdp", "ddp", "composable_ddp"),
    )
    def test_distributed_promotion_waits_for_next_forward(
        self,
        monkeypatch,
        distributed_context,
    ):
        """Let the distributed wrapper finish backward before first capture."""

        class DistributedDataParallel(nn.Module):
            def __init__(self, child):
                super().__init__()
                self.child = child

            def forward(self, value):
                return self.child(value)

        compiled_calls = []

        def track_compiled_call(callable_):
            from torch_neuronx.neuron_dynamo_backend.backend import _AutoCompileBackendAttempt

            attempt = _AutoCompileBackendAttempt()

            def compiled(*args, **kwargs):
                compiled_calls.append(integration._patched_get_optimize_ddp_mode())
                if not attempt.has_real_forward_executor():
                    attempt.record_executor("forward", 3, "forward-key")
                output = callable_(*args, **kwargs)

                def report_backward(gradient):
                    if not attempt.has_real_backward_executor():
                        attempt.record_executor("backward", 2, "backward-key")
                    return gradient

                for tensor in mac._tensor_leaves(output):
                    if tensor.requires_grad:
                        tensor.register_hook(report_backward)
                return output

            setattr(compiled, mac._BACKEND_ATTEMPT_ATTR, attempt)
            return compiled

        monkeypatch.setattr(
            mac,
            "_tensor_is_neuron",
            lambda value: isinstance(value, torch.Tensor),
        )
        monkeypatch.setattr(
            mac,
            "_compile_module_forward",
            track_compiled_call,
        )
        monkeypatch.setattr(
            mac,
            "_compile_module_call_impl",
            lambda module: track_compiled_call(module._call_impl),
        )
        mod = (
            _ComposableFSDPLeafStub()
            if distributed_context == "composable_fsdp"
            else nn.LayerNorm(10)
        )
        if distributed_context == "classic_fsdp":
            mod._is_fsdp_managed_module = True
            mod._fsdp_use_orig_params = True
            target = mod
        elif distributed_context == "composable_fsdp":
            target = mod
        elif distributed_context == "composable_ddp":
            from torch.distributed._composable import replicate

            target = replicate(mod)
            # Keep replicate's in-place DDP class shape without initializing a
            # process group when its lazy runtime hooks execute.
            target._forward_pre_hooks.clear()
            target._forward_hooks.clear()
        else:
            DistributedDataParallel.__module__ = "torch.nn.parallel.fake"
            target = DistributedDataParallel(mod)
        record = _bootstrap_record(mod)
        x = torch.randn(2, 10, requires_grad=True)

        def run_training_step():
            target(x).sum().backward()
            x.grad = None
            mod.zero_grad(set_to_none=True)

        for _ in range(_PROMOTION_INVOCATION_THRESHOLD - 1):
            run_training_step()
        assert record.pending_promotion is False
        assert record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING

        run_training_step()
        assert record.pending_promotion is True
        assert record.state == ModuleAutoCompileState.PROMOTION_READY
        assert record.pending_backward_count == 0

        run_training_step()
        assert record.pending_promotion is False
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
        assert compiled_calls == []

        run_training_step()
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
        expected_capture_mode = (
            "no_optimization"
            if distributed_context in ("ddp", "composable_ddp")
            else integration._original_get_optimize_ddp_mode()
        )
        assert compiled_calls == [expected_capture_mode]

        target(x)
        assert record.state == ModuleAutoCompileState.COMPILED
        assert compiled_calls == [
            expected_capture_mode,
            integration._original_get_optimize_ddp_mode(),
        ]


class TestPreHookDiscoveryPath:
    """Direct discovery and tree seeding must use the same classification."""

    def test_prehook_leaf_discovered(self):
        """A bare leaf observed via the pre-hook path enters BOOTSTRAP_PENDING."""
        leaf = nn.LayerNorm(10)
        _observe_module_forward(leaf)
        record = mac._MODULE_REGISTRY.get(leaf)
        assert record is not None
        assert record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING

    def test_functorch_transform_blocks_observation(self, monkeypatch):
        monkeypatch.setattr(mac, "_functorch_transform_active", lambda: True)
        leaf = nn.LayerNorm(10)
        _observe_module_forward(leaf)
        assert mac._MODULE_REGISTRY.get(leaf) is None

    def test_prehook_compiled_ancestor_blocks(self):
        """A leaf discovered via the pre-hook with a compiled ancestor on the
        stack is BLOCKED (compiled_ancestor)."""

        class Parent(nn.Module):
            def __init__(self, child):
                super().__init__()
                self.child = child

            def forward(self, x):
                # Mark the live parent as compiled only after _call_impl has
                # started, so observation sees both the real call frame and the
                # nested-compile state without bypassing this forward.
                self._compiled_call_impl = lambda *args, **kwargs: None
                try:
                    _observe_module_forward(self.child)
                finally:
                    self._compiled_call_impl = None
                return self.child(x)

        leaf = nn.LayerNorm(10)
        parent = Parent(leaf)
        parent(torch.randn(2, 10))

        record = mac._MODULE_REGISTRY.get(leaf)
        assert record is not None
        assert record.state == ModuleAutoCompileState.BLOCKED
        assert record.terminal_reason == _TERMINAL_REASON_COMPILED_ANCESTOR

    def test_prehook_non_leaf_blocked(self):
        """A non-leaf module observed via the pre-hook path is blocked."""
        model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        _observe_module_forward(model)
        record = mac._MODULE_REGISTRY.get(model)
        assert record is not None
        assert record.state == ModuleAutoCompileState.BLOCKED


class TestModuleToDiscovery:
    """Device materialization is an explicit user-intent discovery signal.

    The tests fake Module.to/Module.to_empty so they run on CPU while still
    exercising the installed patches and the module-tree classifier.
    """

    def _install_with_fake_device_moves(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "1")
        torch_neuronx._install_module_auto_compile_patches()

        def fake_to(self, *args, **kwargs):
            return self

        def fake_to_empty(self, *args, **kwargs):
            return self

        def fake_neuron(self, *args, **kwargs):
            return self

        monkeypatch.setattr(integration, "_original_module_to", fake_to)
        monkeypatch.setattr(integration, "_original_module_to_empty", fake_to_empty)
        monkeypatch.setattr(integration, "_original_module_neuron", fake_neuron)
        return torch_neuronx

    def test_module_to_neuron_discovers_leaf_tree(self, monkeypatch):
        self._install_with_fake_device_moves(monkeypatch)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(8)
                self.act = nn.SiLU()
                self.linear = nn.Linear(8, 8)

        model = MLP()
        assert model.to("neuron:0") is model

        assert mac._MODULE_REGISTRY.get(model) is None
        assert mac._MODULE_REGISTRY[model.norm].state == ModuleAutoCompileState.BOOTSTRAP_PENDING
        assert mac._MODULE_REGISTRY[model.act].state == ModuleAutoCompileState.BOOTSTRAP_PENDING
        assert mac._MODULE_REGISTRY[model.linear].state == ModuleAutoCompileState.BOOTSTRAP_PENDING

    def test_module_neuron_discovers_leaf_tree(self, monkeypatch):
        self._install_with_fake_device_moves(monkeypatch)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(8)
                self.act = nn.SiLU()
                self.linear = nn.Linear(8, 8)

        model = MLP()
        assert model.neuron() is model

        assert mac._MODULE_REGISTRY.get(model) is None
        assert mac._MODULE_REGISTRY[model.norm].state == ModuleAutoCompileState.BOOTSTRAP_PENDING
        assert mac._MODULE_REGISTRY[model.act].state == ModuleAutoCompileState.BOOTSTRAP_PENDING
        assert mac._MODULE_REGISTRY[model.linear].state == ModuleAutoCompileState.BOOTSTRAP_PENDING

    def test_module_to_neuron_discovers_checkpoint_leaves_only(self, monkeypatch):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
        )

        self._install_with_fake_device_moves(monkeypatch)
        region = checkpoint_wrapper(nn.Sequential(nn.Linear(8, 8), nn.SiLU()))

        assert region.to("neuron:0") is region

        assert mac._MODULE_REGISTRY.get(region) is None
        assert mac._MODULE_REGISTRY.get(region._checkpoint_wrapped_module) is None
        for leaf in (region._checkpoint_wrapped_module[0], region._checkpoint_wrapped_module[1]):
            record = mac._MODULE_REGISTRY[leaf]
            assert record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING
            assert record.terminal_reason is None

    def test_direct_checkpoint_keeps_leaf_record(self, monkeypatch):
        from torch.utils.checkpoint import checkpoint

        self._install_with_fake_device_moves(monkeypatch)
        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda value: isinstance(value, torch.Tensor))

        class Inner(nn.Module):
            def forward(self, value):
                return value.sin().cos()

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()

            def forward(self, value):
                return self.inner(value)

        block = Block()
        assert block.to("neuron:0") is block
        assert mac._MODULE_REGISTRY[block.inner].state == ModuleAutoCompileState.BOOTSTRAP_PENDING

        original_forward = block.forward

        def checkpoint_forward(value):
            return checkpoint(original_forward, value, use_reentrant=False)

        block.forward = checkpoint_forward
        value = torch.randn(8, requires_grad=True)
        block(value).sum().backward()

        inner_record = mac._MODULE_REGISTRY[block.inner]
        assert mac._MODULE_REGISTRY.get(block) is None
        assert inner_record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING
        assert inner_record.terminal_reason is None
        assert value.grad is not None

    def test_module_to_empty_neuron_discovers_leaf_tree(self, monkeypatch):
        self._install_with_fake_device_moves(monkeypatch)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(8)
                self.act = nn.SiLU()
                self.linear = nn.Linear(8, 8)

        model = MLP()
        assert model.to_empty(device="neuron:0") is model

        assert mac._MODULE_REGISTRY.get(model) is None
        assert mac._MODULE_REGISTRY[model.norm].state == ModuleAutoCompileState.BOOTSTRAP_PENDING
        assert mac._MODULE_REGISTRY[model.act].state == ModuleAutoCompileState.BOOTSTRAP_PENDING
        assert mac._MODULE_REGISTRY[model.linear].state == ModuleAutoCompileState.BOOTSTRAP_PENDING

    def test_module_to_empty_nonrecursive_does_not_discover_descendants(self, monkeypatch):
        self._install_with_fake_device_moves(monkeypatch)
        model = nn.Sequential(nn.LayerNorm(8), nn.ReLU())

        assert model.to_empty(device="neuron:0", recurse=False) is model

        assert mac._MODULE_REGISTRY.get(model) is None
        assert mac._MODULE_REGISTRY.get(model[0]) is None
        assert mac._MODULE_REGISTRY.get(model[1]) is None

    def test_module_to_neuron_blocks_modules_with_tied_parameters(self, monkeypatch):
        self._install_with_fake_device_moves(monkeypatch)
        model = nn.Sequential(nn.Embedding(16, 8), nn.Linear(8, 16, bias=False))
        model[1].weight = model[0].weight

        assert model.to("neuron:0") is model

        for module in model:
            record = mac._MODULE_REGISTRY[module]
            assert record.state == ModuleAutoCompileState.BLOCKED
            assert record.terminal_reason == "compiler_capability_unavailable"

        _reset_state()
        first = nn.Linear(8, 8, bias=False)
        second = nn.Linear(8, 8, bias=False)
        second.weight = first.weight
        first_record = _ready_record(first)
        _ready_record(second)
        _promote_module(first, first_record)
        assert first_record.state == ModuleAutoCompileState.BLOCKED

    def test_module_to_neuron_allows_shared_buffers(self, monkeypatch):
        self._install_with_fake_device_moves(monkeypatch)
        model = nn.Sequential(nn.ReLU(), nn.Sigmoid())
        shared = torch.ones(1)
        for module in model:
            module.register_buffer("shared", shared)

        assert model.to("neuron:0") is model
        assert all(
            mac._MODULE_REGISTRY[module].state == ModuleAutoCompileState.BOOTSTRAP_PENDING
            for module in model
        )

    def test_module_to_neuron_discovers_uninitialized_lazy_leaf(self, monkeypatch):
        self._install_with_fake_device_moves(monkeypatch)
        model = nn.LazyLinear(8)

        assert model.to("neuron:0") is model

        record = mac._MODULE_REGISTRY.get(model)
        assert record is not None
        assert record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING

    def test_module_to_cpu_without_neuron_state_does_not_discover(self, monkeypatch):
        self._install_with_fake_device_moves(monkeypatch)
        model = nn.LayerNorm(8)

        assert model.to("cpu") is model

        assert mac._MODULE_REGISTRY.get(model) is None

    def test_module_to_empty_cpu_without_neuron_state_does_not_discover(self, monkeypatch):
        self._install_with_fake_device_moves(monkeypatch)
        model = nn.LayerNorm(8)

        assert model.to_empty(device="cpu") is model

        assert mac._MODULE_REGISTRY.get(model) is None

    def test_module_to_neuron_discovery_does_not_promote_later_cpu_calls(self, monkeypatch):
        self._install_with_fake_device_moves(monkeypatch)
        model = nn.LayerNorm(8)

        assert model.to("neuron:0") is model
        record = mac._MODULE_REGISTRY[model]
        assert record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING

        x = torch.randn(2, 8)
        for _ in range(_PROMOTION_INVOCATION_THRESHOLD + 1):
            model(x)

        assert record.successful_invocation_count == 0
        assert record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING
        assert mac._bootstrap_compiled_call_is_owned(model)

    def test_hard_disable_during_tree_traversal_cannot_recreate_records(self, monkeypatch):
        model = nn.Sequential(nn.ReLU(), nn.Sigmoid())

        def disable_during_traversal(_root):
            _disable_feature("injected traversal disable")
            return {}

        monkeypatch.setattr(mac, "_module_parent_map", disable_during_traversal)

        with pytest.warns(UserWarning, match="injected traversal disable"):
            mac.observe_module_tree_after_to(model, target_device=torch.device("neuron"))

        assert len(mac._MODULE_REGISTRY) == 0
        assert all(not leaf._forward_pre_hooks for leaf in model)
        assert all(not leaf._forward_hooks for leaf in model)

    def test_internal_module_compile_scope_does_not_observe_module_to(self, monkeypatch):
        self._install_with_fake_device_moves(monkeypatch)

        model = nn.LayerNorm(8)
        with _internal_module_compile_scope():
            assert model.to("neuron:0") is model

        assert mac._MODULE_REGISTRY.get(model) is None

    def test_module_to_discovery_error_disables_feature_without_breaking_to(self, monkeypatch):
        self._install_with_fake_device_moves(monkeypatch)

        def boom(*args, **kwargs):
            raise RuntimeError("discovery bug")

        monkeypatch.setattr(mac, "observe_module_tree_after_to", boom)
        model = nn.LayerNorm(8)

        with pytest.warns(UserWarning, match="discovery failed in Module.to"):
            assert model.to("neuron:0") is model

        assert not module_auto_compile_enabled()

    def test_module_to_empty_discovery_error_disables_feature_without_breaking_move(
        self, monkeypatch
    ):
        self._install_with_fake_device_moves(monkeypatch)

        def boom(*args, **kwargs):
            raise RuntimeError("discovery bug")

        monkeypatch.setattr(mac, "observe_module_tree_after_to", boom)
        model = nn.LayerNorm(8)

        with pytest.warns(UserWarning, match="discovery failed in Module.to_empty"):
            assert model.to_empty(device="neuron:0") is model

        assert not module_auto_compile_enabled()

    def test_uninstall_restores_module_device_move_patches(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "1")
        torch_neuronx._install_module_auto_compile_patches()
        assert torch.nn.Module.to is not integration._original_module_to
        assert torch.nn.Module.to_empty is not integration._original_module_to_empty
        assert torch.nn.Module.neuron is not integration._original_module_neuron
        assert torch.nn.Module.__getstate__ is integration._patched_module_getstate
        assert (
            getattr(
                integration._optimize_ddp_mode_owner,
                integration._optimize_ddp_mode_attribute,
            )
            is integration._patched_get_optimize_ddp_mode
        )
        coerce_patch_owned = integration._coerce_tangent_patch_owned

        try:
            torch_neuronx._uninstall_module_auto_compile_patches()
            assert torch.nn.Module.to is integration._original_module_to
            assert torch.nn.Module.to_empty is integration._original_module_to_empty
            assert torch.nn.Module.neuron is integration._original_module_neuron
            assert torch.nn.Module.__getstate__ is integration._original_module_getstate
            assert (
                getattr(
                    integration._optimize_ddp_mode_owner,
                    integration._optimize_ddp_mode_attribute,
                )
                is integration._original_get_optimize_ddp_mode
            )
            if coerce_patch_owned:
                assert not hasattr(torch.Tensor, "__coerce_same_metadata_as_tangent__")
        finally:
            monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "1")
            torch_neuronx._install_module_auto_compile_patches()

    def test_uninstall_preserves_replaced_coerce_tangent_method(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "1")
        torch_neuronx._install_module_auto_compile_patches()
        if not integration._coerce_tangent_patch_owned:
            pytest.skip("torch build already owns coerce tangent behavior")

        name = "__coerce_same_metadata_as_tangent__"

        def replacement(*args, **kwargs):
            return None

        setattr(torch.Tensor, name, replacement)
        try:
            torch_neuronx._uninstall_module_auto_compile_patches()
            assert getattr(torch.Tensor, name) is replacement
        finally:
            if getattr(torch.Tensor, name, None) is replacement:
                delattr(torch.Tensor, name)
            torch_neuronx._install_module_auto_compile_patches()


class TestPatchOwnership:
    @pytest.mark.parametrize(
        "target_name",
        (
            "module_compile",
            "module_to",
            "module_to_empty",
            "module_neuron",
            "module_getstate",
            "dynamo_ddp_mode",
            "null_decorator",
        ),
    )
    def test_install_conflict_is_transactional(self, target_name):
        targets = {
            "module_compile": (
                torch.nn.Module,
                "compile",
                integration._original_module_compile,
            ),
            "module_to": (torch.nn.Module, "to", integration._original_module_to),
            "module_to_empty": (
                torch.nn.Module,
                "to_empty",
                integration._original_module_to_empty,
            ),
            "module_neuron": (
                torch.nn.Module,
                "neuron",
                integration._original_module_neuron,
            ),
            "module_getstate": (
                torch.nn.Module,
                "__getstate__",
                integration._original_module_getstate,
            ),
            "dynamo_ddp_mode": (
                integration._optimize_ddp_mode_owner,
                integration._optimize_ddp_mode_attribute,
                integration._original_get_optimize_ddp_mode,
            ),
            "null_decorator": (
                integration._NullDecorator,
                "__call__",
                integration._original_null_decorator_call,
            ),
        }
        owner, attribute, original = targets[target_name]
        if owner is None or original is None:
            pytest.skip(f"{target_name} is unavailable")
        torch_neuronx._uninstall_module_auto_compile_patches()

        def replacement(*args, **kwargs):
            return original(*args, **kwargs)

        setattr(owner, attribute, replacement)
        try:
            with pytest.raises(RuntimeError, match="replaced by another patch"):
                torch_neuronx._install_module_auto_compile_patches()

            assert getattr(owner, attribute) is replacement
            assert integration._module_auto_compile_patches_active is False
            assert integration._applied_module_patches == []
            assert not module_auto_compile_enabled()
        finally:
            setattr(owner, attribute, original)
            mac._reset_feature_disabled()
            torch_neuronx._install_module_auto_compile_patches()

    def test_install_failure_rolls_back_all_targets(self):
        torch_neuronx._uninstall_module_auto_compile_patches()
        original_install = integration._install_coerce_tangent_patch

        def fail_install():
            raise RuntimeError("injected install failure")

        integration._install_coerce_tangent_patch = fail_install
        try:
            with pytest.raises(RuntimeError, match="injected install failure"):
                torch_neuronx._install_module_auto_compile_patches()

            assert torch.nn.Module.compile is integration._original_module_compile
            assert torch.nn.Module.to is integration._original_module_to
            assert torch.nn.Module.to_empty is integration._original_module_to_empty
            assert torch.nn.Module.neuron is integration._original_module_neuron
            assert torch.nn.Module.__getstate__ is integration._original_module_getstate
            assert (
                getattr(
                    integration._optimize_ddp_mode_owner,
                    integration._optimize_ddp_mode_attribute,
                )
                is integration._original_get_optimize_ddp_mode
            )
            assert integration._module_auto_compile_patches_active is False
            assert not module_auto_compile_enabled()
        finally:
            integration._install_coerce_tangent_patch = original_install
            mac._reset_feature_disabled()
            torch_neuronx._install_module_auto_compile_patches()

    def test_uninstall_preserves_later_patch(self):
        stale_module_to = torch.nn.Module.to

        def downstream_module_to(self, *args, **kwargs):
            return stale_module_to(self, *args, **kwargs)

        torch.nn.Module.to = downstream_module_to
        try:
            torch_neuronx._uninstall_module_auto_compile_patches()
            assert torch.nn.Module.to is downstream_module_to
            with pytest.raises(RuntimeError, match="replaced by another patch"):
                torch_neuronx._install_module_auto_compile_patches()
            assert torch.nn.Module.to is downstream_module_to
        finally:
            torch.nn.Module.to = integration._original_module_to
            mac._reset_feature_disabled()
            torch_neuronx._install_module_auto_compile_patches()

    def test_reinstall_after_lost_patch_ownership_fails_closed(self):
        auto_module_to = torch.nn.Module.to

        def downstream_module_to(self, *args, **kwargs):
            return auto_module_to(self, *args, **kwargs)

        torch.nn.Module.to = downstream_module_to
        try:
            with (
                pytest.warns(UserWarning, match="lost ownership of an installed patch"),
                pytest.raises(RuntimeError, match="lost ownership of an installed patch"),
            ):
                torch_neuronx._install_module_auto_compile_patches()

            assert torch.nn.Module.to is downstream_module_to
            assert not integration._module_auto_compile_patches_active
            assert not module_auto_compile_enabled()
            assert mac._FEATURE_HARD_DISABLED
            assert torch.nn.Module.compile is integration._original_module_compile
        finally:
            torch.nn.Module.to = integration._original_module_to
            mac._reset_feature_disabled()
            torch_neuronx._install_module_auto_compile_patches()

    def test_install_and_uninstall_leave_global_torch_compile_untouched(self):
        torch_compile = torch.compile
        torch_compiler_compile = getattr(torch.compiler, "compile", None)

        torch_neuronx._uninstall_module_auto_compile_patches()
        assert torch.compile is torch_compile
        assert getattr(torch.compiler, "compile", None) is torch_compiler_compile

        torch_neuronx._install_module_auto_compile_patches()
        assert torch.compile is torch_compile
        assert getattr(torch.compiler, "compile", None) is torch_compiler_compile

    def test_torch_compile_decorator_accepts_model_keyword(self):
        def fn(value):
            return value + 1

        compiled = torch.compile(backend="eager")(model=fn)

        torch.testing.assert_close(compiled(torch.tensor(1)), torch.tensor(2))


class TestCheckpointHandling:
    """Preserve checkpoint lifecycle and selective checkpoint policy."""

    def test_selective_checkpoint_modes_are_distinguished(self):
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        def policy_fn(_ctx, _op, *_args, **_kwargs):
            return CheckpointPolicy.PREFER_RECOMPUTE

        caching_mode, cached_mode = create_selective_checkpoint_contexts(policy_fn)
        assert not _inside_checkpoint_context()
        assert not _inside_sac_forward()
        assert not _inside_sac_recompute()
        with caching_mode:
            assert _inside_sac_forward()
            assert not _inside_sac_recompute()
        with cached_mode:
            assert not _inside_sac_forward()
            assert _inside_sac_recompute()
            assert _inside_non_reentrant_checkpoint_recompute()

    def test_non_reentrant_checkpoint_forward_and_recompute_are_distinguished(self):
        from torch.utils.checkpoint import checkpoint

        checkpoint_context = []
        recompute_context = []

        def fn(x):
            checkpoint_context.append(_inside_checkpoint_context())
            recompute_context.append(_inside_non_reentrant_checkpoint_recompute())
            return torch.sin(x).cos()

        x = torch.randn(4, requires_grad=True)
        checkpoint(fn, x, use_reentrant=False).sum().backward()

        assert checkpoint_context == [False, False]
        assert recompute_context == [False, True]

    def test_reentrant_checkpoint_stack_is_blocked(self):
        from torch.utils.checkpoint import checkpoint

        seen = []

        def fn(x):
            seen.append(_inside_checkpoint_context())
            return torch.sin(x)

        x = torch.randn(4, requires_grad=True)
        checkpoint(fn, x, use_reentrant=True).sum().backward()

        assert seen == [True, True]

    @pytest.mark.parametrize("use_sac", (False, True), ids=("full", "selective"))
    def test_external_checkpoint_leaf_remains_eligible(self, monkeypatch, use_sac):
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            checkpoint,
            create_selective_checkpoint_contexts,
        )

        execution_modes = []
        policy_trace = []

        class CheckpointLeaf(nn.Module):
            def forward(self, value):
                execution_modes.append("eager")
                return value.sin().cos()

        def policy_fn(ctx, op, *_args, **_kwargs):
            policy_trace.append((ctx.is_recompute, str(op)))
            return CheckpointPolicy.PREFER_RECOMPUTE

        options = {"use_reentrant": False}
        if use_sac:
            options["context_fn"] = lambda: create_selective_checkpoint_contexts(policy_fn)

        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda value: isinstance(value, torch.Tensor))
        module = CheckpointLeaf()
        record = _bootstrap_record(module)
        record.successful_invocation_count = _PROMOTION_INVOCATION_THRESHOLD - 1
        value = torch.randn(4, requires_grad=True)

        checkpoint(module, value, **options).sum().backward()

        assert execution_modes == ["eager", "eager"]
        assert record.state == ModuleAutoCompileState.PROMOTION_READY
        assert record.pending_promotion
        assert record.pending_backward_count == 0
        assert record.terminal_reason is None
        assert value.grad is not None
        if use_sac:
            assert any(not is_recompute for is_recompute, _op in policy_trace)

    def test_external_reentrant_checkpoint_leaf_remains_eligible(self, monkeypatch):
        from torch.utils.checkpoint import checkpoint

        execution_modes = []

        class CheckpointLeaf(nn.Module):
            def forward(self, value):
                execution_modes.append("eager")
                return value.sin().cos()

        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda value: isinstance(value, torch.Tensor))
        module = CheckpointLeaf()
        record = _bootstrap_record(module)
        record.successful_invocation_count = _PROMOTION_INVOCATION_THRESHOLD - 1
        value = torch.randn(4, requires_grad=True)

        checkpoint(module, value, use_reentrant=True).sum().backward()

        assert execution_modes == ["eager", "eager"]
        assert record.state == ModuleAutoCompileState.PROMOTION_READY
        assert record.pending_promotion
        assert record.terminal_reason is None
        assert record.pending_reentrant_checkpoint_count == 0
        assert value.grad is not None

    def test_compiled_sac_invocation_is_hidden_from_outer_checkpoint_cache(self):
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            checkpoint,
            create_selective_checkpoint_contexts,
        )

        from torch_neuronx.neuron_dynamo_backend.backend import _AutoCompileBackendAttempt

        policy_trace = []
        attempt = _AutoCompileBackendAttempt()

        def policy_fn(ctx, op, *_args, **_kwargs):
            policy_trace.append((ctx.is_recompute, op))
            return CheckpointPolicy.PREFER_RECOMPUTE

        def context_fn():
            return create_selective_checkpoint_contexts(policy_fn)

        def function(value):
            context_flags = mac._runtime_context_flags()
            with mac._compiled_invocation_context(attempt, context_flags):
                hidden = value.sin().cos()
            return hidden + value.square()

        value = torch.randn(4, requires_grad=True)
        checkpoint(
            function,
            value,
            use_reentrant=False,
            context_fn=context_fn,
        ).sum().backward()

        observed_ops = {op for _is_recompute, op in policy_trace}
        assert torch.ops.aten.sin.default not in observed_ops
        assert torch.ops.aten.cos.default not in observed_ops
        assert torch.ops.aten.pow.Tensor_Scalar in observed_ops
        assert value.grad is not None
        assert attempt.active_autocompile_context_signature() is None

    def test_recompute_does_not_apply_deferred_promotion(self, monkeypatch):
        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda v: isinstance(v, torch.Tensor))
        mod = nn.LayerNorm(10)
        record = _ready_record(mod)
        _install_bootstrap_hooks(mod, record)
        x = torch.randn(2, 10)
        assert record.pending_promotion is True
        assert record.state == ModuleAutoCompileState.PROMOTION_READY

        monkeypatch.setattr(
            mac,
            "_runtime_context_flags",
            lambda **_kwargs: mac._RUNTIME_CONTEXT_RECOMPUTE,
        )
        _bootstrap_forward_pre_hook(mod, (x,), None)
        assert record.pending_promotion is True
        assert mac._bootstrap_compiled_call_is_owned(mod)
        assert record.state == ModuleAutoCompileState.PROMOTION_READY


class TestAutogradValidationSkip:
    """Gradcheck/gradgradcheck must keep eager autograd semantics."""

    def test_inside_autograd_validation_context_detected(self):
        assert not _inside_autograd_validation_context()
        assert _run_from_fake_gradcheck_frame(_inside_autograd_validation_context)

    def test_deferred_promotion_skipped_inside_gradcheck(self, monkeypatch):
        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda v: isinstance(v, torch.Tensor))
        mod = nn.LayerNorm(10)
        record = _ready_record(mod)
        _install_bootstrap_hooks(mod, record)

        x = torch.randn(2, 10)
        assert record.pending_promotion is True
        assert record.state == ModuleAutoCompileState.PROMOTION_READY

        _run_from_fake_gradcheck_frame(lambda: mod(x))

        assert record.pending_promotion is True, "promotion fired inside gradcheck"
        assert mac._bootstrap_compiled_call_is_owned(mod)
        assert record.state == ModuleAutoCompileState.PROMOTION_READY

    def test_promoted_module_uses_eager_inside_gradcheck(self, monkeypatch):
        class Leaf(nn.Module):
            def forward(self, x):
                return x + 1

        def fake_compiled(*_a, **_k):
            raise AssertionError("compiled callable should not run under gradcheck")

        def fake_compile(_fn, **_kwargs):
            return fake_compiled

        leaf = Leaf()
        record = _ready_record(leaf)
        monkeypatch.setattr(mac.torch, "compile", fake_compile)
        _promote_module(leaf, record)
        wrapper = leaf._compiled_call_impl

        out = _run_from_fake_gradcheck_frame(wrapper, torch.tensor(1.0))
        torch.testing.assert_close(out, torch.tensor(2.0))


class TestNonContiguousInputSkip:
    """Non-contiguous Neuron inputs must keep eager module semantics."""

    def test_deferred_promotion_skipped_for_non_contiguous_input(self, monkeypatch):
        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda v: isinstance(v, torch.Tensor))
        mod = nn.LayerNorm(10)
        record = _ready_record(mod)
        _install_bootstrap_hooks(mod, record)

        x = torch.randn(2, 10)
        assert record.pending_promotion is True
        assert record.state == ModuleAutoCompileState.PROMOTION_READY

        non_contiguous = torch.randn(10, 2).t()
        assert non_contiguous.shape == x.shape
        assert not non_contiguous.is_contiguous()

        mod(non_contiguous)

        assert record.pending_promotion is True, "promotion fired on a non-contiguous input"
        assert mac._bootstrap_compiled_call_is_owned(mod)
        assert record.state == ModuleAutoCompileState.PROMOTION_READY

    def test_promoted_module_uses_eager_for_non_contiguous_input(self, monkeypatch):
        class Leaf(nn.Module):
            def forward(self, x):
                return x + 1

        def fake_compiled(*_a, **_k):
            raise AssertionError("compiled callable should not run for non-contiguous input")

        def fake_compile(_fn, **_kwargs):
            return fake_compiled

        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda v: isinstance(v, torch.Tensor))
        leaf = Leaf()
        record = _ready_record(leaf)
        monkeypatch.setattr(mac.torch, "compile", fake_compile)
        _promote_module(leaf, record)
        wrapper = leaf._compiled_call_impl

        non_contiguous = torch.randn(2, 3).t()
        assert not non_contiguous.is_contiguous()
        torch.testing.assert_close(wrapper(non_contiguous), non_contiguous + 1)

    def test_deferred_promotion_skipped_for_non_contiguous_ancestor_input(self, monkeypatch):
        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.child = nn.LayerNorm(10)

            def forward(self, x):
                return self.child(x.contiguous())

        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda v: isinstance(v, torch.Tensor))
        parent = Parent()
        child = parent.child
        record = _ready_record(child)
        _install_bootstrap_hooks(child, record)

        x = torch.randn(2, 10)
        assert record.pending_promotion is True
        assert record.state == ModuleAutoCompileState.PROMOTION_READY

        non_contiguous = torch.randn(10, 2).t()
        assert non_contiguous.shape == x.shape
        assert not non_contiguous.is_contiguous()

        parent(non_contiguous)

        assert (
            record.pending_promotion is True
        ), "promotion fired under a non-contiguous ancestor input"
        assert mac._bootstrap_compiled_call_is_owned(child)
        assert record.state == ModuleAutoCompileState.PROMOTION_READY

    def test_promoted_module_uses_eager_for_non_contiguous_ancestor_input(self, monkeypatch):
        class Leaf(nn.Module):
            def forward(self, x):
                return x + 1

        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.child = Leaf()

            def forward(self, x):
                return self.child(x.contiguous())

        def fake_compiled(*_a, **_k):
            raise AssertionError(
                "compiled callable should not run under non-contiguous ancestor input"
            )

        def fake_compile(_fn, **_kwargs):
            return fake_compiled

        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda v: isinstance(v, torch.Tensor))
        monkeypatch.setattr(mac.torch, "compile", fake_compile)
        parent = Parent()
        record = _ready_record(parent.child)
        _promote_module(parent.child, record)

        non_contiguous = torch.randn(2, 3).t()
        assert not non_contiguous.is_contiguous()
        torch.testing.assert_close(parent(non_contiguous), non_contiguous.contiguous() + 1)


class TestNeuronTensorGate:
    """The neuron-tensor gate must inspect kwargs, not just positional args, so
    module(input_ids=x) is still discovered. CI has no neuron device, so
    _tensor_is_neuron is monkeypatched to match any tensor."""

    def test_value_contains_neuron_tensor_descends_into_dict(self, monkeypatch):
        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda v: isinstance(v, torch.Tensor))

        t = torch.randn(2, 2)
        # Positional-only (already worked).
        assert mac._value_contains_neuron_tensor((t,))
        # kwargs-only — the case this gate fixes.
        assert mac._value_contains_neuron_tensor({"input_ids": t})
        # Nested in a dict value.
        assert mac._value_contains_neuron_tensor({"batch": [t]})
        # No tensors anywhere.
        assert not mac._value_contains_neuron_tensor({"flag": True})
        assert not mac._value_contains_neuron_tensor(())

    def test_empty_neuron_tensor_detection_is_recursive(self, monkeypatch):
        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda v: isinstance(v, torch.Tensor))

        assert mac._value_contains_empty_neuron_tensor({"batch": [(torch.empty(0),)]})
        assert not mac._value_contains_empty_neuron_tensor({"batch": [torch.ones(1)]})

    def test_tensor_scans_tolerate_cyclic_containers(self):
        cyclic = []
        cyclic.append(cyclic)

        assert not mac._value_contains_neuron_tensor(cyclic)
        assert not mac._value_contains_empty_neuron_tensor(cyclic)
        assert not mac._value_contains_non_contiguous_neuron_tensor(cyclic)

    def test_resident_neuron_state_does_not_make_cpu_call_eligible(self, monkeypatch):
        module = nn.Linear(2, 2)
        monkeypatch.setattr(mac, "_module_has_direct_neuron_state", lambda _module: True)
        monkeypatch.setattr(
            mac,
            "_compiler_can_capture_module",
            lambda _module: True,
        )

        assert not mac._compiled_execution_eligible(module, (torch.randn(1, 2),), {})


class TestDefensiveModuleIntrospection:
    def test_missing_backward_hook_attributes_do_not_disable_capture(self):
        module = nn.ReLU()
        del module._backward_pre_hooks
        del module._backward_hooks

        assert mac._compiler_can_capture_module(module)

    def test_overridden_traversal_apis_are_not_called(self, monkeypatch):
        class HostileLeaf(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))

            def buffers(self, *args, **kwargs):
                raise RuntimeError("buffers override must not run")

            def children(self):
                raise RuntimeError("children override must not run")

            def modules(self):
                raise RuntimeError("modules override must not run")

        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda value: isinstance(value, torch.Tensor))
        module = HostileLeaf()

        mac.observe_module_tree_after_to(module, target_device=None)

        assert mac._MODULE_REGISTRY[module].state == ModuleAutoCompileState.BOOTSTRAP_PENDING

    def test_module_tree_traversal_tolerates_cycles(self):
        parent = nn.Module()
        child = nn.Module()
        parent._modules["child"] = child
        child._modules["parent"] = parent

        assert list(mac._iter_module_tree(parent)) == [parent, child]


class TestUserCompileNotBroken:
    """Auto-compile must not perturb user-owned compile/export paths."""

    def test_bootstrap_entrypoints_are_not_dynamo_disabled(self):
        for entrypoint in (_bootstrap_forward_pre_hook, mac._BootstrapCallImpl.__call__):
            assert not getattr(entrypoint, "_torchdynamo_disable", False), (
                f"{entrypoint.__name__} is @torch._dynamo.disable'd — graph-breaks a "
                "user's fullgraph compile"
            )

    def test_no_global_forward_pre_hook_is_registered(self):
        assert not any(
            getattr(hook, "__module__", "").startswith("torch_neuronx")
            for hook in module_mod._global_forward_pre_hooks.values()
        )

    def test_user_compile_marks_discovered_module_user_owned(self, monkeypatch):
        torch_neuronx._install_module_auto_compile_patches()
        calls = []

        def fake_compile(module, *args, **kwargs):
            calls.append((module, args, kwargs))
            module._compiled_call_impl = lambda *a, **k: None
            return None

        monkeypatch.setattr(integration, "_original_module_compile", fake_compile)
        mod = nn.LayerNorm(8)
        _observe_module_forward(mod)
        record = mac._MODULE_REGISTRY[mod]
        assert record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING

        assert mod.compile(fullgraph=True) is None

        assert calls == [(mod, (), {"fullgraph": True})]
        assert record.state == ModuleAutoCompileState.USER_OWNED
        assert record.terminal_reason == _TERMINAL_REASON_EXPLICIT_COMPILE
        assert record.bootstrap_forward_pre_handle is None
        assert not record.bootstrap_compiled_call_installed
        assert "_call_impl" not in mod.__dict__

    def test_user_compile_before_discovery_marked_preexisting_user_owned(self, monkeypatch):
        torch_neuronx._install_module_auto_compile_patches()

        def fake_compile(module, *args, **kwargs):
            module._compiled_call_impl = lambda *a, **k: None
            return None

        monkeypatch.setattr(integration, "_original_module_compile", fake_compile)
        mod = nn.LayerNorm(8)

        mod.compile()

        record = mac._MODULE_REGISTRY[mod]
        assert record.state == ModuleAutoCompileState.USER_OWNED
        assert record.terminal_reason == _TERMINAL_REASON_PREEXISTING_COMPILED_CALL

    def test_torch_compile_returns_a_separate_wrapper_without_claiming_original(self):
        model = nn.Sequential(nn.LayerNorm(8), nn.SiLU())

        compiled = torch.compile(model, backend="eager")

        assert compiled is not model
        assert model._compiled_call_impl is None
        # torch.compile(module) is an explicit user compile, so the subtree is
        # recorded as USER_OWNED (never promoted). We must not install an auto
        # wrapper or leave bootstrap hooks behind on a module the user owns.
        for module in model.modules():
            record = mac._MODULE_REGISTRY.get(module)
            if record is None:
                continue
            assert record.state == ModuleAutoCompileState.USER_OWNED
            assert record.bootstrap_forward_pre_handle is None
            assert not record.bootstrap_compiled_call_installed
            assert "_call_impl" not in module.__dict__

    def test_functional_torch_compile_of_module_relinquishes_ownership(self):
        """torch.compile(mod) must relinquish like mod.compile() does.

        Regression guard for the upstream failures where a leaf kept our
        bootstrap hooks through a user's trace: the hook dicts alone keep
        nn.Module._call_impl on its hook slow path, which reorders lifted
        placeholders (dynamo test_graph_node_shapes) and multiplies guard
        variants until fullgraph=True callers trip recompile_limit
        (dynamo test_submodule_forward_hooks_with_kwargs).
        """
        torch_neuronx._install_module_auto_compile_patches()

        mod = nn.LayerNorm(8)
        _observe_module_forward(mod)
        record = mac._MODULE_REGISTRY[mod]
        assert record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING
        assert record.bootstrap_forward_pre_handle is not None

        compiled = torch.compile(mod, backend="eager")

        assert compiled is not mod
        assert record.state == ModuleAutoCompileState.USER_OWNED
        assert record.terminal_reason == _TERMINAL_REASON_EXPLICIT_COMPILE
        # The hooks must be gone from the module itself, not just detached from
        # the record -- Dynamo guards on these dicts.
        assert record.bootstrap_forward_pre_handle is None
        assert not record.bootstrap_compiled_call_installed
        assert len(mod._forward_pre_hooks) == 0
        assert len(mod._forward_hooks) == 0
        assert mod._compiled_call_impl is None
        assert "_call_impl" not in mod.__dict__

    def test_optimized_module_patch_is_registered(self):
        """The OptimizedModule.__init__ patch must be owned like the others."""
        torch_neuronx._install_module_auto_compile_patches()

        from torch._dynamo.eval_frame import OptimizedModule

        assert OptimizedModule.__init__ is integration._patched_optimized_module_init
        assert any(
            owner is OptimizedModule and attribute == "__init__"
            for owner, attribute, _original, _replacement in integration._MODULE_PATCHES
        )

    def test_null_decorator_patch_is_registered(self):
        if integration._NullDecorator is None:
            pytest.skip("torch build has no _NullDecorator")
        torch_neuronx._install_module_auto_compile_patches()

        assert integration._NullDecorator.__call__ is integration._patched_null_decorator_call

    @pytest.mark.parametrize("style", ("direct", "decorator"))
    @pytest.mark.parametrize("disable_source", ("argument", "environment"))
    def test_disabled_torch_compile_relinquishes_ownership(
        self,
        monkeypatch,
        style,
        disable_source,
    ):
        if integration._NullDecorator is None:
            pytest.skip("torch build has no _NullDecorator")
        module = nn.LayerNorm(8)
        record = _bootstrap_record(module)
        compile_kwargs = {"disable": True} if disable_source == "argument" else {}
        if disable_source == "environment":
            monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")

        compiled = (
            torch.compile(module, **compile_kwargs)
            if style == "direct"
            else torch.compile(**compile_kwargs)(module)
        )

        assert compiled is module
        assert record.state == ModuleAutoCompileState.USER_OWNED
        assert not module._forward_pre_hooks
        assert "_call_impl" not in module.__dict__

    @pytest.mark.parametrize(
        "copy_module",
        (
            pytest.param(copy.deepcopy, id="deepcopy"),
            pytest.param(lambda module: pickle.loads(pickle.dumps(module)), id="pickle"),
        ),
    )
    def test_explicit_compile_scrubs_copied_bootstrap_state(self, copy_module):
        source = nn.LayerNorm(8)
        _bootstrap_record(source)
        copied = copy_module(source)
        assert copied._forward_pre_hooks
        assert "_call_impl" not in copied.__dict__

        torch.compile(copied, backend="eager")

        record = mac._MODULE_REGISTRY[copied]
        assert record.state == ModuleAutoCompileState.USER_OWNED
        assert not copied._forward_pre_hooks
        assert copied._compiled_call_impl is None
        assert "_call_impl" not in copied.__dict__

    def test_aot_graph_module_is_compiler_owned(self):
        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd

        captured_graphs = []

        def capture_graph(gm, _example_inputs):
            captured_graphs.append(gm)
            return make_boxed_func(gm.forward)

        class AOTLeaf(nn.Module):
            def forward(self, x):
                return x.sin() + 1

        torch._dynamo.reset()
        compiled = torch.compile(
            AOTLeaf(),
            backend=aot_autograd(fw_compiler=capture_graph),
            fullgraph=True,
        )
        compiled(torch.randn(2))
        assert len(captured_graphs) == 1

        graph = captured_graphs[0]
        namespace = {
            "__name__": "torch._functorch._aot_autograd.runtime_wrappers",
            "graph": graph,
            "observe": _observe_module_forward,
        }
        exec("def run():\n    observe(graph)\n", namespace)
        namespace["run"]()

        record = mac._MODULE_REGISTRY[graph]
        assert record.state == ModuleAutoCompileState.USER_OWNED
        assert record.terminal_reason == _TERMINAL_REASON_COMPILER_OWNED_GRAPH
        assert graph._compiled_call_impl is None
        assert not graph._forward_pre_hooks
        assert not graph._forward_hooks

    def test_exported_graph_module_is_compiler_owned(self):
        graph = torch.export.export(nn.ReLU(), (torch.randn(2),)).module()

        _observe_module_forward(graph)

        record = mac._MODULE_REGISTRY[graph]
        assert record.state == ModuleAutoCompileState.USER_OWNED
        assert record.terminal_reason == _TERMINAL_REASON_COMPILER_OWNED_GRAPH

    def test_plain_eager_fx_graph_module_remains_eligible(self, monkeypatch):
        graph = torch.fx.symbolic_trace(nn.ReLU())
        graph(torch.randn(2))

        _observe_module_forward(graph)

        record = mac._MODULE_REGISTRY[graph]
        assert record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING
        assert record.terminal_reason is None

        record.state = ModuleAutoCompileState.PROMOTION_READY
        record.pending_promotion = True
        monkeypatch.setattr(mac, "_compile_module_call_impl", lambda module: module._call_impl)
        mac._promote_module(graph, record)

        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
        assert graph._compiled_call_impl is not None

    def test_compiler_runtime_reclaims_previously_promoted_fx_graph(self, monkeypatch):
        graph = torch.fx.symbolic_trace(nn.ReLU())
        record = _ready_record(graph)
        monkeypatch.setattr(mac, "_compile_module_call_impl", lambda module: module._call_impl)
        mac._promote_module(graph, record)
        assert graph._compiled_call_impl is not None

        namespace = {
            "__name__": "torch._functorch._aot_autograd.runtime_wrappers",
            "graph": graph,
            "x": torch.randn(2),
        }
        exec("def run():\n    return graph(x)\n", namespace)
        namespace["run"]()

        assert record.state == ModuleAutoCompileState.USER_OWNED
        assert record.terminal_reason == _TERMINAL_REASON_COMPILER_OWNED_GRAPH
        assert graph._compiled_call_impl is None

    def test_module_compile_while_temporarily_disabled_stays_user_owned(
        self,
        monkeypatch,
    ):
        sentinel = object()
        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "0")
        monkeypatch.setattr(
            integration,
            "_original_module_compile",
            lambda module, *args, **kwargs: sentinel,
        )
        module = nn.LayerNorm(8)

        assert module.compile(backend="eager") is sentinel
        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "1")

        record = mac._MODULE_REGISTRY[module]
        assert record.state == ModuleAutoCompileState.USER_OWNED
        assert record.terminal_reason == _TERMINAL_REASON_PREEXISTING_COMPILED_CALL

    def test_unhashable_module_compile_bookkeeping_is_best_effort(self, monkeypatch):
        class UnhashableModule(nn.Module):
            __hash__ = None

            def forward(self, x):
                return x

        torch_neuronx._install_module_auto_compile_patches()
        module_result = object()

        def fake_module_compile(module, *args, **kwargs):
            return module_result

        monkeypatch.setattr(integration, "_original_module_compile", fake_module_compile)
        model = UnhashableModule()

        assert model.compile() is module_result

    def test_explicit_compile_wins_concurrent_auto_promotion(self, monkeypatch):
        torch_neuronx._install_module_auto_compile_patches()
        leaf = nn.LayerNorm(8)
        record = _ready_record(leaf)
        auto_compile_started = threading.Event()
        explicit_compile_started = threading.Event()
        release_auto_compile = threading.Event()
        release_explicit_compile = threading.Event()
        errors = []

        def auto_callable(*args, **kwargs):
            return args[0]

        def explicit_callable(*args, **kwargs):
            return args[0]

        def fake_auto_compile(module):
            auto_compile_started.set()
            assert release_auto_compile.wait(timeout=5)
            return auto_callable

        def fake_explicit_compile(module, *args, **kwargs):
            explicit_compile_started.set()
            assert release_explicit_compile.wait(timeout=5)
            module._compiled_call_impl = explicit_callable
            return None

        monkeypatch.setattr(mac, "_compile_module_call_impl", fake_auto_compile)
        monkeypatch.setattr(integration, "_original_module_compile", fake_explicit_compile)

        def run_promotion():
            try:
                _promote_module(leaf, record)
            except Exception as exc:
                errors.append(exc)

        def run_explicit_compile():
            try:
                leaf.compile()
            except Exception as exc:
                errors.append(exc)

        auto_thread = threading.Thread(target=run_promotion)
        explicit_thread = threading.Thread(target=run_explicit_compile)
        auto_thread.start()
        explicit_was_started = False
        try:
            assert auto_compile_started.wait(timeout=5)

            explicit_thread.start()
            explicit_was_started = True
            assert explicit_compile_started.wait(timeout=5)
            assert record.state == ModuleAutoCompileState.USER_OWNED
            assert leaf._compiled_call_impl is None

            release_auto_compile.set()
            auto_thread.join(timeout=10)
            assert not auto_thread.is_alive()
            assert leaf._compiled_call_impl is None

            release_explicit_compile.set()
            explicit_thread.join(timeout=10)
        finally:
            release_auto_compile.set()
            release_explicit_compile.set()
            auto_thread.join(timeout=10)
            if explicit_was_started:
                explicit_thread.join(timeout=10)

        assert not explicit_thread.is_alive()
        assert record.state == ModuleAutoCompileState.USER_OWNED
        assert errors == []
        assert leaf._compiled_call_impl is explicit_callable

    def test_user_fullgraph_compile_does_not_graph_break_when_feature_enabled(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
        x = torch.randn(4, 8)
        torch._dynamo.reset()
        compiled = torch.compile(model, fullgraph=True, dynamic=False)
        torch.testing.assert_close(compiled(x), model(x))

    def test_user_fullgraph_compile_of_parent_with_bootstrap_leaf(self, monkeypatch):
        """A mid-bootstrap leaf (hooks installed, not promoted) must not break a
        user's fullgraph compile of its parent or enter stack walking."""

        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf = nn.Linear(8, 8)

            def forward(self, x):
                return self.leaf(x).relu()

        parent = Parent()
        # Put the leaf into BOOTSTRAP_PENDING with live bootstrap hooks.
        record = _bootstrap_record(parent.leaf)
        assert parent.leaf._forward_pre_hooks  # hooks really installed

        x = torch.randn(4, 8)
        original_recover = mac._recover_innermost_active_module

        def fail_stack_walk():
            raise AssertionError("capture entered stack walking")

        monkeypatch.setattr(
            mac,
            "_recover_innermost_active_module",
            fail_stack_walk,
        )
        torch._dynamo.reset()
        compiled = torch.compile(
            lambda value: parent(value),
            backend="eager",
            fullgraph=True,
            dynamic=False,
        )
        out = compiled(x)
        assert record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING

        monkeypatch.setattr(mac, "_recover_innermost_active_module", original_recover)
        monkeypatch.setattr(mac, "_tensor_is_neuron", lambda value: isinstance(value, torch.Tensor))
        record.successful_invocation_count = _PROMOTION_INVOCATION_THRESHOLD - 1
        with torch.no_grad():
            torch.testing.assert_close(out, parent(x))
        assert record.state == ModuleAutoCompileState.PROMOTION_READY
        assert record.pending_promotion

    def test_user_fullgraph_compile_with_promoted_fsdp_leaf_skips_stack_walk(
        self,
        monkeypatch,
    ):
        leaf = _ComposableFSDPLeafStub()
        record = _ready_record(leaf)
        monkeypatch.setattr(mac, "_compile_module_forward", lambda target: target)
        _promote_module(leaf, record)

        def fail_stack_walk():
            raise AssertionError("capture entered stack walking")

        monkeypatch.setattr(
            mac,
            "_recover_innermost_active_module",
            fail_stack_walk,
        )
        compiled = torch.compile(
            lambda value: leaf(value),
            backend="eager",
            fullgraph=True,
            dynamic=False,
        )

        torch.testing.assert_close(compiled(torch.tensor(1.0)), torch.tensor(2.0))
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL

    def test_user_fullgraph_compile_of_promoted_leaf_uses_forward_fast_path(
        self,
        monkeypatch,
    ):
        leaf = nn.Linear(4, 4)
        record = _ready_record(leaf)

        def fail_nested_compile(*_args, **_kwargs):
            raise AssertionError("external capture entered AutoCompile's graph")

        monkeypatch.setattr(mac, "_compile_module_call_impl", lambda _module: fail_nested_compile)
        _promote_module(leaf, record)

        value = torch.randn(2, 4)
        expected = leaf.forward(value)
        compiled = torch.compile(
            lambda tensor: leaf(tensor),
            backend="eager",
            fullgraph=True,
            dynamic=False,
        )

        torch.testing.assert_close(compiled(value), expected)
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL


class TestPromotionLifecycle:
    def test_first_compiled_call_is_serialized(self, monkeypatch):
        compiled_entered = threading.Event()
        release_compiled = threading.Event()
        compiled_calls = []
        eager_calls = []

        class Leaf(nn.Module):
            def forward(self, x):
                eager_calls.append(threading.get_ident())
                return x + 1

        def fake_compile(_module):
            def compiled(x):
                compiled_calls.append(threading.get_ident())
                compiled_entered.set()
                assert release_compiled.wait(timeout=5)
                return x + 1

            return _reported_callable(compiled)[0]

        module = Leaf()
        record = _ready_record(module)
        monkeypatch.setattr(mac, "_compile_module_call_impl", fake_compile)
        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *args: True)
        mac._promote_module(module, record)

        results = []
        first = threading.Thread(target=lambda: results.append(module(torch.tensor(1.0))))
        first.start()
        assert compiled_entered.wait(timeout=5)
        second = threading.Thread(target=lambda: results.append(module(torch.tensor(2.0))))
        second.start()
        second.join(timeout=0.1)
        assert second.is_alive()
        release_compiled.set()
        first.join(timeout=5)
        second.join(timeout=5)

        assert not first.is_alive()
        assert not second.is_alive()
        assert len(compiled_calls) == 2
        assert len(eager_calls) == 0
        assert record.state == ModuleAutoCompileState.COMPILED
        assert len(results) == 2

    def test_compile_setup_blocks_concurrent_forward(self, monkeypatch):
        compile_started = threading.Event()
        release_compile = threading.Event()
        compiled_calls = []
        eager_calls = []
        errors = []

        class Leaf(nn.Module):
            def forward(self, value):
                eager_calls.append(value)
                return value + 1

        def fake_compile(_module):
            compile_started.set()
            assert release_compile.wait(timeout=5)

            def compiled(value):
                compiled_calls.append(value)
                return value + 1

            return _reported_callable(compiled)[0]

        module = Leaf()
        record = _ready_record(module)
        _install_bootstrap_hooks(module, record)
        monkeypatch.setattr(mac, "_compile_module_call_impl", fake_compile)
        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *_args: True)
        results = []

        def run(value):
            try:
                results.append(module(value))
            except Exception as error:
                errors.append(error)

        first = threading.Thread(target=run, args=(torch.tensor(1.0),))
        second = threading.Thread(target=run, args=(torch.tensor(2.0),))
        first.start()
        assert compile_started.wait(timeout=5)
        second.start()
        second.join(timeout=0.1)
        assert second.is_alive()
        assert eager_calls == []

        release_compile.set()
        first.join(timeout=5)
        second.join(timeout=5)

        assert not first.is_alive()
        assert not second.is_alive()
        assert errors == []
        assert [value.item() for value in eager_calls] == [1.0]
        assert [value.item() for value in compiled_calls] == [2.0]
        assert len(results) == 2
        assert record.state == ModuleAutoCompileState.COMPILED

    def test_reset_wins_in_flight_promotion(self, monkeypatch):
        compile_started = threading.Event()
        release_compile = threading.Event()
        errors = []

        def fake_compile(module):
            compile_started.set()
            assert release_compile.wait(timeout=5)
            return module._call_impl

        module = nn.Linear(8, 8)
        record = _ready_record(module)
        monkeypatch.setattr(mac, "_compile_module_call_impl", fake_compile)

        def promote():
            try:
                mac._promote_module(module, record)
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=promote)
        thread.start()
        assert compile_started.wait(timeout=5)
        mac._reset_state()
        release_compile.set()
        thread.join(timeout=10)

        assert not thread.is_alive()
        assert errors == []
        assert mac._MODULE_REGISTRY.get(module) is None
        assert module._compiled_call_impl is None

    def test_reset_preserves_foreign_compiled_wrapper(self, monkeypatch):
        module = nn.ReLU()
        record = _ready_record(module)
        monkeypatch.setattr(mac, "_compile_module_call_impl", lambda mod: mod._call_impl)
        mac._promote_module(module, record)
        auto_wrapper = module._compiled_call_impl

        def foreign_wrapper(*args, **kwargs):
            return module._call_impl(*args, **kwargs)

        module._compiled_call_impl = foreign_wrapper
        mac._reset_state()

        assert auto_wrapper is not foreign_wrapper
        assert module._compiled_call_impl is foreign_wrapper

    def test_hard_disable_removes_hooks_and_auto_wrappers(self, monkeypatch):
        pending = nn.LayerNorm(8)
        pending_record, _ = _get_or_create_record(pending)
        pending_record.state = ModuleAutoCompileState.BOOTSTRAP_PENDING
        _install_bootstrap_hooks(pending, pending_record)

        promoted = nn.ReLU()
        promoted_record = _ready_record(promoted)
        monkeypatch.setattr(mac, "_compile_module_call_impl", lambda mod: mod._call_impl)
        mac._promote_module(promoted, promoted_record)

        with pytest.warns(UserWarning, match="test disable"):
            mac._disable_feature("test disable")

        assert not pending._forward_pre_hooks
        assert not pending._forward_hooks
        assert promoted._compiled_call_impl is None
        assert len(mac._MODULE_REGISTRY) == 0

    def test_unhashable_module_is_registered_by_identity(self):
        class UnhashableLeaf(nn.Module):
            __hash__ = None

            def __eq__(self, other):
                return self is other

        module = UnhashableLeaf()
        record, is_new = _get_or_create_record(module)

        assert is_new
        assert mac._MODULE_REGISTRY[module] is record


class TestAuditHardening:
    """Regression tests for the hardening fixes."""

    def test_guard_miss_runs_exact_eager_callable_once(self, monkeypatch):
        from torch_neuronx.neuron_dynamo_backend import backend

        backend_calls = []

        def create_backend(attempt):
            def compile_graph(gm, _example_inputs, options=None):
                backend_calls.append(options)
                attempt.record_executor("forward", 3, "forward-key")
                return gm.forward

            return compile_graph

        def function(value):
            if value.shape[0] == 2:
                raise ValueError("exact eager error")
            return value + 1

        monkeypatch.setattr(backend, "_create_autocompile_backend", create_backend)
        monkeypatch.setattr(torch_neuronx, "synchronize", lambda: None)
        compiled = mac._compile_isolated_callable(function)
        attempt = getattr(compiled, mac._BACKEND_ATTEMPT_ATTR)

        torch.testing.assert_close(compiled(torch.ones(1)), torch.full((1,), 2.0))
        with pytest.raises(ValueError, match="^exact eager error$"):
            compiled(torch.ones(2))

        assert len(backend_calls) == 1
        assert attempt.fallback_reasons == [("forward", "guard_miss")]

    def test_env_flag_never_raises_on_hot_path(self, monkeypatch):
        """Invalid feature flags use the default on feature-owned paths."""

        name = "TORCH_NEURONX_MODULE_AUTO_COMPILE"
        for bad in ("", "  ", "yes please", "2", "maybe"):
            monkeypatch.setenv(name, bad)
            # Must not raise for either default.
            assert _env_flag(name, default=False) is False
            assert _env_flag(name, default=True) is True

    def test_coerce_patch_is_neuron_scoped_and_inert_when_disabled(self, monkeypatch):
        """The coerce patch is inert for non-Neuron tensors and when disabled."""
        method = getattr(torch.Tensor, "__coerce_same_metadata_as_tangent__", None)
        if method is None or not integration._coerce_tangent_patch_owned:
            pytest.skip("torch build already owns coerce tangent behavior")

        try:
            from torch.distributed._functional_collectives import AsyncCollectiveTensor
        except ImportError:
            pytest.skip("torch.distributed unavailable")

        class FakeNeuronTensor:
            device = torch.device("neuron")

        cpu_tensor = torch.randn(2, 2)
        neuron_tensor = FakeNeuronTensor()
        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "1")
        assert method(cpu_tensor, None, AsyncCollectiveTensor) is None
        assert method(neuron_tensor, None, AsyncCollectiveTensor) is neuron_tensor

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "0")
        assert method(neuron_tensor, None, AsyncCollectiveTensor) is None

    def test_make_fx_does_not_create_records(self):
        """make_fx must not create or advance a discovery record."""
        from torch.fx.experimental.proxy_tensor import make_fx

        leaf = nn.Linear(8, 8)

        def fn(x):
            # Direct discovery inside the trace must be a no-op.
            _observe_module_forward(leaf)
            return leaf(x)

        make_fx(fn)(torch.randn(4, 8))
        record = mac._MODULE_REGISTRY.get(leaf)
        assert record is None or record.state == ModuleAutoCompileState.COLD

    def test_jit_trace_stack_does_not_create_records(self):
        """JIT invokes modules before is_tracing() is reliably true."""

        leaf = nn.ReLU()
        namespace = {
            "leaf": leaf,
            "observe": _observe_module_forward,
        }
        trace_frame = compile(
            "observe(leaf)",
            "/tmp/torch/jit/_trace.py",
            "exec",
        )
        exec(trace_frame, namespace)

        assert mac._MODULE_REGISTRY.get(leaf) is None

    def test_kill_switch_reset(self):
        """_reset_feature_disabled clears the sticky kill-switch so a later
        test isn't forced feature-off."""

        _disable_feature("test trip")
        assert module_auto_compile_enabled() is False
        _reset_feature_disabled()
        # With the env set to "1" by this test package, the gate is live again.

        os.environ["TORCH_NEURONX_MODULE_AUTO_COMPILE"] = "1"
        assert module_auto_compile_enabled() is True

    def test_compile_targets_have_isolated_dynamo_code_objects(self, monkeypatch):
        compile_targets = []

        def fake_optimize(*_args, **_kwargs):
            def decorate(fn):
                compile_targets.append(fn)
                return fn

            return decorate

        monkeypatch.setattr(mac.torch._dynamo, "optimize", fake_optimize)
        first = mac._compile_module_call_impl(nn.ReLU())
        second = mac._compile_module_call_impl(nn.Sigmoid())

        assert len(compile_targets) == 2
        assert compile_targets[0].__code__ is not compile_targets[1].__code__
        assert first.__code__ is not second.__code__
        torch.testing.assert_close(first(torch.tensor(-1.0)), torch.tensor(0.0))
        torch.testing.assert_close(second(torch.tensor(0.0)), torch.tensor(0.5))

    def test_meta_state_defers_promotion_and_compiled_execution(self, monkeypatch):
        leaf = nn.Linear(1, 1, device="meta")
        record = _ready_record(leaf)
        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *_args: True)

        _promote_module(leaf, record)
        assert record.state == ModuleAutoCompileState.PROMOTION_READY
        assert leaf._compiled_call_impl is None

        leaf.weight = nn.Parameter(torch.ones(1, 1))
        leaf.bias = nn.Parameter(torch.zeros(1))
        assert not mac._module_has_direct_meta_state(leaf)
        _promote_module(leaf, record)
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL

        leaf.weight = nn.Parameter(torch.empty(1, 1, device="meta"))
        assert not mac._compiled_execution_eligible(leaf, (torch.ones(1, 1),), {})

    def test_first_capture_waits_for_base_gradient_then_stays_compiled(
        self,
        monkeypatch,
    ):
        leaf = nn.Linear(1, 1)
        base = torch.ones(1, requires_grad=True)
        leaf._parameters["weight"] = base.view(1, 1)
        leaf.bias = None
        record = _ready_record(leaf)
        compiled_calls = []

        from torch_neuronx.neuron_dynamo_backend.backend import _AutoCompileBackendAttempt

        attempt = _AutoCompileBackendAttempt()

        def compile_module(_module):
            def compiled(value):
                compiled_calls.append(value)
                if not attempt.has_real_forward_executor():
                    attempt.record_executor("forward", 3, "forward-key")
                output = leaf.forward(value)

                def report_backward(gradient):
                    if not attempt.has_real_backward_executor():
                        attempt.record_executor("backward", 2, "backward-key")
                    return gradient

                output.register_hook(report_backward)
                return output

            setattr(compiled, mac._BACKEND_ATTEMPT_ATTR, attempt)
            return compiled

        monkeypatch.setattr(
            mac,
            "_compile_module_call_impl",
            compile_module,
        )
        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *_args: True)

        _promote_module(leaf, record)
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL

        base.grad = torch.ones_like(base)
        eager_output = leaf(torch.ones(1, 1))
        assert compiled_calls == []
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL

        base.grad = None
        second_eager_output = leaf(torch.ones(1, 1))
        assert compiled_calls == []
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL

        (eager_output + second_eager_output).sum().backward()
        base.grad = None
        compiled_output = leaf(torch.ones(1, 1))
        assert len(compiled_calls) == 1
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL

        compiled_output.sum().backward()
        base.grad = None
        leaf(torch.ones(1, 1))
        assert len(compiled_calls) == 2
        assert record.state == ModuleAutoCompileState.COMPILED

        base.grad = torch.ones_like(base)
        leaf(torch.ones(1, 1))
        assert len(compiled_calls) == 3
        assert record.state == ModuleAutoCompileState.COMPILED

    def test_isolated_leaf_compile_ddp_mode_is_thread_local(self):
        expected_mode = integration._original_get_optimize_ddp_mode()
        worker_modes = []
        get_mode = getattr(
            integration._optimize_ddp_mode_owner,
            integration._optimize_ddp_mode_attribute,
        )
        assert get_mode is integration._patched_get_optimize_ddp_mode

        with mac._isolated_leaf_compile_scope():
            assert get_mode() == "no_optimization"
            worker = threading.Thread(target=lambda: worker_modes.append(get_mode()))
            worker.start()
            worker.join()

        assert worker_modes == [expected_mode]
        assert get_mode() == expected_mode

    def test_composable_fsdp_compiles_forward_inside_eager_hooks_and_serializes(
        self,
        monkeypatch,
    ):
        events = []
        leaf = _ComposableFSDPLeafStub()
        leaf.events = events
        leaf.register_forward_pre_hook(_record_forward_pre_hook)
        leaf.register_forward_hook(_record_forward_hook)
        record = _ready_record(leaf)

        def fake_compile(target):
            def compiled(*args, **kwargs):
                events.append("compiled")
                return target(*args, **kwargs)

            return _reported_callable(compiled)[0]

        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *_args: True)
        monkeypatch.setattr(mac, "_compile_module_forward", fake_compile)
        _promote_module(leaf, record)

        restored = pickle.loads(pickle.dumps(leaf))
        clone = copy.deepcopy(leaf)
        torch.testing.assert_close(restored(torch.tensor(1.0)), torch.tensor(2.0))
        torch.testing.assert_close(clone(torch.tensor(1.0)), torch.tensor(2.0))

        assert leaf._compiled_call_impl is None
        torch.testing.assert_close(leaf(torch.tensor(1.0)), torch.tensor(2.0))
        assert events == ["pre", "compiled", "post"]
        assert record.state == ModuleAutoCompileState.COMPILED
        holder = leaf.__dict__.get(mac._AUTO_FORWARD_HOLDER)
        assert isinstance(holder, mac._AutoForwardHolder)
        assert holder.module_ref() is leaf
        assert leaf.__dict__["forward"] is record.auto_wrapper_ref()

        for copied in (restored, clone):
            copied_record = _ready_record(copied)
            _promote_module(copied, copied_record)
            torch.testing.assert_close(copied(torch.tensor(1.0)), torch.tensor(2.0))

        late_fsdp_leaf = nn.Linear(1, 1)
        late_fsdp_leaf.events = []
        late_record = _ready_record(late_fsdp_leaf)
        _promote_module(late_fsdp_leaf, late_record)
        late_fsdp_leaf._is_fsdp_managed_module = True
        late_fsdp_leaf._fsdp_use_orig_params = True
        late_fsdp_leaf.register_forward_pre_hook(_record_forward_pre_hook)
        late_fsdp_leaf.register_forward_hook(_record_forward_hook)
        late_fsdp_leaf(torch.ones(1, 1))
        assert late_record.state == ModuleAutoCompileState.BLOCKED
        assert late_fsdp_leaf.events == ["pre", "post"]

        def replacement_forward(_self, value):
            return value + 2

        replacement = types.MethodType(replacement_forward, clone)
        clone.forward = replacement

        _reset_state()
        assert "forward" not in leaf.__dict__
        assert "forward" not in restored.__dict__
        assert clone.forward is replacement
        assert mac._AUTO_FORWARD_HOLDER not in clone.__dict__
        torch.testing.assert_close(leaf(torch.tensor(1.0)), torch.tensor(2.0))

    def test_forward_only_fallback_restores_assigned_forward(self, monkeypatch):
        class Region(_ComposableFSDPStub):
            def forward(self, value):
                return value

        region = Region()
        original_forward = region.forward

        def assigned_forward(value):
            return original_forward(value) + 1

        region.forward = assigned_forward
        record = _ready_record(region)
        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *_args: True)
        monkeypatch.setattr(
            mac,
            "_compile_module_forward",
            lambda target: _reported_callable(target)[0],
        )

        _promote_module(region, record)
        torch.testing.assert_close(region(torch.tensor(1.0)), torch.tensor(2.0))

        holder = region.__dict__[mac._AUTO_FORWARD_HOLDER]
        state = mac._sanitize_module_getstate(region, region.__dict__.copy())
        assert holder.original_forward is assigned_forward
        assert state["forward"] is assigned_forward
        mac._mark_blocked(
            region,
            record,
            reason=mac._TERMINAL_REASON_BACKEND_FALLBACK,
            clear_compiled_call_impl=True,
        )
        assert region.forward is assigned_forward
        torch.testing.assert_close(region(torch.tensor(1.0)), torch.tensor(2.0))

    @pytest.mark.parametrize("explicit_compile", (False, True))
    def test_composable_fsdp_shallow_copy_uses_copy_state(
        self,
        monkeypatch,
        explicit_compile,
    ):
        class StatefulLeaf(_ComposableFSDPStub):
            def __init__(self):
                super().__init__()
                self._is_fsdp_managed_module = True
                self._fsdp_use_orig_params = True
                self.increment = 1

            def forward(self, value):
                return value + self.increment

        monkeypatch.setattr(
            mac,
            "_compile_module_forward",
            lambda target: _reported_callable(target)[0],
        )
        module = StatefulLeaf()
        record = _ready_record(module)
        _promote_module(module, record)
        copied = copy.copy(module)
        copied.increment = 5

        callable_ = (
            torch.compile(copied, backend="eager", fullgraph=True, dynamic=False)
            if explicit_compile
            else copied
        )
        torch.testing.assert_close(callable_(torch.tensor(1)), torch.tensor(6))
        torch.testing.assert_close(module(torch.tensor(1)), torch.tensor(2))
        assert "forward" not in copied.__dict__
        assert mac._AUTO_FORWARD_HOLDER not in copied.__dict__

    @pytest.mark.parametrize("forward_kind", ("static", "class"))
    def test_composable_fsdp_deepcopy_restores_forward_descriptor(
        self,
        monkeypatch,
        forward_kind,
    ):
        class StaticForwardLeaf(_ComposableFSDPStub):
            def __init__(self):
                super().__init__()
                self._is_fsdp_managed_module = True
                self._fsdp_use_orig_params = True

            @staticmethod
            def forward(value):
                return value + 1

        class ClassForwardLeaf(_ComposableFSDPStub):
            increment = 2

            def __init__(self):
                super().__init__()
                self._is_fsdp_managed_module = True
                self._fsdp_use_orig_params = True

            @classmethod
            def forward(cls, value):
                return value + cls.increment

        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *_args: True)
        monkeypatch.setattr(
            mac.torch,
            "compile",
            lambda target, **_kwargs: lambda *args, **kwargs: target(*args, **kwargs),
        )
        module = StaticForwardLeaf() if forward_kind == "static" else ClassForwardLeaf()
        record = _ready_record(module)
        _promote_module(module, record)

        copied = copy.deepcopy(module)
        expected = torch.tensor(2 if forward_kind == "static" else 3)
        torch.testing.assert_close(copied(torch.tensor(1)), expected)
        assert "forward" not in copied.__dict__
        assert mac._AUTO_FORWARD_HOLDER not in copied.__dict__

    def test_compiled_wrapper_failure_makes_future_calls_eager(self, monkeypatch):
        class Leaf(nn.Module):
            def forward(self, x):
                return x + 1

        class CompileError(Exception):
            pass

        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *args: True)
        from torch_neuronx.neuron_dynamo_backend.backend import _AutoCompileBackendAttempt

        attempt = _AutoCompileBackendAttempt()
        state = {"n": 0}
        failure_reported = threading.Event()
        release_failure = threading.Event()

        def fake_compiled(*a, **k):
            state["n"] += 1
            if state["n"] == 1:
                attempt.record_executor("forward", 3, "forward-key")
                return torch.tensor(1.0)
            assert attempt.begin_executor_execution()
            try:
                attempt.record_failure("forward", "runtime_execution")
                failure_reported.set()
                assert release_failure.wait(timeout=1)
            finally:
                attempt.end_executor_execution()
            raise CompileError()

        setattr(fake_compiled, mac._BACKEND_ATTEMPT_ATTR, attempt)
        leaf = Leaf()
        record = _ready_record(leaf)
        monkeypatch.setattr(mac, "_compile_module_call_impl", lambda _module: fake_compiled)
        _promote_module(leaf, record)

        wrapper = leaf._compiled_call_impl
        assert wrapper is not None
        wrapper(torch.randn(1))
        assert record.state == ModuleAutoCompileState.COMPILED
        assert leaf._compiled_call_impl is wrapper
        failure_errors = queue.Queue()

        def run_failure():
            try:
                leaf(torch.tensor(1.0))
            except BaseException as error:
                failure_errors.put(error)

        failure_thread = threading.Thread(target=run_failure)
        failure_thread.start()
        assert failure_reported.wait(timeout=1)
        assert record.state == ModuleAutoCompileState.DRAINING_TO_EAGER
        assert leaf._compiled_call_impl is wrapper

        eager_result = queue.Queue()
        eager_done = threading.Event()

        def run_eager():
            eager_result.put(leaf(torch.tensor(1.0)))
            eager_done.set()

        eager_thread = threading.Thread(target=run_eager)
        eager_thread.start()
        assert not eager_done.wait(timeout=0.05)

        release_failure.set()
        failure_thread.join(timeout=1)
        eager_thread.join(timeout=1)

        assert not failure_thread.is_alive()
        assert not eager_thread.is_alive()
        assert isinstance(failure_errors.get_nowait(), CompileError)
        torch.testing.assert_close(eager_result.get_nowait(), torch.tensor(2.0))
        assert record.state == ModuleAutoCompileState.BLOCKED
        assert leaf._compiled_call_impl is None

    def test_graph_break_partially_compiles_without_replaying_call(self, monkeypatch):
        forward_calls = []
        captured_graphs = []

        class GraphBreakLeaf(nn.Module):
            def forward(self, value):
                forward_calls.append(True)
                intermediate = value + 1
                torch._dynamo.graph_break()
                return intermediate * 2

        from torch_neuronx.neuron_dynamo_backend.backend import _AutoCompileBackendAttempt

        attempt = _AutoCompileBackendAttempt()
        original_compile = torch.compile

        def compile_with_counting_backend(target, **kwargs):
            def counting_backend(graph_module, _example_inputs):
                captured_graphs.append(graph_module)
                return graph_module.forward

            compiled = original_compile(
                target,
                backend=counting_backend,
                fullgraph=False,
                dynamic=False,
            )
            attempt.record_fallback("forward", "graph_break")
            setattr(compiled, mac._BACKEND_ATTEMPT_ATTR, attempt)
            return compiled

        monkeypatch.setattr(mac, "_compile_isolated_callable", compile_with_counting_backend)
        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *_args: True)
        leaf = GraphBreakLeaf()
        record = _ready_record(leaf)
        _promote_module(leaf, record)

        torch.testing.assert_close(leaf(torch.tensor(1.0)), torch.tensor(4.0))

        assert forward_calls == [True]
        assert captured_graphs
        assert record.state == ModuleAutoCompileState.BLOCKED
        assert leaf._compiled_call_impl is None

    def test_compiled_failure_with_python_side_effect_is_not_replayed(self, monkeypatch):
        class CompileError(Exception):
            pass

        calls = []

        class Leaf(nn.Module):
            def forward(self, value):
                calls.append("eager")
                return value + 1

        def fake_compiled(value):
            calls.append("compiled-side-effect")
            raise CompileError()

        leaf = Leaf()
        record = _ready_record(leaf)
        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *_args: True)
        monkeypatch.setattr(
            mac,
            "_compile_isolated_callable",
            lambda *_args, **_kwargs: _reported_callable(fake_compiled)[0],
        )
        _promote_module(leaf, record)

        with pytest.raises(CompileError):
            leaf(torch.zeros(1))

        assert calls == ["compiled-side-effect"]
        assert record.state == ModuleAutoCompileState.BLOCKED
        torch.testing.assert_close(leaf(torch.zeros(1)), torch.ones(1))
        assert calls == ["compiled-side-effect", "eager"]

    def test_escaped_backend_failure_is_not_replayed(self, monkeypatch):
        stablehlo_error = type(
            "_StableHLOError",
            (Exception,),
            {"__module__": "torch_neuronx.neuron_dynamo_backend.backend"},
        )
        calls = []

        class Leaf(nn.Module):
            def forward(self, value):
                calls.append("eager")
                return value + 1

        def fake_compiled(_value):
            calls.append("compiled")
            try:
                raise stablehlo_error()
            except stablehlo_error as cause:
                raise RuntimeError("backend compilation failed") from cause

        leaf = Leaf()
        record = _ready_record(leaf)
        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *_args: True)
        monkeypatch.setattr(
            mac,
            "_compile_isolated_callable",
            lambda *_args, **_kwargs: _reported_callable(fake_compiled)[0],
        )
        _promote_module(leaf, record)

        with pytest.raises(RuntimeError, match="backend compilation failed"):
            leaf(torch.zeros(1))
        assert calls == ["compiled"]
        assert record.state == ModuleAutoCompileState.BLOCKED
        assert record.terminal_reason == "first_compiled_call_failed"
        assert leaf._compiled_call_impl is None

    def test_compiled_wrapper_uses_eager_during_symbolic_trace(self, monkeypatch):
        """A promoted module must stay traceable by falling back to eager under
        symbolic/JIT tracing contexts."""

        class Leaf(nn.Module):
            def forward(self, x):
                return x + 1

        def fake_compiled(*_a, **_k):
            raise AssertionError("compiled callable should not run while tracing")

        def fake_compile(_fn, **_kwargs):
            return fake_compiled

        leaf = Leaf()
        record = _ready_record(leaf)
        monkeypatch.setattr(mac.torch, "compile", fake_compile)
        _promote_module(leaf, record)
        wrapper = leaf._compiled_call_impl
        monkeypatch.setattr(
            mac,
            "_runtime_context_flags",
            lambda **_kwargs: mac._RUNTIME_CONTEXT_SYMBOLIC_TRACE,
        )
        torch.testing.assert_close(wrapper(torch.tensor(1.0)), torch.tensor(2.0))

    @pytest.mark.parametrize("mode", ("disabled", "cpu", "empty", "functorch"))
    def test_promoted_wrapper_uses_eager_when_call_is_ineligible(self, monkeypatch, mode):
        calls = []

        class Leaf(nn.Module):
            def forward(self, x):
                calls.append("eager")
                return x + 1

        def fake_compiled(x):
            calls.append("compiled")
            return x + 99

        def fake_compile(_fn, **_kwargs):
            return fake_compiled

        leaf = Leaf()
        record = _ready_record(leaf)
        monkeypatch.setattr(mac.torch, "compile", fake_compile)
        _promote_module(leaf, record)
        wrapper = leaf._compiled_call_impl
        assert wrapper is not None

        if mode == "disabled":
            monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "0")
        elif mode in {"empty", "functorch"}:
            monkeypatch.setattr(mac, "_tensor_is_neuron", lambda v: isinstance(v, torch.Tensor))
            if mode == "functorch":
                monkeypatch.setattr(mac, "_functorch_transform_active", lambda: True)

        input_tensor = torch.empty(0) if mode == "empty" else torch.tensor(1.0)
        output = wrapper(input_tensor)
        torch.testing.assert_close(output, input_tensor + 1)
        assert calls == ["eager"]

    def test_disabled_feature_stops_bootstrap_promotion(self, monkeypatch):
        """The bootstrap hooks outlive a disable, so they must re-check the gate:
        a mid-bootstrap module must NOT promote once the feature is turned off."""

        leaf = nn.LayerNorm(10)
        record = _bootstrap_record(leaf)
        record.pending_promotion = True  # queued to promote on next pre-hook

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "0")
        _bootstrap_forward_pre_hook(leaf, (torch.randn(2, 10),), None)
        # Gate off -> no promotion; the inert bootstrap owner remains installed.
        assert record.state == ModuleAutoCompileState.BOOTSTRAP_PENDING
        assert mac._bootstrap_compiled_call_is_owned(leaf)

    def test_late_op_gate_does_not_strand_compiled_callable(self):
        """The late unsupported-op gate must not fire on a PROMOTED_PENDING record
        (which already has _compiled_call_impl), else it would leave that callable
        installed on a BLOCKED record."""

        def sentinel(*a, **k):
            return None

        leaf = nn.LayerNorm(10)
        record, _ = _get_or_create_record(leaf)
        # Simulate a promoted-but-not-yet-run module.
        record.state = ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
        leaf._compiled_call_impl = sentinel

        # An unsupported op is attributed after promotion; the gate must NOT block
        # here (state is post-promotion), leaving the callable intact.
        _classify_and_transition(leaf, record, (leaf,))
        assert record.state == ModuleAutoCompileState.PROMOTED_PENDING_FIRST_COMPILED_CALL
        assert leaf._compiled_call_impl is sentinel


class TestRereviewRegressions:
    @staticmethod
    def _promote(monkeypatch, module, compiled, *, backward=False):
        from torch_neuronx.neuron_dynamo_backend.backend import _AutoCompileBackendAttempt

        attempt = _AutoCompileBackendAttempt()
        attempt.record_executor("forward", 3, "forward-key")
        if backward:
            attempt.record_executor("backward", 2, "backward-key")

        @functools.wraps(compiled)
        def reported(*args, **kwargs):
            return compiled(*args, **kwargs)

        setattr(reported, mac._BACKEND_ATTEMPT_ATTR, attempt)
        monkeypatch.setattr(mac, "_compile_module_call_impl", lambda _module: reported)
        monkeypatch.setattr(mac, "_call_contains_neuron_tensor", lambda *_args: True)
        record = _ready_record(module)
        _promote_module(module, record)
        return record, attempt

    def test_output_lifecycle_preserves_tensor_hooks_and_deduplicates_root(
        self,
        monkeypatch,
    ):
        events = []

        def first_hook(gradient):
            events.append("first")
            return gradient

        def second_hook(gradient):
            events.append("second")
            return gradient

        class Leaf(nn.Module):
            def forward(self, value):
                output = value.sin()
                output.register_hook(first_hook)
                output.register_hook(second_hook)
                return output, output

        leaf = Leaf()
        record, _attempt = self._promote(
            monkeypatch,
            leaf,
            leaf.forward,
            backward=True,
        )

        first, second = leaf(torch.randn(4, requires_grad=True))

        assert tuple(first._backward_hooks.values()) == (first_hook, second_hook)
        assert record.pending_backward_count == 1
        (first + second).sum().backward()
        assert events == ["first", "second"]
        assert record.pending_backward_count == 0

    def test_compiled_retained_graph_drains_after_final_backward(self, monkeypatch):
        leaf = nn.Sigmoid()
        record, _attempt = self._promote(
            monkeypatch,
            leaf,
            leaf.forward,
            backward=True,
        )
        value = torch.randn(4, requires_grad=True)
        output = leaf(value)

        first_gradient = torch.autograd.grad(output.sum(), value, retain_graph=True)[0]
        assert record.pending_backward_count == 1
        mac._request_terminal_eager_locked(
            leaf,
            record,
            mac._TERMINAL_REASON_BACKEND_FALLBACK,
        )
        assert record.state == ModuleAutoCompileState.DRAINING_TO_EAGER
        assert leaf._compiled_call_impl is not None

        second_gradient = torch.autograd.grad(output.sum(), value)[0]

        torch.testing.assert_close(second_gradient, first_gradient)
        assert record.pending_backward_count == 0
        assert record.state == ModuleAutoCompileState.BLOCKED
        assert leaf._compiled_call_impl is None

    def test_budget_expiry_waits_for_live_compiled_graph(self, monkeypatch):
        leaf = nn.SiLU()
        record, _attempt = self._promote(monkeypatch, leaf, leaf.forward)
        value = torch.randn(4, requires_grad=True)
        output = leaf(value)

        with torch.no_grad():
            for _ in range(_OBSERVATION_CALL_BUDGET):
                leaf(torch.randn(4))

        assert record.state == ModuleAutoCompileState.DRAINING_TO_EAGER
        assert record.pending_backward_count == 1
        assert leaf._compiled_call_impl is not None

        output.sum().backward()

        assert record.state == ModuleAutoCompileState.BLOCKED
        assert leaf._compiled_call_impl is None

    def test_backend_failure_waits_for_live_compiled_graph(self, monkeypatch):
        leaf = nn.Tanh()
        record, attempt = self._promote(monkeypatch, leaf, leaf.forward)
        output = leaf(torch.randn(4, requires_grad=True))

        attempt.record_failure("forward", "runtime_execution")

        assert record.state == ModuleAutoCompileState.DRAINING_TO_EAGER
        assert record.pending_backward_count == 1
        assert leaf._compiled_call_impl is not None

        output.sum().backward()

        assert record.state == ModuleAutoCompileState.BLOCKED
        assert leaf._compiled_call_impl is None

    def test_reentrant_compiled_token_lives_through_outer_graph_task(self, monkeypatch):
        from torch.utils.checkpoint import checkpoint

        observed_counts = []

        class OuterProbe(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.clone()

            @staticmethod
            def backward(ctx, gradient):
                observed_counts.append(record.pending_reentrant_checkpoint_count)
                return gradient

        class Leaf(nn.Module):
            def forward(self, value):
                return value.sin().cos()

        leaf = Leaf()
        record, _attempt = self._promote(
            monkeypatch,
            leaf,
            leaf.forward,
            backward=True,
        )
        value = torch.randn(4, requires_grad=True)

        output = checkpoint(leaf, OuterProbe.apply(value), use_reentrant=True)
        assert record.pending_reentrant_checkpoint_count == 1
        output.sum().backward()

        assert observed_counts == [1]
        assert record.pending_reentrant_checkpoint_count == 0
        assert value.grad is not None

    def test_cross_entropy_fake_dispatch_is_autocompile_scoped(self, monkeypatch):
        from torch_neuronx.python_ops.torch_mlir.ops import loss

        op = torch.ops.aten.cross_entropy_loss.default
        dispatch_keys = []

        def record_dispatch(dispatch_key, *_args, **_kwargs):
            dispatch_keys.append(dispatch_key)
            return dispatch_key

        monkeypatch.setattr(op, "_op_dk", record_dispatch)
        monkeypatch.setattr(torch._C, "_get_dispatch_mode", lambda _key: object())
        logits = torch.randn(2, 4)
        targets = torch.tensor([0, 1])

        monkeypatch.setattr(mac, "_internal_module_compile_active", lambda: False)
        loss.cross_entropy_autograd_privateuse1(logits, targets)
        assert dispatch_keys[-1] == torch._C.DispatchKey.PrivateUse1

        monkeypatch.setattr(mac, "_internal_module_compile_active", lambda: True)
        loss.cross_entropy_autograd_privateuse1(logits, targets)
        assert dispatch_keys[-1] == torch._C.DispatchKey.CompositeImplicitAutograd

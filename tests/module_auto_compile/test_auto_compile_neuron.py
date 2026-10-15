"""Integration tests for module auto-compile on Neuron hardware (requires a single NeuronCore)."""

import multiprocessing

import pytest
import torch
import torch._dynamo as dynamo
import torch.nn as nn

import torch_neuronx
from torch_neuronx import module_auto_compile as mac


@pytest.fixture(autouse=True)
def clear_state():
    """Reset AutoCompile and Dynamo state between tests."""
    mac._reset_feature_disabled()
    mac._reset_state()
    dynamo.reset()
    dynamo.utils.counters.clear()
    try:
        yield
    finally:
        try:
            torch.neuron.synchronize()
        finally:
            mac._reset_state()
            mac._reset_feature_disabled()
            dynamo.reset()
            dynamo.utils.counters.clear()


def _backward_outputs(output):
    tensors = [
        value
        for value in torch.utils._pytree.tree_leaves(output)
        if isinstance(value, torch.Tensor) and value.requires_grad
    ]
    if not tensors:
        return
    sum(value.float().sum() for value in tensors).backward()


def _clear_call_gradients(module, args, kwargs):
    module.zero_grad(set_to_none=True)
    for value in torch.utils._pytree.tree_leaves((args, kwargs)):
        if isinstance(value, torch.Tensor) and value.is_leaf and value.grad is not None:
            value.grad = None


def _run_until_compiled(
    module,
    *args,
    compiled_modules=None,
    max_attempts=mac._PROMOTION_INVOCATION_THRESHOLD + 8,
    **kwargs,
):
    targets = (module,) if compiled_modules is None else tuple(compiled_modules)
    output = None
    for _ in range(max_attempts):
        output = module(*args, **kwargs)
        _backward_outputs(output)
        _clear_call_gradients(module, args, kwargs)
        if all(
            (record := mac._MODULE_REGISTRY.get(target)) is not None
            and record.state == mac.ModuleAutoCompileState.COMPILED
            for target in targets
        ):
            for target in targets:
                record = mac._MODULE_REGISTRY[target]
                attempt = record.compile_attempt
                assert attempt is not None
                assert attempt.has_real_forward_executor()
                if record.requires_backward_executor:
                    assert attempt.has_real_backward_executor()
            return output

    states = {
        f"{type(target).__name__}@{id(target):x}": (
            None
            if (record := mac._MODULE_REGISTRY.get(target)) is None
            else (record.state.value, record.terminal_reason)
        )
        for target in targets
    }
    pytest.fail(
        f"modules did not compile in {max_attempts} calls: "
        f"states={states}, dynamo={dict(dynamo.utils.counters)}"
    )


def _run_until_blocked(
    module,
    *args,
    reason=mac._TERMINAL_REASON_BACKEND_FALLBACK,
    max_attempts=mac._PROMOTION_INVOCATION_THRESHOLD + 8,
    **kwargs,
):
    for _ in range(max_attempts):
        output = module(*args, **kwargs)
        _backward_outputs(output)
        _clear_call_gradients(module, args, kwargs)
        record = mac._MODULE_REGISTRY.get(module)
        if record is not None and record.state == mac.ModuleAutoCompileState.BLOCKED:
            assert record.terminal_reason == reason
            assert module._compiled_call_impl is None
            return output

    record = mac._MODULE_REGISTRY.get(module)
    pytest.fail(
        f"module did not become terminal eager in {max_attempts} calls: "
        f"state={None if record is None else record.state.value}, "
        f"reason={None if record is None else record.terminal_reason}"
    )


@pytest.mark.single_device
class TestSingleDeviceAutoCompile:
    """Tests that run on a single NeuronCore."""

    def test_bad_fork_hard_disables_feature(self):
        assert torch_neuronx.is_neuron_runtime_initialized()
        context = multiprocessing.get_context("fork")
        reader, writer = context.Pipe(duplex=False)

        def inspect_child():
            writer.send(
                (
                    torch_neuronx._is_in_bad_fork,
                    mac.module_auto_compile_enabled(),
                    mac._FEATURE_HARD_DISABLED,
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
        assert result == (True, False, True)

    def test_compound_leaf_gets_real_forward_and_backward_executors(self):
        class CompoundLeaf(nn.Module):
            def forward(self, value):
                return value.sin().cos() + value.square()

        model = CompoundLeaf().neuron()
        x = torch.randn(4, 64, device="neuron:0", requires_grad=True)

        _run_until_compiled(model, x)

        record = mac._MODULE_REGISTRY.get(model)
        assert record is not None
        assert record.state == mac.ModuleAutoCompileState.COMPILED
        assert model._compiled_call_impl is not None
        assert record.compile_attempt.has_real_forward_executor()
        assert record.compile_attempt.has_real_backward_executor()

    def test_higher_order_backward_uses_compatibility_path(self):
        class CompoundLeaf(nn.Module):
            def forward(self, value):
                return value.sin() + value.square()

        model = CompoundLeaf().to("neuron:0")
        warmup = torch.randn(4, 64, device="neuron:0", requires_grad=True)
        _run_until_compiled(model, warmup)

        record = mac._MODULE_REGISTRY[model]
        attempt = record.compile_attempt
        value = torch.randn(4, 64, device="neuron:0", requires_grad=True)
        output = model(value)
        gradient = torch.autograd.grad(output.sum(), value, create_graph=True)[0]
        second_gradient = torch.autograd.grad(gradient.sum(), value)[0]

        assert record.state == mac.ModuleAutoCompileState.COMPILED
        assert record.terminal_reason is None
        assert model._compiled_call_impl is not None
        assert ("backward", "higher_order_autograd") in attempt.compatibility_reasons
        torch.testing.assert_close(
            second_gradient,
            2 - value.sin(),
            rtol=1e-2,
            atol=1e-2,
        )

        eager_value = torch.randn(4, 64, device="neuron:0", requires_grad=True)
        eager_gradient = torch.autograd.grad(
            model(eager_value).sum(),
            eager_value,
            create_graph=True,
        )[0]
        torch.testing.assert_close(
            torch.autograd.grad(eager_gradient.sum(), eager_value)[0],
            2 - eager_value.sin(),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_checkpoint_forward_and_gradients_match_eager(self, monkeypatch):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
        )

        class CheckpointLeaf(nn.Module):
            def forward(self, value):
                sine = value.sin()
                return sine.cos() + sine.square()

        torch.manual_seed(44)
        source = torch.randn(4, 64, device="neuron:0")

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "0")
        eager_model = checkpoint_wrapper(CheckpointLeaf()).to("neuron:0")
        eager_input = source.detach().clone().requires_grad_(True)
        eager_output = eager_model(eager_input)
        eager_output.float().sum().backward()

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "1")
        compiled_leaf = CheckpointLeaf()
        compiled_model = checkpoint_wrapper(compiled_leaf).to("neuron:0")
        warmup_input = source.detach().clone().requires_grad_(True)
        _run_until_compiled(
            compiled_model,
            warmup_input,
            compiled_modules=(compiled_leaf,),
        )

        compiled_input = source.detach().clone().requires_grad_(True)
        compiled_output = compiled_model(compiled_input)
        compiled_output.float().sum().backward()

        assert mac._MODULE_REGISTRY.get(compiled_model) is None
        record = mac._MODULE_REGISTRY.get(compiled_leaf)
        assert record is not None
        assert record.state == mac.ModuleAutoCompileState.COMPILED
        assert record.compile_attempt.has_real_forward_executor()
        assert record.compile_attempt.has_real_backward_executor()
        torch.testing.assert_close(compiled_output, eager_output, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(compiled_input.grad, eager_input.grad, rtol=1e-2, atol=1e-2)

    def test_selective_checkpoint_policy_and_gradients_match_eager(self, monkeypatch):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
        )
        from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

        class CheckpointLeaf(nn.Module):
            def forward(self, value):
                sine = value.sin()
                return sine.cos() + sine.square()

        save_op = torch.ops.aten.sin.default

        def make_context_fn(trace):
            def policy_fn(ctx, op, *_args, **_kwargs):
                decision = (
                    CheckpointPolicy.MUST_SAVE
                    if op == save_op
                    else CheckpointPolicy.PREFER_RECOMPUTE
                )
                trace.append((ctx.is_recompute, str(op), decision.name))
                return decision

            return lambda: create_selective_checkpoint_contexts(policy_fn)

        def make_model(trace):
            leaf = CheckpointLeaf()
            model = checkpoint_wrapper(
                leaf,
                context_fn=make_context_fn(trace),
            )
            return model, leaf

        def run_model(module, value):
            output = module(value)
            output.float().sum().backward()
            return output

        torch.manual_seed(45)
        source = torch.randn(4, 64, device="neuron:0")

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "0")
        eager_trace = []
        eager_model, _eager_leaf = make_model(eager_trace)
        eager_model.to("neuron:0")
        eager_input = source.detach().clone().requires_grad_(True)
        eager_output = run_model(eager_model, eager_input)

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "1")
        compiled_trace = []
        compiled_model, compiled_leaf = make_model(compiled_trace)
        compiled_model.to("neuron:0")
        warmup_input = source.detach().clone().requires_grad_(True)
        _run_until_compiled(
            compiled_model,
            warmup_input,
            compiled_modules=(compiled_leaf,),
        )

        assert mac._MODULE_REGISTRY.get(compiled_model) is None
        record = mac._MODULE_REGISTRY[compiled_leaf]

        attempt = record.compile_attempt
        assert attempt.has_real_forward_executor()
        assert attempt.has_real_backward_executor()
        expected_policy_trace = tuple(
            (op, decision) for is_recompute, op, decision in eager_trace if not is_recompute
        )
        assert attempt.forward_sac_policy_trace() == expected_policy_trace

        compiled_trace.clear()
        compiled_input = source.detach().clone().requires_grad_(True)
        compiled_output = run_model(compiled_model, compiled_input)

        torch.testing.assert_close(compiled_output, eager_output, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(compiled_input.grad, eager_input.grad, rtol=1e-2, atol=1e-2)
        assert any(
            decision == CheckpointPolicy.MUST_SAVE.name for _op, decision in expected_policy_trace
        )
        assert any(
            decision == CheckpointPolicy.PREFER_RECOMPUTE.name
            for _op, decision in expected_policy_trace
        )

    def test_one_op_linear_terminalizes_eager_and_matches_gradients(self, monkeypatch):
        torch.manual_seed(42)
        eager_model = nn.Linear(64, 32)
        compiled_model = nn.Linear(64, 32)
        compiled_model.load_state_dict(eager_model.state_dict())
        input_data = torch.randn(4, 64, device="neuron:0")
        grad_output = torch.randn(4, 32, device="neuron:0")

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "0")
        eager_model.to("neuron:0")
        eager_input = input_data.detach().clone().requires_grad_(True)
        eager_output = eager_model(eager_input)
        (eager_output * grad_output).sum().backward()

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "1")
        compiled_model.to("neuron:0")
        _run_until_blocked(compiled_model, input_data)
        record = mac._MODULE_REGISTRY.get(compiled_model)
        assert record is not None
        assert record.state == mac.ModuleAutoCompileState.BLOCKED

        compiled_model.zero_grad(set_to_none=True)
        compiled_input = input_data.detach().clone().requires_grad_(True)
        compiled_output = compiled_model(compiled_input)
        (compiled_output * grad_output).sum().backward()

        torch.testing.assert_close(compiled_output, eager_output, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(compiled_input.grad, eager_input.grad, rtol=1e-2, atol=1e-2)
        for (eager_name, eager_parameter), (compiled_name, compiled_parameter) in zip(
            eager_model.named_parameters(),
            compiled_model.named_parameters(),
            strict=True,
        ):
            assert eager_name == compiled_name
            torch.testing.assert_close(
                compiled_parameter.grad,
                eager_parameter.grad,
                rtol=1e-2,
                atol=1e-2,
            )

    def test_resident_neuron_module_preserves_cpu_input_error(self, monkeypatch):
        class CompoundLinear(nn.Linear):
            def forward(self, value):
                return super().forward(value).square()

        model = CompoundLinear(8, 8).to("neuron:0")
        neuron_input = torch.randn(2, 8, device="neuron:0")
        cpu_input = torch.randn(2, 8)

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "0")
        with pytest.raises(Exception) as eager_error:
            model(cpu_input)

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "1")
        _run_until_compiled(model, neuron_input)
        record = mac._MODULE_REGISTRY.get(model)
        assert record is not None
        assert record.state == mac.ModuleAutoCompileState.COMPILED

        with pytest.raises(type(eager_error.value)) as compiled_error:
            model(cpu_input)

        assert str(compiled_error.value) == str(eager_error.value)
        assert record.state == mac.ModuleAutoCompileState.BLOCKED
        assert model._compiled_call_impl is None

        expected = model.forward(neuron_input)
        torch.testing.assert_close(model(neuron_input), expected)
        assert record.state == mac.ModuleAutoCompileState.BLOCKED
        assert record.terminal_reason == mac._TERMINAL_REASON_BACKEND_FALLBACK
        assert model._compiled_call_impl is None

    def test_compound_embedding_forward_and_backward_match_eager(self, monkeypatch):
        class CompoundEmbedding(nn.Embedding):
            def forward(self, indices):
                return super().forward(indices).square()

        torch.manual_seed(43)
        eager_model = CompoundEmbedding(128, 32)
        compiled_model = CompoundEmbedding(128, 32)
        compiled_model.load_state_dict(eager_model.state_dict())
        indices = torch.tensor([[1, 3, 7], [2, 3, 9]], device="neuron:0")
        grad_output = torch.randn(2, 3, 32, device="neuron:0")

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "0")
        eager_model.to("neuron:0")
        eager_output = eager_model(indices)
        (eager_output * grad_output).sum().backward()

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "1")
        compiled_model.to("neuron:0")
        _run_until_compiled(compiled_model, indices)
        record = mac._MODULE_REGISTRY.get(compiled_model)
        assert record is not None
        assert record.state == mac.ModuleAutoCompileState.COMPILED

        compiled_model.zero_grad(set_to_none=True)
        compiled_output = compiled_model(indices)
        (compiled_output * grad_output).sum().backward()

        torch.testing.assert_close(compiled_output, eager_output, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(
            compiled_model.weight.grad,
            eager_model.weight.grad,
            rtol=1e-2,
            atol=1e-2,
        )

    def test_unsupported_embedding_bag_backward_terminalizes_eager(self, monkeypatch):
        class CompoundEmbeddingBag(nn.EmbeddingBag):
            def forward(self, indices, offsets):
                return super().forward(indices, offsets).square()

        torch.manual_seed(46)
        eager_model = CompoundEmbeddingBag(128, 32, mode="sum")
        model = CompoundEmbeddingBag(128, 32, mode="sum")
        model.load_state_dict(eager_model.state_dict())
        indices = torch.tensor([1, 3, 7, 2, 3, 9], device="neuron:0")
        offsets = torch.tensor([0, 3], device="neuron:0")

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "0")
        eager_model.to("neuron:0")
        expected = eager_model(indices, offsets)
        expected.float().sum().backward()

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "1")
        model.to("neuron:0")
        _run_until_blocked(model, indices, offsets)
        record = mac._MODULE_REGISTRY.get(model)
        assert record is not None
        assert record.state == mac.ModuleAutoCompileState.BLOCKED

        model.zero_grad(set_to_none=True)
        actual = model(indices, offsets)
        actual.float().sum().backward()
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(model.weight.grad, eager_model.weight.grad, rtol=1e-2, atol=1e-2)

    def test_functional_embedding_leaf_compiles(self):
        class FunctionalEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(128, 32))

            def forward(self, indices):
                return nn.functional.embedding(indices, self.weight).square()

        model = FunctionalEmbedding().to("neuron:0")
        indices = torch.tensor([[1, 2, 3]], device="neuron:0")
        _run_until_compiled(model, indices)

        record = mac._MODULE_REGISTRY.get(model)
        assert record is not None
        assert record.state == mac.ModuleAutoCompileState.COMPILED

    def test_leaf_graph_module_forward_and_backward_match_eager(self, monkeypatch):
        class FunctionalLinearSquare(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(32, 64))
                self.bias = nn.Parameter(torch.randn(32))

            def forward(self, value):
                return nn.functional.linear(value, self.weight, self.bias).square()

        torch.manual_seed(44)
        eager_model = torch.fx.symbolic_trace(FunctionalLinearSquare())
        compiled_model = torch.fx.symbolic_trace(FunctionalLinearSquare())
        compiled_model.load_state_dict(eager_model.state_dict())
        input_data = torch.randn(4, 64, device="neuron:0")
        grad_output = torch.randn(4, 32, device="neuron:0")

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "0")
        eager_model.to("neuron:0")
        eager_input = input_data.detach().clone().requires_grad_(True)
        eager_output = eager_model(eager_input)
        (eager_output * grad_output).sum().backward()

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "1")
        compiled_model.to("neuron:0")
        _run_until_compiled(compiled_model, input_data)

        record = mac._MODULE_REGISTRY.get(compiled_model)
        assert record is not None
        assert record.state == mac.ModuleAutoCompileState.COMPILED

        compiled_model.zero_grad(set_to_none=True)
        compiled_input = input_data.detach().clone().requires_grad_(True)
        compiled_output = compiled_model(compiled_input)
        (compiled_output * grad_output).sum().backward()

        torch.testing.assert_close(compiled_output, eager_output, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(compiled_input.grad, eager_input.grad, rtol=1e-2, atol=1e-2)
        for (eager_name, eager_parameter), (compiled_name, compiled_parameter) in zip(
            eager_model.named_parameters(),
            compiled_model.named_parameters(),
            strict=True,
        ):
            assert eager_name == compiled_name
            torch.testing.assert_close(
                compiled_parameter.grad,
                eager_parameter.grad,
                rtol=1e-2,
                atol=1e-2,
            )

    def test_one_op_mse_loss_terminalizes_eager_and_matches_gradients(self, monkeypatch):
        prediction = torch.randn(8, 16, device="neuron:0", requires_grad=True)
        target = torch.randn(8, 16, device="neuron:0")

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "0")
        eager_prediction = prediction.detach().clone().requires_grad_(True)
        expected = nn.MSELoss().to("neuron:0")(eager_prediction, target)
        expected.backward()

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "1")
        model = nn.MSELoss().to("neuron:0")
        warmup_prediction = prediction.detach().clone().requires_grad_(True)
        _run_until_blocked(model, warmup_prediction, target)

        record = mac._MODULE_REGISTRY.get(model)
        assert record is not None
        assert record.state == mac.ModuleAutoCompileState.BLOCKED

        compiled_prediction = prediction.detach().clone().requires_grad_(True)
        output = model(compiled_prediction, target)
        output.backward()
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(
            compiled_prediction.grad,
            eager_prediction.grad,
            rtol=1e-2,
            atol=1e-2,
        )

    def test_batch_norm_cumulative_average_stays_eager(self):
        model = nn.BatchNorm1d(8, momentum=None).to("neuron:0")
        model.train()
        inputs = torch.randn(4, 8, device="neuron:0")

        for _ in range(4):
            model(inputs)

        record = mac._MODULE_REGISTRY.get(model)
        assert record is not None
        assert record.state == mac.ModuleAutoCompileState.BLOCKED
        assert record.terminal_reason == mac._TERMINAL_REASON_COMPILER_CAPABILITY_UNAVAILABLE
        assert model._compiled_call_impl is None
        assert model.num_batches_tracked.cpu().item() == 4

    def test_cross_entropy_loss_compiles_and_matches_eager(self, monkeypatch):
        logits = torch.randn(8, 16, device="neuron:0")
        targets = torch.randint(0, 16, (8,), device="neuron:0")

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "0")
        eager_logits = logits.detach().clone().requires_grad_(True)
        expected = nn.CrossEntropyLoss().to("neuron:0")(eager_logits, targets)
        expected.backward()

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "1")
        model = nn.CrossEntropyLoss().to("neuron:0")
        warmup_logits = logits.detach().clone().requires_grad_(True)
        _run_until_compiled(model, warmup_logits, targets)

        record = mac._MODULE_REGISTRY.get(model)
        assert record is not None
        assert record.state == mac.ModuleAutoCompileState.COMPILED
        assert model._compiled_call_impl is not None

        compiled_logits = logits.detach().clone().requires_grad_(True)
        output = model(compiled_logits, targets)
        output.backward()
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(
            compiled_logits.grad,
            eager_logits.grad,
            rtol=1e-2,
            atol=1e-2,
        )

    def test_tp_layout_sdpa_wrapper_stays_eager_and_preserves_view(self):
        class ScaledDotProductAttentionWrapper(nn.Module):
            def forward(self, query, key, value):
                return nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    is_causal=True,
                    dropout_p=0.0,
                )

        batch_size = 1
        sequence_length = 16
        num_heads = 4
        head_dim = 16
        inputs = [
            torch.randn(
                batch_size,
                sequence_length,
                num_heads,
                head_dim,
                device="neuron:0",
            ).transpose(1, 2)
            for _ in range(3)
        ]
        assert all(not value.is_contiguous() for value in inputs)

        model = ScaledDotProductAttentionWrapper().to("neuron:0")
        output = None
        for _ in range(3):
            output = model(*inputs)

        record = mac._MODULE_REGISTRY.get(model)
        assert record is not None
        assert record.state == mac.ModuleAutoCompileState.BOOTSTRAP_PENDING
        assert mac._bootstrap_compiled_call_is_owned(model)
        assert record.bootstrap_forward_pre_handle is not None
        reshaped = output.transpose(1, 2).view(batch_size, sequence_length, -1)
        assert reshaped.shape == (batch_size, sequence_length, num_heads * head_dim)

    def test_no_recompilation(self):
        """After initial compile, no new graphs should be captured."""
        model = nn.LayerNorm(64).to("neuron:0")
        x = torch.randn(4, 64, device="neuron:0")

        _run_until_compiled(model, x)

        # Record current compile count
        count_before = dynamo.utils.counters["stats"]["calls_captured"]
        assert count_before > 0

        for _ in range(10):
            model(x)

        count_after = dynamo.utils.counters["stats"]["calls_captured"]
        assert (
            count_after == count_before
        ), f"Recompilation detected: {count_after - count_before} new compilations"

    def test_backward_matches_eager_after_compile(self, monkeypatch):
        """Compiled output, input gradients, and parameter gradients match eager."""
        torch.manual_seed(42)
        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "0")
        eager_model = nn.LayerNorm(64)
        compiled_model = nn.LayerNorm(64)
        compiled_model.load_state_dict(eager_model.state_dict())
        eager_model.to("neuron:0")
        input_data = torch.randn(4, 64, device="neuron:0")
        grad_output = torch.randn(4, 64, device="neuron:0")

        eager_input = input_data.detach().clone().requires_grad_(True)
        eager_output = eager_model(eager_input)
        (eager_output * grad_output).sum().backward()
        eager_input_grad = eager_input.grad.detach().clone()
        eager_param_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in eager_model.named_parameters()
        }

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "1")
        compiled_model.to("neuron:0")
        warmup_input = input_data.detach()
        _run_until_compiled(compiled_model, warmup_input)

        record = mac._MODULE_REGISTRY.get(compiled_model)
        assert record.state == mac.ModuleAutoCompileState.COMPILED

        compiled_model.zero_grad(set_to_none=True)
        compiled_input = input_data.detach().clone().requires_grad_(True)
        compiled_output = compiled_model(compiled_input)
        (compiled_output * grad_output).sum().backward()

        torch.testing.assert_close(compiled_output, eager_output, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(compiled_input.grad, eager_input_grad, rtol=1e-2, atol=1e-2)
        for name, parameter in compiled_model.named_parameters():
            assert parameter.grad is not None
            torch.testing.assert_close(
                parameter.grad, eager_param_grads[name], rtol=1e-2, atol=1e-2
            )

    def test_multi_module_model(self):
        """Multiple leaf modules in a model should each be compiled."""

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(64)
                self.act = CompoundActivation()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                return self.linear(self.act(self.norm(x)))

        class CompoundActivation(nn.Module):
            def forward(self, value):
                return nn.functional.silu(value) * value

        model = MLP().to("neuron:0")
        x = torch.randn(4, 64, device="neuron:0", requires_grad=True)
        _run_until_compiled(model, x, compiled_modules=(model.norm, model.act))

        norm_record = mac._MODULE_REGISTRY.get(model.norm)
        act_record = mac._MODULE_REGISTRY.get(model.act)
        linear_record = mac._MODULE_REGISTRY.get(model.linear)

        assert norm_record.state == mac.ModuleAutoCompileState.COMPILED
        assert act_record.state == mac.ModuleAutoCompileState.COMPILED
        assert linear_record.state == mac.ModuleAutoCompileState.BLOCKED
        assert linear_record.terminal_reason == mac._TERMINAL_REASON_BACKEND_FALLBACK
        assert model.linear._compiled_call_impl is None
        assert dynamo.utils.counters["stats"]["calls_captured"] > 0

    def test_to_empty_materialization_discovers_and_compiles_leaf_tree(self):
        """Meta-initialized training models are materialized with Module.to_empty."""

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(64)
                self.act = CompoundActivation()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                return self.linear(self.act(self.norm(x)))

        class CompoundActivation(nn.Module):
            def forward(self, value):
                return nn.functional.silu(value) * value

        with torch.device("meta"):
            model = MLP()
        model.to_empty(device="neuron:0")

        leaves = (model.norm, model.act, model.linear)
        for leaf in leaves:
            record = mac._MODULE_REGISTRY.get(leaf)
            assert record is not None
            assert record.state == mac.ModuleAutoCompileState.BOOTSTRAP_PENDING

        for module in model.modules():
            reset_parameters = getattr(module, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()

        x = torch.randn(4, 64, device="neuron:0", requires_grad=True)
        _run_until_compiled(model, x, compiled_modules=(model.norm, model.act))

        for leaf in (model.norm, model.act):
            record = mac._MODULE_REGISTRY.get(leaf)
            assert record is not None
            assert record.state == mac.ModuleAutoCompileState.COMPILED
            assert leaf._compiled_call_impl is not None
        linear_record = mac._MODULE_REGISTRY[model.linear]
        assert linear_record.state == mac.ModuleAutoCompileState.BLOCKED
        assert linear_record.terminal_reason == mac._TERMINAL_REASON_BACKEND_FALLBACK
        assert model.linear._compiled_call_impl is None

    def test_disabled_by_env(self, monkeypatch):
        """The disabled feature must not discover a module."""
        # Sanity: feature is on at import (conftest set the env to "1").
        assert mac.module_auto_compile_enabled()

        monkeypatch.setenv("TORCH_NEURONX_MODULE_AUTO_COMPILE", "0")
        assert not mac.module_auto_compile_enabled()

        model = nn.LayerNorm(64).to("neuron:0")
        x = torch.randn(4, 64, device="neuron:0")
        for _ in range(8):
            model(x)

        record = mac._MODULE_REGISTRY.get(model)
        assert record is None, (
            "AutoCompile created a record despite TORCH_NEURONX_MODULE_AUTO_COMPILE=0 "
            f"(state={None if record is None else record.state.value})"
        )

    def test_compile_context_skipped(self):
        """A leaf used only inside a torch.compile region must not be
        auto-promoted — _compile_context_active() suppresses observation
        mid-trace."""
        # At rest the predicate is false.
        assert not mac._compile_context_active()

        inner = nn.LayerNorm(64).to("neuron:0")

        def fn(t):
            return inner(t)

        compiled = torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
        x = torch.randn(4, 64, device="neuron:0")
        for _ in range(8):
            compiled(x)

        record = mac._MODULE_REGISTRY.get(inner)
        # Explicit compilation must own this module.
        assert record is None or record.state != mac.ModuleAutoCompileState.COMPILED, (
            "AutoCompile compiled a leaf owned by torch.compile "
            f"(state={None if record is None else record.state.value})"
        )

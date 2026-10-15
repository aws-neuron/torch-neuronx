"""Distributed integration tests for module auto-compile.

Runs under the standard ``tests/distributed`` harness (``pytest --forked -n 8``),
where each xdist worker owns 8 NeuronCores and ``DistributedTester`` spawns
``world_size`` processes — one visible core per process. Topology is TP4/FSDP2
over 8 ranks.

Each spawned worker sees a single core, so models target the bare ``"neuron"``
device (resolves to ``neuron:0`` in-process), not ``neuron:{rank}``.
"""

import math
import os
from collections import Counter

import pytest
import torch
import torch.nn as nn

import torch_neuronx
import torch_neuronx.module_auto_compile as mac
from tests.distributed.collective_ops.base_collective_op import BaseCollectiveOpTest
from torch_neuronx.neuron_dynamo_backend import backend as neuron_backend

pytestmark = pytest.mark.neuron_cores(8)

_DEVICE = "neuron"

_TP_SIZE = 4
_HIDDEN = 1024
_INTERMEDIATE = 2048
_VOCAB = 3000


class LlamaRMSNorm(nn.Module):
    """Simplified RMSNorm for testing (no HF dependency)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(self.weight.dtype)


class SimpleLlamaLayer(nn.Module):
    """Simplified transformer layer for testing."""

    def __init__(self, hidden_size=_HIDDEN, intermediate_size=_INTERMEDIATE):
        super().__init__()
        self.input_norm = LlamaRMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.post_norm = LlamaRMSNorm(hidden_size)
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.input_norm(x)
        x = self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        x = residual + x
        x = self.post_norm(x)
        return x


class SimpleModel(nn.Module):
    def __init__(
        self, num_layers=2, hidden_size=_HIDDEN, intermediate_size=_INTERMEDIATE, vocab_size=_VOCAB
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [SimpleLlamaLayer(hidden_size, intermediate_size) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _build_tp_fsdp_model(
    world_size,
    num_layers,
    hidden_size,
    intermediate_size,
    vocab_size=_VOCAB,
    use_sac=False,
    use_mixed_precision=True,
):
    """Construct a TP4/FSDP2 model and return its checkpoint evidence."""
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed._tensor import init_device_mesh
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        checkpoint,
        create_selective_checkpoint_contexts,
    )

    tp_size = _TP_SIZE
    fsdp_size = world_size // tp_size
    mesh = init_device_mesh(_DEVICE, (fsdp_size, tp_size), mesh_dim_names=("dp", "tp"))

    torch.manual_seed(42)
    model = SimpleModel(
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
    ).to(_DEVICE)

    # Apply TP
    for layer in model.layers:
        parallelize_module(
            layer,
            mesh["tp"],
            {
                "gate_proj": ColwiseParallel(),
                "up_proj": ColwiseParallel(),
                "down_proj": RowwiseParallel(),
            },
        )

    policy_trace = []
    if use_sac:
        save_op = torch.ops.aten.mul.Tensor

        for layer in model.layers:
            original_forward = layer.forward

            def make_checkpoint_forward(forward):
                def policy_fn(ctx, op, *_args, **_kwargs):
                    decision = (
                        CheckpointPolicy.MUST_SAVE
                        if op == save_op
                        else CheckpointPolicy.PREFER_RECOMPUTE
                    )
                    policy_trace.append((ctx.is_recompute, str(op), decision.name))
                    return decision

                def context_fn():
                    return create_selective_checkpoint_contexts(policy_fn)

                def checkpoint_forward(value):
                    return checkpoint(
                        forward,
                        value,
                        use_reentrant=False,
                        context_fn=context_fn,
                    )

                return checkpoint_forward

            layer.forward = make_checkpoint_forward(original_forward)

    checkpoint_leaves = tuple(
        module
        for layer in model.layers
        for module in layer.modules()
        if module is not layer and not any(module.children())
    )

    # Apply FSDP. The main no-crash path uses mixed precision to match the
    # target training setup; the deterministic correctness path keeps full
    # precision to avoid comparing against a numerically unstable baseline.
    fsdp_kwargs = {"mesh": mesh["dp"]}
    if use_mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
        )
    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    eager_owners = tuple(model.layers)
    return model, policy_trace, checkpoint_leaves, eager_owners


def _train_steps(model, steps, vocab_size, seq_len, rank):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = []
    torch.manual_seed(100 + rank)
    for _ in range(steps):
        x = torch.randint(0, vocab_size, (1, seq_len), device=_DEVICE)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), x.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    return optimizer, losses


def _layout_signature(value):
    if value is None:
        return None
    placements = tuple(str(placement) for placement in getattr(value, "placements", ()))
    mesh = getattr(value, "device_mesh", None)
    if mesh is None:
        mesh = getattr(getattr(value, "_spec", None), "mesh", None)
    mesh_tensor = getattr(mesh, "mesh", None)
    mesh_shape = () if mesh_tensor is None else tuple(mesh_tensor.shape)
    return type(value).__name__, placements, mesh_shape


def _local_clone(value):
    to_local = getattr(value, "to_local", None)
    if callable(to_local):
        value = to_local()
    wait = getattr(value, "wait", None)
    if callable(wait) and type(value).__name__ == "AsyncCollectiveTensor":
        value = wait()
    return value.detach().float().cpu().clone()


def _snapshot_model(model):
    parameters = {}
    gradients = {}
    layouts = {}
    for name, parameter in model.named_parameters():
        parameters[name] = _local_clone(parameter)
        gradient = parameter.grad
        gradients[name] = None if gradient is None else _local_clone(gradient)
        layouts[name] = (
            _layout_signature(parameter),
            _layout_signature(gradient),
        )
    return parameters, gradients, layouts


def _assert_real_compiled_records(target_modules, eager_owners):
    compiled_items = [
        (module, record)
        for module, record in mac._MODULE_REGISTRY.items()
        if record.state == mac.ModuleAutoCompileState.COMPILED
    ]
    target_records = [
        mac._MODULE_REGISTRY.get(module)
        for module in target_modules
        if mac._MODULE_REGISTRY.get(module) is not None
    ]
    compiled_target_records = [
        record for record in target_records if record.state == mac.ModuleAutoCompileState.COMPILED
    ]
    assert compiled_target_records, "AutoCompile did not compile a target TP/FSDP leaf"
    assert compiled_items
    for module, record in compiled_items:
        assert mac._module_is_compile_region(module)
        attempt = record.compile_attempt
        assert attempt is not None
        assert attempt.has_real_forward_executor()
        assert attempt.has_real_backward_executor()
    for owner in eager_owners:
        owner_record = mac._MODULE_REGISTRY.get(owner)
        assert owner_record is None or owner_record.state != mac.ModuleAutoCompileState.COMPILED
        assert owner._compiled_call_impl is None
    policy_trace = tuple(
        entry
        for record in compiled_target_records
        for entry in record.compile_attempt.forward_sac_policy_trace()
    )
    return len(compiled_items), policy_trace


def _run_no_crash(rank, world_size, kwargs):
    """Run the full-size mixed-precision TP4/FSDP2 path."""
    del kwargs
    mac._reset_state()

    model, _policy_trace, compile_targets, eager_owners = _build_tp_fsdp_model(
        world_size,
        num_layers=2,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
    )
    steps = mac._PROMOTION_INVOCATION_THRESHOLD + 4
    _train_steps(model, steps=steps, vocab_size=_VOCAB, seq_len=128, rank=rank)
    _assert_real_compiled_records(compile_targets, eager_owners)


def _run_post_promotion_matches_baseline(rank, world_size, kwargs):
    """Compare TP/FSDP values, state, and SAC policy on each rank."""
    import torch._dynamo as dynamo

    use_sac = kwargs.get("use_sac", False)
    steps = mac._PROMOTION_INVOCATION_THRESHOLD + 4

    def run_case(enable_ac):
        mac._reset_state()
        dynamo.reset()
        os.environ["TORCH_NEURONX_MODULE_AUTO_COMPILE"] = "1" if enable_ac else "0"

        model, policy_trace, compile_targets, eager_owners = _build_tp_fsdp_model(
            world_size,
            num_layers=1,
            hidden_size=128,
            intermediate_size=256,
            vocab_size=500,
            use_sac=use_sac,
            use_mixed_precision=False,
        )
        optimizer, losses = _train_steps(
            model,
            steps=steps,
            vocab_size=500,
            seq_len=32,
            rank=rank,
        )
        policy_trace.clear()
        x = torch.randint(0, 500, (1, 32), device=_DEVICE)
        logits = model(x)
        probe_loss = nn.functional.cross_entropy(logits.view(-1, 500), x.view(-1))
        probe_loss.backward()
        torch_neuronx.synchronize()
        parameters, gradients, layouts = _snapshot_model(model)

        if enable_ac:
            compiled_count, compiler_policy_trace = _assert_real_compiled_records(
                compile_targets,
                eager_owners,
            )
        else:
            assert not list(mac._MODULE_REGISTRY.values())
            compiled_count = 0
            compiler_policy_trace = ()

        optimizer.zero_grad(set_to_none=True)
        return {
            "losses": losses,
            "probe_loss": probe_loss.item(),
            "parameters": parameters,
            "gradients": gradients,
            "layouts": layouts,
            "policy_trace": tuple(policy_trace),
            "compiler_policy_trace": compiler_policy_trace,
            "compiled_count": compiled_count,
        }

    try:
        baseline = run_case(False)
        candidate = run_case(True)
    finally:
        os.environ["TORCH_NEURONX_MODULE_AUTO_COMPILE"] = "1"

    assert candidate["compiled_count"] > 0
    assert all(math.isfinite(value) for value in baseline["losses"])
    assert all(math.isfinite(value) for value in candidate["losses"])
    for step, (baseline_loss, candidate_loss) in enumerate(
        zip(baseline["losses"], candidate["losses"], strict=True)
    ):
        assert abs(baseline_loss - candidate_loss) < 0.05, (
            f"Loss mismatch at step {step}: "
            f"baseline={baseline_loss}, AutoCompile={candidate_loss}"
        )
    assert abs(baseline["probe_loss"] - candidate["probe_loss"]) < 0.05
    assert baseline["layouts"] == candidate["layouts"]

    for evidence_name in ("parameters", "gradients"):
        baseline_evidence = baseline[evidence_name]
        candidate_evidence = candidate[evidence_name]
        assert baseline_evidence.keys() == candidate_evidence.keys()
        for name in baseline_evidence:
            expected = baseline_evidence[name]
            actual = candidate_evidence[name]
            assert (expected is None) == (actual is None)
            if expected is not None:
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    if use_sac:
        expected_policy_trace = Counter(
            (neuron_backend._canonical_sac_policy_target(op), decision)
            for is_recompute, op, decision in baseline["policy_trace"]
            if not is_recompute
        )
        candidate_policy_trace = Counter(
            (neuron_backend._canonical_sac_policy_target(op), decision)
            for is_recompute, op, decision in candidate["policy_trace"]
            if not is_recompute
        )
        candidate_policy_trace.update(candidate["compiler_policy_trace"])
        assert candidate_policy_trace == expected_policy_trace, (
            "SAC policy trace mismatch: "
            f"expected={expected_policy_trace!r}, "
            f"actual={candidate_policy_trace!r}"
        )
        assert any(
            not is_recompute and decision == "MUST_SAVE"
            for is_recompute, _op, decision in baseline["policy_trace"]
        )
        # Torch 2.13 stores the forward policy. Its recompute mode consumes those decisions.
        assert any(
            not is_recompute and decision == "PREFER_RECOMPUTE"
            for is_recompute, _op, decision in baseline["policy_trace"]
        )
        assert any(decision == "MUST_SAVE" for _op, decision in expected_policy_trace)
        assert any(decision == "PREFER_RECOMPUTE" for _op, decision in expected_policy_trace)


@pytest.mark.multi_device
class TestAutoCompileDistributed(BaseCollectiveOpTest):
    """TP4/FSDP2 auto-compile tests over 8 ranks via DistributedTester."""

    # TP4 x FSDP2 needs exactly 8 ranks.
    @property
    def world_size(self):
        return 8

    def test_tp_fsdp_no_crash(self):
        self.distributed_tester.run_test(_run_no_crash)

    def test_post_promotion_matches_baseline(self):
        self.distributed_tester.run_test(_run_post_promotion_matches_baseline, use_sac=False)

    def test_tp_fsdp_sac_matches_baseline(self):
        self.distributed_tester.run_test(_run_post_promotion_matches_baseline, use_sac=True)


if __name__ == "__main__":
    pytest.main(["-vs", __file__])

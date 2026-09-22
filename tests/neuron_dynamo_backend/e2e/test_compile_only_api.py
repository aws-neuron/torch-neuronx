"""Compile-only API behavior tests, device-free.

Each test class pins one behavior a compile-only consumer relies on:

- Backward through the public artifact: gradient count and None re-insertion
  must match what AOTAutograd expects (``none_output_info`` in the handle).
- bf16: production models compile in bf16, not fp32.
- MoE-style routing (top-k gather/scatter): the routing pattern, without the
  weight of a full gpt-oss checkpoint.
- Dynamic shapes: the backend forces dynamic=False; a caller passing
  dynamic=True must get a working compile or a loud error, never a silent
  miscompile.
- CPU-tensor inputs under backend="neuron": the on-device executor auto-copies
  cpu inputs; the compile-only handle must tolerate them too.
- Graph breaks: Dynamo splitting a model into multiple subgraphs must produce
  an artifact per compiled subgraph, reachable from the outputs.

Device-free (compile_only() context), gated on neuronxcc like the rest of the
compile-only suite.
"""

import pytest
import torch
import torch.nn as nn

from tests.neuron_dynamo_backend.compile_only_utils import (
    COMPILE_ONLY_FAKE_INSTANCE,
    COMPILE_ONLY_LOGICAL_NC,
    COMPILE_ONLY_TARGET,
    parse_result_line,
    requires_compiler,
    run_in_subprocess,
)


def _opts():
    from torch_neuronx.compiler import CompileOptions

    return CompileOptions(target=COMPILE_ONLY_TARGET)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@requires_compiler
class TestCompileOnlyBackwardStructure:
    """.backward() under compile-only: gradients must land with correct
    structure — one grad per differentiable input, None for the rest.

    The compile-only handle synthesizes backward outputs from FX metadata and
    re-inserts None gradients (none_output_info). If the count or positions
    were wrong, autograd would raise or write grads to the wrong params.
    """

    def test_backward_populates_all_param_grads(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        from torch_neuronx.compiler import compile_only

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 16)
                self.fc2 = nn.Linear(16, 8)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        torch._dynamo.reset()
        with compile_only(_opts()):
            m = MLP().to("neuron").train()
            x = torch.randn(4, 8, device="neuron", requires_grad=True)
            out = torch.compile(m, backend="neuron")(x)
            out.sum().backward()

            for name, p in m.named_parameters():
                assert p.grad is not None, f"{name}: no gradient after backward"
                assert (
                    p.grad.shape == p.shape
                ), f"{name}: grad shape {tuple(p.grad.shape)} != param {tuple(p.shape)}"
                assert p.grad.dtype == p.dtype
            assert x.grad is not None and x.grad.shape == x.shape

    def test_backward_with_non_differentiable_input(self, tmp_path, monkeypatch):
        """A non-differentiable (integer) input must get NO gradient while the
        float path still does — the None re-insertion positions must be right."""
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        from torch_neuronx.compiler import compile_only

        class Gather(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(32, 8)
                self.fc = nn.Linear(8, 8)

            def forward(self, ids, scale):
                return self.fc(self.emb(ids)) * scale

        torch._dynamo.reset()
        with compile_only(_opts()):
            m = Gather().to("neuron").train()
            ids = torch.randint(0, 32, (4,), device="neuron")
            scale = torch.randn(4, 1, device="neuron", requires_grad=True)
            out = torch.compile(m, backend="neuron")(ids, scale)
            out.sum().backward()

            assert scale.grad is not None and scale.grad.shape == scale.shape
            assert m.emb.weight.grad is not None
            assert ids.grad is None  # integer input: no gradient, no crash

    def test_backward_with_cpu_gradient_device_mismatch(self, tmp_path, monkeypatch):
        """A CPU grad_output against a neuron output must be REJECTED, not
        silently mis-bound.

        The compiled output is on neuron but the incoming grad_output is on CPU.
        PyTorch's autograd engine (validate_outputs) requires grad_output to
        match the output's device, so backward must raise a device-mismatch
        error. This guards that the compile-only backward path still surfaces
        that check rather than accepting a wrong-device gradient.
        """
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        from torch_neuronx.compiler import compile_only

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 8)

            def forward(self, x):
                return self.fc(x)

        torch._dynamo.reset()
        with compile_only(_opts()):
            m = MLP().to("neuron").train()
            x = torch.randn(4, 8, device="neuron", requires_grad=True)
            out = torch.compile(m, backend="neuron")(x)
            grad = torch.ones(4, 8)  # CPU
            with pytest.raises(RuntimeError, match=r"invalid gradient|device"):
                out.backward(grad)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@requires_compiler
class TestCompileOnlyBF16:
    """Production models compile in bf16; the suite otherwise only runs fp32."""

    def test_bf16_forward_spec_parity(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        from torch_neuronx.compiler import compile_only

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 32)
                self.fc2 = nn.Linear(32, 16)

            def forward(self, x):
                return self.fc2(torch.nn.functional.silu(self.fc1(x)))

        model = MLP().to(torch.bfloat16).eval()
        x_cpu = torch.randn(2, 16, dtype=torch.bfloat16)
        with torch.no_grad():
            eager = model(x_cpu)

        torch._dynamo.reset()
        with compile_only(_opts()):
            m = model.to("neuron")
            x = x_cpu.to("neuron")
            with torch.no_grad():
                out = torch.compile(m, backend="neuron")(x)

            assert out.dtype == torch.bfloat16, f"expected bf16 out, got {out.dtype}"
            assert tuple(out.shape) == tuple(eager.shape)
            art = getattr(out, "_neff_artifact", None)
            assert art is not None and art.cache_key
            torch.neuron.synchronize()
            import torch_neuronx

            neff_info = torch_neuronx._C.get_neff_info(art.cache_key)
            neff = neff_info[1] if neff_info else None
            assert neff and len(neff) > 0
            assert all(
                s.dtype == torch.bfloat16 for s in art.io_spec.inputs
            ), f"io_spec input dtypes: {[s.dtype for s in art.io_spec.inputs]}"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@requires_compiler
class TestCompileOnlyMoERouting:
    """Top-k routing with gather/scatter — the MoE pattern (gpt-oss/qwen3-moe)
    without a full checkpoint. Exercises topk, one_hot, index arithmetic, and
    weighted expert mixing, which dominate the MoE lowering risk."""

    def test_topk_routing_compiles(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        from torch_neuronx.compiler import compile_only

        class TinyMoE(nn.Module):
            def __init__(self, hidden=32, n_experts=4, k=2):
                super().__init__()
                self.k = k
                self.router = nn.Linear(hidden, n_experts, bias=False)
                self.w_up = nn.Parameter(torch.randn(n_experts, hidden, hidden * 2))
                self.w_down = nn.Parameter(torch.randn(n_experts, hidden * 2, hidden))

            def forward(self, x):
                # x: [tokens, hidden]
                logits = self.router(x)
                weights, idx = torch.topk(logits, self.k, dim=-1)  # [T, k]
                weights = torch.softmax(weights, dim=-1)
                out = torch.zeros_like(x)
                # dense-gather formulation (static shapes, no data-dependent loop)
                for slot in range(self.k):
                    e = idx[:, slot]  # [T]
                    up = self.w_up[e]  # [T, H, 2H]
                    down = self.w_down[e]  # [T, 2H, H]
                    h = torch.nn.functional.silu(torch.bmm(x.unsqueeze(1), up))
                    y = torch.bmm(h, down).squeeze(1)
                    out = out + weights[:, slot : slot + 1] * y
                return out

        torch._dynamo.reset()
        with compile_only(_opts()):
            m = TinyMoE().to("neuron").eval()
            x = torch.randn(8, 32, device="neuron")
            with torch.no_grad():
                out = torch.compile(m, backend="neuron", fullgraph=True)(x)

            assert tuple(out.shape) == (8, 32)
            art = getattr(out, "_neff_artifact", None)
            assert art is not None and art.cache_key
            torch.neuron.synchronize()
            import torch_neuronx

            neff_info = torch_neuronx._C.get_neff_info(art.cache_key)
            neff = neff_info[1] if neff_info else None
            assert neff and len(neff) > 0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@requires_compiler
class TestCompileOnlyDynamicShapes:
    """The backend forces dynamic=False. A caller who passes dynamic=True must
    get either a correct static compile or a loud error — never a silent
    wrong-shape NEFF."""

    def test_dynamic_true_compiles_or_fails_loud(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        from torch_neuronx.compiler import compile_only

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 8)

            def forward(self, x):
                return torch.relu(self.fc(x))

        torch._dynamo.reset()
        with compile_only(_opts()):
            m = MLP().to("neuron")
            x = torch.randn(4, 8, device="neuron")
            try:
                with torch.no_grad():
                    out = torch.compile(m, backend="neuron", dynamic=True)(x)
            except Exception:
                return  # loud failure is acceptable
            assert tuple(out.shape) == (4, 8)
            art = getattr(out, "_neff_artifact", None)
            assert art is not None
            assert all(
                all(isinstance(d, int) for d in s.shape) for s in art.io_spec.inputs
            ), f"symbolic dims leaked into io_spec: {art.io_spec.inputs}"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@requires_compiler
class TestCompileOnlyCpuInputs:
    """The on-device executor auto-copies cpu inputs at EXECUTION time; under
    compile-only there is no execution, and mixed cpu/neuron devices fail loud
    at Dynamo fake-tensor tracing (matching eager semantics). Pin that: a cpu
    input into a neuron model raises a clear device error, never a silent
    wrong-device compile."""

    def test_cpu_inputs_fail_loud(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        from torch_neuronx.compiler import compile_only

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 8)

            def forward(self, x):
                return torch.relu(self.fc(x))

        torch._dynamo.reset()
        with compile_only(_opts()):
            m = MLP().to("neuron")
            x_cpu = torch.randn(4, 8)  # deliberately NOT moved to neuron
            with pytest.raises(Exception, match=r"(?i)device"), torch.no_grad():
                torch.compile(m, backend="neuron")(x_cpu)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@requires_compiler
class TestCompileOnlyGraphBreaks:
    """Dynamo graph breaks split a model into subgraphs; each non-trivial
    subgraph compiles separately. The artifacts must be reachable so a caller
    can collect every NEFF."""

    def test_graph_break_yields_artifacts(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        monkeypatch.setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "1")
        from torch_neuronx.compiler import compile_only

        class Broken(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 8)
                self.fc2 = nn.Linear(8, 8)

            def forward(self, x):
                h = torch.relu(self.fc1(x))
                torch._dynamo.graph_break()
                return self.fc2(h)

        torch._dynamo.reset()
        with compile_only(_opts()):
            m = Broken().to("neuron")
            x = torch.randn(4, 8, device="neuron")
            with torch.no_grad():
                out = torch.compile(m, backend="neuron")(x)

            assert tuple(out.shape) == (4, 8)
            art = getattr(out, "_neff_artifact", None)
            assert art is not None and art.cache_key
            from tests.neuron_dynamo_backend.compile_only_utils import count_artifacts

            num_stablehlo, _ = count_artifacts(str(tmp_path))
            assert (
                num_stablehlo >= 2
            ), f"expected one StableHLO per compiled subgraph, got {num_stablehlo}"

    def test_graph_break_with_eager_op_between_subgraphs(self, tmp_path, monkeypatch):
        """A graph-break scalar read warns because compile-only has no runtime value."""
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        monkeypatch.setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "1")
        from torch_neuronx.compiler import compile_only

        class BreakWithEager(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 8)
                self.fc2 = nn.Linear(8, 8)

            def forward(self, x):
                h = torch.relu(self.fc1(x))
                if h.sum().item() > 0:
                    h = h + 1
                return self.fc2(h)

        torch._dynamo.reset()
        with compile_only(_opts()):
            m = BreakWithEager().to("neuron")
            x = torch.randn(4, 8, device="neuron")
            warn_always = torch.is_warn_always_enabled()
            torch.set_warn_always(True)
            try:
                with torch.no_grad(), pytest.warns(UserWarning, match="device-to-host read"):
                    out = torch.compile(m, backend="neuron")(x)
            finally:
                torch.set_warn_always(warn_always)
            assert tuple(out.shape) == (4, 8)
            from tests.neuron_dynamo_backend.compile_only_utils import count_artifacts

            num_stablehlo, _ = count_artifacts(str(tmp_path))
            assert num_stablehlo >= 1, f"expected pre-break artifact, got {num_stablehlo}"


@requires_compiler
class TestCompileOnlyCleanShutdown:
    """A device-free compile-only subprocess returns normally (no os._exit) and
    exits within a timeout. A teardown hang from an un-retired compile op fails
    as a timeout here instead of being masked by a hard exit."""

    def test_process_exits_normally_without_os_exit(self, tmp_path):
        body = """
        import torch
        import torch.nn as nn
        import torch_neuronx

        assert not torch_neuronx._C._is_neuron_runtime_initialized()  # lazy: nothing created yet

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 8)

            def forward(self, x):
                return self.fc(x).relu()

        model = MLP().to("neuron")
        x = torch.randn(4, 8, device="neuron")
        with torch.no_grad():
            out = torch.compile(model, backend="neuron")(x)

        art = getattr(out, "_neff_artifact", None)
        assert art is not None and art.cache_key
        # Runtime is up (fake instance device-free, real on device hosts).
        assert torch_neuronx._C._is_neuron_runtime_initialized()
        assert torch_neuronx._C._vnc_count() > 0
        print("RESULT:" + art.cache_key)
        """
        proc = run_in_subprocess(
            body,
            env_overrides={
                "TORCH_NEURONX_COMPILE_ONLY": "1",
                "NEURON_PLATFORM_TARGET_OVERRIDE": COMPILE_ONLY_TARGET,
                "NEURON_LOGICAL_NC_CONFIG": COMPILE_ONLY_LOGICAL_NC,
                "NEURON_RT_FAKE_INSTANCE_TYPE": COMPILE_ONLY_FAKE_INSTANCE,
                "TORCH_NEURONX_DEBUG_DIR": str(tmp_path),
            },
            timeout=120,
        )
        assert proc.returncode == 0, "compile-only subprocess did not exit cleanly"
        assert parse_result_line(proc.stdout), "no cache_key: compile did not complete"

    def test_bare_eager_op_process_exits_normally(self, tmp_path):
        """A BARE eager op (no torch.compile) also compiles device-free and must
        not hang teardown. Eager ops take XLACompilableKernelExecution, not
        CompileOnlyKernelExecution — regression for the retire-at-compile gate in
        OperationExecutionEngine::HandleCompilationComplete covering both."""
        body = """
        import torch
        import torch_neuronx

        a = torch.randn(4, 8, device="neuron")
        b = torch.randn(8, 16).to("neuron")
        out = torch.matmul(a, b)  # eager: dispatch -> lower -> compile, no execute
        assert tuple(out.shape) == (4, 16)
        assert torch_neuronx._C._is_neuron_runtime_initialized()
        print("RESULT:ok")
        """
        proc = run_in_subprocess(
            body,
            env_overrides={
                "TORCH_NEURONX_COMPILE_ONLY": "1",
                "NEURON_PLATFORM_TARGET_OVERRIDE": COMPILE_ONLY_TARGET,
                "NEURON_LOGICAL_NC_CONFIG": COMPILE_ONLY_LOGICAL_NC,
                "NEURON_RT_FAKE_INSTANCE_TYPE": COMPILE_ONLY_FAKE_INSTANCE,
                "TORCH_NEURONX_DEBUG_DIR": str(tmp_path),
            },
            timeout=300,
        )
        assert proc.returncode == 0, "eager compile-only subprocess did not exit cleanly"
        assert parse_result_line(proc.stdout) == "ok"

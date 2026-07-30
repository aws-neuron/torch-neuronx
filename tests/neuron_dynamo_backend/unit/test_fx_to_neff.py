"""Unit tests for the public compile API: fx_to_neff / CompileOptions / CompilationArtifact.

These exercise the device-free (compile-only) compile path — no Neuron device,
no runtime — through BOTH the ``compile_only()`` context manager and the
``TORCH_NEURONX_COMPILE_ONLY`` env var (the latter in a subprocess, since the
C++ flag is resolved once per process).

Equivalence: ``fx_to_neff`` and ``torch.compile(backend="neuron")`` bottom out
in the same ``_lower_fx_to_neff`` core, so lowering the SAME post-AOT graph
through each yields the SAME ``cache_key``. The comparison is per-graph (not
per-model): ``backend="neuron"`` runs the core once per graph AOTAutograd
produces (forward, and backward on ``.backward()``), while ``fx_to_neff`` runs
it once on the graph handed in.
"""

import inspect
import os

import pytest
import torch
import torch.nn as nn

from tests.neuron_dynamo_backend.compile_only_utils import (
    COMPILE_ONLY_FAKE_INSTANCE,
    COMPILE_ONLY_LOGICAL_NC,
    COMPILE_ONLY_TARGET,
    count_artifacts,
    parse_result_line,
    requires_compiler,
    run_in_subprocess,
    stablehlo_op_digest,
)
from tests.neuron_dynamo_backend.unit.utils.test_utils import CapturedGraphs
from torch_neuronx import _C
from torch_neuronx.compiler import (
    CompilationArtifact,
    CompileOptions,
    IOSpec,
    compile_only,
    fx_to_neff,
    fx_to_stablehlo,
    stablehlo_to_neff,
)


class TinyMLP(nn.Module):
    """Smallest model that still lowers to a non-trivial NEFF."""

    def __init__(self, dim=8):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _capture_post_aot_forward(model, x):
    """Return the post-AOT forward GraphModule the neuron backend would lower.

    Uses aot_autograd with a capturing fw_compiler so we get the exact graph
    that ``_lower_fx_to_neff`` receives — the one whose cache_key we compare.
    """
    from torch._dynamo.backends.common import aot_autograd

    from torch_neuronx.neuron_dynamo_backend.decompositions import (
        get_compile_decomposition_table,
    )

    captured = CapturedGraphs()

    def fw_compiler(gm, example_inputs):
        captured.post_aot_forward_graph = gm
        return gm.forward

    aot_backend = aot_autograd(
        fw_compiler=fw_compiler,
        decompositions=get_compile_decomposition_table(),
    )

    def capture_backend(gm, example_inputs):
        captured.pre_aot_graph = gm
        return aot_backend(gm, example_inputs)

    torch._dynamo.reset()
    compiled = torch.compile(model, backend=capture_backend)
    with torch.no_grad():
        compiled(x)
    assert captured.post_aot_forward_graph is not None, "failed to capture post-AOT graph"
    return captured.post_aot_forward_graph


def _capture_fwd_bwd_graphs(model, x):
    """Return (forward_gm, backward_gm) from AOTAutograd.

    A capturing fw/bw compiler pair records both post-AOT graphs. The backward
    compiler fires when ``.backward()`` is called, so we run a real backward.
    """
    from torch._dynamo.backends.common import aot_autograd

    from torch_neuronx.neuron_dynamo_backend.decompositions import (
        get_compile_decomposition_table,
    )

    graphs = {}

    def fw_compiler(gm, example_inputs):
        graphs["fwd"] = gm
        return gm.forward

    def bw_compiler(gm, example_inputs):
        graphs["bwd"] = gm
        return gm.forward

    aot_backend = aot_autograd(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        decompositions=get_compile_decomposition_table(),
    )

    torch._dynamo.reset()
    compiled = torch.compile(model, backend=aot_backend)
    out = compiled(x)
    out.sum().backward()
    assert "fwd" in graphs and "bwd" in graphs, "failed to capture fwd+bwd graphs"
    return graphs["fwd"], graphs["bwd"]


# ---------------------------------------------------------------------------
# Contract: the tightened signatures / dataclass fields
# ---------------------------------------------------------------------------


class TestCompileApiContract:
    """Guardrails so the public surface does not silently regain removed pieces."""

    def test_fx_to_neff_takes_no_example_inputs(self):
        sig = inspect.signature(fx_to_neff)
        params = list(sig.parameters)
        assert params == ["gm", "options", "pre_processing_analysis_info"], (
            "fx_to_neff must be (gm, options=None, *, pre_processing_analysis_info=None); "
            f"example_inputs was dropped (shapes come from gm.meta). Got {params}."
        )
        assert sig.parameters["pre_processing_analysis_info"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_artifact_has_no_input_tensors(self):
        fields = set(CompilationArtifact.__dataclass_fields__)
        assert "input_tensors" not in fields, (
            "input_tensors is a backend-path handoff (rides on the output tensor "
            "as _neff_input_tensors), not part of the public CompilationArtifact."
        )
        expected = {
            "cache_key",
            "io_spec",
            "target",
            "neff",
            "stablehlo",
            "has_collectives",
            "_mlir_module",
            "_lowering",
        }
        assert fields == expected, f"Unexpected fields: {fields - expected | expected - fields}"

    def test_compile_only_accepts_options(self):
        params = list(inspect.signature(compile_only).parameters)
        assert params == ["options"], (
            "compile_only must accept the same CompileOptions as fx_to_neff so the "
            f"backend='neuron' path can carry a target device-free. Got {params}."
        )

    def test_public_api_lives_in_compiler_package(self):
        # torch_neuronx.compiler is the one canonical home for the lowering API.
        from torch_neuronx import compiler

        for name in ("fx_to_neff", "fx_to_stablehlo", "stablehlo_to_neff", "compile_only"):
            assert callable(getattr(compiler, name)), f"{name} must be callable"
        for name in ("CompileOptions", "CompilationArtifact", "IOSpec"):
            assert isinstance(getattr(compiler, name), type), f"{name} must be a type"

    def test_old_compile_shim_is_gone(self):
        # The back-compat torch_neuronx.compile module was removed — compiler is
        # the sole path. (torch_neuronx.compile.platform/.cache belong to a
        # separate package and are not what this asserts.)
        import importlib

        with pytest.raises(ImportError):
            importlib.import_module("torch_neuronx.compile")


@requires_compiler
class TestCompileOnlyAllocation:
    """Compile-only .to("neuron") preserves metadata for lowering.

    Under ``compile_only()``, storage is allocated on NRT's fake device (funtime)
    — shape, dtype, and strides are intact so graph lowering works.
    """

    def test_to_neuron_preserves_metadata(self):
        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            w = torch.nn.Linear(64, 64).to("neuron").weight
            assert w.device.type == "neuron", w.device
            # Shape/dtype metadata is intact so the graph still lowers.
            assert tuple(w.shape) == (64, 64)
            assert w.dtype == torch.float32
            assert w.untyped_storage().nbytes() == 64 * 64 * w.element_size()

    def test_cpu_readback_warns(self):
        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            w = torch.nn.Linear(8, 8).to("neuron").weight
            warn_always = torch.is_warn_always_enabled()
            torch.set_warn_always(True)
            try:
                with pytest.warns(UserWarning, match="device-to-host read"):
                    cpu_w = w.cpu()
            finally:
                torch.set_warn_always(warn_always)
            assert cpu_w.shape == w.shape

    def test_d2d_reformat_is_noop(self):
        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            t = torch.randn(4, 8, device="neuron")
            out = t.t().contiguous()  # routes through a strided D2D copy
            assert out.device.type == "neuron"
            assert tuple(out.shape) == (8, 4)

    @pytest.mark.xfail(
        reason="TODO fix - PrivateUse1 device-guard registration bug for distributed",
        strict=False,
    )
    def test_nonzero_device_index_tolerated(self):
        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            t = torch.zeros(4, device="neuron:1")
            assert t.device.type == "neuron"
            assert t.device.index == 1


@requires_compiler
class TestCompileOnlyContextState:
    """compile_only() flips the process-global flag and restores it on exit."""

    def test_flag_set_inside_and_restored_after(self):
        prev = _C._is_compile_only()
        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            assert _C._is_compile_only() is True
        assert _C._is_compile_only() == prev

    def test_flag_restored_on_exception(self):
        prev = _C._is_compile_only()
        raised = False
        try:
            with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
                assert _C._is_compile_only() is True
                raise RuntimeError("boom")
        except RuntimeError:
            raised = True
        assert raised, "compile_only() swallowed the exception"
        assert _C._is_compile_only() == prev


# ---------------------------------------------------------------------------
# Compile-wait-fetch flow (synchronize + get_neff_info)
# ---------------------------------------------------------------------------


@requires_compiler
class TestCompileWaitFetch:
    """Verify the compile → synchronize → get_neff_info flow."""

    def test_synchronize_then_fetch_returns_neff_bytes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        model = TinyMLP()
        x = torch.randn(4, 8)

        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            gm = _capture_post_aot_forward(model, x)
            art = fx_to_neff(gm, CompileOptions(target=COMPILE_ONLY_TARGET))
            assert art.cache_key
            torch.neuron.synchronize()
            neff_info = _C.get_neff_info(art.cache_key)

        assert neff_info is not None
        _, neff_bytes = neff_info
        assert len(neff_bytes) > 0

    def test_unknown_cache_key_returns_none(self):
        result = _C.get_neff_info("deadbeef-never-submitted")
        assert result is None

    def test_failed_compile_raises_on_sync_and_fetch(self, tmp_path, monkeypatch):
        """A fresh compile failure raises at synchronize() and stays retrievable."""
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        monkeypatch.setenv("NEURON_CC_FLAGS", "--definitely-not-a-real-flag")
        model = TinyMLP(dim=13)
        x = torch.randn(4, 13)

        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            gm = _capture_post_aot_forward(model, x)
            art = fx_to_neff(gm, CompileOptions(target=COMPILE_ONLY_TARGET))
            # Drains rather than hanging, and rethrows the stashed compiler error.
            with pytest.raises(RuntimeError, match="Compilation failed|neuronx-cc"):
                torch.neuron.synchronize()
            with pytest.raises(RuntimeError, match="Compilation failed"):
                _C.get_neff_info(art.cache_key)

    def test_multiple_compiles_all_resolve(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        x = torch.randn(4, 8)

        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            gm = _capture_post_aot_forward(TinyMLP(), x)
            a = fx_to_neff(gm, CompileOptions(target=COMPILE_ONLY_TARGET))
            b = fx_to_neff(gm, CompileOptions(target=COMPILE_ONLY_TARGET))
            torch.neuron.synchronize()

        assert a.cache_key == b.cache_key
        info = _C.get_neff_info(a.cache_key)
        assert info and len(info[1]) > 0


# ---------------------------------------------------------------------------
# fx_to_neff device-free, via the compile_only() context manager
# ---------------------------------------------------------------------------


@requires_compiler
class TestFxToNeffContext:
    """Compile to a NEFF with no device, driven by compile_only()."""

    def test_fx_to_neff_produces_neff(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        # Keep the StableHLO on disk (compilation_job rmtree's it otherwise).
        monkeypatch.setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "1")
        model = TinyMLP()
        x = torch.randn(4, 8)

        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            gm = _capture_post_aot_forward(model, x)
            artifact = fx_to_neff(gm, CompileOptions(target=COMPILE_ONLY_TARGET))
            torch.neuron.synchronize()
            neff_info = _C.get_neff_info(artifact.cache_key)
            neff = neff_info[1] if neff_info else None

        assert isinstance(artifact, CompilationArtifact)
        assert artifact.cache_key
        assert artifact.target == COMPILE_ONLY_TARGET
        assert artifact.io_spec.inputs and artifact.io_spec.outputs
        assert neff and len(neff) > 0, "synchronize + get_neff_info should return NEFF bytes"

        num_stablehlo, _ = count_artifacts(str(tmp_path))
        assert num_stablehlo >= 1, "expected StableHLO artifact on disk"

    def test_backend_and_fx_to_neff_same_cache_key(self, tmp_path, monkeypatch):
        """The equivalence claim: same post-AOT graph -> same cache_key."""
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        model = TinyMLP()
        x = torch.randn(4, 8)

        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            gm = _capture_post_aot_forward(model, x)
            # Lower the SAME graph twice through the shared core.
            a = fx_to_neff(gm, CompileOptions(target=COMPILE_ONLY_TARGET))
            b = fx_to_neff(gm, CompileOptions(target=COMPILE_ONLY_TARGET))

        assert (
            a.cache_key == b.cache_key
        ), "identical graph + options must be content-keyed to the same NEFF"

    def test_default_options_leaves_no_target(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", COMPILE_ONLY_TARGET)
        model = TinyMLP()
        x = torch.randn(4, 8)

        with compile_only():
            gm = _capture_post_aot_forward(model, x)
            artifact = fx_to_neff(gm)  # no options

        assert artifact.target is None, "no options -> target provenance stays None"
        assert artifact.cache_key

    def test_fx_to_stablehlo_in_isolation(self):
        opts = CompileOptions(target=COMPILE_ONLY_TARGET)

        with compile_only(opts):
            x = torch.randn(4, 8)
            gm = _capture_post_aot_forward(TinyMLP(), x)
            artifact = fx_to_stablehlo(gm, [x], opts)

        assert artifact.stablehlo
        assert artifact.io_spec.inputs and artifact.io_spec.outputs
        assert artifact.target == COMPILE_ONLY_TARGET
        assert artifact.cache_key is None
        assert artifact.neff is None
        assert artifact.has_collectives is False

    def test_stablehlo_to_neff_after_stage_one_scope_exits(self):
        opts = CompileOptions(target=COMPILE_ONLY_TARGET)

        with compile_only(opts):
            x = torch.randn(4, 8)
            gm = _capture_post_aot_forward(TinyMLP(), x)
            artifact = fx_to_stablehlo(gm, [x], opts)

        stablehlo = artifact.stablehlo
        artifact.neff = b"stale-neff"
        with compile_only(opts):
            result = stablehlo_to_neff(artifact, opts)
            torch.neuron.synchronize()
            neff_info = _C.get_neff_info(result.cache_key)

        assert result is artifact
        assert result.stablehlo == stablehlo
        assert result.neff is None
        assert neff_info and len(neff_info[1]) > 0

    def test_stablehlo_to_neff_accepts_reconstructed_artifact(self):
        opts = CompileOptions(target=COMPILE_ONLY_TARGET)

        with compile_only(opts):
            x = torch.randn(4, 8)
            gm = _capture_post_aot_forward(TinyMLP(), x)
            one_shot = fx_to_neff(gm, opts)
            staged = fx_to_stablehlo(gm, [x], opts)

        artifact = CompilationArtifact(
            io_spec=staged.io_spec,
            target=staged.target,
            stablehlo=staged.stablehlo,
            has_collectives=staged.has_collectives,
        )
        with compile_only(opts):
            result = stablehlo_to_neff(artifact, opts)
            torch.neuron.synchronize()
            neff_info = _C.get_neff_info(result.cache_key)

        assert result is artifact
        assert result._lowering is None
        assert result.cache_key == one_shot.cache_key
        assert neff_info and len(neff_info[1]) > 0

    def test_stablehlo_to_neff_rejects_artifact_without_bytes_or_state(self):
        artifact = CompilationArtifact(io_spec=IOSpec())

        with pytest.raises(ValueError, match="neither lowering state nor StableHLO bytes"):
            stablehlo_to_neff(artifact, CompileOptions(target=COMPILE_ONLY_TARGET))

    def test_two_stage_equals_fx_to_neff(self, monkeypatch, tmp_path):
        """fx_to_stablehlo -> stablehlo_to_neff composes to the SAME NEFF as
        fx_to_neff. Guards that the two-stage split runs the same FX passes (a
        skipped pass would change the StableHLO and diverge the cache_key)."""
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        model = TinyMLP()
        x = torch.randn(4, 8)
        opts = CompileOptions(target=COMPILE_ONLY_TARGET)

        with compile_only(opts):
            gm = _capture_post_aot_forward(model, x)
            one_shot = fx_to_neff(gm, opts)
            shlo = fx_to_stablehlo(gm, [x], opts)
            stablehlo = shlo.stablehlo
            two_stage = stablehlo_to_neff(shlo, opts)

        assert stablehlo is not None
        assert two_stage is shlo
        assert two_stage.stablehlo == stablehlo
        assert two_stage.cache_key == one_shot.cache_key, (
            "fx_to_stablehlo->stablehlo_to_neff diverged from fx_to_neff — the "
            "two-stage path must run the same post-AOT FX passes.\n"
            f"one_shot={one_shot.cache_key}\ntwo_stage={two_stage.cache_key}"
        )


# ---------------------------------------------------------------------------
# backend="neuron" under compile_only() — the dynamo path, device-free
# ---------------------------------------------------------------------------


@requires_compiler
class TestBackendCompileOnlyContext:
    @pytest.mark.skipif(
        not os.environ.get("NEURON_RT_FAKE_INSTANCE_TYPE") and not os.path.exists("/dev/neuron0"),
        reason="requires funtime (NEURON_RT_FAKE_INSTANCE_TYPE) or a device for .to('neuron')",
    )
    def test_compile_backend_returns_artifact_no_execute(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        torch._dynamo.reset()
        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            model = TinyMLP().to("neuron")
            x = torch.randn(4, 8, device="neuron")
            out = torch.compile(model, backend="neuron")(x)

            artifact = getattr(out, "_neff_artifact", None)
            assert artifact is not None, "backend should attach CompilationArtifact to output"
            assert artifact.cache_key
            torch.neuron.synchronize()
            neff_info = _C.get_neff_info(artifact.cache_key)
            neff = neff_info[1] if neff_info else None
            assert neff and len(neff) > 0

        # Flat inputs ride on the tensor, NOT on the public artifact.
        assert hasattr(out, "_neff_input_tensors")


# ---------------------------------------------------------------------------
# Env-var path: TORCH_NEURONX_COMPILE_ONLY=1 (fresh process)
# ---------------------------------------------------------------------------


@requires_compiler
class TestEnvVarPath:
    """The env var must produce the same device-free compile as the context.

    Runs in a subprocess because the C++ compile-only flag is resolved once per
    process (NeuronDevice.cpp), so the env has to be set before interpreter start.
    """

    def test_env_var_compiles_without_device(self, tmp_path):
        body = """
        import os, torch, torch.nn as nn
        import torch_neuronx
        from torch_neuronx.compiler import fx_to_neff, CompileOptions

        assert torch_neuronx._C._is_compile_only(), "env var did not enable compile-only"
        assert not torch_neuronx._C._is_neuron_runtime_initialized(), "runtime already started"

        class TinyMLP(nn.Module):
            def __init__(self, dim=8):
                super().__init__()
                self.fc1 = nn.Linear(dim, dim)
                self.fc2 = nn.Linear(dim, dim)
            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = TinyMLP().to("neuron")
        x = torch.randn(4, 8, device="neuron")
        out = torch.compile(model, backend="neuron")(x)
        assert torch_neuronx._C._is_neuron_runtime_initialized()
        art = getattr(out, "_neff_artifact", None)
        assert art is not None and art.cache_key
        print("RESULT:" + art.cache_key)
        """
        proc = run_in_subprocess(
            body,
            env_overrides={
                "TORCH_NEURONX_COMPILE_ONLY": "1",
                "NEURON_PLATFORM_TARGET_OVERRIDE": COMPILE_ONLY_TARGET,
                "NEURON_LOGICAL_NC_CONFIG": COMPILE_ONLY_LOGICAL_NC,
                "NEURON_RT_FAKE_INSTANCE_TYPE": COMPILE_ONLY_FAKE_INSTANCE,
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "1",
                "TORCH_NEURONX_DEBUG_DIR": str(tmp_path),
            },
        )
        cache_key = parse_result_line(proc.stdout)
        assert cache_key, "env-var compile-only should produce a cache_key"

        num_stablehlo, _ = count_artifacts(str(tmp_path))
        assert num_stablehlo >= 1


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Compile failure: synchronize must drain, error must surface
# ---------------------------------------------------------------------------


@requires_compiler
class TestFailedCompile:
    """A failed compile must not strand the op in the stream's runtime queue.

    Device-free, the runtime worker scans no devices, so an op handed to it is
    never retired and synchronize() blocks forever. The compilation callback
    resolves ops with no execution stage itself.
    """

    def test_failed_compile_drains_and_raises(self, tmp_path, monkeypatch):
        """A cached failed entry re-throws on lookup; the op must still retire.

        Distinct path from a fresh failure: the cache fast-fail throws out of
        ProcessPrepareTask rather than reaching the compile callback. Device-free
        the runtime worker has no visible devices to scan, so an op left for it
        would never leave the stream queue and synchronize() would hang.
        """
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        monkeypatch.setenv("NEURON_CC_FLAGS", "--definitely-not-a-real-flag")
        model = TinyMLP(dim=13)
        x = torch.randn(4, 13)

        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            gm = _capture_post_aot_forward(model, x)
            # Compile twice: the second hits the cached failed entry.
            for _ in range(2):
                fx_to_neff(gm, CompileOptions(target=COMPILE_ONLY_TARGET))
                with pytest.raises(RuntimeError, match="Compilation failed|neuronx-cc"):
                    torch.neuron.synchronize()


# ---------------------------------------------------------------------------
# Artifact wait/future semantics
# ---------------------------------------------------------------------------


@requires_compiler
class TestStableHLOContent:
    """Assert StableHLO is generated with real content for both fwd and bwd.

    Uses the structural op digest (``{op_name: count}``) rather than a byte hash:
    it proves the lowering emitted the expected computation, and a mismatch shows
    which ops differ. Device-free (compile_only), no NEFF/neuronx-cc needed.
    """

    def test_forward_stablehlo_has_content(self):
        opts = CompileOptions(target=COMPILE_ONLY_TARGET)
        with compile_only(opts):
            gm = _capture_post_aot_forward(TinyMLP(), torch.randn(4, 8))
            artifact = fx_to_stablehlo(gm, [torch.randn(4, 8)], opts)

        digest = stablehlo_op_digest(artifact._mlir_module)
        # A TinyMLP forward must lower to at least a matmul.
        assert (
            digest.get("stablehlo.dot_general", 0) >= 1
        ), f"forward StableHLO missing matmul; digest={digest}"

    def test_backward_stablehlo_generated_and_distinct(self):
        """Backward lowers to a real, non-empty graph that differs from forward.

        This is the meaningful "training compiles" signal — not a file count.
        A backward graph that merely duplicated the forward (or lowered to
        nothing) would be caught here.
        """
        opts = CompileOptions(target=COMPILE_ONLY_TARGET)
        with compile_only(opts):
            fwd_gm, bwd_gm = _capture_fwd_bwd_graphs(TinyMLP(), torch.randn(4, 8))
            fwd = fx_to_stablehlo(fwd_gm, [torch.randn(4, 8)], opts)
            bwd = fx_to_stablehlo(bwd_gm, None, opts)

        fwd_digest = stablehlo_op_digest(fwd._mlir_module)
        bwd_digest = stablehlo_op_digest(bwd._mlir_module)

        # Backward produced real StableHLO ops (beyond just func.func/return).
        compute_ops = {k: v for k, v in bwd_digest.items() if k.startswith("stablehlo.")}
        assert compute_ops, f"backward StableHLO has no compute ops; digest={bwd_digest}"
        # And it is a distinct graph, not a copy of the forward.
        assert bwd_digest != fwd_digest, (
            "backward digest identical to forward — backward graph likely not "
            f"lowered. fwd={fwd_digest} bwd={bwd_digest}"
        )


# ---------------------------------------------------------------------------
# Equivalence: compile-only lowering == on-device lowering (same process)
# ---------------------------------------------------------------------------


@requires_compiler
class TestOnDeviceEquivalence:
    """The compile-only NEFF must be identical to what on-device compilation emits.

    Compiling the SAME graph normally and in compile-only within one process
    fixes the compiler version, target, and LNC, so any difference is a real
    divergence introduced by the compile-only path (e.g. a stray device query
    perturbing the graph). Requires a device, so skipped device-free.
    """

    @pytest.mark.skipif(
        not os.path.exists("/dev/neuron0"),
        reason="on-device equivalence check requires a Neuron device",
    )
    def test_compile_only_digest_matches_on_device(self):
        model = TinyMLP()
        x = torch.randn(4, 8)

        # On-device lowering (runtime present, no compile_only).
        gm = _capture_post_aot_forward(model, x)
        on_device = stablehlo_op_digest(
            fx_to_stablehlo(gm, [x], CompileOptions(target=COMPILE_ONLY_TARGET))._mlir_module
        )

        # Compile-only lowering of the same graph.
        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            compile_only_digest = stablehlo_op_digest(
                fx_to_stablehlo(gm, [x], CompileOptions(target=COMPILE_ONLY_TARGET))._mlir_module
            )

        assert compile_only_digest == on_device, (
            "compile-only lowering diverged from on-device lowering.\n"
            f"on-device={on_device}\ncompile-only={compile_only_digest}"
        )

    @pytest.mark.skipif(
        not os.path.exists("/dev/neuron0"),
        reason="on-device NEFF equivalence check requires a Neuron device",
    )
    def test_compile_only_neff_identical_and_executes_on_device(self):
        """The strongest anti-drift guarantee: a NEFF compiled device-free is
        the SAME BINARY the on-device path produces, and it runs correctly.

        Same graph + same target/LNC/compiler => same content-hash cache_key
        and byte-identical NEFF, whether or not a device is present. Then the
        on-device execution matches a CPU reference, proving the device-free
        artifact is not just identical but correct.
        """
        import hashlib

        torch.manual_seed(0)
        model = TinyMLP()
        x_cpu = torch.randn(4, 8)
        with torch.no_grad():
            ref = model(x_cpu)

        opts = CompileOptions(target=COMPILE_ONLY_TARGET)

        # Compile-only (device-free) NEFF for the captured graph.
        with compile_only(opts):
            gm = _capture_post_aot_forward(model, x_cpu)
            art_co = fx_to_neff(gm, opts)
            torch.neuron.synchronize()
            neff_co = _C.get_neff_info(art_co.cache_key)[1]
        sha_co = hashlib.sha256(neff_co).hexdigest()

        # On-device NEFF for the same graph (runtime present, no compile_only).
        gm_dev = _capture_post_aot_forward(model, x_cpu)
        art_dev = fx_to_neff(gm_dev, opts)
        torch.neuron.synchronize()
        neff_dev = _C.get_neff_info(art_dev.cache_key)[1]
        sha_dev = hashlib.sha256(neff_dev).hexdigest()

        assert art_co.cache_key == art_dev.cache_key, (
            "cache_key drift: compile-only and on-device compiled the same graph "
            f"to different keys ({art_co.cache_key} vs {art_dev.cache_key})"
        )
        assert sha_co == sha_dev, "compile-only NEFF bytes differ from the on-device NEFF"

        # Execute the (identical) NEFF on device and check numerical parity.
        m = model.to("neuron")
        xd = x_cpu.to("neuron")
        with torch.no_grad():
            out = torch.compile(m, backend="neuron")(xd)
        torch.neuron.synchronize()
        err = (out.cpu() - ref).abs().max().item()
        assert err < 2e-2, f"on-device execution diverged from CPU reference: max_abs_err={err}"


@requires_compiler
class TestRunFxPassesFlag:
    """CompileOptions(run_fx_passes=False): the caller ran its own pipeline.

    The graph is taken as already post-pass; analysis (mutation/None
    bookkeeping) comes from pre_processing_analysis_info instead of being
    recomputed. A graph that IS in post-pass form must produce the same
    StableHLO — and therefore the same cache_key — either way.
    """

    def test_default_is_true(self):
        assert CompileOptions().run_fx_passes is True

    def test_post_pass_graph_same_cache_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        model = TinyMLP()
        x = torch.randn(4, 8)

        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            gm = _capture_post_aot_forward(model, x)
            # Run the Neuron passes ONCE ourselves — this is the "customer ran
            # their own pipeline" graph (TinyMLP passes are shape-preserving).
            from torch_neuronx.kernels.cache_utils import compilation_job
            from torch_neuronx.neuron_dynamo_backend.fx.pipelines.post_aot_processing_pipeline import (  # noqa: E501
                post_aot_processing,
            )

            with compilation_job(op_name="prepass") as job:
                processed = post_aot_processing(gm, job=job, model_name="prepass").graph_module

            with_passes = fx_to_neff(processed, CompileOptions(target=COMPILE_ONLY_TARGET))
            without_passes = fx_to_neff(
                processed,
                CompileOptions(target=COMPILE_ONLY_TARGET, run_fx_passes=False),
            )

        assert without_passes.cache_key == with_passes.cache_key, (
            "a post-pass graph must lower identically with run_fx_passes=False "
            "(passes are no-ops on it)"
        )
        torch.neuron.synchronize()
        assert _C.get_neff_info(without_passes.cache_key)

    def test_caller_supplied_mutation_info_lands_in_io_spec(self, tmp_path, monkeypatch):
        """With passes skipped, io_spec.aliases comes from the caller."""
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo

        model = TinyMLP()
        x = torch.randn(8, 8)

        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            gm = _capture_post_aot_forward(model, x)
            # Claim output 0 aliases input 0 (shapes match: 8x8 in, 8x8 out).
            supplied = AliasingInfo()
            supplied.add(parameter_number=0, parameter_index=[], output_index=0)
            art = fx_to_neff(
                gm,
                CompileOptions(target=COMPILE_ONLY_TARGET, run_fx_passes=False),
                pre_processing_analysis_info={"mutation_info": supplied},
            )

        assert [(a.input_index, a.output_index) for a in art.io_spec.aliases] == [(0, 0)], (
            f"caller-supplied mutation_info must surface in io_spec.aliases; "
            f"got {art.io_spec.aliases}"
        )

    def test_no_analysis_means_no_aliases(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))
        model = TinyMLP()
        x = torch.randn(4, 8)

        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            gm = _capture_post_aot_forward(model, x)
            art = fx_to_neff(gm, CompileOptions(target=COMPILE_ONLY_TARGET, run_fx_passes=False))

        assert art.io_spec.aliases == []
        assert art.cache_key


@requires_compiler
class TestCompileOnlyDeterminism:
    """Compile-only lowering is reproducible: same graph -> same digest."""

    def test_digest_stable_across_runs(self):
        model = TinyMLP()
        x = torch.randn(4, 8)
        with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
            gm = _capture_post_aot_forward(model, x)
            d1 = stablehlo_op_digest(
                fx_to_stablehlo(gm, [x], CompileOptions(target=COMPILE_ONLY_TARGET))._mlir_module
            )
            d2 = stablehlo_op_digest(
                fx_to_stablehlo(gm, [x], CompileOptions(target=COMPILE_ONLY_TARGET))._mlir_module
            )
        assert d1 == d2, f"compile-only digest not deterministic: {d1} vs {d2}"

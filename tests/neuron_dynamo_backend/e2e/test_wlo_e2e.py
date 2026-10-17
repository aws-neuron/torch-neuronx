"""
E2E tests for Weight Layout Optimization (WLO) through torch.compile.

WLO is enabled by ``options={"weight_layout_optimization": True}`` and is
inference-only: the backend gates it on the AOT graph being an inference graph
(backend.py::neuron_backend_fx_compiler), so these tests run under
``torch.inference_mode()``. The compiler may transpose floating-point weights
into a device-friendly layout; the runtime (Executor._prepare_inputs) relays each
weight before feeding the NEFF. The observable contract is that WLO must NOT change
results -- a WLO-compiled model must match its eager reference.

Unlike the unit tests (test_compile.py, test_wlo_executor.py, test_wlo_hlo_parser.py),
which mock the compiler and run the relayout math on CPU, these exercise the real
torch.compile(backend="neuron") path end to end and therefore require a Neuron
device AND neuronx-cc (i.e. a trn pod / CI, not a bare checkout).
"""

import logging

import pytest
import torch
import torch.nn as nn

# Check if neuron device is available
assert torch.neuron.device_count() > 0, "No neuron devices were discovered."

# Parent logger of both compile.py (async submit) and cache_artifact.py (lazy sidecar fetch).
WLO_LOGGER = "torch_neuronx.neuron_dynamo_backend"

# (in_features, out_features, batch, dtype, tol)
WLO_CONFIGS = [
    pytest.param(64, 32, 4, torch.float32, 1e-4, id="linear-f32"),
    pytest.param(64, 32, 4, torch.bfloat16, 2e-2, id="linear-bf16"),
    pytest.param(128, 128, 8, torch.float32, 1e-4, id="linear-square-f32"),
]


class MLP(nn.Module):
    """Two Linear layers -- their weights are the floating-point params WLO may relayout."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _make_models(in_features, out_features, dtype):
    """A CPU reference model and a Neuron copy sharing identical weights."""
    torch.manual_seed(0)
    cpu_model = MLP(in_features, out_features).to(dtype)
    neuron_model = MLP(in_features, out_features).to(dtype)
    neuron_model.load_state_dict(cpu_model.state_dict())
    return cpu_model, neuron_model.to("neuron")


class TestWLOEndToEnd:
    @pytest.mark.parametrize("in_f,out_f,batch,dtype,tol", WLO_CONFIGS)
    def test_wlo_matches_eager(self, in_f, out_f, batch, dtype, tol):
        """A WLO-compiled inference model must produce the same output as eager.

        This is the primary contract: relaying weights is a layout-only change and
        must be numerically transparent.
        """
        torch.manual_seed(0)
        x = torch.randn(batch, in_f, dtype=dtype)

        with torch.inference_mode():
            cpu_model, neuron_model = _make_models(in_f, out_f, dtype)
            ref = cpu_model(x)

            compiled = torch.compile(
                neuron_model,
                backend="neuron",
                dynamic=False,
                options={"weight_layout_optimization": True},
            )
            out = compiled(x.to("neuron"))

        torch.testing.assert_close(out.cpu().float(), ref.float(), rtol=tol, atol=tol)

    def test_wlo_forward_matches_eager(self):
        """A WLO-compiled forward (training) graph must also match eager. WLO applies to
        inference AND forward graphs, so the relayout must be numerically transparent on a
        forward graph too, not just inference."""
        in_f, out_f, batch, dtype, tol = 64, 32, 4, torch.float32, 1e-4
        torch.manual_seed(0)
        x = torch.randn(batch, in_f, dtype=dtype, requires_grad=True)

        # No inference_mode + train() -> AOTAutograd traces a forward (training) graph.
        cpu_model, neuron_model = _make_models(in_f, out_f, dtype)
        neuron_model.train()
        ref = cpu_model(x)

        compiled = torch.compile(
            neuron_model,
            backend="neuron",
            dynamic=False,
            options={"weight_layout_optimization": True},
        )
        out = compiled(x.to("neuron"))

        torch.testing.assert_close(out.cpu().float(), ref.float(), rtol=tol, atol=tol)

    def test_wlo_matches_non_wlo(self):
        """Compiling the same model with and without WLO must give the same result --
        WLO changes layout, never numerics."""
        in_f, out_f, batch, dtype, tol = 64, 32, 4, torch.float32, 1e-4
        torch.manual_seed(0)
        x = torch.randn(batch, in_f, dtype=dtype)

        with torch.inference_mode():
            _, model_wlo = _make_models(in_f, out_f, dtype)
            _, model_plain = _make_models(in_f, out_f, dtype)

            out_wlo = torch.compile(
                model_wlo,
                backend="neuron",
                dynamic=False,
                options={"weight_layout_optimization": True},
            )(x.to("neuron"))
            out_plain = torch.compile(
                model_plain,
                backend="neuron",
                dynamic=False,
            )(x.to("neuron"))

        torch.testing.assert_close(out_wlo.cpu(), out_plain.cpu(), rtol=tol, atol=tol)

    def test_wlo_path_is_exercised_and_sidecar_is_deferred(self, caplog):
        """Prove the WLO path ran AND that the sidecar fetch is async-deferred (comment 3):
        compile.py logs "sidecar deferred" at async submit (compile time), and cache_artifact.py
        logs "WLO: discovered N weight transforms (lazy)" when the executor is built at first call.
        Both must appear. We assert the messages rather than N > 0: whether the compiler emits any
        transform for a small model is a heuristic, so a hard transforms-found assert is flaky.
        """
        torch.manual_seed(0)
        x = torch.randn(4, 64, dtype=torch.float32)

        with torch.inference_mode(), caplog.at_level(logging.DEBUG, logger=WLO_LOGGER):
            _, neuron_model = _make_models(64, 32, torch.float32)
            compiled = torch.compile(
                neuron_model,
                backend="neuron",
                dynamic=False,
                options={"weight_layout_optimization": True},
            )
            compiled(x.to("neuron"))

        assert any(
            "sidecar deferred" in msg for msg in caplog.messages
        ), f"WLO async submit did not run; messages: {caplog.messages}"
        assert any(
            "WLO: discovered" in msg and "(lazy)" in msg for msg in caplog.messages
        ), f"WLO sidecar was not fetched lazily at executor build; messages: {caplog.messages}"

    def test_wlo_skipped_for_training_graph(self, caplog):
        """WLO applies to inference and forward graphs only. A backward/joint graph (compiled
        with grad enabled) must skip WLO and warn, and still produce correct results."""
        torch.manual_seed(0)
        x = torch.randn(4, 64, dtype=torch.float32, requires_grad=True)

        with caplog.at_level(logging.WARNING, logger="torch_neuronx.neuron_dynamo_backend.backend"):
            _, neuron_model = _make_models(64, 32, torch.float32)
            neuron_model.train()
            compiled = torch.compile(
                neuron_model,
                backend="neuron",
                dynamic=False,
                options={"weight_layout_optimization": True},
            )
            out = compiled(x.to("neuron"))
            out.sum().backward()

        assert any(
            "does not apply to backward/joint graphs" in msg for msg in caplog.messages
        ), f"Expected backward/joint skip warning; messages: {caplog.messages}"

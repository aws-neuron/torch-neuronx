import logging
import os

import pytest

logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption(
        "--dynamic",
        action="store_true",
        default=False,
        help="Enable dynamic shapes (pass dynamic=True to torch.compile)",
    )
    parser.addoption(
        "--stablehlo-executor",
        action="store_true",
        default=False,
        help="Use the StableHLO reference interpreter executor",
    )
    parser.addoption(
        "--hilo-infergoldens",
        action="store_true",
        default=False,
        help="Use the hilo frontend + XLAInferGoldens executor (requires neuronx-cc)",
    )


@pytest.fixture(autouse=True)
def setup_env_vars(pytestconfig):
    if pytestconfig.getoption("--hilo-infergoldens", default=False):
        os.environ["TORCH_NEURONX_DYNAMO_HILO_INFERGOLDENS"] = "1"
        os.environ.pop("TORCH_NEURONX_DYNAMO_STABLEHLO_EXECUTOR", None)
    elif pytestconfig.getoption("--stablehlo-executor", default=False):
        os.environ["TORCH_NEURONX_DYNAMO_STABLEHLO_EXECUTOR"] = "1"
        os.environ.pop("TORCH_NEURONX_DYNAMO_HILO_INFERGOLDENS", None)
    else:
        os.environ.pop("TORCH_NEURONX_DYNAMO_STABLEHLO_EXECUTOR", None)
        os.environ.pop("TORCH_NEURONX_DYNAMO_HILO_INFERGOLDENS", None)
    if pytestconfig.getoption("--preserve-hilo-workdir", default=False):
        os.environ["TORCH_NEURONX_PRESERVE_HILO_WORKDIR"] = "1"
    if pytestconfig.getoption("--dynamic", default=False):
        os.environ["NEURON_COMPILE_COVERAGE_DYNAMIC"] = "1"


@pytest.fixture(autouse=True)
def seed_and_reset_dynamo():
    import torch
    import torch._dynamo

    torch.manual_seed(0)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


# StableHLO reference interpreter cannot evaluate ops with data-dependent dynamic
# shapes (e.g. when SymInt parameters produce dynamic_iota/dynamic_broadcast).
# Error: "LLVM ERROR: dynamic result types are not supported at the moment"
# This is a hard process abort so xfail cannot catch it — must skip.
skip_data_dependent_dynamic = pytest.mark.skip(
    reason="StableHLO interpreter: data-dependent dynamic shapes not supported "
    "(LLVM ERROR: dynamic result types are not supported at the moment)"
)


@pytest.fixture
def compile_and_compare():
    """Fixture that compiles a model, runs it, and compares against CPU."""
    import torch

    use_dynamic = os.environ.get("NEURON_STABLEHLO_GEN_DYNAMIC") == "1"

    def _run(model, *inputs):
        cpu_model = model
        cpu_inputs = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]

        with torch.no_grad():
            cpu_out = cpu_model(*cpu_inputs)

        compiled = torch.compile(cpu_model, backend="neuron", dynamic=use_dynamic)

        with torch.no_grad():
            compiled_out = compiled(*inputs)

        if isinstance(cpu_out, torch.Tensor):
            torch.testing.assert_close(compiled_out, cpu_out, rtol=1e-5, atol=1e-5)
        elif isinstance(cpu_out, tuple | list):
            for c, r in zip(cpu_out, compiled_out, strict=False):
                if isinstance(c, torch.Tensor):
                    torch.testing.assert_close(r, c, rtol=1e-5, atol=1e-5)
        return compiled_out

    return _run

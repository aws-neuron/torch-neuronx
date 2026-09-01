"""E2E tests for StableHLO executor backends.

Validates that torch.compile(backend='neuron') produces correct results through
the StableHLO reference interpreter and the Hilo + XLAInferGoldens executor

"""

import os

import pytest
import torch
import torch._dynamo

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class AddMul(torch.nn.Module):
    def forward(self, x, y):
        return (x + y) * x


class ReluLinear(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class SoftmaxReduce(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1).sum(dim=0)


class Conv2dBnRelu(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(out_c)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class TransformerBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn_proj = torch.nn.Linear(dim, dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.fc1 = torch.nn.Linear(dim, dim * 4)
        self.fc2 = torch.nn.Linear(dim * 4, dim)
        self.act = torch.nn.GELU()

    def forward(self, x):
        h = x + self.attn_proj(self.norm1(x))
        h = h + self.fc2(self.act(self.fc1(self.norm2(h))))
        return h


# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

# Each entry: (model_name, model_factory, input_shapes)
# input_shapes is a list of shapes — one per forward() argument.
MODEL_CONFIGS = [
    ("add_mul", lambda shapes: AddMul(), [(4, 16), (4, 16)]),
    ("add_mul", lambda shapes: AddMul(), [(2, 64), (2, 64)]),
    ("relu_linear", lambda shapes: ReluLinear(shapes[0][-1]), [(4, 16)]),
    ("relu_linear", lambda shapes: ReluLinear(shapes[0][-1]), [(8, 32)]),
    ("softmax_reduce", lambda shapes: SoftmaxReduce(), [(4, 16)]),
    ("softmax_reduce", lambda shapes: SoftmaxReduce(), [(8, 64)]),
    ("conv2d_bn_relu", lambda shapes: Conv2dBnRelu(shapes[0][1], 8).eval(), [(1, 3, 8, 8)]),
    ("conv2d_bn_relu", lambda shapes: Conv2dBnRelu(shapes[0][1], 8).eval(), [(2, 3, 16, 16)]),
    ("transformer_block", lambda shapes: TransformerBlock(shapes[0][-1]), [(2, 8, 32)]),
    ("transformer_block", lambda shapes: TransformerBlock(shapes[0][-1]), [(4, 4, 64)]),
]

DTYPES = [torch.float32, torch.bfloat16]

BACKENDS = ["stablehlo_executor", "hilo_infergoldens"]


def _make_params():
    params = []
    for backend in BACKENDS:
        for model_name, factory, input_shapes in MODEL_CONFIGS:
            for dtype in DTYPES:
                shapes_str = "x".join(str(s) for s in input_shapes)
                params.append(
                    pytest.param(
                        backend,
                        factory,
                        input_shapes,
                        dtype,
                        id=f"{backend}-{model_name}-{dtype}-{shapes_str}",
                    )
                )
    return params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_backend_env(backend: str):
    """Set env vars to activate the given executor backend."""
    os.environ.pop("TORCH_NEURONX_DYNAMO_STABLEHLO_EXECUTOR", None)
    os.environ.pop("TORCH_NEURONX_DYNAMO_HILO_INFERGOLDENS", None)
    if backend == "stablehlo_executor":
        os.environ["TORCH_NEURONX_DYNAMO_STABLEHLO_EXECUTOR"] = "1"
    elif backend == "hilo_infergoldens":
        os.environ["TORCH_NEURONX_DYNAMO_HILO_INFERGOLDENS"] = "1"


def _clear_backend_env():
    os.environ.pop("TORCH_NEURONX_DYNAMO_STABLEHLO_EXECUTOR", None)
    os.environ.pop("TORCH_NEURONX_DYNAMO_HILO_INFERGOLDENS", None)


def _compile_and_compare(model, *inputs):
    """Compile with neuron backend and compare against eager CPU."""
    cpu_inputs = [x.clone() for x in inputs]
    with torch.no_grad():
        cpu_out = model(*cpu_inputs)

    compiled = torch.compile(model, backend="neuron")
    with torch.no_grad():
        compiled_out = compiled(*inputs)

    tol = 5e-2 if any(inp.dtype == torch.bfloat16 for inp in cpu_inputs) else 1e-2
    if isinstance(cpu_out, torch.Tensor):
        torch.testing.assert_close(compiled_out, cpu_out, rtol=tol, atol=tol)
    elif isinstance(cpu_out, tuple | list):
        for c, r in zip(cpu_out, compiled_out, strict=False):
            if isinstance(c, torch.Tensor):
                torch.testing.assert_close(r, c, rtol=tol, atol=tol)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend,factory,input_shapes,dtype", _make_params())
def test_stablehlo_executors(backend, factory, input_shapes, dtype):
    torch.manual_seed(0)
    torch._dynamo.reset()

    _set_backend_env(backend)
    try:
        model = factory(input_shapes).to(dtype)
        model.eval()
        inputs = [torch.randn(shape, dtype=dtype) for shape in input_shapes]
        _compile_and_compare(model, *inputs)
    finally:
        _clear_backend_env()
        torch._dynamo.reset()

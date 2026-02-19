"""Extra op specs for Neuron ops not in PyTorch's op_db."""

import itertools
from functools import partial

import torch
from torch.testing._internal.common_utils import make_tensor
from torch.testing._internal.opinfo.core import OpInfo, SampleInput

# Default dtypes for extra ops
NEURON_DEFAULT_DTYPES = (torch.float32, torch.bfloat16)


def _sample_inputs_sdpa(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for scaled_dot_product_fused_attention_overrideable.

    Two categories of configs:
    1. NKI-compatible: seq_len % 512 == 0 and head_dim <= 128 -> uses NKI flash attention
    2. Non-NKI: seq_len not divisible by 512 or head_dim > 128 -> falls back to MLIR
    """
    configs = [
        # NKI-compatible configs (seq % 512 == 0, head_dim <= 128)
        (1, 4, 512, 64, "nki"),  # batch=1, heads=4, seq=512, head_dim=64
        (2, 8, 1024, 128, "nki"),  # batch=2, heads=8, seq=1024, head_dim=128
        # Non-NKI configs (seq not % 512 or head_dim > 128) -> MLIR fallback
        (1, 4, 256, 64, "mlir"),  # seq=256 not divisible by 512
        (2, 4, 512, 192, "mlir"),  # head_dim=192 > 128
    ]
    for batch, heads, seq, head_dim, impl in configs:
        shape = (batch, heads, seq, head_dim)
        query = make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        key = make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        value = make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        for is_causal in [False, True]:
            # name encodes impl type and shape for CSV distinction
            name = f"{impl}_b{batch}h{heads}s{seq}d{head_dim}_causal{is_causal}"
            yield SampleInput(
                query,
                args=(key, value),
                kwargs={
                    "attn_bias": None,
                    "dropout_p": 0.0,
                    "is_causal": is_causal,
                    "scale": None,
                },
                name=name,
            )


def _sample_inputs_native_mha(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for _native_multi_head_attention."""
    # (batch, seq, embed_dim), num_heads
    configs = [
        ((2, 64, 256), 4),
        ((1, 128, 512), 8),
    ]
    for (batch, seq, embed_dim), num_heads in configs:
        query = make_tensor(
            (batch, seq, embed_dim), device=device, dtype=dtype, requires_grad=requires_grad
        )
        key = make_tensor(
            (batch, seq, embed_dim), device=device, dtype=dtype, requires_grad=requires_grad
        )
        value = make_tensor(
            (batch, seq, embed_dim), device=device, dtype=dtype, requires_grad=requires_grad
        )
        # qkv weights
        qkv_weight = make_tensor(
            (3 * embed_dim, embed_dim), device=device, dtype=dtype, requires_grad=requires_grad
        )
        qkv_bias = make_tensor(
            (3 * embed_dim,), device=device, dtype=dtype, requires_grad=requires_grad
        )
        proj_weight = make_tensor(
            (embed_dim, embed_dim), device=device, dtype=dtype, requires_grad=requires_grad
        )
        proj_bias = make_tensor(
            (embed_dim,), device=device, dtype=dtype, requires_grad=requires_grad
        )
        yield SampleInput(
            query,
            args=(key, value, embed_dim, num_heads, qkv_weight, qkv_bias, proj_weight, proj_bias),
            kwargs={"mask": None, "need_weights": False},
            name=f"b{batch}s{seq}e{embed_dim}h{num_heads}",
        )


def _sample_inputs_grouped_mm(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for _grouped_mm.

    Two variants based on tensor dimensions:
    1. NKI (2D x 2D): a (d1, t), b (t, d2), offs (g+1,) -> output (g, d1, d2)
       - t must be divisible by 128, offs are group boundaries [0, ..., t]
    2. MLIR (2D x 3D): a (t, d1), b (g, d1, d2), offs (g,) -> output (t, d2)
       - offs are starting offsets for each group, must be divisible by align
    """
    # NKI variant: 2D x 2D
    # a: (d1, t), b: (t, d2), offs: group boundaries (g+1 elements)
    d1, t, d2 = 64, 256, 128  # t must be divisible by 128
    a_nki = make_tensor((d1, t), device=device, dtype=dtype, requires_grad=requires_grad)
    b_nki = make_tensor((t, d2), device=device, dtype=dtype, requires_grad=requires_grad)
    # offs: group boundaries [0, 128, 256] means 2 groups: [0,128) and [128,256)
    offs_nki = torch.tensor([0, 128, 256], dtype=torch.int32, device=device)
    yield SampleInput(a_nki, args=(b_nki, offs_nki), name="nki_2dx2d_d64t256d128")

    # MLIR variant: 2D x 3D
    # a: (t, d1), b: (g, d1, d2), offs: starting offset per group (g elements)
    t, d1, d2, g = 256, 64, 128, 2
    a_mlir = make_tensor((t, d1), device=device, dtype=dtype, requires_grad=requires_grad)
    b_mlir = make_tensor((g, d1, d2), device=device, dtype=dtype, requires_grad=requires_grad)
    # offs: starting offset for each group [0, 128] for 2 groups
    offs_mlir = torch.tensor([0, 128], dtype=torch.int32, device=device)
    yield SampleInput(a_mlir, args=(b_mlir, offs_mlir), name="mlir_2dx3d_t256d64d128g2")


def _sample_inputs_simple_unary(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for simple unary ops (relu, gelu, silu, etc.)."""
    shapes = [(32,), (8, 64), (2, 4, 128)]
    for shape in shapes:
        yield SampleInput(
            make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad),
            name=f"shape{'x'.join(map(str, shape))}",
        )


def _sample_inputs_linear(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for linear op."""
    # (batch, in_features), (out_features, in_features)
    configs = [
        ((8, 64), 128),
        ((2, 16, 256), 512),
    ]
    for input_shape, out_features in configs:
        in_features = input_shape[-1]
        inp = make_tensor(input_shape, device=device, dtype=dtype, requires_grad=requires_grad)
        weight = make_tensor(
            (out_features, in_features), device=device, dtype=dtype, requires_grad=requires_grad
        )
        bias = make_tensor((out_features,), device=device, dtype=dtype, requires_grad=requires_grad)
        yield SampleInput(inp, args=(weight, bias), name=f"in{input_shape}_out{out_features}")
        yield SampleInput(inp, args=(weight,), name=f"in{input_shape}_out{out_features}_nobias")


def _sample_inputs_embedding(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for embedding op."""
    # weight: (num_embeddings, embedding_dim), indices: (*)
    configs = [
        (1000, 128, (8, 16)),  # vocab=1000, dim=128, indices shape (8, 16)
        (5000, 256, (4, 32)),
    ]
    for num_embeddings, embedding_dim, indices_shape in configs:
        weight = make_tensor(
            (num_embeddings, embedding_dim), device=device, dtype=dtype, requires_grad=requires_grad
        )
        indices = torch.randint(0, num_embeddings, indices_shape, device=device)
        yield SampleInput(
            weight,
            args=(indices,),
            name=f"v{num_embeddings}d{embedding_dim}_idx{'x'.join(map(str, indices_shape))}",
        )


def _sample_inputs_convolution(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for convolution op."""
    # input: (N, C_in, H, W), weight: (C_out, C_in, kH, kW)
    configs = [
        ((1, 3, 32, 32), (16, 3, 3, 3), 1, 1),  # 3x3 conv, stride=1, padding=1
        ((2, 64, 16, 16), (128, 64, 3, 3), 1, 1),
    ]
    for input_shape, weight_shape, stride, padding in configs:
        inp = make_tensor(input_shape, device=device, dtype=dtype, requires_grad=requires_grad)
        weight = make_tensor(weight_shape, device=device, dtype=dtype, requires_grad=requires_grad)
        bias = make_tensor(
            (weight_shape[0],), device=device, dtype=dtype, requires_grad=requires_grad
        )
        yield SampleInput(
            inp,
            args=(weight, bias, (stride, stride), (padding, padding), (1, 1), False, (0, 0), 1),
            name=f"in{input_shape}_k{weight_shape[2]}x{weight_shape[3]}",
        )


def _sample_inputs_native_dropout(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for native_dropout op."""
    shapes = [(8, 64), (2, 4, 128)]
    for shape in shapes:
        inp = make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        for p in [0.0, 0.5]:
            yield SampleInput(inp, args=(p, True), name=f"shape{'x'.join(map(str, shape))}_p{p}")


def _sample_inputs_softmax(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for _softmax and _log_softmax."""
    shapes = [(8, 64), (2, 4, 128)]
    for shape in shapes:
        inp = make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        yield SampleInput(inp, args=(-1, False), name=f"shape{'x'.join(map(str, shape))}")


def _sample_inputs_nll_loss(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for nll_loss_forward."""
    # input: (N, C), target: (N,)
    configs = [
        (8, 10),  # batch=8, classes=10
        (16, 100),
    ]
    for batch, num_classes in configs:
        inp = make_tensor(
            (batch, num_classes), device=device, dtype=dtype, requires_grad=requires_grad
        )
        target = torch.randint(0, num_classes, (batch,), device=device)
        weight = None
        yield SampleInput(inp, args=(target, weight, 1, -100), name=f"b{batch}c{num_classes}")


def _sample_inputs_threshold(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for threshold op."""
    shapes = [(32,), (8, 64)]
    for shape in shapes:
        inp = make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        yield SampleInput(inp, args=(0.5, 0.0), name=f"shape{'x'.join(map(str, shape))}")


def _sample_inputs_softplus(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for softplus op."""
    shapes = [(32,), (8, 64)]
    for shape in shapes:
        inp = make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        yield SampleInput(inp, args=(1.0, 20.0), name=f"shape{'x'.join(map(str, shape))}")


def _sample_inputs_one_hot(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for one_hot op."""
    configs = [
        ((8,), 10),
        ((4, 8), 20),
    ]
    for shape, num_classes in configs:
        inp = torch.randint(0, num_classes, shape, device=device)
        yield SampleInput(
            inp, args=(num_classes,), name=f"shape{'x'.join(map(str, shape))}_c{num_classes}"
        )


def _sample_inputs_linalg_vector_norm(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for linalg_vector_norm op."""
    shapes = [(32,), (8, 64)]
    for shape in shapes:
        inp = make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        yield SampleInput(inp, args=(2.0, None, False), name=f"shape{'x'.join(map(str, shape))}_l2")


def _sample_inputs_foreach_unary(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for foreach unary ops (norm, etc.)."""
    # List of tensors with various shapes
    shapes_list = [
        [(8,), (16,), (32,)],
        [(4, 8), (8, 16)],
    ]
    for shapes in shapes_list:
        tensors = [
            make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
            for shape in shapes
        ]
        yield SampleInput(tensors, name=f"list{len(shapes)}")


def _sample_inputs_foreach_scalar(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for foreach ops with scalar (add, mul, div)."""
    shapes_list = [
        [(8,), (16,), (32,)],
        [(4, 8), (8, 16)],
    ]
    for shapes in shapes_list:
        tensors = [
            make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
            for shape in shapes
        ]
        yield SampleInput(tensors, args=(2.0,), name=f"list{len(shapes)}_scalar")


def _sample_inputs_foreach_list(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for foreach ops with two tensor lists (add, mul, div)."""
    shapes_list = [
        [(8,), (16,), (32,)],
        [(4, 8), (8, 16)],
    ]
    for shapes in shapes_list:
        tensors1 = [
            make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
            for shape in shapes
        ]
        tensors2 = [
            make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
            for shape in shapes
        ]
        yield SampleInput(tensors1, args=(tensors2,), name=f"list{len(shapes)}_list")


def _sample_inputs_fused_adamw(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for _fused_adamw_ optimizer op."""
    shapes = [(64,), (128,)]
    params = tuple(
        make_tensor(shape, device=device, dtype=dtype, requires_grad=False) for shape in shapes
    )
    grads = tuple(
        make_tensor(shape, device=device, dtype=dtype, requires_grad=False) for shape in shapes
    )
    exp_avgs = tuple(torch.zeros(shape, device=device, dtype=dtype) for shape in shapes)
    exp_avg_sqs = tuple(torch.zeros(shape, device=device, dtype=dtype) for shape in shapes)
    max_exp_avg_sqs = ()  # Empty when amsgrad=False
    state_steps = tuple(torch.tensor([1], dtype=torch.int64, device=device) for _ in shapes)
    yield SampleInput(
        params,
        args=(grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps),
        kwargs={
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01,
            "eps": 1e-8,
            "amsgrad": False,
            "maximize": False,
        },
        name="adamw_basic",
    )


# Extra op specifications
NEURON_EXTRA_OPS = {
    # Attention ops
    "_scaled_dot_product_fused_attention_overrideable": {
        "op": torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
        "sample_inputs_func": _sample_inputs_sdpa,
        "dtypes": (torch.bfloat16,),
        "supports_autograd": True,  # backward tested via autograd
    },
    # Note: _scaled_dot_product_fused_attention_overrideable_backward is NOT included
    # because it requires forward outputs (logsumexp, etc.) as inputs. It's tested
    # implicitly via autograd when testing the forward op with supports_autograd=True.
    "_native_multi_head_attention": {
        "op": torch.ops.aten._native_multi_head_attention.default,
        "sample_inputs_func": _sample_inputs_native_mha,
        "dtypes": (torch.bfloat16, torch.float32),
        "supports_autograd": False,
    },
    "_grouped_mm": {
        "op": torch.ops.aten._grouped_mm.default,
        "sample_inputs_func": _sample_inputs_grouped_mm,
        "dtypes": (torch.bfloat16,),
        "supports_autograd": False,
    },
    # Activation ops
    "relu": {
        "op": torch.ops.aten.relu.default,
        "sample_inputs_func": _sample_inputs_simple_unary,
        "dtypes": (torch.float32, torch.bfloat16),
        "supports_autograd": True,
    },
    "gelu": {
        "op": torch.ops.aten.gelu.default,
        "sample_inputs_func": _sample_inputs_simple_unary,
        "dtypes": (torch.float32, torch.bfloat16),
        "supports_autograd": True,
    },
    "silu": {
        "op": torch.ops.aten.silu.default,
        "sample_inputs_func": _sample_inputs_simple_unary,
        "dtypes": (torch.float32, torch.bfloat16),
        "supports_autograd": True,
    },
    "softplus": {
        "op": torch.ops.aten.softplus.default,
        "sample_inputs_func": _sample_inputs_softplus,
        "dtypes": (torch.float32, torch.bfloat16),
        "supports_autograd": True,
    },
    "threshold": {
        "op": torch.ops.aten.threshold.default,
        "sample_inputs_func": _sample_inputs_threshold,
        "dtypes": (torch.float32, torch.bfloat16),
        "supports_autograd": True,
    },
    # Linear/embedding ops
    "linear": {
        "op": torch.ops.aten.linear.default,
        "sample_inputs_func": _sample_inputs_linear,
        "dtypes": (torch.float32, torch.bfloat16),
        "supports_autograd": True,
    },
    "embedding": {
        "op": torch.ops.aten.embedding.default,
        "sample_inputs_func": _sample_inputs_embedding,
        "dtypes": (torch.float32, torch.bfloat16),
        "supports_autograd": True,
    },
    # Convolution
    "convolution": {
        "op": torch.ops.aten.convolution.default,
        "sample_inputs_func": _sample_inputs_convolution,
        "dtypes": (torch.float32, torch.bfloat16),
        "supports_autograd": True,
    },
    # Softmax ops
    "_softmax": {
        "op": torch.ops.aten._softmax.default,
        "sample_inputs_func": _sample_inputs_softmax,
        "dtypes": (torch.float32, torch.bfloat16),
        "supports_autograd": True,
    },
    "_log_softmax": {
        "op": torch.ops.aten._log_softmax.default,
        "sample_inputs_func": _sample_inputs_softmax,
        "dtypes": (torch.float32, torch.bfloat16),
        "supports_autograd": True,
    },
    # Dropout
    "native_dropout": {
        "op": torch.ops.aten.native_dropout.default,
        "sample_inputs_func": _sample_inputs_native_dropout,
        "dtypes": (torch.float32, torch.bfloat16),
    },
    # Loss ops
    "nll_loss_forward": {
        "op": torch.ops.aten.nll_loss_forward.default,
        "sample_inputs_func": _sample_inputs_nll_loss,
        "dtypes": (torch.float32, torch.bfloat16),
    },
    # Other ops
    "one_hot": {
        "op": torch.ops.aten.one_hot.default,
        "sample_inputs_func": _sample_inputs_one_hot,
        "dtypes": (torch.int64,),
    },
    "linalg_vector_norm": {
        "op": torch.ops.aten.linalg_vector_norm.default,
        "sample_inputs_func": _sample_inputs_linalg_vector_norm,
        "dtypes": (torch.float32, torch.bfloat16),
    },
    # Foreach ops
    "_foreach_norm": {
        "op": torch._foreach_norm,
        "sample_inputs_func": _sample_inputs_foreach_unary,
        "dtypes": (torch.float32, torch.bfloat16),
    },
    "_foreach_add": {
        "op": torch._foreach_add,
        "sample_inputs_func": _sample_inputs_foreach_list,
        "dtypes": (torch.float32, torch.bfloat16),
    },
    "_foreach_mul": {
        "op": torch._foreach_mul,
        "sample_inputs_func": _sample_inputs_foreach_list,
        "dtypes": (torch.float32, torch.bfloat16),
    },
    "_foreach_div": {
        "op": torch._foreach_div,
        "sample_inputs_func": _sample_inputs_foreach_list,
        "dtypes": (torch.float32, torch.bfloat16),
    },
    "_fused_adamw_": {
        "op": torch._fused_adamw_,
        "sample_inputs_func": _sample_inputs_fused_adamw,
        "dtypes": (torch.float32, torch.bfloat16),
    },
}


def get_neuron_extra_op_db() -> list[OpInfo]:
    """Convert extra op specs to OpInfo objects."""
    extra_ops = []
    for name, spec in NEURON_EXTRA_OPS.items():
        op_info = OpInfo(
            name=name,
            op=spec.get("op"),  # Actual callable (e.g., torch.ops.aten.xxx)
            sample_inputs_func=spec["sample_inputs_func"],
            dtypes=spec.get("dtypes", NEURON_DEFAULT_DTYPES),
            supports_autograd=spec.get("supports_autograd", False),
        )
        extra_ops.append(op_info)
    return extra_ops

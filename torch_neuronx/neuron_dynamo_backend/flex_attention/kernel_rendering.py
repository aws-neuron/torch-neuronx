import hashlib
import importlib.util
import logging
import sys
from pathlib import Path

import jinja2
import torch

from torch_neuronx.kernels.cache_utils import get_artifacts_directory
from torch_neuronx.neuron_dynamo_backend._nki.ast import NKIArray
from torch_neuronx.neuron_dynamo_backend._nki.fx_to_nki import make_new_array
from torch_neuronx.neuron_dynamo_backend._nki.subgraph_lowering import NISAPointwiseSubgraphLowering

logger = logging.getLogger(__name__)

# Jinja2 environment with autoescape enabled to satisfy static analysis.
# These templates render NKI kernel code (not HTML), so escaping is a no-op
# in practice, but keeps security scanners happy.
jinja_env = jinja2.Environment(autoescape=True)

# dtype of the score tile the flex kernels hand to score_mod / joint_mod. The kernel templates
# allocate their score tiles with it ({{ score_tile_dtype }} in flex_attention.py.jinja and
# flex_attention_bwd.py.jinja) and the lowering declares the score_mod "score" placeholder with
# it, so the kernel and the lowering cannot drift apart. It is not the query dtype: torch traces
# score_mod with a placeholder of the query dtype (query.new_empty(())), which does not describe
# what the kernels pass.
SCORE_TILE_DTYPE = torch.float32

# The kernels also hand score_mod the raw matmul result straight from PSUM (fwd, scale == 1.0),
# and PSUM accumulates in fp32, so the score tile cannot be narrower than that.
_PSUM_ACCUM_DTYPE = torch.float32
if SCORE_TILE_DTYPE != _PSUM_ACCUM_DTYPE:
    raise ImportError(
        f"SCORE_TILE_DTYPE must be {_PSUM_ACCUM_DTYPE} (PSUM accumulation dtype), "
        f"got {SCORE_TILE_DTYPE}"
    )


def nl_dtype_name(dtype: torch.dtype) -> str:
    """Convert a PyTorch dtype to its generated NKI source spelling.

    Args:
        dtype (torch.dtype): PyTorch dtype to format.

    Returns:
        str: Corresponding ``nl.<dtype>`` spelling.
    """
    return str(dtype).replace("torch.", "nl.")


"""
input_spec: list of (name, dtype)
  e.g.: input_spec = [
                    ('b', torch.int32),
                    ('h', torch.int32),
                    ('q_idx', torch.int32),
                    ('kv_idx', torch.int32),
                ]
"""


# Ops that read a placeholder for shape/dtype only (they emit their own
# constant), so a scalar b/h feeding only these never needs a real tile.
_METADATA_ONLY_PACKETS = (
    torch.ops.aten.new_ones,
    torch.ops.aten.new_zeros,
    torch.ops.aten.ones_like,
    torch.ops.aten.zeros_like,
)


def _scalar_needs_materialization(node: torch.fx.Node) -> bool:
    # b/h arrive as Python ints; materialize to a tile only if some op consumes
    # their value (e.g. h*2). Otherwise the memset is dead — and crashes when a
    # mod is invoked directly with a tile (nisa.memset wants an int value).
    return any(
        getattr(u.target, "overloadpacket", u.target) not in _METADATA_ONLY_PACKETS
        for u in node.users
    )


def generate_graph_lowering(gm: torch.fx.GraphModule, input_spec, shape: tuple):
    """Lower an FX graph to NKI pointwise statements.

    Args:
        gm (torch.fx.GraphModule): FX graph module to lower.
        input_spec (Sequence[tuple]): Placeholder names, dtypes, and optional capture shapes.
        shape (tuple): Default tile shape for placeholder arrays.

    Returns:
        NISAPointwiseSubgraphLowering: Completed lowering for the graph.

    Note:
        Scalar ``b`` and ``h`` placeholders are materialized only when an operation reads
        their values; three-item input specifications use their captured shapes.
    """
    graph_lowering = NISAPointwiseSubgraphLowering(gm, shape)

    # input_spec entries are (name, dtype), or (name, dtype, capture_shape) for
    # free-variable captures (lowered at their captured rank/shape). Scalars b/h
    # become memset tiles when their value is used; everything else
    # is a tile-shape NKIArray placeholder.
    scalar_names = {"b", "h"}
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    placeholder_inps = []
    for idx, spec in enumerate(input_spec):
        name, dtype = spec[0], spec[1]
        var = "{{" + name + "}}"
        node = placeholders[idx] if idx < len(placeholders) else None

        if len(spec) == 3:
            placeholder_inps.append(NKIArray(graph_lowering.ctx, var, dtype, tuple(spec[2])))
        elif name in scalar_names and node is not None and _scalar_needs_materialization(node):
            placeholder_inps.append(make_new_array(graph_lowering.ctx, tuple(shape), var, dtype))
        else:
            placeholder_inps.append(NKIArray(graph_lowering.ctx, var, dtype, shape))

    graph_lowering.run(*placeholder_inps)
    return graph_lowering


def generate_mod_template(gm: torch.fx.GraphModule, input_spec, shape: tuple):
    """Generate an NKI modifier body template from an FX graph.

    Args:
        gm (torch.fx.GraphModule): FX graph module to lower.
        input_spec (Sequence[tuple]): Placeholder names, dtypes, and optional capture shapes.
        shape (tuple): Default tile shape for placeholder arrays.

    Returns:
        str: Jinja-renderable source code ending with the graph output.
    """
    graph_lowering = generate_graph_lowering(gm, input_spec, shape)
    ctx = graph_lowering.ctx
    parts = []
    for bufname in ctx.buffers:
        parts.append(ctx.buffers[bufname].codegen())
    for stmt in ctx.code_body:
        parts.append(stmt.codegen())
    parts.append("return " + graph_lowering.graph_outputs.name)
    return "\n".join(parts)


def generate_buffer_allocator(graph_lowering: NISAPointwiseSubgraphLowering, prefix, manual=True):
    """Generate source code that allocates lowering buffers.

    Args:
        graph_lowering (NISAPointwiseSubgraphLowering): Lowering whose buffers to allocate.
        prefix (str): Prefix value supplied while rendering buffer expressions.
        manual (bool, optional): Use the manual SBUF allocator instead of ``nl.ndarray``.
            Defaults to ``True``.

    Returns:
        str: Rendered Python source that allocates each lowering buffer.

    Note:
        Boolean buffers use ``nl.uint8`` because bit-vector operations need one-byte
        operands and ``nl.bool_`` has no SBUF allocator size.
    """
    ctx = graph_lowering.ctx
    parts = ["#GENERATED: Buffer Allocator#"]
    for bufname in ctx.buffers:
        buffer = ctx.buffers[bufname]
        nl_dtype = str(buffer.dtype).replace("torch.", "nl.")
        # The nisa bitvec ops need 1-byte operands. nl.bool_ has no sbuf allocator size.
        if buffer.dtype == torch.bool:
            nl_dtype = "nl.uint8"
        if manual:
            parts.append(
                bufname
                + " = allocator.alloc_sbuf_tensor"
                + f"(shape={buffer.shape}, dtype={nl_dtype})"
            )
        else:
            parts.append(
                bufname + " = nl.ndarray" + f"({buffer.shape}, dtype={nl_dtype}, buffer=nl.sbuf)"
            )
    return jinja_env.from_string("\n".join(parts)).render(prefix=prefix)


def generate_buffer_initializer(graph_lowering: NISAPointwiseSubgraphLowering, prefix):
    """Generate source code that initializes lowering buffers.

    Args:
        graph_lowering (NISAPointwiseSubgraphLowering): Lowering whose buffers to initialize.
        prefix (str): Prefix value supplied while rendering buffer expressions.

    Returns:
        str: Rendered Python source that assigns ``None`` to each buffer.
    """
    ctx = graph_lowering.ctx
    parts = ["#GENERATED: Initialize Buffers#"]
    for bufname in ctx.buffers:
        parts.append(bufname + " = None")
    return jinja_env.from_string("\n".join(parts)).render(prefix=prefix)


def generate_body_template(graph_lowering: NISAPointwiseSubgraphLowering):
    """Generate an NKI function body template from a completed lowering.

    Args:
        graph_lowering (NISAPointwiseSubgraphLowering): Lowering whose statements to emit.

    Returns:
        str: Jinja-renderable statements ending with the graph output.
    """
    ctx = graph_lowering.ctx
    parts = ["#GENERATED: Function Body#"]
    for stmt in ctx.code_body:
        parts.append(stmt.codegen())
    parts.append("return " + graph_lowering.graph_outputs.name)
    return "\n".join(parts)


def save_and_import_kernel(
    rendered_kernel: str, module_name: str = "temp_mask_mod", dir_name: str = "nki"
):
    """Save rendered kernel source and import it as a Python module.

    Args:
        rendered_kernel (str): Rendered kernel source code.
        module_name (str, optional): Name for the imported module. Defaults to
            ``"temp_mask_mod"``.
        dir_name (str, optional): Artifacts subdirectory for generated files. Defaults to
            ``"nki"``.

    Returns:
        ModuleType: Imported module containing the rendered kernel.

    Note:
        Uses a file lock to serialize concurrent writers, names the source from its SHA-256
        hash, and registers the imported module in ``sys.modules`` under ``module_name``.
    """
    import fcntl

    kernel_hash = hashlib.sha256(rendered_kernel.encode()).hexdigest()
    kernel_dir = Path(get_artifacts_directory()) / dir_name
    kernel_dir.mkdir(parents=True, exist_ok=True)

    temp_file = kernel_dir / f"kernel{kernel_hash}.py"
    lock_file = kernel_dir / f"kernel{kernel_hash}.lock"

    with open(lock_file, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if not temp_file.exists() or temp_file.read_text() != rendered_kernel:
                with open(temp_file, "w") as f:
                    f.write(rendered_kernel)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)

    logger.debug(f"Generated python module available at: {temp_file}")

    spec = importlib.util.spec_from_file_location(module_name, temp_file)
    temp_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = temp_module
    spec.loader.exec_module(temp_module)

    return temp_module

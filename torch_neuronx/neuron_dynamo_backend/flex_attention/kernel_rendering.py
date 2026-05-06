import hashlib
import importlib.util
import logging
import sys
from pathlib import Path

import jinja2
import torch

from torch_neuronx.kernels.cache_utils import get_artifacts_directory
from torch_neuronx.neuron_dynamo_backend._nki.ast import NKIArray
from torch_neuronx.neuron_dynamo_backend._nki.subgraph_lowering import NISAPointwiseSubgraphLowering

logger = logging.getLogger(__name__)

# Jinja2 environment with autoescape enabled to satisfy static analysis.
# These templates render NKI kernel code (not HTML), so escaping is a no-op
# in practice, but keeps security scanners happy.
jinja_env = jinja2.Environment(autoescape=True)

"""
input_spec: list of (name, dtype)
  e.g.: input_spec = [
                    ('b', torch.int32),
                    ('h', torch.int32),
                    ('q_idx', torch.int32),
                    ('kv_idx', torch.int32),
                ]
"""


def generate_graph_lowering(gm: torch.fx.GraphModule, input_spec, shape: tuple):
    graph_lowering = NISAPointwiseSubgraphLowering(gm, shape)

    placeholder_inps = [
        NKIArray(graph_lowering.ctx, "{{" + name + "}}", dtype, shape) for name, dtype in input_spec
    ]

    graph_lowering.run(*placeholder_inps)
    return graph_lowering


def generate_mod_template(gm: torch.fx.GraphModule, input_spec, shape: tuple):
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
    ctx = graph_lowering.ctx
    parts = ["#GENERATED: Buffer Allocator#"]
    for bufname in ctx.buffers:
        buffer = ctx.buffers[bufname]
        nl_dtype = str(buffer.dtype).replace("torch.", "nl.")
        # TODO: nl.bool_ is not correctly handled yet in flexattention kernel
        if buffer.dtype == torch.bool:
            nl_dtype = "nl.int32"
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
    ctx = graph_lowering.ctx
    parts = ["#GENERATED: Initialize Buffers#"]
    for bufname in ctx.buffers:
        parts.append(bufname + " = None")
    return jinja_env.from_string("\n".join(parts)).render(prefix=prefix)


def generate_body_template(graph_lowering: NISAPointwiseSubgraphLowering):
    ctx = graph_lowering.ctx
    parts = ["#GENERATED: Function Body#"]
    for stmt in ctx.code_body:
        parts.append(stmt.codegen())
    parts.append("return " + graph_lowering.graph_outputs.name)
    return "\n".join(parts)


def save_and_import_kernel(
    rendered_kernel: str, module_name: str = "temp_mask_mod", dir_name: str = "nki"
):
    """Save a rendered kernel to disk and import it as a Python module.

    Uses a file lock to prevent races when multiple ranks write concurrently.

    Args:
        rendered_kernel: The rendered kernel source code.
        module_name: The module name to use when importing.
        dir_name: Subdirectory name under the artifacts directory.

    Returns:
        The imported module.
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

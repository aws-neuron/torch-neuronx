"""Compile-only handle for the ``neuron`` dynamo backend.

In compile-only mode the backend lowers the FX graph to a NEFF and returns
without executing it — there is no device and no runtime. It must still hand
AOTAutograd a boxed callable that returns the tensors the compiled graph
declares (AOTAutograd's runtime wrapper and the autograd engine validate the
returned tensors' count, shape, dtype, and device). This module builds that
callable, synthesizing the outputs from graph metadata rather than running the
graph.
"""

from __future__ import annotations

import logging

import torch

from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import apply_output_mutations

logger = logging.getLogger(__name__)


def _output_specs(gm: torch.fx.GraphModule) -> list:
    """Return per-output specs from the graph's output node.

    Each element is the output's ``meta['val']`` (a FakeTensor carrying shape and
    dtype) for a tensor output, or ``None`` for a non-tensor position. Reading
    metadata means the graph is never executed to discover its output shapes.
    """
    output_node = next(n for n in gm.graph.nodes if n.op == "output")
    outputs = output_node.args[0]
    if not isinstance(outputs, tuple | list):
        outputs = (outputs,)

    specs = []
    for out in outputs:
        val = out.meta.get("val") if hasattr(out, "meta") else None
        if val is not None and hasattr(val, "shape") and hasattr(val, "dtype"):
            specs.append(val)
        else:
            specs.append(None)
    return specs


def _placeholder_outputs(output_specs: list, device) -> list:
    """Build placeholder output tensors matching ``output_specs`` on ``device``.

    The caller only ever reads metadata (shape/dtype/device) off these tensors,
    never the data. So each placeholder is a single-element tensor expanded to
    the declared shape with zero strides: right metadata, one element of real
    storage. Allocating full-size buffers would waste memory on data nobody
    reads.

    ``device`` must be a concrete ``neuron:N`` taken from the runtime inputs.
    A bare ``"neuron"`` needs a current device to resolve, and in compile-only
    mode there is none. Backward also validates gradient devices against the
    forward inputs, so the placeholders must live on the same device as those
    inputs. Non-tensor positions come back as ``None``.
    """
    outputs = []
    for spec in output_specs:
        if spec is None:
            outputs.append(None)
        else:
            base = torch.empty((), dtype=spec.dtype, device=device)
            outputs.append(base.as_strided(tuple(spec.shape), (0,) * len(spec.shape)))
    return outputs


def build_compile_only_handle(gm: torch.fx.GraphModule, lowering, artifact):
    """Build a boxed callable for Dynamo backend compile-only mode.

    Args:
        gm (torch.fx.GraphModule): Pre-lowering graph accepted for backend signature
            compatibility and otherwise unused.
        lowering (object): Lowering result containing the graph, cache key, and mutations.
        artifact (CompilationArtifact): Submitted compilation artifact for caller-driven
            execution.

    Returns:
        Callable: Boxed callable that synthesizes the graph's declared outputs.

    Note:
        The callable does not execute the graph. It attaches ``artifact`` and the flat
        inputs to the first tensor output for use by a caller-owned runtime.
    """
    logger.warning(
        "Compile-only mode: NEFF compiling (cache_key=%s); returning placeholder "
        "outputs with artifact attached for caller-driven execution.",
        lowering.cache_key,
    )

    # Output specs from the post-pass graph; apply_output_mutations restores
    # the original signature (same call as executor.py).
    output_specs = _output_specs(lowering.gm)
    pass_mutations = lowering.pass_mutations

    def handle(inputs):
        """Return placeholder outputs carrying the compiled artifact and inputs.

        Args:
            inputs (Iterable): Flat inputs used to select a device and restore outputs.

        Returns:
            tuple: Declared outputs after applying the lowering pass mutations.
        """
        flat_inputs = list(inputs)
        device = next((t.device for t in flat_inputs if isinstance(t, torch.Tensor)), "meta")
        outputs = _placeholder_outputs(output_specs, device)
        device_str = device.type if hasattr(device, "type") else str(device)
        outputs = apply_output_mutations(outputs, pass_mutations, device_str, flat_inputs)
        # Attach CompilationArtifact to first tensor output.
        for out in outputs:
            if isinstance(out, torch.Tensor):
                out._neff_artifact = artifact
                out._neff_input_tensors = flat_inputs
                break
        return tuple(outputs)

    handle._boxed_call = True
    return handle

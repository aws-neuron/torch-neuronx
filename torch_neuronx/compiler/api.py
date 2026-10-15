"""Standalone lowering API: FX GraphModule → StableHLO → NEFF.

Thin wrappers over the SAME internals the `neuron` dynamo backend uses
(`convert_fx_to_stablehlo`, `CompileGraph`), so a consumer that owns its own
`torch.compile` backend can lower without reimplementing it, and cannot drift
from the backend:

    fx_to_stablehlo(gm, example_inputs, options)   -> CompilationArtifact
    stablehlo_to_neff(artifact, options)             -> CompilationArtifact
    fx_to_neff(gm, options)                          -> CompilationArtifact

The caller produces the FX graph (via torch.export / AOTAutograd) and must trace
with `decompositions=get_neuron_decompositions()` so composite ops (e.g. SDPA)
lower correctly.
"""

from __future__ import annotations

import contextlib

import torch

from torch_neuronx.compiler.api_models import (
    CompilationArtifact,
    CompileOptions,
    InputOutputAlias,
    IOSpec,
    TensorSpec,
)

# Compile-only mode gates every device touch on one process-global flag
# (c10_neuron::is_compile_only), set via compile_only() or TORCH_NEURONX_COMPILE_ONLY.


@contextlib.contextmanager
def compile_only(options: CompileOptions | None = None):
    """Compile to NEFF without device execution.

        with torch_neuronx.compiler.compile_only():
            model = Model().to("neuron")
            torch.compile(model, backend="neuron")(x)

    Pass `options` to cross-compile for a specific target without a live device.
    Device-free operation requires NEURON_RT_FAKE_INSTANCE_TYPE set externally.
    """
    import torch_neuronx

    prev = torch_neuronx._C._is_compile_only()
    torch_neuronx._C._set_compile_only(True)
    prev_env = _apply_target_env(options) if options is not None else {}
    try:
        yield
    finally:
        if options is not None:
            _restore_env(prev_env)
        torch_neuronx._C._set_compile_only(prev)


def get_neuron_decompositions() -> dict:
    """Return the Neuron decomposition table.

    Pass this as `decompositions=` to `torch.export` / AOTAutograd when tracing
    the FX graph you hand to `fx_to_neff`. It lowers device-conditional composite
    ops (e.g. SDPA's overrideable overload) into primitives torch-mlir can
    legalize; without it the graph carries opaque ops that fail at lowering.
    """
    from torch_neuronx.neuron_dynamo_backend.decompositions import (
        get_compile_decomposition_table,
    )

    return get_compile_decomposition_table()


def _make_io_spec(function_io, alias_buffer_map=None) -> IOSpec:
    """Build the public IOSpec from the internal FunctionIO.

    `alias_buffer_map`, when provided, is the backend's unified alias map
    `{neff_output_index: neff_input_index}` for in-place mutations
    (e.g. a KV cache written in place and surfaced as an output) and pass-through
    aliases, mapped to public `InputOutputAlias(input_index, output_index)` entries.
    """
    inputs = [TensorSpec(tuple(t.shape), t.dtype) for t in getattr(function_io, "inputs", [])]
    outputs = [TensorSpec(tuple(t.shape), t.dtype) for t in getattr(function_io, "outputs", [])]
    aliases: list[InputOutputAlias] = []
    if alias_buffer_map:
        for output_index, input_index in alias_buffer_map.items():
            aliases.append(InputOutputAlias(input_index, output_index))
    return IOSpec(inputs=inputs, outputs=outputs, aliases=aliases)


def _make_artifact(lowering, target=None, neff=None, stablehlo=None) -> CompilationArtifact:
    """Build a CompilationArtifact from the internal lowering result."""
    return CompilationArtifact(
        cache_key=lowering.cache_key if lowering.cache_key else None,
        io_spec=_make_io_spec(
            lowering.io_spec, alias_buffer_map=getattr(lowering, "alias_buffer_map", None)
        ),
        target=target,
        stablehlo=stablehlo,
        neff=neff,
        has_collectives=lowering.has_collectives,
        pass_mutations=list(getattr(lowering, "pass_mutations", None) or []),
    )


def _apply_target_env(options: CompileOptions):
    """Thread target/lnc to the C++ compile via the platform-override env.

    The C++ GetPlatformTarget()/GetLogicalNeuronCores() read
    NEURON_PLATFORM_TARGET_OVERRIDE / NEURON_LOGICAL_NC_CONFIG (see
    PlatformUtils.cpp), so `target` can cross-compile without a live instance.
    Returns the previous values for restoration.
    """
    import os

    prev = {}
    if options.target is not None:
        prev["NEURON_PLATFORM_TARGET_OVERRIDE"] = os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE")
        os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = options.target
    if options.logical_nc is not None:
        prev["NEURON_LOGICAL_NC_CONFIG"] = os.environ.get("NEURON_LOGICAL_NC_CONFIG")
        os.environ["NEURON_LOGICAL_NC_CONFIG"] = options.logical_nc
    if options.optimization_level:
        prev["NEURON_COMPILER_OPT_LEVEL"] = os.environ.get("NEURON_COMPILER_OPT_LEVEL")
        os.environ["NEURON_COMPILER_OPT_LEVEL"] = options.optimization_level
    return prev


def _restore_env(prev: dict):
    import os

    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@contextlib.contextmanager
def _target_env(options: CompileOptions):
    """Apply the target/LNC platform-override env for the duration of a compile,
    restoring the previous values after. No-op when `target`/`logical_nc` are None.
    """
    prev = _apply_target_env(options)
    try:
        yield
    finally:
        _restore_env(prev)


@contextlib.contextmanager
def _compile_scope(options: CompileOptions, gm: torch.fx.GraphModule):
    """Shared scope for the graph-in APIs (`fx_to_stablehlo`, `fx_to_neff`).

    Applies the target/LNC env, opens a `compilation_job`, and persists the FX
    graph + IO summary — the scaffolding common to every graph-in compile. Yields
    the job so each API supplies only its own `_lower_*` call.
    """
    from torch_neuronx.kernels.cache_utils import compilation_job

    with _target_env(options), compilation_job(op_name=options.model_name) as job:
        job.save_fx_graph(options.model_name, gm)
        yield job


def fx_to_stablehlo(
    gm: torch.fx.GraphModule,
    example_inputs,
    options: CompileOptions | None = None,
) -> CompilationArtifact:
    """Lower an FX GraphModule to StableHLO MLIR (stage 1 of the compile).

    Runs the SAME front half of the lowering core (`_lower_fx_to_stablehlo`) as
    `fx_to_neff` and `torch.compile(backend="neuron")` — including the post-AOT FX
    passes — so the StableHLO matches the production path. Pair with
    `stablehlo_to_neff` to finish the NEFF.

    Ops are lowered by schema; the tensor device tag need not be `neuron` (a
    meta-tagged graph lowers identically), but the graph must carry Neuron op
    semantics — trace with `get_neuron_decompositions()`.
    """
    from torch_neuronx.neuron_dynamo_backend.backend import _lower_fx_to_stablehlo
    from torch_neuronx.neuron_dynamo_backend.compile import _serialize_mlir_module

    options = options or CompileOptions()

    with _compile_scope(options, gm) as job:
        lowering = _lower_fx_to_stablehlo(
            gm,
            example_inputs,
            job,
            model_name=options.model_name,
            run_fx_passes=options.run_fx_passes,
        )

    return CompilationArtifact(
        io_spec=_make_io_spec(
            lowering.io_spec,
            alias_buffer_map=getattr(lowering, "alias_buffer_map", None),
        ),
        target=options.target,
        stablehlo=_serialize_mlir_module(lowering.stablehlo_mlir),
        has_collectives=lowering.has_collectives,
        pass_mutations=list(getattr(lowering, "pass_mutations", None) or []),
        _mlir_module=lowering.stablehlo_mlir,
        _lowering=lowering,
    )


def stablehlo_to_neff(
    artifact: CompilationArtifact,
    options: CompileOptions | None = None,
) -> CompilationArtifact:
    """Compile a StableHLO artifact (from `fx_to_stablehlo`) to a NEFF.

    In process, resumes the shared lowering core carried by `fx_to_stablehlo`.
    A reconstructed artifact without private lowering state compiles directly
    from its canonical StableHLO bytes. Populates and returns the supplied
    artifact with cache_key set; await bytes via synchronize(). Detached
    artifacts using the tensorizer path are not supported because that path
    requires the in-process MLIR module.
    """
    options = options or CompileOptions(target=artifact.target)

    if artifact._lowering is None:
        if not artifact.stablehlo:
            raise ValueError(
                "CompilationArtifact has neither lowering state nor StableHLO bytes to compile"
            )
        from torch_neuronx.neuron_dynamo_backend.compile import _compile_stablehlo_bytes

        with _target_env(options):
            artifact.cache_key = _compile_stablehlo_bytes(
                artifact.stablehlo,
                has_collectives=artifact.has_collectives,
                use_tensorizer=options.use_tensorizer,
            )
        artifact.target = options.target
        artifact.neff = None
        return artifact

    from torch_neuronx.neuron_dynamo_backend.backend import _lower_stablehlo_to_neff

    with _target_env(options):
        lowering = _lower_stablehlo_to_neff(
            artifact._lowering,
            model_name=options.model_name,
            options={"use_tensorizer_backend": options.use_tensorizer},
        )

    artifact.cache_key = lowering.cache_key or None
    artifact.io_spec = _make_io_spec(
        lowering.io_spec, alias_buffer_map=getattr(lowering, "alias_buffer_map", None)
    )
    artifact.target = options.target
    artifact.neff = None
    artifact.has_collectives = lowering.has_collectives
    artifact.pass_mutations = list(getattr(lowering, "pass_mutations", None) or [])
    artifact._lowering = lowering
    return artifact


def fx_to_neff(
    gm: torch.fx.GraphModule,
    options: CompileOptions | None = None,
    *,
    pre_processing_analysis_info: dict | None = None,
) -> CompilationArtifact:
    """Compile an FX GraphModule directly to a NEFF.

    The entry point most consumers want: hand it the FX graph (traced with the
    neuron decompositions) and an optional target, get a NEFF back.

        from torch_neuronx.compiler import fx_to_neff, CompileOptions, get_neuron_decompositions
        artifact = fx_to_neff(gm, CompileOptions(target="trn2"))

    Shapes come from `node.meta['val']` (populated by torch.export / AOTAutograd),
    not sample tensors — so no `example_inputs`.

    `pre_processing_analysis_info` carries the pre-AOT `{"aliasing_analysis": ...}`
    from `preprocess_graph`; when set, it becomes the `mhlo.input_output_alias` so
    the compiler donates in-place-mutated buffers (e.g. KV cache). A caller using
    its own backend must run `preprocess_graph` and thread the result here.

    Runs the SAME core (`_lower_fx_to_neff`) as `torch.compile(backend="neuron")`
    — same passes, same cache key — so the API and backend cannot diverge.
    """
    from torch_neuronx.neuron_dynamo_backend.backend import _lower_fx_to_neff

    options = options or CompileOptions()

    with _compile_scope(options, gm) as job:
        lowering = _lower_fx_to_neff(
            gm,
            [],
            job,
            model_name=options.model_name,
            pre_processing_analysis_info=pre_processing_analysis_info,
            options={"use_tensorizer_backend": options.use_tensorizer},
            run_fx_passes=options.run_fx_passes,
        )

    # Async compile; bytes not resident yet. Caller synchronizes then reads cache.
    return _make_artifact(lowering, target=options.target, neff=None)

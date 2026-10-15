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
    """Compile to a NEFF without executing on a Neuron device.

    Args:
        options (CompileOptions, optional): Cross-compilation settings. Defaults
            to ``None``.

    Yields:
        None: Activates compile-only mode for the context lifetime.

    Example::

        >>> with torch_neuronx.compiler.compile_only():  # doctest: +SKIP
        ...     compiled = torch.compile(model, backend="neuron")(x)

    Note:
        This context sets a process-wide flag and restores its previous value,
        including when an exception occurs. When supplied, ``options`` also
        temporarily overrides the target, logical NeuronCore configuration,
        and compiler optimization level. Device-free operation requires
        ``NEURON_RT_FAKE_INSTANCE_TYPE`` to be set externally.
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
    """Return the Neuron decomposition table for FX graph tracing.

    Returns:
        dict: Decompositions for composite operations supported by lowering.

    Note:
        Pass this table as ``decompositions=`` to AOTAutograd, or as
        ``decomp_table=`` to ``ExportedProgram.run_decompositions``. It replaces
        device-conditional composite operations, such as the overrideable SDPA
        overload, with primitives that torch-mlir can legalize; otherwise opaque
        operations can fail during lowering.
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
    """Lower an FX graph module to StableHLO MLIR.

    Args:
        gm (torch.fx.GraphModule): FX graph module carrying Neuron semantics.
        example_inputs (object): Example inputs used to derive graph inputs.
        options (CompileOptions, optional): Compilation settings. Defaults to
            ``None``.

    Returns:
        CompilationArtifact: Stage-one artifact containing StableHLO and IO
        metadata, without a NEFF cache key.

    Raises:
        Exception: If FX-to-StableHLO conversion fails.

    Note:
        This runs the same post-AOT passes and lowering core as ``fx_to_neff``
        and the Dynamo backend. Operations lower by schema, so tensor device
        tags need not be ``neuron``; trace composite operations with
        ``get_neuron_decompositions`` before calling this function.
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
    """Compile a StableHLO artifact to a NEFF.

    Args:
        artifact (CompilationArtifact): Artifact to compile and update in place.
        options (CompileOptions, optional): Settings; defaults to artifact target.

    Returns:
        CompilationArtifact: The supplied artifact with its cache key and
        compilation metadata populated and ``neff`` reset to ``None``.

    Raises:
        ValueError: If neither lowering state nor StableHLO exists, or detached
            tensorizer lowering is requested.

    Note:
        In-process artifacts resume the shared lowering core; reconstructed
        artifacts compile from canonical StableHLO bytes. Compilation is
        asynchronous, so call ``torch.neuron.synchronize`` before fetching NEFF
        bytes. Detached artifacts cannot use the tensorizer path because it
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
    """Compile an FX graph module directly to a NEFF.

    Args:
        gm (torch.fx.GraphModule): Post-AOT graph with shapes in ``node.meta['val']``.
        options (CompileOptions, optional): Compilation settings. Defaults to ``None``.
        pre_processing_analysis_info (dict, optional): Caller analysis supplying
            pre-AOT mutations or all bookkeeping when passes are skipped. Defaults to ``None``.

    Returns:
        CompilationArtifact: Artifact with an asynchronous cache key and IO metadata.

    Raises:
        Exception: If FX-to-StableHLO conversion fails.

    Note:
        No example inputs are accepted because shapes come from graph metadata.
        Pass ``preprocess_graph`` analysis to preserve pre-AOT aliases. With
        ``options.run_fx_passes=False``, also supply post-pass alias bookkeeping.
        The lowering core and cache key match the Dynamo backend; synchronize
        before fetching NEFF bytes from the cache.
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

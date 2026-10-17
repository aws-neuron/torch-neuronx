"""E2E tests for ``torch_neuronx._C.execute_compiled_graph``.

Covers behavior a caller sees when submitting a pre-compiled NEFF on
the current neuron stream:

* Correctness with the new ``input_names`` / ``output_names`` kwargs —
  arbitrary NEFF port names route to the right tensors, sizes are
  validated synchronously, and the ``input{i}`` / ``output{i}`` fallback
  applies when the kwargs are omitted or empty.
* Stream-pipeline integration — output visible to downstream eager ops
  without an explicit ``synchronize()``, preceding eager writes complete
  before the kernel reads them, and ``torch.neuron.synchronize()`` blocks
  until kernel completion.

Every test stages the NEFF into the persistent cache via
``put_neff_cache(cache_key, neff_bytes)`` (see ``_stage_neff``) before
submitting. ``put_neff_cache`` derives the persistent-cache key
internally, matching what ``NeffDirectKernelExecution`` will look up.

The NEFFs are produced by the NKI beta frontend — chosen so we can
control the port names via source-level parameter names + the
``output_names`` compile option. The choice of producer is incidental:
the API is general-purpose.
"""

import os

# NKI-beta kernels used by the tests below. They must live at module
# scope so NKI's ParserFrontend can resolve them by qualified name at
# compile time. Each kernel's source-level parameter names surface as
# NEFF input port names, and the ``output_names`` compile option renames
# the return value's port — together they let a test pick which NEFF
# ports match the runtime's ``input{i}`` / ``output{i}`` fallback.
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
import torch

import torch_neuronx._C as _C


@nki.jit
def _default_named_add(input0, input1):
    sa = nl.ndarray(input0.shape, dtype=input0.dtype, buffer=nl.sbuf)
    sb = nl.ndarray(input1.shape, dtype=input1.dtype, buffer=nl.sbuf)
    sc = nl.ndarray(input0.shape, dtype=input0.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sa, src=input0)
    nisa.dma_copy(dst=sb, src=input1)
    nisa.tensor_tensor(dst=sc, data1=sa, data2=sb, op=nl.add)
    output0 = nl.ndarray(input0.shape, dtype=input0.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=output0, src=sc)
    return output0


# Same computation as ``_default_named_add`` but with source-level parameter
# names that will surface as arbitrary NEFF input port names (``lhs`` / ``rhs``).
# Used together with the ``output_names=["output0"]`` override at compile time
# to produce a NEFF whose inputs need explicit ``input_names`` but whose
# output can be resolved via the runtime's fallback.
@nki.jit
def _custom_input_named_add(lhs, rhs):
    sa = nl.ndarray(lhs.shape, dtype=lhs.dtype, buffer=nl.sbuf)
    sb = nl.ndarray(rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)
    sc = nl.ndarray(lhs.shape, dtype=lhs.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sa, src=lhs)
    nisa.dma_copy(dst=sb, src=rhs)
    nisa.tensor_tensor(dst=sc, data1=sa, data2=sb, op=nl.add)
    out = nl.ndarray(lhs.shape, dtype=lhs.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=sc)
    return out


_NEFF_CACHE: dict = {}


def _stage_neff(neff_bytes, cache_key):
    """Stage ``neff_bytes`` so ``execute_compiled_graph(cache_key=...)`` can find it.

    No compile step runs here, so nothing binds ``cache_key`` to a persistent key.
    Pass ``base_key`` to bind it. ``cache_key`` doubles as the persistent key: it is
    unique and filesystem-safe."""
    _C.put_neff_cache(cache_key, neff_bytes, base_key=cache_key)


def _compile_beta_add_kernel(shape, fn, input_port_names, output_port_names):
    """Compile an NKI-beta add ``fn`` to a NEFF and stage it in the
    persistent cache under a unique ``cache_key``.

    ``input_port_names`` must exactly match the source-level parameter names
    of ``fn`` — ParserFrontend uses the ``inputs`` dict keys as both the
    Python binding and the NEFF input port names.
    ``output_port_names`` is the ``output_names`` override on the return
    value.

    Returns ``cache_key`` — pass it directly to ``execute_compiled_graph``.
    Memoized on the (shape, fn identity, names) tuple to amortize
    neuronx-cc cost across tests.
    """
    fn_id = id(fn.func) if hasattr(fn, "func") else id(fn)
    memo_key = (tuple(shape), fn_id, tuple(input_port_names), tuple(output_port_names))
    if memo_key in _NEFF_CACHE:
        return _NEFF_CACHE[memo_key]

    import tempfile

    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff
    from nki.framework.kernel import Kernel as _NkiBetaKernel

    raw_fn = fn.func if hasattr(fn, "func") else fn
    kernel = _NkiBetaKernel(raw_fn)

    with tempfile.TemporaryDirectory() as td:
        np_a = np.empty(shape, dtype=np.float32)
        opts = CompileOptions(
            target=_C._get_platform_target(),
            verbose=False,
            artifacts_dir=td,
            output_path=os.path.join(td, "out.neff"),
        )
        nir = compile_to_bir(
            kernel,
            frontend=ParserFrontend(),
            inputs={input_port_names[0]: np_a, input_port_names[1]: np_a},
            compile_opts=opts,
            output_names=list(output_port_names),
        )
        compile_bir_to_neff(
            opts,
            nir,
            input_arrays=[],
            argument_names=[s.name for s in nir.descriptor.input_specs],
            output_arg_names=[s.name for s in nir.descriptor.output_specs],
        )
        with open(os.path.join(td, "out.neff"), "rb") as f:
            neff_bytes = f.read()

    cache_key = "test_ecg_io_names_" + "_".join(
        [
            "x".join(str(d) for d in shape),
            str(fn_id),
            "-".join(input_port_names),
            "-".join(output_port_names),
        ]
    )
    _stage_neff(neff_bytes, cache_key)
    _NEFF_CACHE[memo_key] = cache_key
    return cache_key


def _compile_default_named_add_kernel(shape):
    """NEFF ports match the runtime's fallback names
    (``input0`` / ``input1`` / ``output0``)."""
    return _compile_beta_add_kernel(shape, _default_named_add, ["input0", "input1"], ["output0"])


def _compile_custom_input_named_add_kernel(shape):
    """NEFF with custom input port names (``lhs`` / ``rhs``) but default
    output name (``output0``)."""
    return _compile_beta_add_kernel(shape, _custom_input_named_add, ["lhs", "rhs"], ["output0"])


def _compile_custom_output_named_add_kernel(shape):
    """NEFF with default input port names (``input0`` / ``input1``) but a
    custom output port name (``sum``)."""
    return _compile_beta_add_kernel(shape, _default_named_add, ["input0", "input1"], ["sum"])


def test_execute_compiled_graph_names_both_provided():
    """Happy path — supply both ``input_names`` and ``output_names`` that
    exactly match the NEFF's port names."""
    shape = (4, 8)
    dtype = torch.float32
    cache_key = _compile_default_named_add_kernel(shape)

    torch.manual_seed(0)
    a_cpu = torch.randn(shape, dtype=dtype)
    b_cpu = torch.randn(shape, dtype=dtype)
    a = a_cpu.to("neuron")
    b = b_cpu.to("neuron")
    out = torch.empty(shape, dtype=dtype, device="neuron")

    _C.execute_compiled_graph(
        "test_add",
        cache_key,
        [a, b],
        [out],
        False,  # has_collectives
        input_names=["input0", "input1"],
        output_names=["output0"],
    )
    torch.testing.assert_close(out.cpu(), a_cpu + b_cpu)


def test_execute_compiled_graph_input_count_mismatch_raises():
    """When ``input_names`` is supplied, the binding TORCH_CHECKs that its
    length matches ``inputs`` and refuses to submit if they don't."""
    shape = (4, 4)
    dtype = torch.float32
    cache_key = _compile_default_named_add_kernel(shape)

    a = torch.zeros(shape, dtype=dtype, device="neuron")
    out_tensor = torch.empty(shape, dtype=dtype, device="neuron")

    # Pass one tensor but a two-element input_names list — count mismatch.
    with pytest.raises(RuntimeError, match="input_names has .* entries but inputs has"):
        _C.execute_compiled_graph(
            "bad_call",
            cache_key,
            [a],
            [out_tensor],
            False,
            input_names=["input0", "input1"],
            output_names=["output0"],
        )


def test_execute_compiled_graph_output_count_mismatch_raises():
    """Symmetric to the input check — output_names, when provided, must
    match the outputs count."""
    shape = (4, 4)
    dtype = torch.float32
    cache_key = _compile_default_named_add_kernel(shape)

    a = torch.zeros(shape, dtype=dtype, device="neuron")
    b = torch.zeros(shape, dtype=dtype, device="neuron")

    with pytest.raises(RuntimeError, match="output_names has .* entries but outputs has"):
        _C.execute_compiled_graph(
            "bad_call",
            cache_key,
            [a, b],
            [],  # empty outputs — output_names has an entry → mismatch
            False,
            input_names=["input0", "input1"],
            output_names=["output0"],
        )


def test_execute_compiled_graph_default_names_fallback():
    """When the caller omits ``input_names`` / ``output_names``, the runtime
    falls back to the positional ``input{i}`` / ``output{i}`` port names.

    The NEFF's ports are already ``input0`` / ``input1`` / ``output0``, so
    a submission without name kwargs must produce the correct result."""
    shape = (4, 4)
    dtype = torch.float32
    cache_key = _compile_default_named_add_kernel(shape)

    torch.manual_seed(0)
    a_cpu = torch.randn(shape, dtype=dtype)
    b_cpu = torch.randn(shape, dtype=dtype)
    a = a_cpu.to("neuron")
    b = b_cpu.to("neuron")
    out = torch.empty(shape, dtype=dtype, device="neuron")

    _C.execute_compiled_graph(
        "default_named_add",
        cache_key,
        [a, b],
        [out],
        False,  # has_collectives
    )
    torch.testing.assert_close(out.cpu(), a_cpu + b_cpu)


def test_execute_compiled_graph_default_names_empty_lists_equivalent():
    """Passing explicit empty ``input_names=[]`` / ``output_names=[]`` must
    behave the same as omitting them — both should go through the fallback.
    """
    shape = (4, 4)
    dtype = torch.float32
    cache_key = _compile_default_named_add_kernel(shape)

    torch.manual_seed(1)
    a_cpu = torch.randn(shape, dtype=dtype)
    b_cpu = torch.randn(shape, dtype=dtype)
    a = a_cpu.to("neuron")
    b = b_cpu.to("neuron")
    out = torch.empty(shape, dtype=dtype, device="neuron")

    _C.execute_compiled_graph(
        "default_named_add_empty_lists",
        cache_key,
        [a, b],
        [out],
        False,
        input_names=[],
        output_names=[],
    )
    torch.testing.assert_close(out.cpu(), a_cpu + b_cpu)


def test_execute_compiled_graph_only_input_names_provided():
    """Mixed mode — the NEFF's input ports have custom names (``lhs`` /
    ``rhs``) but its output uses the default (``output0``). The caller
    supplies ``input_names`` explicitly and lets the output fall back."""
    shape = (4, 4)
    dtype = torch.float32
    cache_key = _compile_custom_input_named_add_kernel(shape)

    torch.manual_seed(2)
    a_cpu = torch.randn(shape, dtype=dtype)
    b_cpu = torch.randn(shape, dtype=dtype)
    a = a_cpu.to("neuron")
    b = b_cpu.to("neuron")
    out = torch.empty(shape, dtype=dtype, device="neuron")

    _C.execute_compiled_graph(
        "custom_input_names",
        cache_key,
        [a, b],
        [out],
        False,
        input_names=["lhs", "rhs"],  # required — NEFF has non-default input names
        # output_names omitted — falls back to "output0", matching the NEFF
    )
    torch.testing.assert_close(out.cpu(), a_cpu + b_cpu)


def test_execute_compiled_graph_only_output_names_provided():
    """Mixed mode — the NEFF's inputs use the default names (``input0`` /
    ``input1``) but the output has a custom name (``sum``). The caller
    supplies ``output_names`` and lets the inputs fall back."""
    shape = (4, 4)
    dtype = torch.float32
    cache_key = _compile_custom_output_named_add_kernel(shape)

    torch.manual_seed(3)
    a_cpu = torch.randn(shape, dtype=dtype)
    b_cpu = torch.randn(shape, dtype=dtype)
    a = a_cpu.to("neuron")
    b = b_cpu.to("neuron")
    out = torch.empty(shape, dtype=dtype, device="neuron")

    _C.execute_compiled_graph(
        "custom_output_name",
        cache_key,
        [a, b],
        [out],
        False,
        # input_names omitted — falls back to "input0" / "input1"
        output_names=["sum"],  # required — NEFF has non-default output name
    )
    torch.testing.assert_close(out.cpu(), a_cpu + b_cpu)


# ---------------------------------------------------------------------------
# Stream-pipeline integration
#
# The binding submits via ``SubmitOperationContext`` onto the current
# ``NeuronStream``, so tensor lifetimes, cross-XU ordering, and ``.cpu()``
# observability all flow through the normal eager scheduler. The tests below
# verify that from a caller's perspective.
# ---------------------------------------------------------------------------


def test_execute_compiled_graph_output_visible_to_subsequent_eager_op():
    """Outputs of ``execute_compiled_graph`` are visible to a downstream
    eager op without an explicit ``.synchronize()`` — confirms the kernel
    rides on the same neuron stream as every other eager op."""
    shape = (4, 4)
    dtype = torch.float32
    cache_key = _compile_default_named_add_kernel(shape)

    a = torch.full(shape, 2.0, dtype=dtype, device="neuron")
    b = torch.full(shape, 3.0, dtype=dtype, device="neuron")
    out = torch.empty(shape, dtype=dtype, device="neuron")

    _C.execute_compiled_graph(
        "add_then_scale",
        cache_key,
        [a, b],
        [out],
        False,
        input_names=["input0", "input1"],
        output_names=["output0"],
    )

    # Downstream eager op: scale by 2. Its kRead from the kernel's output
    # must defer until our submitted op completes via the stream's
    # cross-XU ordering. No explicit synchronize() required.
    scaled = out * 2.0
    expected = torch.full(shape, (2.0 + 3.0) * 2.0, dtype=dtype)
    torch.testing.assert_close(scaled.cpu(), expected)


def test_execute_compiled_graph_waits_for_preceding_eager_op():
    """An eager op produces a tensor → ``execute_compiled_graph`` reads
    it. The submitted kernel must defer until the preceding eager op's
    write completes; otherwise it would read stale HBM."""
    shape = (4, 4)
    dtype = torch.float32
    cache_key = _compile_default_named_add_kernel(shape)

    base = torch.full(shape, 1.0, dtype=dtype, device="neuron")
    # Eager op produces ``a`` — the kernel must observe its result, not
    # the uninitialized HBM behind ``a``'s allocation.
    a = base * 7.0  # → 7.0
    b = torch.full(shape, 2.0, dtype=dtype, device="neuron")
    out = torch.empty(shape, dtype=dtype, device="neuron")

    _C.execute_compiled_graph(
        "read_after_eager_write",
        cache_key,
        [a, b],
        [out],
        False,
        input_names=["input0", "input1"],
        output_names=["output0"],
    )
    expected = torch.full(shape, 7.0 + 2.0, dtype=dtype)
    torch.testing.assert_close(out.cpu(), expected)


def test_execute_compiled_graph_synchronize_waits_for_kernel():
    """``torch.neuron.synchronize()`` after ``execute_compiled_graph`` must
    block until the kernel completes — not just until submission returns.

    After ``synchronize()`` the stream's ``query()`` must return True
    (drained) and the output tensor's HBM must hold the final result.
    """
    shape = (4, 4)
    dtype = torch.float32
    cache_key = _compile_default_named_add_kernel(shape)

    # Drain anything that was already in flight so this test reads only
    # the post-submit state of the stream.
    torch.neuron.synchronize()

    a = torch.full(shape, 4.0, dtype=dtype, device="neuron")
    b = torch.full(shape, 5.0, dtype=dtype, device="neuron")
    out = torch.empty(shape, dtype=dtype, device="neuron")

    _C.execute_compiled_graph(
        "sync_waits",
        cache_key,
        [a, b],
        [out],
        False,
        input_names=["input0", "input1"],
        output_names=["output0"],
    )

    # Block until the submitted kernel completes.
    torch.neuron.synchronize()

    # Stream is now drained. ``query()`` must return True; nothing should
    # be in flight that could mutate the output tensor after we read it.
    stream = torch.neuron.current_stream()
    assert stream.query(), "stream should be empty after synchronize()"

    expected = torch.full(shape, 4.0 + 5.0, dtype=dtype)
    torch.testing.assert_close(out.cpu(), expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

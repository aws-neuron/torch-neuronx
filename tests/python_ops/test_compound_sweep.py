# ruff: noqa: N803, N806
"""
Parametrized sweep to validate compound stride decomposition.

Tests all combinations of:
  - Tensor patterns (contiguous, pure transpose, pure slice, compound, broadcast variants)
  - Operations (neg, mul, add, clone)
  - Shapes

Patterns covered:
  Single transformations (handled by existing TryCreateTranspose / TryCreateSlice):
    contiguous, transpose_only, slice_first_half, slice_second_half

  Compound: transpose + slice (handled by TryCreateCompound):
    transpose_slice_first, transpose_slice_second (RoPE x2), transpose_slice_seq

  Compound: multi-axis slice (handled by TryCreateCompound):
    multi_slice

  Pure broadcast (handled by TryCreateBroadcast):
    broadcast_expand

  Compound: broadcast + non-trivial non-broadcast dims (handled by TryCreateCompound):
    broadcast_transpose          -- expand then transpose
    broadcast_slice              -- expand then slice
    broadcast_transpose_slice    -- expand then transpose then slice (RoPE cos/sin pattern)

Run with:
  pytest tests/test_compound_sweep.py -v --tb=short -s 2>&1 | tee /tmp/compound_sweep.txt

NOTE: These tests require the prologue system to be enabled. They are skipped when:
    - NEURON_LAUNCH_BLOCKING=1 (sync mode disables prologue)
    - TORCH_NEURONX_ENABLE_STABLEHLO=0 (StableHLO disabled)
    - TORCH_NEURONX_ENABLE_PROLOGUE=0 (prologue explicitly disabled)
"""

import os

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Skip condition for prologue-dependent tests
# ---------------------------------------------------------------------------

# These tests require the prologue system to be enabled. The prologue is disabled in:
# 1. Sync mode (NEURON_LAUNCH_BLOCKING=1)
# 2. When StableHLO is disabled (TORCH_NEURONX_ENABLE_STABLEHLO=0)
# 3. When prologue is explicitly disabled (TORCH_NEURONX_ENABLE_PROLOGUE=0)
SKIP_PROLOGUE_DISABLED = pytest.mark.skipif(
    os.environ.get("NEURON_LAUNCH_BLOCKING") == "1"
    or os.environ.get("TORCH_NEURONX_ENABLE_STABLEHLO", "1") in ("0", "false")
    or os.environ.get("TORCH_NEURONX_ENABLE_PROLOGUE", "1") in ("0", "false"),
    reason=(
        "Compound sweep tests require prologue to be enabled (disabled in sync mode "
        "or when TORCH_NEURONX_ENABLE_PROLOGUE=0) and StableHLO to be enabled"
    ),
)

# ---------------------------------------------------------------------------
# Fixture: enforce no CPU fallback during tests
# ---------------------------------------------------------------------------

# Ops that are expected to fall back to CPU — these are not related to contiguous handling.
# RNG ops (torch.randn, torch.rand, etc.) are not implemented on Neuron and always fall back.
_EXPECTED_FALLBACK_OPS = frozenset(
    {
        "aten::normal_",  # torch.randn()
        "aten::uniform_",  # torch.rand()
        "aten::random_",  # other RNG
        "aten::bernoulli_",  # torch.bernoulli()
        "aten::randperm",
        "aten::randint",
    }
)


@pytest.fixture(autouse=True)
def no_cpu_fallback(monkeypatch, request):
    """Enforce that the prologue system handles all non-contiguous tensors on-device.

    Three complementary enforcement layers:

    1. NEURON_FALLBACK_ENABLED=0: Disables the C++ async error-recovery CPU fallback.
       Any async kernel failure raises immediately instead of silently falling back.

    2. get_fallback_ops() check: Catches Python-level op dispatch fallbacks
       (e.g., via the neuron_cpu_fallback C++ handler). RNG ops are expected.

    3. contiguous_generic_kernel patch: Catches the contiguous_internal CPU roundtrip
       (device->CPU->device copy). This fires when ProcessInputTransformations falls
       through all prologue creators (TryCreateTranspose/Slice/Broadcast/Compound)
       AND all on-device contiguous implementations (TransposeMLIR/BroadcastMLIR/SliceMLIR).
       If this is called, it means the prologue did NOT handle the tensor pattern.

    Together these ensure that every non-contiguous tensor in the sweep is handled
    by the prologue MLIR system (fused with the op, no copies).
    """
    try:
        import torch_neuronx

        torch_neuronx.clear_op_tracking()
    except ImportError:
        yield
        return

    # Layer 1: disable C++ async CPU fallback
    monkeypatch.setenv("NEURON_FALLBACK_ENABLED", "0")

    # Layer 3: patch contiguous_generic_kernel to fail if called
    # This detects when the prologue system falls through to the CPU roundtrip path
    import torch_neuronx.python_ops.nki_kernels.contiguous_generic as _cg_module

    # Globally track fallback calls on the module so the fuzzer test can access it
    _cg_module._fuzzer_fallback_triggered = []

    def _failing_contiguous_generic_kernel(*args, **kwargs):
        _cg_module._fuzzer_fallback_triggered.append(True)
        # Still execute to get correct results (so other assertions can run)
        return _cg_module._original_contiguous_generic_kernel(*args, **kwargs)

    # Save original and patch
    _cg_module._original_contiguous_generic_kernel = _cg_module.contiguous_generic_kernel
    monkeypatch.setattr(_cg_module, "contiguous_generic_kernel", _failing_contiguous_generic_kernel)

    # Also patch the reference in contiguous.py which imports it directly
    import torch_neuronx.python_ops.contiguous as _contiguous_module

    monkeypatch.setattr(
        _contiguous_module, "contiguous_generic_kernel", _failing_contiguous_generic_kernel
    )

    yield  # run the test

    # Layer 2: check Python-level fallback tracking
    fallback_ops = torch_neuronx.get_fallback_ops()
    unexpected = [op for op in fallback_ops if op not in _EXPECTED_FALLBACK_OPS]
    assert not unexpected, (
        f"CPU fallback detected for unexpected ops: {unexpected}\n"
        f"This means the lazy materialization (prologue) did not handle the "
        f"non-contiguous tensor pattern correctly.\n"
        f"(Allowed expected fallbacks: {sorted(_EXPECTED_FALLBACK_OPS)})"
    )

    # Layer 3: check if contiguous_generic_kernel was called (CPU roundtrip)
    # Note: 'clone' is excluded because aten::clone IS a contiguous copy operation --
    # it legitimately calls contiguous_generic_kernel as its implementation.
    # The prologue system handles inputs to compute ops (neg/mul/add), not clone itself.
    # Skip for the fuzzer test -- it checks fallbacks inline per-iteration.
    current_op = (
        request.node.callspec.params.get("op_name", "") if hasattr(request.node, "callspec") else ""
    )
    if (
        current_op != "clone"
        and "test_random_mixed_sequences" not in request.node.name
        and "test_fallback_detection_self_test" not in request.node.name
        and _cg_module._fuzzer_fallback_triggered
    ):
        pytest.fail(
            "contiguous_generic_kernel was called -- this means the prologue system "
            "(TryCreateTranspose/Slice/Broadcast/Compound) did NOT handle the non-contiguous "
            "tensor pattern. The tensor went through a device->CPU->device roundtrip instead "
            "of being handled on-device via MLIR prologue fusion."
        )


# ---------------------------------------------------------------------------
# Tensor pattern factories
# ---------------------------------------------------------------------------


def make_contiguous(B, H, S, D, dtype, device):
    """Plain contiguous tensor [B, H, S, D]."""
    return torch.randn(B, H, S, D, dtype=dtype, device=device)


def make_transpose_only(B, H, S, D, dtype, device):
    """[B,S,H,D] -> transpose(1,2) -> [B,H,S,D]: pure transpose."""
    return torch.randn(B, S, H, D, dtype=dtype, device=device).transpose(1, 2)


def make_slice_first_half(B, H, S, D, dtype, device):
    """Contiguous[..., :D//2]: pure slice, offset=0."""
    return torch.randn(B, H, S, D, dtype=dtype, device=device)[..., : D // 2]


def make_slice_second_half(B, H, S, D, dtype, device):
    """Contiguous[..., D//2:]: pure slice, offset>0."""
    return torch.randn(B, H, S, D, dtype=dtype, device=device)[..., D // 2 :]


def make_transpose_slice_first(B, H, S, D, dtype, device):
    """transpose(1,2)[..., :D//2]: compound transpose+slice, offset=0."""
    return torch.randn(B, S, H, D, dtype=dtype, device=device).transpose(1, 2)[..., : D // 2]


def make_transpose_slice_second(B, H, S, D, dtype, device):
    """transpose(1,2)[..., D//2:]: compound transpose+slice, offset>0. (RoPE x2 pattern)"""
    return torch.randn(B, S, H, D, dtype=dtype, device=device).transpose(1, 2)[..., D // 2 :]


def make_transpose_slice_seq(B, H, S, D, dtype, device):
    """transpose(1,2)[:,:,:S//2,:]: compound, slice in seq dim."""
    return torch.randn(B, S, H, D, dtype=dtype, device=device).transpose(1, 2)[:, :, : S // 2, :]


def make_multi_slice(B, H, S, D, dtype, device):
    """[..., :D//2][:,:,:S//2,:]: two slices on contiguous, no transpose."""
    return torch.randn(B, H, S, D, dtype=dtype, device=device)[..., : D // 2][:, :, : S // 2, :]


def make_broadcast_expand(B, H, S, D, dtype, device):
    """unsqueeze(1).expand: pure broadcast (stride=0 on H dim).
    Simulates cos/sin in RoPE before unsqueeze_dim is applied."""
    # [B, S, D] -> unsqueeze(1) -> [B, 1, S, D] -> expand -> [B, H, S, D]
    return torch.randn(B, S, D, dtype=dtype, device=device).unsqueeze(1).expand(B, H, S, D)


def make_broadcast_transpose(B, H, S, D, dtype, device):
    """Broadcast + transpose: expand then permute dims.
    [B, S, D] -> unsqueeze(1) -> [B,1,S,D] -> expand -> [B,H,S,D] -> transpose(2,3)
    Results in stride=0 on H dim AND transposed S/D dims."""
    return (
        torch.randn(B, S, D, dtype=dtype, device=device)
        .unsqueeze(1)
        .expand(B, H, S, D)
        .transpose(2, 3)
    )  # [B, H, D, S]


def make_broadcast_slice(B, H, S, D, dtype, device):
    """Broadcast + slice: expand then slice last dim.
    [B, S, D] -> unsqueeze(1) -> [B,1,S,D] -> expand -> [B,H,S,D] -> [..., :D//2]"""
    return (
        torch.randn(B, S, D, dtype=dtype, device=device)
        .unsqueeze(1)
        .expand(B, H, S, D)[..., : D // 2]
    )


def make_broadcast_transpose_slice(B, H, S, D, dtype, device):
    """Broadcast + transpose + slice: the full RoPE cos/sin compound pattern.
    cos has shape [B, S, D], gets unsqueeze(1) -> [B,1,S,D] -> expand -> [B,H,S,D]
    Then in apply_rotary_pos_emb it's multiplied with q which is [B,H,S,D].
    Here we simulate the cos tensor going through rotate_half's slice:
    [B,S,D] -> unsqueeze(1) -> expand [B,H,S,D] -> transpose(1,2) -> [B,S,H,D] -> [..., D//2:]
    """
    return (
        torch.randn(B, S, D, dtype=dtype, device=device)
        .unsqueeze(1)
        .expand(B, H, S, D)
        .transpose(1, 2)[..., D // 2 :]
    )  # [B, S, H, D//2]


# --- Folded patterns (narrow→view→permute) ---


def make_folded_narrow_view_permute(B, H, S, D, dtype, device):
    """Narrow a dim, then view-split it, then permute.
    This creates coprime stride ratios that the GCD check rejects,
    handled by TryFoldedDecomposition (SLICE→RESHAPE→TRANSPOSE).
    [B, H, D] -> narrow(2, 0, D-1) -> view(B, H, D//2-1, 2) -> permute(2, 0, 3, 1)
    Requires D >= 6 for the factorization to work."""
    base_d = D + 1  # ensure narrow produces something factorable
    t = torch.randn(B, H, base_d, dtype=dtype, device=device)
    narrowed_size = D  # narrow to D elements
    # Factor D into two parts
    factor_a = 2
    factor_b = D // 2
    return t.narrow(2, 0, narrowed_size).view(B, H, factor_a, factor_b).permute(2, 0, 3, 1)


def make_folded_padded_sequence(B, H, S, D, dtype, device):
    """Padded sequence pattern: base[:, :seq, :].view(B, h1, h2, w, c).permute(0, 4, 1, 2, 3).
    Simulates the common NLP/vision pattern where variable-length sequences are packed.
    Uses smaller dimensions to keep test fast."""
    # Create a base with a padding dim slightly larger than needed
    seq_len = S * 2  # total sequence slots
    padded_len = seq_len + S  # add padding
    c_dim = min(D, 8)  # channel dim
    t = torch.randn(B, padded_len, c_dim, dtype=dtype, device=device)
    # narrow to seq_len, then reshape into multi-dim sequence
    h1 = 2
    h2 = seq_len // h1
    return t[:, :seq_len, :].view(B, h1, h2, c_dim).permute(0, 3, 1, 2)


# --- View-based patterns ---


def make_view_split_select(B, H, S, D, dtype, device):
    """View to split last dim into (D//2, 2), then select index 0.
    Creates stride-2 access pattern: strides [..., 2] instead of [..., 1].
    [B,H,S,D] -> view [B,H,S,D//2,2] -> [..., 0] -> [B,H,S,D//2] with stride[-1]=2."""
    return torch.randn(B, H, S, D, dtype=dtype, device=device).view(B, H, S, D // 2, 2)[..., 0]


def make_view_split_transpose(B, H, S, D, dtype, device):
    """View to split last dim, then transpose the split parts.
    [B,H,S,D] -> view [B,H,S,D//2,2] -> transpose(-1,-2) -> [B,H,S,2,D//2]
    Non-contiguous: innermost stride becomes D//2 instead of 1."""
    return (
        torch.randn(B, H, S, D, dtype=dtype, device=device)
        .view(B, H, S, D // 2, 2)
        .transpose(-1, -2)
    )


def make_view_on_sliced(B, H, S, D, dtype, device):
    """Slice first (creating non-contiguous), then view with same shape.
    view on a non-contiguous tensor with the same shape is a no-op that preserves
    the non-contiguous strides. Tests that the prologue handles this correctly.
    [B,H,S,D] -> [:,:,:,D//4:3*D//4] -> view(B,H,S,D//2) -> still non-contiguous."""
    sliced = torch.randn(B, H, S, D, dtype=dtype, device=device)[:, :, :, D // 4 : 3 * D // 4]
    return sliced.view(B, H, S, D // 2)


def make_view_split_select_transpose(B, H, S, D, dtype, device):
    """Compound: view split + select + transpose.
    [B,S,H,D] -> view [B,S,H,D//2,2] -> [..., 0] -> [B,S,H,D//2] (stride-2)
    -> transpose(1,2) -> [B,H,S,D//2] (stride-2 + transposed).
    Exercises the compound decomposition with both step factor and transpose."""
    return (
        torch.randn(B, S, H, D, dtype=dtype, device=device)
        .view(B, S, H, D // 2, 2)[..., 0]
        .transpose(1, 2)
    )


PATTERNS = {
    # --- Single transformations (existing paths) ---
    "contiguous": make_contiguous,
    "transpose_only": make_transpose_only,
    "slice_first_half": make_slice_first_half,
    "slice_second_half": make_slice_second_half,
    # --- Compound: transpose + slice ---
    "transpose_slice_first": make_transpose_slice_first,
    "transpose_slice_second": make_transpose_slice_second,  # RoPE x2
    "transpose_slice_seq": make_transpose_slice_seq,
    # --- Compound: multi-axis slice ---
    "multi_slice": make_multi_slice,
    # --- Pure broadcast ---
    "broadcast_expand": make_broadcast_expand,
    # --- Compound: broadcast + non-trivial non-broadcast dims ---
    "broadcast_transpose": make_broadcast_transpose,
    "broadcast_slice": make_broadcast_slice,
    "broadcast_transpose_slice": make_broadcast_transpose_slice,  # RoPE cos/sin
    # --- Patterns that were initially expected to be limitations but pass ---
    "outermost_dim_slice": lambda B, H, S, D, dtype, device: (
        # Batch-dim slice on transposed tensor. When B==1 there's nothing to slice,
        # so fall back to a plain transpose (still exercises the prologue path).
        torch.randn(B, S, H, D, dtype=dtype, device=device).transpose(1, 2)[max(0, B - 1) :, ...]
        if B >= 2
        else torch.randn(B, S, H, D, dtype=dtype, device=device).transpose(1, 2)
    ),
    "step_slice": lambda B, H, S, D, dtype, device: (
        # Step slice on last dim of transposed tensor -- stride=2, handled via contiguous path
        torch.randn(B, H, S, D, dtype=dtype, device=device).transpose(1, 2)[:, :, :, ::2]
    ),
    "flip": lambda B, H, S, D, dtype, device: (
        # torch.flip() on Neuron is a registered compute op (aten::flip in tensor_ops.py),
        # NOT a view op. It compiles to MLIR, executes on-device, and returns a new
        # contiguous tensor -- no negative strides. This tests end-to-end correctness.
        torch.randn(B, S, H, D, dtype=dtype, device=device).transpose(1, 2).flip(-1)
    ),
    # --- Folded patterns (narrow→view→permute, handled by TryFoldedDecomposition) ---
    "folded_narrow_view_permute": make_folded_narrow_view_permute,
    "folded_padded_sequence": make_folded_padded_sequence,
    # --- View-based patterns (view can preserve/create non-contiguous strides) ---
    "view_split_select": make_view_split_select,
    "view_split_transpose": make_view_split_transpose,
    "view_on_sliced": make_view_on_sliced,
    "view_split_select_transpose": make_view_split_select_transpose,
}

# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

OPS = {
    "neg": lambda x: -x,
    "mul": lambda x: x * x,
    "add": lambda x: x + x,
    "clone": lambda x: x.clone(),
}

# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

SHAPES = [
    (1, 32, 128, 128),  # Llama-like
    (2, 8, 64, 64),  # Smaller
]

# ---------------------------------------------------------------------------
# Full RoPE integration test
# ---------------------------------------------------------------------------


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Parametrized sweep
# ---------------------------------------------------------------------------


@SKIP_PROLOGUE_DISABLED
@pytest.mark.parametrize("pattern_name", list(PATTERNS.keys()))
@pytest.mark.parametrize("op_name", list(OPS.keys()))
@pytest.mark.parametrize("shape", SHAPES, ids=[f"B{b}H{h}S{s}D{d}" for b, h, s, d in SHAPES])
def test_compound_op(shape, op_name, pattern_name):
    """
    For each (shape, pattern, op):
      1. Build tensor on CPU using the pattern factory
      2. Run op on CPU -> reference
      3. Build same tensor on Neuron
      4. Run op on Neuron -> result
      5. Compare numerically
    """
    B, H, S, D = shape
    dtype = torch.bfloat16

    try:
        import torch_neuronx
    except ImportError:
        pytest.skip("torch_neuronx not available")

    torch.manual_seed(42)
    op = OPS[op_name]
    factory = PATTERNS[pattern_name]

    # --- CPU reference ---
    cpu_tensor = factory(B, H, S, D, dtype, "cpu")
    cpu_result = op(cpu_tensor)

    print(
        f"\n  pattern={pattern_name} op={op_name} shape={shape}"
        f"\n  tensor: shape={cpu_tensor.shape} strides={cpu_tensor.stride()} "
        f"offset={cpu_tensor.storage_offset()} contiguous={cpu_tensor.is_contiguous()}"
    )

    # --- Neuron ---
    torch.manual_seed(42)
    neuron_tensor = factory(B, H, S, D, dtype, "neuron")

    try:
        neuron_result = op(neuron_tensor)
        neuron_cpu = neuron_result.cpu()
    except RuntimeError as e:
        pytest.fail(f"Neuron execution failed with RuntimeError: {e}")

    # --- Compare ---
    has_nan = torch.isnan(neuron_cpu).any().item()
    has_inf = torch.isinf(neuron_cpu).any().item()

    assert not has_nan, (
        f"NaN in Neuron output for pattern={pattern_name} op={op_name}\n"
        f"  CPU sample: {cpu_result.float().flatten()[:8]}\n"
        f"  Neuron sample: {neuron_cpu.float().flatten()[:8]}"
    )
    assert not has_inf, f"Inf in Neuron output for pattern={pattern_name} op={op_name}"

    # View ops (slice, transpose, broadcast) are data-preserving — results must
    # be bitwise identical. Compute ops (neg, mul, add) are deterministic on
    # identical inputs, so atol=0, rtol=0 is correct.
    torch.testing.assert_close(neuron_cpu, cpu_result, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Full RoPE end-to-end test
# ---------------------------------------------------------------------------


@SKIP_PROLOGUE_DISABLED
@pytest.mark.parametrize("shape", SHAPES, ids=[f"B{b}H{h}S{s}D{d}" for b, h, s, d in SHAPES])
def test_full_rope(shape):
    """End-to-end test of apply_rotary_pos_emb matching the Llama/Qwen3 pattern.

    This exercises:
      - q/k: [B,S,H,D] -> transpose(1,2) -> [B,H,S,D]  (pure transpose)
      - rotate_half(q): x1=[..., :D//2] (compound: transpose+slice, offset=0)
                        x2=[..., D//2:] (compound: transpose+slice, offset>0)
      - cos/sin: [B,S,D] -> unsqueeze(1) -> [B,1,S,D]   (broadcast)
      - q*cos: broadcast multiply
    """
    B, H, S, D = shape
    dtype = torch.bfloat16

    try:
        import torch_neuronx
    except ImportError:
        pytest.skip("torch_neuronx not available")

    torch.manual_seed(0)

    # Build CPU tensors
    q_cpu = torch.randn(B, S, H, D, dtype=dtype).transpose(1, 2)  # [B,H,S,D]
    k_cpu = torch.randn(B, S, H, D, dtype=dtype).transpose(1, 2)  # [B,H,S,D]
    cos_cpu = torch.randn(B, S, D, dtype=dtype)
    sin_cpu = torch.randn(B, S, D, dtype=dtype)

    q_embed_cpu, k_embed_cpu = apply_rotary_pos_emb(q_cpu, k_cpu, cos_cpu, sin_cpu)

    # Build Neuron tensors
    torch.manual_seed(0)
    q_neuron = torch.randn(B, S, H, D, dtype=dtype, device="neuron").transpose(1, 2)
    k_neuron = torch.randn(B, S, H, D, dtype=dtype, device="neuron").transpose(1, 2)
    cos_neuron = torch.randn(B, S, D, dtype=dtype, device="neuron")
    sin_neuron = torch.randn(B, S, D, dtype=dtype, device="neuron")

    try:
        q_embed_neuron, k_embed_neuron = apply_rotary_pos_emb(
            q_neuron, k_neuron, cos_neuron, sin_neuron
        )
        q_embed_neuron_cpu = q_embed_neuron.cpu()
        k_embed_neuron_cpu = k_embed_neuron.cpu()
    except RuntimeError as e:
        pytest.fail(f"RoPE Neuron execution failed: {e}")

    torch.testing.assert_close(q_embed_neuron_cpu, q_embed_cpu, atol=0, rtol=0)
    torch.testing.assert_close(k_embed_neuron_cpu, k_embed_cpu, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Complex compound stress tests (within current capabilities)
# ---------------------------------------------------------------------------


@SKIP_PROLOGUE_DISABLED
@pytest.mark.parametrize("shape", SHAPES, ids=[f"B{b}H{h}S{s}D{d}" for b, h, s, d in SHAPES])
def test_complex_transpose_two_slices(shape):
    """Compound: transpose + 2 independent slices (seq dim AND head_dim).

    Decomposition: base -> SLICE(head_dim) -> SLICE(seq) -> TRANSPOSE
    """
    B, H, S, D = shape
    dtype = torch.bfloat16
    try:
        import torch_neuronx
    except ImportError:
        pytest.skip("torch_neuronx not available")

    torch.manual_seed(1)
    # [B,S,H,D] -> transpose(1,2) -> [B,H,S,D] -> [:,:,:S//2,:] -> [..., D//4:3*D//4]
    cpu = torch.randn(B, S, H, D, dtype=dtype).transpose(1, 2)[:, :, : S // 2, :][
        ..., D // 4 : 3 * D // 4
    ]
    cpu_result = -cpu

    torch.manual_seed(1)
    neuron = torch.randn(B, S, H, D, dtype=dtype, device="neuron").transpose(1, 2)[
        :, :, : S // 2, :
    ][..., D // 4 : 3 * D // 4]
    try:
        neuron_result = -neuron
        neuron_cpu = neuron_result.cpu()
    except RuntimeError as e:
        pytest.fail(f"complex_transpose_two_slices failed: {e}")

    torch.testing.assert_close(neuron_cpu, cpu_result, atol=0, rtol=0)


@SKIP_PROLOGUE_DISABLED
@pytest.mark.parametrize("shape", SHAPES, ids=[f"B{b}H{h}S{s}D{d}" for b, h, s, d in SHAPES])
def test_complex_broadcast_transpose_two_slices(shape):
    """Compound: broadcast + transpose + 2 slices -- the most complex 4D pattern.

    [B,S,D] -> unsqueeze(1) -> expand [B,H,S,D] -> transpose(1,2) -> [B,S,H,D]
             -> [:, :S//2, :, :] -> [..., D//4:]
    Decomposition: base -> SLICE(head_dim) -> SLICE(seq) -> TRANSPOSE -> RESHAPE -> BROADCAST
    """
    B, H, S, D = shape
    dtype = torch.bfloat16
    try:
        import torch_neuronx
    except ImportError:
        pytest.skip("torch_neuronx not available")

    torch.manual_seed(2)
    cpu = (
        torch.randn(B, S, D, dtype=dtype)
        .unsqueeze(1)
        .expand(B, H, S, D)
        .transpose(1, 2)[:, : S // 2, :, :][..., D // 4 :]
    )
    cpu_result = cpu * cpu

    torch.manual_seed(2)
    neuron = (
        torch.randn(B, S, D, dtype=dtype, device="neuron")
        .unsqueeze(1)
        .expand(B, H, S, D)
        .transpose(1, 2)[:, : S // 2, :, :][..., D // 4 :]
    )
    try:
        neuron_result = neuron * neuron
        neuron_cpu = neuron_result.cpu()
    except RuntimeError as e:
        pytest.fail(f"complex_broadcast_transpose_two_slices failed: {e}")

    torch.testing.assert_close(neuron_cpu, cpu_result, atol=0, rtol=0)


@SKIP_PROLOGUE_DISABLED
@pytest.mark.parametrize("shape", SHAPES, ids=[f"B{b}H{h}S{s}D{d}" for b, h, s, d in SHAPES])
def test_complex_two_broadcast_dims_transpose_slice(shape):
    """Compound: 2 broadcast dims + transpose + slice -- 6D tensor.

    [B, S, D] -> unsqueeze(1) -> unsqueeze(3) -> expand [B,H,S,G,D]
              -> transpose(1,2) -> [B,S,H,G,D] -> [..., D//2:]
    Decomposition: 2 broadcast dims + transpose + slice
    """
    B, H, S, D = shape
    G = 4  # extra group dim
    dtype = torch.bfloat16
    try:
        import torch_neuronx
    except ImportError:
        pytest.skip("torch_neuronx not available")

    torch.manual_seed(3)
    cpu = (
        torch.randn(B, S, D, dtype=dtype)
        .unsqueeze(1)
        .unsqueeze(3)  # [B, 1, S, 1, D]
        .expand(B, H, S, G, D)  # [B, H, S, G, D] -- 2 broadcast dims
        .transpose(1, 2)[  # [B, S, H, G, D]
            ..., D // 2 :
        ]
    )  # [B, S, H, G, D//2]
    cpu_result = cpu + cpu

    torch.manual_seed(3)
    neuron = (
        torch.randn(B, S, D, dtype=dtype, device="neuron")
        .unsqueeze(1)
        .unsqueeze(3)
        .expand(B, H, S, G, D)
        .transpose(1, 2)[..., D // 2 :]
    )
    try:
        neuron_result = neuron + neuron
        neuron_cpu = neuron_result.cpu()
    except RuntimeError as e:
        pytest.fail(f"complex_two_broadcast_dims_transpose_slice failed: {e}")

    torch.testing.assert_close(neuron_cpu, cpu_result, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Known limitation tests — xfail (expected to fail until implemented)
#
# True remaining limitation of TryCreateCompound:
#   1. Overlapping strides (unfold with step < size): elements share memory,
#      cannot be represented as slice+transpose+broadcast.
#
# Note: Negative strides are NOT a practical limitation on Neuron.
#   On CPU, torch.flip() and x[::-1] produce negative-stride views.
#   On Neuron, both are registered compute ops (aten::flip in tensor_ops.py)
#   that compile to MLIR and return new contiguous tensors — no negative strides.
#   The prologue system's negative-stride guard (strides[i] < 0) is a defensive
#   safety check that is never triggered in practice.
# ---------------------------------------------------------------------------


@SKIP_PROLOGUE_DISABLED
@pytest.mark.xfail(
    strict=False,
    reason=(
        "KNOWN LIMITATION: Overlapping strides (unfold with step < size). "
        "x.unfold(dim, size, step) where step < size creates overlapping memory regions "
        "that cannot be represented as a simple slice+transpose+broadcast chain. "
        "TryCreateCompound rejects this (implied_size <= current_size), so it falls "
        "through to contiguous_generic_kernel which does a device→CPU→device roundtrip. "
        "The result is numerically correct but not on-device efficient. "
        "Fix: add a GATHER/STRIDED_SLICE transformation type."
    ),
)
@pytest.mark.parametrize("shape", SHAPES, ids=[f"B{b}H{h}S{s}D{d}" for b, h, s, d in SHAPES])
def test_xfail_overlapping_strides(shape):
    """Overlapping strides via unfold(step < size).

    Numerically correct but uses contiguous_generic_kernel (CPU roundtrip) instead
    of the prologue MLIR system. The no_cpu_fallback fixture's Layer 3 check catches this.
    """
    B, H, S, D = shape
    dtype = torch.bfloat16
    try:
        import torch_neuronx
    except ImportError:
        pytest.skip("torch_neuronx not available")

    torch.manual_seed(20)
    cpu = torch.randn(B, H, S, D, dtype=dtype).unfold(2, 4, 2)  # [B, H, S//2-1, D, 4]
    cpu_result = -cpu

    torch.manual_seed(20)
    neuron = torch.randn(B, H, S, D, dtype=dtype, device="neuron").unfold(2, 4, 2)
    neuron_result = -neuron
    neuron_cpu = neuron_result.cpu()

    torch.testing.assert_close(neuron_cpu, cpu_result, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Broadcast + compound interaction test
# ---------------------------------------------------------------------------


@SKIP_PROLOGUE_DISABLED
@pytest.mark.parametrize("shape", SHAPES, ids=[f"B{b}H{h}S{s}D{d}" for b, h, s, d in SHAPES])
def test_broadcast_compound_mul(shape):
    """Test the exact RoPE multiply pattern: q * cos where:
      - q has shape [B,H,S,D] (transposed, non-contiguous)
      - cos has shape [B,1,S,D] (broadcast, stride=0 on H dim)
    This exercises the interaction between compound (q) and broadcast (cos) inputs.
    """
    B, H, S, D = shape
    dtype = torch.bfloat16

    try:
        import torch_neuronx
    except ImportError:
        pytest.skip("torch_neuronx not available")

    torch.manual_seed(7)

    # q: transposed [B,H,S,D]
    q_cpu = torch.randn(B, S, H, D, dtype=dtype).transpose(1, 2)
    # cos: broadcast [B,1,S,D]
    cos_cpu = torch.randn(B, S, D, dtype=dtype).unsqueeze(1)

    result_cpu = q_cpu * cos_cpu

    torch.manual_seed(7)
    q_neuron = torch.randn(B, S, H, D, dtype=dtype, device="neuron").transpose(1, 2)
    cos_neuron = torch.randn(B, S, D, dtype=dtype, device="neuron").unsqueeze(1)

    try:
        result_neuron = q_neuron * cos_neuron
        result_neuron_cpu = result_neuron.cpu()
    except RuntimeError as e:
        pytest.fail(f"broadcast_compound_mul failed: {e}")

    torch.testing.assert_close(result_neuron_cpu, result_cpu, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Random mixed sequence fuzz test
#
# Generates random sequences of 1-10 transformations (transpose, slice,
# expand/broadcast) applied to a contiguous tensor, then runs a simple
# element-wise operation and compares CPU vs Neuron results.
#
# Uses numpy.random with a random seed (printed on failure for reproduction).
# ---------------------------------------------------------------------------

# Available random operations for the fuzz test.
# Excluded ops and why:
#   - diagonal, unfold: create overlapping strides (known limitation of TryCreateCompound)
#   - unbind, chunk, split: return tuples/lists, can't chain on a single tensor
#   - flatten, unflatten: equivalent to view merge/split, already covered by "view"
_FUZZ_OPS = [
    "transpose",
    "permute",
    "slice",
    "select",
    "narrow",
    "expand",
    "unsqueeze",
    "squeeze",
    "view",
]


def _generate_and_apply_sequence(tensor, rng, num_ops):
    """Apply a random sequence of view ops and return the replayable actions.

    Instead of string parsing, this records the exact operations as lambdas
    so they can be perfectly replayed on the Neuron tensor.
    """
    op_records = []
    op_log = []

    for _ in range(num_ops):
        ndim = tensor.dim()
        if ndim < 1:
            break

        # Filter valid ops based on rank
        valid_ops = list(_FUZZ_OPS)
        if ndim < 2:
            # transpose and permute need at least 2 dims
            for remove_op in ["transpose", "permute"]:
                if remove_op in valid_ops:
                    valid_ops.remove(remove_op)
            # select reduces ndim by 1, so need ndim >= 2 to avoid 0-dim
            if "select" in valid_ops:
                valid_ops.remove("select")

        op = rng.choice(valid_ops)

        try:
            if op == "transpose":
                dims = rng.choice(ndim, size=2, replace=False).tolist()
                d0, d1 = int(dims[0]), int(dims[1])
                tensor = tensor.transpose(d0, d1)
                op_records.append(lambda t, d0=d0, d1=d1: t.transpose(d0, d1))
                op_log.append(f"transpose({d0},{d1}) -> shape={list(tensor.shape)}")

            elif op == "permute":
                # Random permutation of all dims (generalization of transpose)
                perm = rng.permutation(ndim).tolist()
                perm = [int(p) for p in perm]
                tensor = tensor.permute(perm)
                op_records.append(lambda t, p=perm: t.permute(p))
                op_log.append(f"permute({perm}) -> shape={list(tensor.shape)}")

            elif op == "select":
                # Pick one index along a random dim, reducing ndim by 1
                # Guard: only if ndim >= 2 (filtered above, but double-check)
                if ndim < 2:
                    op_log.append("select SKIPPED (ndim < 2)")
                    continue
                dim = int(rng.integers(0, ndim))
                dim_size = tensor.size(dim)
                if dim_size <= 1:
                    op_log.append(f"select(dim={dim}) SKIPPED (size=1)")
                    continue
                idx = int(rng.integers(0, dim_size))
                tensor = tensor.select(dim, idx)
                op_records.append(lambda t, d=dim, i=idx: t.select(d, i))
                op_log.append(f"select(dim={dim}, idx={idx}) -> shape={list(tensor.shape)}")

            elif op == "narrow":
                # Narrow a random dim to a sub-range (like slice but explicit API)
                dim = int(rng.integers(0, ndim))
                dim_size = tensor.size(dim)
                if dim_size <= 1:
                    op_log.append(f"narrow(dim={dim}) SKIPPED (size=1)")
                    continue
                start = int(rng.integers(0, dim_size - 1))
                length = int(rng.integers(1, dim_size - start + 1))
                tensor = torch.narrow(tensor, dim, start, length)
                op_records.append(
                    lambda t, d=dim, s=start, len_=length: torch.narrow(t, d, s, len_)
                )
                op_log.append(
                    f"narrow(dim={dim}, start={start}, length={length}) "
                    f"-> shape={list(tensor.shape)}"
                )

            elif op == "slice":
                dim = int(rng.integers(0, ndim))
                dim_size = tensor.size(dim)
                if dim_size <= 1:
                    op_log.append(f"slice(dim={dim}) SKIPPED (size=1)")
                    continue

                start = int(rng.integers(0, dim_size - 1))
                end = int(rng.integers(start + 1, dim_size + 1))

                # 50% chance of a step slice if range is large enough
                step = int(rng.integers(2, 4)) if rng.random() > 0.5 and (end - start) >= 2 else 1

                # Construct the N-dimensional slice object
                idx = [slice(None)] * ndim
                idx[dim] = slice(start, end, step)
                idx = tuple(idx)

                tensor = tensor[idx]
                op_records.append(lambda t, idx=idx: t[idx])
                op_log.append(
                    f"slice(dim={dim}, {start}:{end}:{step}) -> shape={list(tensor.shape)}"
                )

            elif op == "expand":
                dim = int(rng.integers(0, ndim))
                dim_size = tensor.size(dim)

                if dim_size > 1:
                    # Narrow to size 1 first so we can legally expand it (stride=0 broadcast)
                    start = int(rng.integers(0, dim_size))
                    tensor = torch.narrow(tensor, dim, start, 1)
                    op_records.append(
                        lambda t, dim=dim, start=start: torch.narrow(t, dim, start, 1)
                    )
                    op_log.append(f"narrow_to_1(dim={dim}, start={start})")

                expand_size = int(rng.integers(2, 7))
                expand_shape = list(tensor.shape)
                expand_shape[dim] = expand_size
                tensor = tensor.expand(expand_shape)
                op_records.append(lambda t, shape=expand_shape: t.expand(shape))
                op_log.append(
                    f"expand(dim={dim}, size={expand_size}) -> shape={list(tensor.shape)}"
                )

            elif op == "unsqueeze":
                dim = int(rng.integers(0, ndim + 1))
                tensor = tensor.unsqueeze(dim)
                op_records.append(lambda t, dim=dim: t.unsqueeze(dim))
                op_log.append(f"unsqueeze(dim={dim}) -> shape={list(tensor.shape)}")

            elif op == "squeeze":
                size_1_dims = [i for i in range(ndim) if tensor.size(i) == 1]
                if size_1_dims:
                    dim = int(rng.choice(size_1_dims))
                    tensor = tensor.squeeze(dim)
                    op_records.append(lambda t, dim=dim: t.squeeze(dim))
                    op_log.append(f"squeeze(dim={dim}) -> shape={list(tensor.shape)}")
                else:
                    op_log.append("squeeze SKIPPED (no size-1 dims)")

            elif op == "view":
                # view/reshape can split dims, merge dims, or keep same shape.
                # On contiguous tensors: any reshape works (just reinterprets strides).
                # On non-contiguous tensors: view with same shape is a no-op that
                #   preserves strides; view with incompatible shape raises RuntimeError.
                #
                # We try multiple strategies randomly:
                #   1. Split a dim: [.., N, ..] -> [.., A, B, ..] where A*B=N
                #   2. Merge adjacent dims: [.., A, B, ..] -> [.., A*B, ..]
                #   3. Same-shape view (no-op on non-contiguous, tests view passthrough)
                #
                # All are wrapped in try/except since non-contiguous tensors may reject
                # incompatible reshapes.

                view_strategy = int(rng.integers(0, 3))

                if view_strategy == 0:
                    # Split a random dim into two factors.
                    # View-split on non-contiguous tensors creates stride patterns where
                    # GCD-based factorization fails (coprime stride ratios). These are
                    # handled by TryFoldedDecomposition which detects compact
                    # sub-dimension groups and emits SLICE→RESHAPE→TRANSPOSE.
                    #
                    # Example:
                    #   x = torch.randn(2, 3, 5)
                    #   y = x.narrow(2, 0, 4).view(2, 3, 2, 2).permute(2, 0, 3, 1)
                    #   # shape=[2, 2, 2, 3], strides=(2, 15, 1, 5)
                    #   # GCD(5, 2) = 1 → DecomposeStrides rejects
                    #   # TryFoldedDecomposition: base [2, 3, 5] → SLICE → RESHAPE → TRANSPOSE

                    def _get_factors(n):
                        """Get all factor pairs (a, b) where a*b=n, a>1, b>1."""
                        factors = []
                        for a in range(2, int(n**0.5) + 1):
                            if n % a == 0:
                                factors.append((a, n // a))
                        return factors

                    splittable = [
                        (i, _get_factors(tensor.size(i)))
                        for i in range(ndim)
                        if tensor.size(i) >= 4
                    ]
                    splittable = [(i, fs) for i, fs in splittable if fs]

                    if splittable:
                        dim, factors = splittable[int(rng.integers(0, len(splittable)))]
                        a, b = factors[int(rng.integers(0, len(factors)))]
                        new_shape = list(tensor.shape)
                        new_shape[dim : dim + 1] = [a, b]
                        tensor = tensor.view(new_shape)
                        op_records.append(lambda t, s=new_shape: t.view(s))
                        op_log.append(
                            f"view(split dim={dim}, {a}x{b}) -> shape={list(tensor.shape)}"
                        )
                    else:
                        op_log.append("view(split) SKIPPED (no splittable dims)")

                elif view_strategy == 1:
                    # Merge two adjacent dims.
                    # Only on contiguous tensors — same GCD-based limitation as
                    # split above. Merging dims on a non-contiguous tensor creates
                    # stride patterns where the merged stride doesn't factor cleanly
                    # into the original base dimensions.
                    if not tensor.is_contiguous():
                        op_log.append("view(merge) SKIPPED (non-contiguous, known limitation)")
                        continue
                    if ndim >= 2:
                        dim = int(rng.integers(0, ndim - 1))
                        merged = tensor.size(dim) * tensor.size(dim + 1)
                        new_shape = list(tensor.shape)
                        new_shape[dim : dim + 2] = [merged]
                        tensor = tensor.view(new_shape)
                        op_records.append(lambda t, s=new_shape: t.view(s))
                        op_log.append(
                            f"view(merge dims {dim},{dim+1} -> {merged}) "
                            f"-> shape={list(tensor.shape)}"
                        )
                    else:
                        op_log.append("view(merge) SKIPPED (ndim < 2)")

                else:
                    # Same-shape view (no-op, preserves non-contiguous strides)
                    same_shape = list(tensor.shape)
                    tensor = tensor.view(same_shape)
                    op_records.append(lambda t, s=same_shape: t.view(s))
                    op_log.append(f"view(same_shape) -> shape={list(tensor.shape)}")

        except RuntimeError:
            # Catch cases where PyTorch naturally rejects the view combination
            pass

    return tensor, op_records, op_log


# ---------------------------------------------------------------------------
# Fallback detection self-test
#
# Verifies that the _fuzzer_fallback_triggered mechanism actually works by
# intentionally triggering a known fallback case (unfold with overlapping
# strides) and asserting that the detection fires.
# ---------------------------------------------------------------------------


@SKIP_PROLOGUE_DISABLED
@pytest.mark.xfail(
    os.environ.get("TORCH_NEURONX_CPP_DISPATCH", "1") != "0",
    strict=True,
    reason=(
        "C++ dispatch routes contiguous through DispatchContiguousImpl which "
        "falls back to ATen CPU fallback, never reaching the Python "
        "contiguous_generic_kernel. The fallback detection mechanism only "
        "works with the Python dispatch path (TORCH_NEURONX_CPP_DISPATCH=0)."
    ),
)
def test_fallback_detection_self_test():
    """Meta-test: verify that our fallback detection infrastructure actually works.

    Creates a tensor with overlapping strides (unfold, step < size) which is a
    known limitation that falls through to contiguous_generic_kernel. Then checks
    that _fuzzer_fallback_triggered correctly caught it.

    If this test FAILS, it means our detection mechanism is broken and the fuzz
    tests cannot be trusted to catch regressions.
    """
    try:
        import torch_neuronx
    except ImportError:
        pytest.skip("torch_neuronx not available")

    import torch_neuronx.python_ops.nki_kernels.contiguous_generic as _cg_module

    _cg_module._fuzzer_fallback_triggered.clear()

    # unfold with step < size creates overlapping strides — a known fallback case
    torch.manual_seed(99)
    neuron_tensor = torch.randn(2, 4, 16, 8, dtype=torch.bfloat16, device="neuron")
    unfolded = neuron_tensor.unfold(2, 4, 2)  # overlapping strides

    # This neg op will trigger contiguous_generic_kernel for the unfolded tensor
    result = -unfolded
    _ = result.cpu()

    assert _cg_module._fuzzer_fallback_triggered, (
        "SELF-TEST FAILED: contiguous_generic_kernel was NOT called for a known "
        "fallback case (unfold with overlapping strides). This means the fallback "
        "detection mechanism is broken and fuzz tests cannot be trusted."
    )
    print("\n  [fallback_detection_self_test] Detection mechanism verified ✓")


_FUZZ_NUM_ITERATIONS = 200


@SKIP_PROLOGUE_DISABLED
@pytest.mark.parametrize("iteration", range(_FUZZ_NUM_ITERATIONS))
def test_random_mixed_sequences(iteration):
    """Fuzz test: random view-op sequences to find prologue system bugs.

    Each iteration generates a random seed, applies random view ops (transpose,
    slice, permute, select, expand, etc.) to create a non-contiguous tensor, then
    runs neg() on Neuron and compares against CPU. Any failure is a real bug —
    the prologue system should handle ALL standard view patterns on-device.

    When this test fails:
      1. Note the seed printed in the failure message
      2. File a ticket for test_random_mixed_sequences with the seed
      3. Fix the bug in the prologue system (Transformations.cpp)

    Run parallelized:  pytest -k test_random_mixed_sequences -n 8
    """
    try:
        import torch_neuronx
    except ImportError:
        pytest.skip("torch_neuronx not available")

    # Each iteration gets its own independent random seed
    seed = np.random.SeedSequence().entropy
    rng = np.random.default_rng(seed)

    dtype = torch.bfloat16
    rank = int(rng.integers(2, 5))
    shape = [int(rng.integers(2, 9)) for _ in range(rank)]
    num_ops = int(rng.integers(1, 11))
    torch_seed = int(rng.integers(0, 2**31))

    # --- CPU ---
    torch.manual_seed(torch_seed)
    cpu_tensor = torch.randn(shape, dtype=dtype)

    # Apply ops and get the replayable list of lambdas
    cpu_transformed, op_records, op_log = _generate_and_apply_sequence(cpu_tensor, rng, num_ops)

    if cpu_transformed.is_contiguous() or any(s == 0 for s in cpu_transformed.shape):
        pytest.skip("Generated contiguous or zero-size tensor")

    cpu_result = -cpu_transformed

    # --- Neuron ---
    torch.manual_seed(torch_seed)
    neuron_tensor = torch.randn(shape, dtype=dtype, device="neuron")

    # Flawlessly replay the exact same ops on the Neuron tensor
    neuron_transformed = neuron_tensor
    for op_func in op_records:
        neuron_transformed = op_func(neuron_transformed)

    try:
        import torch_neuronx.python_ops.nki_kernels.contiguous_generic as _cg_module

        _cg_module._fuzzer_fallback_triggered.clear()  # Clear before execution

        # THIS is the operation guarded by the no_cpu_fallback fixture
        neuron_result = -neuron_transformed
        neuron_cpu = neuron_result.cpu()

        # Fail immediately if the prologue fell through to the CPU roundtrip path.
        if _cg_module._fuzzer_fallback_triggered:
            pytest.fail(
                f"\n{'='*70}\n"
                f"PROLOGUE BUG: CPU fallback triggered for non-contiguous pattern\n"
                f"{'='*70}\n"
                f"Seed: {seed}\n"
                f"Please file a ticket with this seed to track the fix.\n"
                f"Shape: {shape}, num_ops: {num_ops}\n"
                f"Operations:\n"
                + "\n".join([f"  {entry}" for entry in op_log])
                + f"\nFinal shape: {list(cpu_transformed.shape)}\n"
                f"Final strides: {list(cpu_transformed.stride())}\n"
                f"Offset: {cpu_transformed.storage_offset()}\n"
                f"contiguous_generic_kernel was called — the prologue system did NOT "
                f"handle this non-contiguous tensor pattern on-device.\n"
                f"{'='*70}\n"
            )

    except RuntimeError as e:
        pytest.fail(
            f"\n{'='*70}\n"
            f"PROLOGUE BUG: Runtime error on non-contiguous pattern\n"
            f"{'='*70}\n"
            f"Seed: {seed}\n"
            f"Please file a ticket with this seed to track the fix.\n"
            f"Shape: {shape}, num_ops: {num_ops}\n"
            f"Operations:\n"
            + "\n".join([f"  {entry}" for entry in op_log])
            + f"\nFinal shape: {list(cpu_transformed.shape)}\n"
            f"Error: {e}\n"
            f"{'='*70}\n"
        )

    try:
        torch.testing.assert_close(neuron_cpu, cpu_result, atol=0, rtol=0)
    except AssertionError as e:
        pytest.fail(
            f"\n{'='*70}\n"
            f"PROLOGUE BUG: Numerical mismatch on non-contiguous pattern\n"
            f"{'='*70}\n"
            f"Seed: {seed}\n"
            f"Please file a ticket with this seed to track the fix.\n"
            f"Shape: {shape}, num_ops: {num_ops}\n"
            f"Operations:\n"
            + "\n".join([f"  {entry}" for entry in op_log])
            + f"\nFinal shape: {list(cpu_transformed.shape)}\n"
            f"Error: {e}\n"
            f"{'='*70}\n"
        )

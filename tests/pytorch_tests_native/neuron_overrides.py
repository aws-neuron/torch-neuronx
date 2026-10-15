"""Neuron runtime patches for upstream Dynamo PyTorch tests.

Temporary workaround until upstream tests are refactored to be backend-agnostic.
These patches require confirmation from the PyTorch WorkGroup

Patches torch.compile, torch.allclose, torch.equal, and TestCase.assertEqual to
handle device-agnostic test execution on Neuron without modifying upstream test
files.

- torch.compile: redirects standard backends (eager, inductor, etc.) to "neuron"
- torch.allclose / torch.equal / assertEqual: move tensors to the same device
  before comparison
"""

import torch
import torch.testing._internal.common_utils as _common_utils

_original_compile = torch.compile
_COMPILE_REDIRECT = frozenset({"eager", "aot_eager", "inductor", "aot_eager_decomp_partition"})

# Noise floor for a float32 gradient computed on Neuron. The float32 truncation
# floor is O(eps^2) ≈ 1e-6, but condition-number amplification, non-associative
# summation, and catastrophic cancellation push the observed error to ~1e-3 in
# the near-zero regime where rtol can't help. Every gradient tolerance we widen
# for Neuron is bounded by this same physical limit, so they share one value.
# Retune here and both gradcheck and split-backward comparisons move together.
_FLOAT32_GRAD_NOISE_FLOOR = 1e-3


def _neuron_compile(model=None, *, backend="eager", **kwargs):
    """Redirect torch.compile backends to Neuron"""
    if backend in _COMPILE_REDIRECT:
        backend = "neuron"
    return _original_compile(model, backend=backend, **kwargs)


_original_allclose = torch.allclose


def _cross_device_allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Patch torch.allclose to handle cross-device comparison (neuron vs cpu).

    The body runs under ``DisableTorchFunction`` for the same reason as
    ``_cross_device_equal`` below. ``test_meta.py`` keys its
    ``meta_function_expected_failures`` table by function identity at import
    time, which happens after this patch. The table therefore holds this shim,
    but the meta crossref mode intercepts the original builtin. The lookup
    misses the entry, so the mode runs the meta comparison and fails:
    ``allclose`` calls ``item()``, and ``aten::_local_scalar_dense`` has no
    meta kernel. Hide the inner call to keep the expected-failure semantics.
    """
    with torch._C.DisableTorchFunction():
        if input.device != other.device:
            other = other.to(input.device)
        return _original_allclose(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan)


_original_equal = torch.equal


def _cross_device_equal(input, other):
    """Patch torch.equal to handle cross-device comparison (neuron vs cpu).

    Upstream dynamo tests build a CPU reference and compare it against a
    compiled result that the redirected ``backend="neuron"`` path returns on
    ``neuron:0`` (e.g. ``test_exceptions.py::test_autocast_with_exception``).
    The real ``aten::equal`` kernel refuses cross-device operands by contract,
    so — as with ``allclose``/``assertEqual`` above — co-locate before
    comparing. Non-tensor operands fall through to the original.

    The body runs under ``DisableTorchFunction`` so ``TorchFunctionMode``
    harnesses never intercept the inner call. ``test_meta.py`` keys its
    ``meta_function_skips`` table by function identity at import time — after
    this patch is applied — so the table holds *this shim* while the meta
    crossref mode intercepts the *original* builtin, misses the skip entry,
    and crashes (``aten::equal`` has no meta kernel on any backend). Hiding
    the inner call reproduces the skip semantics upstream already mandates
    for ``torch.equal`` across all dtypes.
    """
    with torch._C.DisableTorchFunction():
        if (
            isinstance(input, torch.Tensor)
            and isinstance(other, torch.Tensor)
            and input.device != other.device
        ):
            other = other.to(input.device)
        return _original_equal(input, other)


_original_assert_equal = _common_utils.TestCase.assertEqual


def _cross_device_assert_equal(self, x, y, *args, **kwargs):
    """Patch TestCase.assertEqual to handle cross-device tensor comparison"""
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.device != y.device:
        y = y.to(x.device)
    return _original_assert_equal(self, x, y, *args, **kwargs)


_original_gradcheck = _common_utils.gradcheck
_original_gradgradcheck = _common_utils.gradgradcheck

# Capture the upstream defaults before override_gradcheck_defaults() replaces them.
_UPSTREAM_GRADCHECK_DEFAULTS = {
    "eps": _common_utils.GRADCHECK_DEFAULT_EPS,
    "atol": _common_utils.GRADCHECK_DEFAULT_ATOL,
    "rtol": _common_utils.GRADCHECK_DEFAULT_RTOL,
}


def _has_neuron_input(inputs) -> bool:
    """True when any tensor in a nested gradcheck input is on a Neuron device."""
    stack = [inputs]
    while stack:
        obj = stack.pop()
        if isinstance(obj, torch.Tensor):
            if obj.device.type == "neuron":
                return True
        elif isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, list | tuple | set):
            stack.extend(obj)
    return False


def _dtype_aware_gradcheck(original):
    """Keep the upstream gradcheck defaults for a check that no Neuron tensor reaches.

    ``override_gradcheck_defaults`` widens eps and atol for the whole session. A
    test that runs only on the CPU keeps true float64 precision and does not need
    the wider values. The wider eps also breaks such a test: it perturbs an input
    far enough to leave the valid range. For example, gradcheck of a
    differentiable Adam moves beta2 from 0.999 to 1.0, and Adam then rejects it.
    """

    def wrapper(fn, inputs, *args, **kwargs):
        if not _has_neuron_input(inputs):
            for key, value in _UPSTREAM_GRADCHECK_DEFAULTS.items():
                if kwargs.get(key) is None:
                    kwargs[key] = value
        return original(fn, inputs, *args, **kwargs)

    return wrapper


def override_gradcheck_defaults():
    """Relax gradcheck tolerances for Neuron's float32 precision.

    Neuron autocasts float64 to float32, so PyTorch's float64-tuned
    defaults don't apply. Use the historical Neuron values:

      eps=1e-3   — For central-difference gradcheck the optimal step
                   is eps_opt = (3*u)^(1/3) where u is machine epsilon.
                   float32 u ≈ 1.19e-7 → eps_opt ≈ 5.6e-3; 1e-3 is
                   within an order of magnitude and is the historical
                   value used for Neuron hardware.
      atol      — ``_FLOAT32_GRAD_NOISE_FLOOR``; see its definition for
                   why float32 gradient error on Neuron lands at ~1e-3.

    rtol is left at the upstream default (1e-3) — relative error is
    not the binding constraint here.

    Also forces ``TEST_WITH_SLOW_GRADCHECK=False`` as a safety net in
    case ``PYTORCH_TEST_WITH_SLOW_GRADCHECK=1`` leaks in from the
    environment.

    Installs ``_dtype_aware_gradcheck`` so a CPU-only check keeps the
    upstream values.
    """
    _common_utils.GRADCHECK_DEFAULT_EPS = 1e-3
    _common_utils.GRADCHECK_DEFAULT_ATOL = _FLOAT32_GRAD_NOISE_FLOOR
    _common_utils.TEST_WITH_SLOW_GRADCHECK = False
    _common_utils.gradcheck = _dtype_aware_gradcheck(_original_gradcheck)
    _common_utils.gradgradcheck = _dtype_aware_gradcheck(_original_gradgradcheck)


def override_split_backward_grad_tolerances():
    """Relax split-backward gradient tolerances for Neuron's float32 precision.

    Tests that split a backward pass compare the result against a monolithic
    .backward() reference. The two accumulate their sums in a different
    order, and float32 addition is not associative, so the two disagree by
    more than the upstream float32 default (rtol=1.3e-6, atol=1e-5). CPU and
    CUDA hide this because their matmul accumulates in full float32 width;
    Neuron's lower-precision matmul accumulation surfaces it. The results are
    still correct to float32 hardware epsilon.
    """
    _common_utils.SPLIT_BACKWARD_GRAD_ATOL = _FLOAT32_GRAD_NOISE_FLOOR
    _common_utils.SPLIT_BACKWARD_GRAD_RTOL = _FLOAT32_GRAD_NOISE_FLOOR


def _enable_neuron_attention_flags():
    """Force PLATFORM_SUPPORTS_*_ATTENTION=True on Neuron.

    torch.testing._internal.common_cuda has LazyVals driven by
    `evaluate_platform_supports_*()` that only recognise CUDA/ROCm/XPU.
    Neuron supports SDPA / flash / mem-efficient attention at the op level
    (via the PrivateUse1 dispatcher), so attention tests should exercise
    real code paths instead of short-circuiting to a CUDA-only skip.
    """
    import os

    if os.environ.get("PYTORCH_TESTING_DEVICE_ONLY_FOR") != "neuron":
        return
    from torch.testing._internal import common_cuda

    common_cuda.PLATFORM_SUPPORTS_FLASH_ATTENTION = True
    common_cuda.PLATFORM_SUPPORTS_FUSED_ATTENTION = True
    common_cuda.PLATFORM_SUPPORTS_CUDNN_ATTENTION = False


def _filter_complex_scalars_in_opinfo_samples():
    """Filter OpInfo samples that inject complex Python scalars for non-complex dtypes.

    Upstream ``reference_inputs_where`` in
    ``torch.testing._internal.common_methods_invocations`` yields a ``2j``
    (complex) Python scalar as part of its type-promotion loop for every
    dtype::

        for scalar in (0, 0.0, 2j, False):
            yield SampleInput(scalar, ...)

    On CUDA this silently promotes to a complex output via CPU fallback. On
    Neuron, ``TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1`` disables
    fallback for implemented ops, so the complex scalar raises an error even
    when the parameterized dtype (e.g. ``bool``, ``int32``, ``float32``) is
    not complex.

    We wrap ``reference_inputs_where`` to drop samples whose first positional
    argument is a complex Python scalar when the test dtype is non-complex.
    Complex-dtype variants (``complex64``/``complex128``/``chalf``) still
    receive the original samples unchanged.

    TODO: Remove once the upstream PR guards the complex scalar behind
    ``if dtype.is_complex:`` and lands in our fork.
    """
    from torch.testing._internal import common_methods_invocations as cmi

    original = cmi.reference_inputs_where

    def _patched_reference_inputs_where(op_info, device, dtype, requires_grad, **kwargs):
        for sample in original(op_info, device, dtype, requires_grad, **kwargs):
            if not dtype.is_complex:
                # Use type() instead of isinstance() so only true ``complex``
                # scalars are filtered; int/float/bool scalars still pass
                # through to exercise their type-promotion paths.
                input_val = sample.input
                if type(input_val) is complex:
                    continue
                args = sample.args or ()
                if args and type(args[-1]) is complex:
                    continue
            yield sample

    cmi.reference_inputs_where = _patched_reference_inputs_where


def apply_patches():
    """Apply all Neuron test patches."""
    torch.compile = _neuron_compile
    torch.allclose = _cross_device_allclose
    torch.equal = _cross_device_equal
    _common_utils.TestCase.assertEqual = _cross_device_assert_equal
    override_gradcheck_defaults()
    override_split_backward_grad_tolerances()
    _enable_neuron_attention_flags()
    _filter_complex_scalars_in_opinfo_samples()

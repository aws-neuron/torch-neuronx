"""aten::equal — whole-tensor equality comparison returning a host bool.

Unlike most ATen ops, ``equal(Tensor self, Tensor other) -> bool`` returns a
Python bool, not a tensor, so it cannot go through the MLIR lowering path.
Instead it composes host-side checks with device work:

- device mismatch  -> RuntimeError (see "Why the device check is here" below)
- shape mismatch   -> False, with no device dispatch
- empty tensors    -> True, with no device dispatch
- otherwise        -> ``(self == other).all()`` on device, materialized to a
  host bool via ``item()`` (``_local_scalar_dense``)

Why the device check is here
---------------------------
Same-device is a *precondition of the operator*, not a policy this backend
picks. ``native_functions.yaml`` declares ``equal`` without a ``device_check``
entry, so it defaults to ``DeviceCheckType.ExactSame``, and torchgen emits a
``c10::impl::check_and_update_common_device`` call per tensor argument into the
generated dispatcher wrapper — above the kernel, before any kernel body runs.
Backends that go through codegen therefore refuse mismatched devices for free,
and do so even when their own kernel contains no such check.

Neuron does not inherit that wrapper: torchgen gates it on the CUDA and XPU
dispatch keys, and this op is registered by hand via ``aten_lib.impl(...,
"PrivateUse1")``, which bypasses codegen entirely. The explicit raise below is
this backend hand-implementing the precondition it would otherwise be the sole
backend to skip.

Refusing is also the only behavior consistent with the documented contract
("True if two tensors have the same size and elements", with dtype explicitly
not differentiated). Device is not part of that predicate, so for two co-valued
tensors on different devices returning ``False`` would misstate the data — and
be indistinguishable from a genuine value difference — while returning ``True``
would require a host/device transfer the caller never requested. The composition
below would refuse anyway, since ``aten::eq`` enforces co-location through
TensorIterator; the check exists to fail with a clearer message than the inner
op would give.

Other semantics
---------------
Mixed dtypes are promoted to the common dtype before comparison, because
``torch.equal`` compares values and does not differentiate dtypes.

NaN handling is version-sensitive and deliberately left to ``==``: elementwise
comparison makes ``torch.equal(nan_tensor, other_nan_tensor)`` False. Note that
newer torch (2.13) short-circuits ``equal(t, t)`` on tensor identity, so the
same-object case is True there regardless of NaN content; do not assume a single
NaN answer across versions.
"""

import torch


def equal_neuron(self: torch.Tensor, other: torch.Tensor) -> bool:
    """Compare two same-device tensors and return a host Boolean.

    Args:
        self (torch.Tensor): First tensor to compare.
        other (torch.Tensor): Second tensor to compare.

    Returns:
        bool: ``True`` when shapes match and all promoted values compare equal.

    Raises:
        RuntimeError: If the tensors are on different devices.

    Note:
        Mixed dtypes are promoted. NaNs compare unequal according to elementwise ``==``
        semantics.
    """
    import torch_neuronx

    if self.device != other.device:
        raise RuntimeError(
            "Cannot compare two tensors on different devices. "
            f"Got: {self.device} and {other.device}"
        )

    torch_neuronx._C._log_executed_op("aten::equal")
    if self.shape != other.shape:
        return False
    if self.numel() == 0:
        return True

    if self.dtype != other.dtype:
        common_dtype = torch.promote_types(self.dtype, other.dtype)
        self = self.to(common_dtype)
        other = other.to(common_dtype)

    return bool((self == other).all().item())

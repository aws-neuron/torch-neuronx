"""aten::equal — whole-tensor equality comparison returning a host bool.

Unlike most ATen ops, ``equal(Tensor self, Tensor other) -> bool`` returns a
Python bool, not a tensor, so it cannot go through the MLIR lowering path.
Instead it composes host-side checks with device work:

- device mismatch  -> RuntimeError (mirrors ``cpu_equal``/``cuda_equal``)
- shape mismatch   -> False, with no device dispatch
- empty tensors    -> True, with no device dispatch (matches ``cuda_equal``)
- otherwise        -> ``(self == other).all()`` on device, materialized to a
  host bool via ``item()`` (``_local_scalar_dense``)

Mixed dtypes are promoted to the common dtype before comparison, matching
``cpu_equal``'s ``promote_inputs_to_common_dtype`` behavior (torch.equal
compares values, not dtypes). NaN compares unequal to itself (``==``
semantics), so ``torch.equal(nan, nan)`` is False, matching CPU.
"""

import torch


def equal_neuron(self: torch.Tensor, other: torch.Tensor) -> bool:
    """Neuron implementation of ``aten::equal``."""
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

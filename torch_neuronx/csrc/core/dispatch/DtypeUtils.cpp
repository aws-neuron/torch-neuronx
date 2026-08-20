#include "torch_neuronx/csrc/core/dispatch/DtypeUtils.h"

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/NeuronOpTracking.h"
#include "torch_neuronx/csrc/core/dispatch/DispatchUtils.h"
#include "torch_neuronx/csrc/core/utils/CopyUtils.h"
#include "torch_neuronx/csrc/core/utils/PackedInt64Utils.h"
#include "torch_neuronx/csrc/ops/InternalOps.h"

namespace torch_neuronx {
namespace dispatch {

torch::Tensor NeuronSafeCast(const torch::Tensor& t, c10::ScalarType target_dtype) {
  if (t.scalar_type() == target_dtype) {
    return t;
  }

  // Packed int64 graph input: keep it *logical int64* here. Both the low-word
  // extraction (in-graph NKI gather) and the int32[..., 2] physical-view
  // reinterpret are consolidated at the graph boundary — the fake input is
  // generated as int32[..., 2] (so the NEFF parameter is int32) and the int64
  // arg's physical view is bound to the NEFF at execution (FlattenAndFilterInputs).
  // Returning the int64 tensor unchanged (rather than reinterpreting to int32
  // here) keeps the dispatch arg's logical shape intact, so ops that read the
  // input shape at the host level (nonzero, cross_entropy, ...) are not confused
  // by the trailing packed lane dim. Only for *contiguous* int64 — a
  // non-contiguous int64 (broadcast, slice, transpose, permuted) falls through
  // to the CPU round-trip below (copies to CPU, restrides, casts to the low word).
  //
  // NOTE: the explicit _to_copy Int<-Long cast (DispatchToCopyImpl) still calls
  // execution_view directly — it produces the actual gathered int32 result.
  static const bool packed_int64 = IsPackedInt64Enabled();
  if (packed_int64 && t.scalar_type() == c10::ScalarType::Long &&
      target_dtype == c10::ScalarType::Int && t.device().type() == c10::DeviceType::PrivateUse1) {
    if (t.is_contiguous()) {
      return t;
    }
    // Non-contiguous packed int64 (slice/broadcast/transpose/permute): return a
    // *contiguous* int64 copy so it flows through the same physical-view binding
    // as the contiguous case. The fused NEFF expects the int32[..., 2] physical
    // view (fake input + in-graph gather + materialized output); binding the old
    // CPU-bounced int32[numel] low-word buffer is half the required size (e.g.
    // "output0 too small: 4104 vs 8208").
    //
    // Contiguify the int32 *physical view* (well-defined 4-byte element strides)
    // via contiguous_internal — the fast on-device (MLIR/transpose) contiguify
    // used by DispatchContiguousImpl. Doing it on the int32 view (not the int64)
    // is essential: contiguous_internal redispatches, and on int32
    // NeuronSafeCast(Int->Int) is a no-op, so it can't recurse back into this
    // int64 branch (t.contiguous() on the int64 would). The physical view's
    // [low, high] pairs are value-preserving, so logical_view of the contiguous
    // result equals t.contiguous(). For in-place ops the result is written back
    // into this non-contiguous tensor by UpcastExecutionOutputs ->
    // CopyBackToOriginal (CopyToNonContiguous).
    auto phys = torch_neuronx::packed_int64::physical_view(t);  // int32[..., 2], strided
    auto phys_contig = torch_neuronx::ops::contiguous_internal(phys, c10::MemoryFormat::Contiguous);
    return torch_neuronx::packed_int64::logical_view(phys_contig);  // int64, contiguous
  }

  // CPU int64 operand feeding a packed-int64 graph. Kept *Long* so PreprocessArgs
  // materializes it on-device as packed int64 and the graph boundary binds its
  // int32[..., 2] physical view (matching the NEFF parameter). Two cases:
  //   1. 0-dim wrapped Python scalars — a binary op mixing a neuron int64 with a
  //      Python-int / CPU scalar (e.g. `len(x) - offsets[i]`) wraps the int as a
  //      0-dim CPU int64.
  //   2. Multi-element CPU int64 *indices* — index/index_select/index_copy/
  //      index_add accept a CPU index against a neuron self, reaching here as a
  //      multi-element CPU int64.
  // Downcasting either to int32 (the t.to() fallback below) would bind a
  // half-width low word and trip NRT "input0 too small: 4 vs 8".
  //
  // (A non-index multi-element CPU tensor feeding a neuron op is a genuine device
  // mismatch that PyTorch rejects before reaching here, so this does not swallow
  // real downcast requests.)
  if (packed_int64 && t.scalar_type() == c10::ScalarType::Long &&
      target_dtype == c10::ScalarType::Int && t.device().is_cpu()) {
    return t;
  }

  // Packed int64 output: int32 execution result → int64 packed storage.
  // Allocates the int64 output and writes sign-extended (low, high) word pairs.
  // materialize_from_i32 uses torch::empty (the empty_strided_neuron factory),
  // which does NOT redispatch — this is essential to avoid infinite recursion
  // back through UpcastExecutionOutputs -> NeuronSafeCast.
  if (packed_int64 && t.scalar_type() == c10::ScalarType::Int &&
      target_dtype == c10::ScalarType::Long && t.device().type() == c10::DeviceType::PrivateUse1) {
    return torch_neuronx::packed_int64::materialize_from_i32(t);
  }

  // Float64 always needs CPU bounce. Int64 needs CPU bounce only when
  // native int64 is disabled and packed int64 is disabled.
  static const bool native_int64 = IsNativeInt64Enabled();
  bool is_unsupported_src = t.scalar_type() == c10::ScalarType::Double ||
                            (!native_int64 && t.scalar_type() == c10::ScalarType::Long);
  bool is_unsupported_dst = target_dtype == c10::ScalarType::Double ||
                            (!native_int64 && target_dtype == c10::ScalarType::Long);
  bool needs_cpu_roundtrip = t.is_privateuseone() && (is_unsupported_src || is_unsupported_dst);

  if (needs_cpu_roundtrip) {
    auto orig_device = t.device();
    TORCH_NEURONX_DISPATCH_DEBUG("[NeuronSafeCast] CPU round-trip via NRT",
                                 " from=", t.scalar_type(), " to=", target_dtype,
                                 " device=", orig_device.str());

    // Use NRT helpers directly instead of .to() to avoid re-dispatching
    // through aten::_to_copy.  Matches Python convert_for_neuron() which
    // calls cast_policy.copy_neuron_to_cpu + cast_policy.copy_cpu_to_neuron.
    //
    // IMPORTANT: Do NOT call t.contiguous() here — it dispatches through
    // DispatchContiguousImpl → DowncastArgs → NeuronSafeCast → infinite
    // recursion for non-contiguous int64 tensors.  Instead, copy the full
    // storage to CPU via NRT and restride on the CPU side (same pattern as
    // NeuronToContiguousCPU in NeuronDispatcher.cpp).
    torch::Tensor cpu_tmp;
    if (t.is_contiguous()) {
      cpu_tmp = torch::empty(t.sizes(),
                             torch::TensorOptions().dtype(t.scalar_type()).device(torch::kCPU));
      utils::copy_neuron_to_cpu(t, cpu_tmp, /*non_blocking=*/false);
    } else {
      // Non-contiguous: copy full storage to CPU, restride there.
      auto storage_bytes = t.storage().nbytes();
      auto elem_size = t.element_size();
      int64_t storage_numel = static_cast<int64_t>(storage_bytes / elem_size);

      auto flat_neuron = t.as_strided({storage_numel}, {1}, 0);
      auto flat_cpu = torch::empty(
          {storage_numel}, torch::TensorOptions().dtype(t.scalar_type()).device(torch::kCPU));
      utils::copy_neuron_to_cpu(flat_neuron, flat_cpu, /*non_blocking=*/false);

      cpu_tmp =
          flat_cpu.as_strided(t.sizes().vec(), t.strides().vec(), t.storage_offset()).contiguous();
    }

    // Dtype cast on CPU (cheap)
    if (cpu_tmp.scalar_type() != target_dtype) {
      cpu_tmp = cpu_tmp.to(target_dtype);
    }
    if (!cpu_tmp.is_contiguous()) {
      cpu_tmp = cpu_tmp.contiguous();
    }

    // Upload back to Neuron
    auto result = torch::empty(
        cpu_tmp.sizes(), torch::TensorOptions().dtype(cpu_tmp.scalar_type()).device(orig_device));
    utils::copy_cpu_to_neuron(cpu_tmp, result, /*non_blocking=*/false);
    torch_neuronx::markFallbackLogged("aten::_to_copy");
    return result;
  }

  return t.to(target_dtype);
}

void CopyBackToOriginal(torch::Tensor& orig, const torch::Tensor& result,
                        const std::string& debug_context) {
  if (!orig.defined() || !result.defined()) return;

  // Same storage — no copy needed.
  if (result.data_ptr() == orig.data_ptr()) return;

  TORCH_NEURONX_DISPATCH_DEBUG("[CopyBackToOriginal] ", debug_context,
                               " result_dtype=", result.scalar_type(),
                               " orig_dtype=", orig.scalar_type(), " result_shape=", result.sizes(),
                               " orig_shape=", orig.sizes());

  // Resize if shape changed (e.g., some out ops allow shape inference).
  ResizeOutputWithWarning(orig, result.sizes());

  // Copy with dtype conversion if needed.
  if (orig.scalar_type() != result.scalar_type()) {
    auto casted = NeuronSafeCast(result, orig.scalar_type());
    if (!orig.is_contiguous() && orig.is_privateuseone() && casted.is_contiguous() &&
        casted.is_privateuseone()) {
      // Non-contiguous Neuron dst: use CopyToNonContiguous directly to avoid
      // re-dispatching aten::copy_ through DispatchCopy, which can infinitely
      // recurse when the epilogue fallback path leads back here.
      torch_neuronx::utils::CopyToNonContiguous(orig, casted);
    } else {
      orig.copy_(casted);
    }
  } else {
    if (!orig.is_contiguous() && orig.is_privateuseone() && result.is_contiguous() &&
        result.is_privateuseone()) {
      // Non-contiguous Neuron dst: use CopyToNonContiguous directly to avoid
      // re-dispatching aten::copy_ through DispatchCopy, which can infinitely
      // recurse when the epilogue fallback path leads back here.
      torch_neuronx::utils::CopyToNonContiguous(orig, result);
    } else {
      orig.copy_(result);
    }
  }
}

void ResizeOutputWithWarning(torch::Tensor& out, c10::IntArrayRef target_shape) {
  if (!out.defined() || out.sizes() == target_shape) {
    return;
  }
  // PyTorch's out= contract: emit a UserWarning when a non-empty out=
  // tensor is resized. Matches the message produced by
  // at::native::resize_output_check in aten/src/ATen/native/Resize.cpp.
  if (out.numel() != 0) {
    TORCH_WARN("An output with one or more elements was resized since it had ", "shape ",
               out.sizes(), ", which does not match the required ", "output shape ", target_shape,
               ". ", "This behavior is deprecated, and in a future PyTorch release outputs ",
               "will not be resized unless they have zero elements. You can explicitly ",
               "reuse an out tensor t by resizing it, inplace, to zero elements with ",
               "t.resize_(0).");
  }
  out.resize_(target_shape);
}

}  // namespace dispatch
}  // namespace torch_neuronx

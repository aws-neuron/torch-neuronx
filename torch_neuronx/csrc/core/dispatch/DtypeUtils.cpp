#include "torch_neuronx/csrc/core/dispatch/DtypeUtils.h"

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/dispatch/DispatchUtils.h"
#include "torch_neuronx/csrc/core/utils/CopyUtils.h"

namespace torch_neuronx {
namespace dispatch {

torch::Tensor NeuronSafeCast(const torch::Tensor& t, c10::ScalarType target_dtype) {
  if (t.scalar_type() == target_dtype) {
    return t;
  }

  // Float64 always needs CPU bounce. Int64 needs CPU bounce only when
  // native int64 is disabled — when enabled, MLIR handles S64 natively.
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

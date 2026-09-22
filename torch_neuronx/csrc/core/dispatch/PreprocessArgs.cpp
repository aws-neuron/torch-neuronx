#include "torch_neuronx/csrc/core/dispatch/PreprocessArgs.h"

#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/NeuronOpsApi.h"
#include "torch_neuronx/csrc/core/dispatch/DispatchUtils.h"
#include "torch_neuronx/csrc/core/dispatch/Int64Policy.h"

namespace torch_neuronx {
namespace dispatch {

// ═══════════════════════════════════════════════════════════════
//  PreprocessSingleTensor — convert_for_neuron equivalent
// ═══════════════════════════════════════════════════════════════

// Shared helper: preprocess a single tensor.
// Handles: CPU 0-dim → device move, contiguous, int64→int32, float64→float32.
// Returns true if the tensor was modified (caller must write it back).
static bool PreprocessSingleTensor(at::Tensor& t, const torch::Device& target_device,
                                   bool skip_contiguous, bool is_write_output,
                                   const char* debug_label) {
  if (!t.defined()) return false;
  bool changed = false;

  TORCH_NEURONX_DISPATCH_DEBUG("[PreprocessSingleTensor] ", debug_label, " ENTER dim=", t.dim(),
                               " dtype=", t.scalar_type(), " device=", t.device().str());

  // out= / inplace-self device contract: a write-aliased tensor argument must be
  // on the Neuron device. PyTorch raises for a mismatched-device output — its
  // TensorIterator same-device check only exempts CPU 0-dim *inputs*
  // (`!op.is_output`), never outputs. We must check here rather than rely on a
  // downstream execution-time check, because a CPU output would otherwise be
  // silently relocated to Neuron by the 0-dim fast path below (masking the
  // mismatch, e.g. logsumexp(neuron, out=cpu-scalar)).
  if (is_write_output && !t.is_privateuseone()) {
    TORCH_CHECK(false,
                "Expected all tensors to be on the same device, but the output tensor is on ",
                t.device().str(), " and the operation was dispatched to the Neuron device");
  }

  // ════════════════════════════════════════════════════════════
  // CPU 0-dim tensors (PyTorch-wrapped scalars)
  //
  // When you write `neuron_tensor + 1`, PyTorch wraps the Python `1` as a
  // 0-dim CPU tensor before dispatch (because the schema says Tensor other).
  // These are conceptually scalars — they have no device affinity. Only
  // *inputs* reach here as CPU 0-dim (a CPU 0-dim output was rejected above),
  // matching PyTorch's `allow_cpu_scalars_` for inputs.
  //
  // We handle them by extracting the scalar value and creating a new tensor
  // directly on the target device with the correct dtype. This avoids going
  // through the device copy machinery (cpu_to_neuron) which can fail or be
  // expensive for a single scalar value.
  // ════════════════════════════════════════════════════════════
  if (t.device().is_cpu() && t.dim() == 0) {
    auto src_dtype = t.scalar_type();

    TORCH_NEURONX_DISPATCH_DEBUG("[PreprocessSingleTensor] ", debug_label,
                                 " CPU 0-dim scalar tensor → create on device",
                                 " dtype=", src_dtype, " target_device=", target_device.str());

    // Extract scalar value and create on CPU first, then move to device.
    auto cpu_opts = torch::TensorOptions().dtype(src_dtype);
    auto scalar = t.item();
    if (scalar.isBoolean()) {
      t = torch::tensor(scalar.toBool(), cpu_opts).to(target_device);
    } else if (scalar.isIntegral(/*includeBool=*/false)) {
      t = torch::tensor(scalar.toLong(), cpu_opts).to(target_device);
    } else {
      t = torch::tensor(scalar.toDouble(), cpu_opts).to(target_device);
    }

    TORCH_NEURONX_DISPATCH_DEBUG("[PreprocessSingleTensor] ", debug_label,
                                 " EXIT (CPU 0-dim fast path)", " dtype=", t.scalar_type(),
                                 " device=", t.device().str());
    return true;
  }

  // ════════════════════════════════════════════════════════════
  // Device tensors (dim >= 1, or already on Neuron device)
  // ════════════════════════════════════════════════════════════

  // ── Make contiguous unless skipped ──
  // Prologue normally handles non-contiguous layouts in-graph, so contiguify is
  // skipped when it's enabled. int64 is the exception: it executes on the packed
  // int32[..., 2] physical view, which the prologue's strided transforms can't
  // express (they'd leave an i64 tensor where the NEFF expects i32[..., 2], an
  // i64↔i32x2 boundary mismatch). So always contiguify non-contiguous int64 —
  // DispatchContiguousImpl's CPU-roundtrip yields a clean contiguous int64 that
  // the packed view binds correctly.
  const bool force_contig = (t.scalar_type() == c10::ScalarType::Long);
  if ((!skip_contiguous || force_contig) && !t.is_contiguous()) {
    TORCH_NEURONX_DISPATCH_DEBUG("[PreprocessSingleTensor] ", debug_label, " making contiguous");
    t = t.contiguous();
    changed = true;
  }

  TORCH_NEURONX_DISPATCH_DEBUG("[PreprocessSingleTensor] ", debug_label, " EXIT changed=", changed,
                               " dtype=", t.scalar_type(), " device=", t.device().str());

  return changed;
}

// ═══════════════════════════════════════════════════════════════
//  PreprocessArgs — mirrors Python ArgumentProcessor.preprocess_inputs
// ═══════════════════════════════════════════════════════════════

void PreprocessArgs(std::vector<c10::IValue>& args, const c10::FunctionSchema& schema,
                    const std::unordered_set<size_t>& static_argnums, bool skip_contiguous,
                    const std::string& op_name) {
  // ── Step 1: Determine target device from first non-scalar Neuron tensor ──
  torch::Device target_device(c10::DeviceType::PrivateUse1,
                              static_cast<c10::DeviceIndex>(at::neuron::current_device()));
  bool found_target_device = false;
  ForEachTensor(args, [&](const at::Tensor& t) {
    if (!found_target_device && t.is_privateuseone() && t.dim() > 0) {
      target_device = t.device();
      found_target_device = true;
    }
  });

  TORCH_NEURONX_DISPATCH_DEBUG("[PreprocessArgs] ENTER", " target_device=", target_device.str(),
                               " num_args=", args.size());

  // When prologue is enabled, non-contiguous inputs are handled later by
  // ProcessInputTransformations. When the caller sets skip_contiguous (e.g.
  // aten::contiguous), the kernel handles it itself. Either way, skip here.
  skip_contiguous = skip_contiguous || torch_neuronx::IsPrologueEnabled();

  const auto& schema_args = schema.arguments();

  for (size_t i = 0; i < args.size() && i < schema_args.size(); ++i) {
    auto& arg = args[i];
    const bool is_kwarg = schema_args[i].kwarg_only();
    std::string debug_label = "arg[" + std::to_string(i) + "]";

    TORCH_NEURONX_DISPATCH_DEBUG("[PreprocessArgs] ", debug_label, " name=", schema_args[i].name(),
                                 " kwarg_only=", is_kwarg, " isTensor=", arg.isTensor(),
                                 " isInt=", arg.isInt(), " isDouble=", arg.isDouble(),
                                 " isScalar=", arg.isScalar(), " isTensorList=", arg.isTensorList(),
                                 " isNone=", arg.isNone(), " tagKind=", arg.tagKind());

    // ════════════════════════════════════════════════════════════
    // 0. Static args — must remain as scalars (compile-time constants).
    // ════════════════════════════════════════════════════════════
    if (static_argnums.count(i)) {
      TORCH_NEURONX_DISPATCH_DEBUG("[PreprocessArgs] ", debug_label,
                                   " is static_argnum — skip (keep as scalar)");
      continue;
    }

    // ════════════════════════════════════════════════════════════
    // A. Scalar arguments (raw int / double / bool IValues, NOT tensors)
    //
    // Positional scalar args: always convert to a 0-dim tensor on the target
    //   device. Kwarg-only scalar args: convert only when the schema type is
    //   NumberType (Scalar) or FloatType, so the value becomes a runtime NEFF
    //   input reusable across values (e.g. addcmul's `value`, a fused
    //   optimizer's `lr` — an LR scheduler does not recompile). Other kwarg
    //   types (int dtype, bool flags) are left as-is and go into the cache key.
    // ════════════════════════════════════════════════════════════
    // NOTE: We create scalar tensors on CPU first, then .to(device).
    // torch::tensor(val, opts.device(neuron)) internally calls
    // at::scalar_tensor(scalar, opts) via TensorDataContainer::convert_to_tensor
    // (TensorDataContainer.h:239), which dispatches through PrivateUse1 →
    // NeuronDispatcher::Dispatch → PreprocessArgs → torch::tensor() →
    // infinite recursion / stack overflow.
    //
    // This matches Python torch.tensor() which creates on CPU then .to(device)
    // (see tensor_new.cpp internal_new_from_data lines 437-465).
    auto kwarg_kind = schema_args[i].type()->kind();
    bool is_number_kwarg = is_kwarg && (kwarg_kind == c10::TypeKind::NumberType ||
                                        kwarg_kind == c10::TypeKind::FloatType);
    bool convert_scalar = !is_kwarg || is_number_kwarg;

    if (arg.isInt() && convert_scalar) {
      bool use_int64 = SupportsNativeInt64(op_name);
      auto val = arg.toInt();
      auto int_dtype = use_int64 ? torch::kInt64 : torch::kInt32;
      auto t = torch::tensor(val, torch::TensorOptions().dtype(int_dtype)).to(target_device);
      args[i] = c10::IValue(std::move(t));
      TORCH_NEURONX_DISPATCH_DEBUG("[PreprocessArgs] ", debug_label, " int scalar → ", int_dtype,
                                   " tensor on ", target_device.str(), " val=", val);
      continue;
    }
    if (arg.isDouble() && convert_scalar) {
      auto val = arg.toDouble();
      auto t = torch::tensor(val, torch::TensorOptions().dtype(torch::kFloat32)).to(target_device);
      args[i] = c10::IValue(std::move(t));
      TORCH_NEURONX_DISPATCH_DEBUG("[PreprocessArgs] ", debug_label,
                                   " double scalar → float32 tensor on ", target_device.str(),
                                   " val=", val);
      continue;
    }
    // Catch remaining Scalar types (e.g. Bool) not matched by isInt()/isDouble().
    // aten::full's fill_value arrives as Scalar with tagKind=Bool.
    // Mirrors Python preprocess_inputs(scalar_to_tensor=True) where
    // isinstance(True, int) is True.
    if (arg.isScalar() && convert_scalar) {
      auto scalar = arg.toScalar();
      torch::Tensor t;
      if (scalar.isBoolean()) {
        t = torch::tensor(scalar.toBool(), torch::TensorOptions().dtype(torch::kBool));
      } else if (scalar.isIntegral(/*includeBool=*/false)) {
        bool use_int64 = SupportsNativeInt64(op_name);
        auto int_dtype = use_int64 ? torch::kInt64 : torch::kInt32;
        t = torch::tensor(scalar.toLong(), torch::TensorOptions().dtype(int_dtype));
      } else {
        t = torch::tensor(scalar.toDouble(), torch::TensorOptions().dtype(torch::kFloat32));
      }
      t = t.to(target_device);
      args[i] = c10::IValue(std::move(t));
      TORCH_NEURONX_DISPATCH_DEBUG("[PreprocessArgs] ", debug_label, " Scalar → tensor on ",
                                   target_device.str());
      continue;
    }

    // ════════════════════════════════════════════════════════════
    // B. Tensor arguments — preprocess via shared helper
    // ════════════════════════════════════════════════════════════
    if (arg.isTensor()) {
      auto t = arg.toTensor();
      const auto& alias_info = schema_args[i].alias_info();
      const bool is_write_output = alias_info != nullptr && alias_info->isWrite();
      if (PreprocessSingleTensor(t, target_device, skip_contiguous, is_write_output,
                                 debug_label.c_str())) {
        args[i] = c10::IValue(std::move(t));
      }
      continue;
    }

    // ════════════════════════════════════════════════════════════
    // C. TensorList arguments — preprocess each element
    // ════════════════════════════════════════════════════════════
    if (arg.isTensorList()) {
      auto tl = arg.toTensorList();
      bool list_changed = false;
      const auto& tl_alias = schema_args[i].alias_info();
      const bool tl_is_write = tl_alias != nullptr && tl_alias->isWrite();
      for (size_t j = 0; j < tl.size(); ++j) {
        at::Tensor t = tl[j];
        std::string elem_label = debug_label + "[" + std::to_string(j) + "]";
        if (PreprocessSingleTensor(t, target_device, skip_contiguous, tl_is_write,
                                   elem_label.c_str())) {
          tl[j] = std::move(t);
          list_changed = true;
        }
      }
      if (list_changed) {
        args[i] = c10::IValue(std::move(tl));
      }
      continue;
    }

    // ════════════════════════════════════════════════════════════
    // C2. GenericList (Tensor?[]) — preprocess tensor elements
    //     Covers optional-tensor lists like _index_put_impl_ indices.
    // ════════════════════════════════════════════════════════════
    if (arg.isList()) {
      auto list = arg.toList();
      bool list_changed = false;
      const auto& list_alias = schema_args[i].alias_info();
      const bool list_is_write = list_alias != nullptr && list_alias->isWrite();
      for (size_t j = 0; j < list.size(); ++j) {
        auto elem = list.get(j);
        if (elem.isTensor()) {
          at::Tensor t = elem.toTensor();
          if (!t.defined()) continue;
          std::string elem_label = debug_label + "[" + std::to_string(j) + "]";

          // A list element is bound straight to the NEFF, so a CPU element has to
          // be moved. PreprocessSingleTensor only relocates 0-dim CPU tensors,
          // because for a plain Tensor argument a device mismatch is an error to
          // raise, not repair. An index list is the exception: callers do pass CPU
          // indices, and aten::index.Tensor already moves them (DispatchIndexImpl).
          bool changed = false;
          if (t.device().is_cpu() && t.dim() > 0) {
            t = t.to(target_device);
            changed = true;
          }
          if (PreprocessSingleTensor(t, target_device, skip_contiguous, list_is_write,
                                     elem_label.c_str())) {
            changed = true;
          }
          if (changed) {
            list.set(j, c10::IValue(std::move(t)));
            list_changed = true;
          }
        }
      }
      if (list_changed) {
        args[i] = c10::IValue(list);
      }
      continue;
    }

    // ════════════════════════════════════════════════════════════
    // D. Everything else — pass through as-is.
    // ════════════════════════════════════════════════════════════
    TORCH_NEURONX_DISPATCH_DEBUG("[PreprocessArgs] ", debug_label,
                                 " passthrough (no preprocessing)");
  }

  TORCH_NEURONX_DISPATCH_DEBUG("[PreprocessArgs] EXIT");
}

}  // namespace dispatch
}  // namespace torch_neuronx

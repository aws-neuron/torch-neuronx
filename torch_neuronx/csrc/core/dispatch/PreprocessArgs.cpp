#include "torch_neuronx/csrc/core/dispatch/PreprocessArgs.h"

#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/dispatch/DispatchUtils.h"

namespace torch_neuronx {
namespace dispatch {

// ═══════════════════════════════════════════════════════════════
//  PreprocessSingleTensor — convert_for_neuron equivalent
// ═══════════════════════════════════════════════════════════════

// Shared helper: preprocess a single tensor.
// Handles: CPU 0-dim → device move, contiguous, int64→int32, float64→float32.
// Returns true if the tensor was modified (caller must write it back).
static bool PreprocessSingleTensor(at::Tensor& t, const torch::Device& target_device,
                                   bool skip_contiguous, const char* debug_label) {
  if (!t.defined()) return false;
  bool changed = false;

  TORCH_NEURONX_DISPATCH_DEBUG("[PreprocessSingleTensor] ", debug_label, " ENTER dim=", t.dim(),
                               " dtype=", t.scalar_type(), " device=", t.device().str());

  // ════════════════════════════════════════════════════════════
  // CPU 0-dim tensors (PyTorch-wrapped scalars)
  //
  // When you write `neuron_tensor + 1`, PyTorch wraps the Python `1` as a
  // 0-dim CPU tensor before dispatch (because the schema says Tensor other).
  // These are conceptually scalars — they have no device affinity.
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
  if (!skip_contiguous && !t.is_contiguous()) {
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
                    const std::unordered_set<size_t>& static_argnums, bool skip_contiguous) {
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
  static const bool prologue_enabled = []() {
    const char* env = std::getenv("TORCH_NEURONX_ENABLE_PROLOGUE");
    // Prologue is enabled by default; only disabled if explicitly set to "0"/"false".
    return env == nullptr || (std::string(env) != "0" && std::string(env) != "false");
  }();
  skip_contiguous = skip_contiguous || prologue_enabled;

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
    // Positional scalar args: always convert to 0-dim tensor on target device.
    // Kwarg-only scalar args: convert to 0-dim tensor ONLY when the schema
    //   type is NumberType (Scalar), so the value becomes a runtime input to
    //   the compiled NEFF rather than a compile-time constant. This allows
    //   the same NEFF to be reused with different scalar values (e.g.
    //   addcmul's `value` kwarg). Other kwarg types (int for dtype, bool
    //   for pin_memory, etc.) are left as-is.
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
    bool is_number_kwarg = is_kwarg && schema_args[i].type()->kind() == c10::TypeKind::NumberType;
    bool convert_scalar = !is_kwarg || is_number_kwarg;

    if (arg.isInt() && convert_scalar) {
      static const bool native_int64 = IsNativeInt64Enabled();
      auto val = arg.toInt();
      auto int_dtype = native_int64 ? torch::kInt64 : torch::kInt32;
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
        static const bool native_int64 = IsNativeInt64Enabled();
        auto int_dtype = native_int64 ? torch::kInt64 : torch::kInt32;
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
      if (PreprocessSingleTensor(t, target_device, skip_contiguous, debug_label.c_str())) {
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
      for (size_t j = 0; j < tl.size(); ++j) {
        at::Tensor t = tl[j];
        std::string elem_label = debug_label + "[" + std::to_string(j) + "]";
        if (PreprocessSingleTensor(t, target_device, skip_contiguous, elem_label.c_str())) {
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
      for (size_t j = 0; j < list.size(); ++j) {
        auto elem = list.get(j);
        if (elem.isTensor()) {
          at::Tensor t = elem.toTensor();
          std::string elem_label = debug_label + "[" + std::to_string(j) + "]";
          if (t.defined() &&
              PreprocessSingleTensor(t, target_device, skip_contiguous, elem_label.c_str())) {
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

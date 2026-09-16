#include "torch_neuronx/csrc/core/dispatch/DispatchUtils.h"

#include <ATen/EmptyTensor.h>
#include <torch/csrc/autograd/python_variable.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/dispatch/DtypeUtils.h"
#include "torch_neuronx/csrc/core/dispatch/Int64Policy.h"
#include "torch_neuronx/csrc/utils/TypeConversion.h"

namespace torch_neuronx {
namespace dispatch {

// ═══════════════════════════════════════════════════════════════
//  Empty Tensor Detection
// ═══════════════════════════════════════════════════════════════

bool HasEmptyTensorInput(const std::vector<c10::IValue>& args) {
  bool found = false;
  ForEachTensor(args, [&](const at::Tensor& t) {
    if (t.numel() == 0) found = true;
  });
  return found;
}

bool HasEmptyOutputTensor(const std::vector<c10::IValue>& args, const c10::FunctionSchema& schema) {
  const auto& schema_args = schema.arguments();

  for (size_t i = 0; i < args.size() && i < schema_args.size(); ++i) {
    const auto& alias_info = schema_args[i].alias_info();
    if (!alias_info || !alias_info->isWrite()) {
      continue;
    }

    if (args[i].isTensor()) {
      const auto& t = args[i].toTensor();
      if (t.defined() && t.numel() == 0) {
        return true;
      }
    }
  }
  return false;
}

bool RawNrtCopyOverlaps(const at::Tensor& src, const at::Tensor& dst) {
  if (!src.has_storage() || !dst.has_storage() || src.storage().data() != dst.storage().data()) {
    return false;
  }
  const int64_t src_es = src.element_size();
  const int64_t dst_es = dst.element_size();
  const int64_t a_lo = src.storage_offset() * src_es;
  const int64_t a_hi = (src.storage_offset() + src.numel()) * src_es;
  const int64_t b_lo = dst.storage_offset() * dst_es;
  const int64_t b_hi = (dst.storage_offset() + dst.numel()) * dst_es;
  return std::max(a_lo, b_lo) < std::min(a_hi, b_hi);
}

bool IsIndexOp(const std::string& op_name) {
  // Index-family ops where PyTorch's Meta dispatch does NOT validate
  // shapes/bounds.  CPU dispatch errors on these when shapes are invalid,
  // but Meta silently produces wrong output shapes.
  //
  // Ops in GetIndexOpConfig (scatter, scatter_add, _index_put_impl_) are
  // NOT listed here — they have their own empty-index handling branch
  // inside TryHandleEmptyTensors.
  static const std::unordered_set<std::string> kIndexOps = {
      "index_copy",   "index_copy_", "index_add",  "index_add_", "index_fill", "index_fill_",
      "index_select", "index_put",   "index_put_", "gather",     "put",        "put_",
  };
  auto base = ExtractBaseOpName(op_name);
  return kIndexOps.count(base) > 0;
}

// ═══════════════════════════════════════════════════════════════
//  Complex Dtype Detection
// ═══════════════════════════════════════════════════════════════

bool HasComplexInput(const std::vector<c10::IValue>& args) {
  bool found = false;
  ForEachTensor(args, [&](const at::Tensor& t) {
    if (!found && t.is_complex()) {
      found = true;
    }
  });
  return found;
}

bool ComplexDecompositionEnabled() {
  const char* value = std::getenv("TORCH_NEURONX_COMPLEX_DECOMPOSE");
  if (value == nullptr) return false;
  std::string normalized(value);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
  return normalized == "1" || normalized == "true" || normalized == "yes";
}

bool HasComplexNonOutputInput(const std::vector<c10::IValue>& args,
                              const c10::FunctionSchema& schema) {
  // Filter out out= params and delegate to HasComplexInput.
  const auto& schema_args = schema.arguments();
  std::vector<c10::IValue> inputs_only;
  inputs_only.reserve(args.size());
  for (size_t i = 0; i < args.size() && i < schema_args.size(); ++i) {
    const auto& alias_info = schema_args[i].alias_info();
    if (alias_info && alias_info->isWrite()) continue;
    inputs_only.push_back(args[i]);
  }
  return HasComplexInput(inputs_only);
}

static const std::unordered_set<std::string>& NativeComplexOpsAllowlist() {
  static const std::unordered_set<std::string> kOps = {
      "mul.Tensor", "mul.Scalar", "mul.out",      "mul.Scalar_out", "mul_.Tensor",
      "div.Tensor", "div.out",    "div.out_mode", "div_.Tensor",
  };
  return kOps;
}

bool HasNativeComplexLowering(const std::string& lookup_name) {
  return NativeComplexOpsAllowlist().count(lookup_name) > 0;
}

bool ShouldUseComplexDecomposition(bool has_lower_func) {
  return ComplexDecompositionEnabled() && has_lower_func;
}

// ═══════════════════════════════════════════════════════════════
//  64-bit Dtype Detection and Downcasting
// ═══════════════════════════════════════════════════════════════

// Neuron-supported dtypes for MLIR lowering. Matches Python copy.py / to_copy.py.
// Dtypes NOT in this set are handled via CPU bounce in DispatchCopy/DispatchToCopy.
bool IsSupportedDtype(c10::ScalarType dt) {
  if (dt == c10::ScalarType::Long)
    return torch_neuronx::dispatch::IsS64Passthrough() ||
           torch_neuronx::dispatch::IsPackedInt64Enabled();
  return dt == c10::ScalarType::Float || dt == c10::ScalarType::Half ||
         dt == c10::ScalarType::BFloat16 || dt == c10::ScalarType::Int ||
         dt == c10::ScalarType::Bool || dt == c10::ScalarType::ComplexFloat;
}

static c10::ScalarType CastTarget(c10::ScalarType dt) {
  // float64 → float32; all unsupported int types → int32
  if (dt == c10::ScalarType::Double) return c10::ScalarType::Float;
  return c10::ScalarType::Int;
}

static bool IsScalarTypeArg(const c10::Argument& arg) {
  auto kind = arg.real_type()->kind();
  if (kind == c10::TypeKind::ScalarTypeType) return true;
  if (kind == c10::TypeKind::OptionalType) {
    auto inner = arg.real_type()->cast<c10::OptionalType>()->getElementType();
    return inner->kind() == c10::TypeKind::ScalarTypeType;
  }
  return false;
}

void DowncastArgs(std::vector<c10::IValue>& args, const c10::FunctionSchema& schema) {
  const auto& schema_args = schema.arguments();
  for (size_t i = 0; i < args.size() && i < schema_args.size(); ++i) {
    auto& arg = args[i];

    // Downcast any ScalarType kwarg with unsupported dtype
    if (IsScalarTypeArg(schema_args[i]) && arg.isInt()) {
      auto dt = static_cast<c10::ScalarType>(arg.toInt());
      if (!IsSupportedDtype(dt)) {
        auto target = CastTarget(dt);
        args[i] = c10::IValue(static_cast<int64_t>(target));
        TORCH_NEURONX_DISPATCH_DEBUG("[DowncastArgs] ", schema_args[i].name(), " kwarg ", dt, " → ",
                                     target);
      }
      continue;
    }

    if (arg.isTensor()) {
      auto t = arg.toTensor();
      if (!t.defined()) continue;
      auto dt = t.scalar_type();
      if (!IsSupportedDtype(dt)) {
        auto target = CastTarget(dt);
        args[i] = c10::IValue(NeuronSafeCast(t, target));
        TORCH_NEURONX_DISPATCH_DEBUG("[DowncastArgs] arg[", i, "] ", dt, " → ", target);
      }
    } else if (arg.isTensorList()) {
      auto tl = arg.toTensorList();
      bool changed = false;
      for (size_t j = 0; j < tl.size(); ++j) {
        at::Tensor t = tl[j];
        if (!t.defined()) continue;
        auto dt = t.scalar_type();
        if (!IsSupportedDtype(dt)) {
          tl[j] = NeuronSafeCast(t, CastTarget(dt));
          changed = true;
        }
      }
      if (changed) args[i] = c10::IValue(std::move(tl));
    } else if (arg.isList()) {
      // GenericList — covers Tensor?[] (e.g. _index_put_impl_ indices).
      auto list = arg.toList();
      bool changed = false;
      for (size_t j = 0; j < list.size(); ++j) {
        auto elem = list.get(j);
        if (!elem.isTensor()) continue;
        auto t = elem.toTensor();
        if (!t.defined()) continue;
        auto dt = t.scalar_type();
        if (!IsSupportedDtype(dt)) {
          list.set(j, c10::IValue(NeuronSafeCast(t, CastTarget(dt))));
          changed = true;
        }
      }
      if (changed) args[i] = c10::IValue(list);
    }
  }
}

std::vector<c10::IValue> CloneIValueListContainers(const std::vector<c10::IValue>& args) {
  auto cloned = args;
  for (size_t i = 0; i < cloned.size(); ++i) {
    if (cloned[i].isTensorList()) {
      cloned[i] = c10::IValue(cloned[i].toTensorList().copy());
    } else if (cloned[i].isList()) {
      cloned[i] = c10::IValue(cloned[i].toList().copy());
    }
  }
  return cloned;
}

bool DowncastTensorList(std::vector<at::Tensor>& tensors) {
  bool any_changed = false;
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto& t = tensors[i];
    if (!t.defined()) continue;
    auto dt = t.scalar_type();
    // Always downcast int64 to int32 here, regardless of mode: collectives are
    // handled in int32. Unlike the op-aware DowncastArgs path (which keeps S64
    // for native compute ops), this list is schema-less. Other dtypes follow
    // IsSupportedDtype as usual (e.g. float64 → float32).
    const bool needs_downcast = !IsSupportedDtype(dt) || dt == c10::ScalarType::Long;
    if (!needs_downcast) continue;
    auto target = CastTarget(dt);
    tensors[i] = NeuronSafeCast(t, target);
    any_changed = true;
    TORCH_NEURONX_DISPATCH_DEBUG("[DowncastTensorList] idx=", i, " ", dt, " → ", target);
  }
  return any_changed;
}

// ═══════════════════════════════════════════════════════════════
//  PromoteBinaryCompareOperands
// ═══════════════════════════════════════════════════════════════

// Mirrors PyTorch's binary-comparison whitelist (CPU/CUDA promote both
// operands to common dtype before the kernel; Neuron bypasses
// TensorIterator so we apply it here).
//
// Source: aten/src/ATen/native/BinaryOps.cpp:365-370 (compare ops),
//         aten/src/ATen/native/BinaryOps.cpp:1497-1507 (logical ops).
static bool IsBinaryCompareOpForPromotion(const c10::FunctionSchema& schema) {
  static const std::unordered_set<std::string> kCompareOps = {
      "aten::eq",          "aten::ne",           "aten::lt",          "aten::le",
      "aten::gt",          "aten::ge",           "aten::logical_and", "aten::logical_or",
      "aten::logical_xor", "aten::eq_",          "aten::ne_",         "aten::lt_",
      "aten::le_",         "aten::gt_",          "aten::ge_",         "aten::logical_and_",
      "aten::logical_or_", "aten::logical_xor_",
  };
  return kCompareOps.count(schema.operator_name().name) > 0;
}

void PromoteBinaryCompareOperands(std::vector<c10::IValue>& args,
                                  const c10::FunctionSchema& schema) {
  if (!IsBinaryCompareOpForPromotion(schema)) return;

  // Locate the two leading positional Tensor operands ("self" and "other"
  // in the schemas of all whitelisted ops). After PreprocessArgs, scalar
  // operands have already been materialized as 0-dim tensors on the device,
  // so we can rely on both arg[0] and arg[1] being tensors.
  if (args.size() < 2 || !args[0].isTensor() || !args[1].isTensor()) return;
  auto a = args[0].toTensor();
  auto b = args[1].toTensor();
  if (!a.defined() || !b.defined()) return;

  auto dt_a = a.scalar_type();
  auto dt_b = b.scalar_type();
  if (dt_a == dt_b) return;

  auto common = c10::promoteTypes(dt_a, dt_b);

  // In packed int64 mode, comparisons operate on the low int32 word.
  // Promoting to int64 builds a high word that's immediately discarded.
  if (common == c10::ScalarType::Long && UsePackedInt64ViewForOp(schema.operator_name().name)) {
    common = c10::ScalarType::Int;
  }

  // The whitelist contains only ops whose result dtype is bool, so the
  // inputs are always non-bool/non-complex value tensors. Cast each operand
  // to the common dtype if it differs. NeuronSafeCast handles unsupported
  // source/target dtypes via CPU round-trip.
  if (dt_a != common) {
    auto promoted = NeuronSafeCast(a, common);
    args[0] = c10::IValue(std::move(promoted));
    TORCH_NEURONX_DISPATCH_DEBUG("[PromoteBinaryCompareOperands] ", schema.operator_name().name,
                                 " arg[0] ", dt_a, " → ", common);
  }
  if (dt_b != common) {
    auto promoted = NeuronSafeCast(b, common);
    args[1] = c10::IValue(std::move(promoted));
    TORCH_NEURONX_DISPATCH_DEBUG("[PromoteBinaryCompareOperands] ", schema.operator_name().name,
                                 " arg[1] ", dt_b, " → ", common);
  }
}

// ═══════════════════════════════════════════════════════════════
//  Dispatch snapshot helpers
// ═══════════════════════════════════════════════════════════════

std::vector<std::pair<size_t, torch::Tensor>> SnapshotOutTensors(
    const c10::FunctionSchema& schema, const std::vector<c10::IValue>& args) {
  std::vector<std::pair<size_t, torch::Tensor>> result;
  const auto& returns = schema.returns();
  const auto& schema_args = schema.arguments();
  for (size_t ret_idx = 0; ret_idx < returns.size(); ++ret_idx) {
    const auto& ret = returns[ret_idx];
    if (!ret.alias_info() || !ret.alias_info()->isWrite()) continue;

    for (size_t arg_idx = 0; arg_idx < schema_args.size() && arg_idx < args.size(); ++arg_idx) {
      const auto& arg_alias = schema_args[arg_idx].alias_info();
      if (!arg_alias || !arg_alias->isWrite()) continue;
      if (ret.alias_info()->beforeSets() != arg_alias->beforeSets()) continue;
      if (!args[arg_idx].isTensor()) continue;

      auto t = args[arg_idx].toTensor();
      if (t.defined()) {
        result.push_back({ret_idx, t});
        TORCH_NEURONX_DISPATCH_DEBUG("[SnapshotOutTensors] snapshot out/inplace tensor",
                                     " ret_idx=", ret_idx, " arg_idx=", arg_idx,
                                     " dtype=", t.scalar_type(), " data_ptr=", t.data_ptr());
      }
      break;
    }
  }

  // Handle TensorList out args (e.g. split_with_sizes_copy.out where out is Tensor[])
  // These have no return aliases (void return) but the arg itself is aliased as write.
  if (result.empty()) {
    for (size_t arg_idx = 0; arg_idx < schema_args.size() && arg_idx < args.size(); ++arg_idx) {
      const auto& arg_alias = schema_args[arg_idx].alias_info();
      if (!arg_alias || !arg_alias->isWrite()) continue;
      if (!args[arg_idx].isTensorList()) continue;

      auto tensor_list = args[arg_idx].toTensorList();
      for (size_t i = 0; i < tensor_list.size(); ++i) {
        at::Tensor t = tensor_list[i];
        if (t.defined()) {
          result.push_back({i, t});
          TORCH_NEURONX_DISPATCH_DEBUG("[SnapshotOutTensors] snapshot out TensorList element",
                                       " list_idx=", i, " arg_idx=", arg_idx,
                                       " dtype=", t.scalar_type(), " data_ptr=", t.data_ptr());
        }
      }
      break;
    }
  }

  return result;
}

bool HasExplicitDtypeKwarg(const c10::FunctionSchema& schema,
                           const std::vector<c10::IValue>& args) {
  const auto& schema_args = schema.arguments();
  for (size_t i = 0; i < schema_args.size() && i < args.size(); ++i) {
    if (schema_args[i].kwarg_only() && schema_args[i].name() == "dtype" && !args[i].isNone()) {
      return true;
    }
  }
  return false;
}

static void ExtractTensorInfoForFallback(const at::Tensor& tensor,
                                         at::neuron::CPUFallbackContext& ctx) {
  void* storage_va = const_cast<void*>(tensor.storage().data());
  ctx.tensor_metadata.push_back(at::neuron::TensorContext::FromTensor(tensor));
  // findTensor may return nullptr if the underlying lazy allocation has not
  // yet been bound on the runtime worker (kAlloc has been queued but not
  // scheduled). We capture the storage VA so the fallback executor can
  // re-resolve at execute time, after kAlloc binds.
  ctx.tensor_data_ptrs.push_back(c10_neuron::NeuronCachingAllocator::findTensor(storage_va));
  ctx.tensor_storage_ptrs.push_back(storage_va);
}

// GIL deadlock prevention: CPU tensors from Python dispatch may hold PyObject*
// references. When OperationContext is destroyed on the RuntimeWorker thread,
// decref_pyobject() needs the GIL, but the main thread holds it while waiting
// for the stream to drain — causing a deadlock. .detach().clone() creates a
// new tensor without Python object backing, safe for worker thread destruction.
static c10::IValue DetachCloneCPUTensor(const c10::IValue& val) {
  if (val.isTensor()) {
    auto t = val.toTensor();
    if (t.defined() && t.device().is_cpu()) {
      return c10::IValue(t.detach().clone());
    }
  }
  return val;
}

at::neuron::CPUFallbackContext BuildCPUFallbackContext(const std::vector<c10::IValue>& args,
                                                       const c10::FunctionSchema& schema) {
  using namespace at::neuron;
  CPUFallbackContext ctx;
  const auto& schema_args = schema.arguments();

  for (size_t i = 0; i < schema_args.size() && i < args.size(); ++i) {
    const auto& val = args[i];
    const auto& arg_info = schema_args[i];
    bool is_kwarg = arg_info.kwarg_only();

    if (val.isTensor()) {
      auto tensor = val.toTensor();
      if (tensor.defined() && tensor.device().type() == c10::DeviceType::PrivateUse1) {
        // Neuron tensors: store as TensorContext + nrt_tensor_t* (no Python refs)
        if (is_kwarg) {
          ctx.tensor_kwarg_names.push_back(arg_info.name());
        }
        ExtractTensorInfoForFallback(tensor, ctx);
        ctx.input_metadata.push_back(TensorMarker{});
      } else if (tensor.defined()) {
        // CPU tensor: detach().clone() to sever Python object references
        auto safe_val = DetachCloneCPUTensor(val);
        ctx.input_metadata.push_back(ScalarValue(safe_val));
        if (is_kwarg) {
          ctx.original_kwargs.insert(arg_info.name(), safe_val);
        }
      } else {
        ctx.input_metadata.push_back(ScalarValue(c10::IValue()));
      }
    } else if (val.isTensorList()) {
      auto tl = val.toTensorVector();
      for (const auto& t : tl) {
        ExtractTensorInfoForFallback(t, ctx);
      }
      ctx.input_metadata.push_back(TensorListMarker(tl.size()));
    } else if (val.isList()) {
      // GenericList — covers Tensor?[] (e.g. _index_put_impl_ indices)
      // and scalar lists (e.g. SymInt[] pad in constant_pad_nd).
      // Extract tensors for fallback; pure-scalar lists are stored as ScalarValue.
      auto list = val.toListRef();
      size_t tensor_count = 0;
      for (const auto& elem : list) {
        if (elem.isTensor()) {
          auto t = elem.toTensor();
          if (t.defined() && t.device().type() == c10::DeviceType::PrivateUse1) {
            ExtractTensorInfoForFallback(t, ctx);
            ++tensor_count;
          }
        }
      }
      if (tensor_count > 0) {
        ctx.input_metadata.push_back(TensorListMarker(tensor_count));
      } else {
        // Pure scalar list (no Neuron tensors). May contain CPU tensors that
        // need detach().clone() to avoid GIL deadlock on async destruction.
        auto safe_val = DetachCloneCPUTensor(val);
        ctx.input_metadata.push_back(ScalarValue(safe_val));
        if (is_kwarg) {
          ctx.original_kwargs.insert(arg_info.name(), safe_val);
        }
      }
    } else {
      ctx.input_metadata.push_back(ScalarValue(val));
      if (is_kwarg) {
        ctx.original_kwargs.insert(arg_info.name(), val);
      }
    }
  }

  return ctx;
}

// ═══════════════════════════════════════════════════════════════
//  Debug logging helpers
// ═══════════════════════════════════════════════════════════════

void LogArgs(const char* prefix, const std::vector<c10::IValue>& args) {
  for (size_t i = 0; i < args.size(); ++i) {
    const auto& arg = args[i];
    if (arg.isTensor()) {
      auto t = arg.toTensor();
      TORCH_NEURONX_DISPATCH_DEBUG(prefix, " arg[", i, "] Tensor", " shape=", t.sizes(),
                                   " dtype=", t.scalar_type(), " device=", t.device());
    } else if (arg.isScalar()) {
      TORCH_NEURONX_DISPATCH_DEBUG(prefix, " arg[", i,
                                   "] Scalar value=", arg.toScalar().toDouble());
    } else if (arg.isDouble()) {
      TORCH_NEURONX_DISPATCH_DEBUG(prefix, " arg[", i, "] Double value=", arg.toDouble());
    } else if (arg.isInt()) {
      TORCH_NEURONX_DISPATCH_DEBUG(prefix, " arg[", i, "] Int value=", arg.toInt());
    } else if (arg.isNone()) {
      TORCH_NEURONX_DISPATCH_DEBUG(prefix, " arg[", i, "] None");
    } else {
      TORCH_NEURONX_DISPATCH_DEBUG(prefix, " arg[", i, "] other type");
    }
  }
}

void LogTensors(const char* prefix, const std::vector<torch::Tensor>& tensors) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].defined()) {
      TORCH_NEURONX_DISPATCH_DEBUG(
          prefix, "[", i, "]", " data_ptr=", tensors[i].data_ptr(), " shape=", tensors[i].sizes(),
          " dtype=", tensors[i].scalar_type(), " device=", tensors[i].device());
    } else {
      TORCH_NEURONX_DISPATCH_DEBUG(prefix, "[", i, "] None");
    }
  }
}

// ═══════════════════════════════════════════════════════════════
//  Empty Tensor Handling — op-type configs
//  C++ port of Python ReductionOps/IndexOps from config.py.
// ═══════════════════════════════════════════════════════════════

std::string ExtractBaseOpName(const std::string& op_name) {
  // "aten::sum.dim_IntList" → "sum"
  auto pos = op_name.rfind("::");
  std::string after_ns = (pos != std::string::npos) ? op_name.substr(pos + 2) : op_name;
  auto dot = after_ns.find('.');
  if (dot != std::string::npos) after_ns = after_ns.substr(0, dot);
  std::transform(after_ns.begin(), after_ns.end(), after_ns.begin(), ::tolower);
  return after_ns;
}

const ReductionConfig* GetReductionConfig(const std::string& op_name) {
  using FT = ReductionConfig::FillType;
  // Mirrors Python config.py: ReductionOps.IDENTITY_OPS, NAN_ON_EMPTY_OPS, NO_IDENTITY_OPS
  static const std::unordered_map<std::string, ReductionConfig> kConfigs = {
      // Identity ops (fill with identity value)
      {"sum", {FT::kIdentity, 0.0}},
      {"add", {FT::kIdentity, 0.0}},
      {"prod", {FT::kIdentity, 1.0}},
      {"product", {FT::kIdentity, 1.0}},
      {"all", {FT::kIdentity, 1.0}},  // True
      {"any", {FT::kIdentity, 0.0}},  // False
      {"norm", {FT::kIdentity, 0.0}},
      // NaN ops (fill with NaN)
      {"mean", {FT::kNaN, 0.0}},
      {"std", {FT::kNaN, 0.0}},
      {"var", {FT::kNaN, 0.0}},
      // Error ops (should not be called on empty tensors)
      {"max", {FT::kError, 0.0}},
      {"min", {FT::kError, 0.0}},
      {"amax", {FT::kError, 0.0}},
      {"amin", {FT::kError, 0.0}},
      {"argmax", {FT::kError, 0.0}},
      {"argmin", {FT::kError, 0.0}},
      {"std_mean", {FT::kError, 0.0}},
      {"var_mean", {FT::kError, 0.0}},
  };
  auto base = ExtractBaseOpName(op_name);
  auto it = kConfigs.find(base);
  return (it != kConfigs.end()) ? &it->second : nullptr;
}

const MatmulAccumulateConfig* GetMatmulAccumulateConfig(const std::string& op_name) {
  // beta * self + alpha * (matmul) — see MatmulAccumulateConfig in the header.
  // Keyed on the base name, so all overloads (.out/.dtype/.dtype_out) match.
  //
  // Only natively-lowered ops belong here. `addbmm` has the same semantics but
  // is NOT registered (it CPU-falls-back, which already gives the right answer
  // for a zero contraction dim), so an entry for it would be dead code — add
  // one if/when addbmm gains a native lowering.
  static const std::unordered_map<std::string, MatmulAccumulateConfig> kConfigs = {
      {"baddbmm", {0}},
      {"addmm", {0}},
  };
  auto base = ExtractBaseOpName(op_name);
  auto it = kConfigs.find(base);
  return (it != kConfigs.end()) ? &it->second : nullptr;
}

const FillValueConfig* GetFillValueConfig(const std::string& op_name) {
  static const std::unordered_map<std::string, FillValueConfig> kConfigs = {
      // constant_pad_nd(input, pad, value=0) — pad value is a scalar
      {"constant_pad_nd", {"value", 0.0, false}},
      // isin(elements, test_elements, *, invert=False) — with an empty test set every
      // element is a non-member, so the eager result is all-`invert` (shaped like
      // `elements`; a 0-element operand can't be compiled, so it's filled here). `invert`
      // is a bool kwarg read as a scalar fill (True->1.0, False->0.0). The torch.compile
      // path handles empty in the aten.isin decomposition instead (this branch is eager-only).
      {"isin", {"invert", 0.0, false}},
      // linear(input, weight, bias=None) — bias is a tensor
      {"linear", {"bias", 0.0, true}},
      // nonzero_static(input, size, fill_value=-1): an empty input has no
      // nonzero element, so every row of the (size, ndim) output is padding
      {"nonzero_static", {"fill_value", -1.0, false}},
  };
  auto base = ExtractBaseOpName(op_name);
  auto it = kConfigs.find(base);
  return (it != kConfigs.end()) ? &it->second : nullptr;
}

bool HasZeroDim(const std::vector<int64_t>& shape) {
  return std::any_of(shape.begin(), shape.end(), [](int64_t d) { return d == 0; });
}

c10::Scalar ExtractScalarArg(const std::vector<c10::IValue>& args,
                             const c10::FunctionSchema& schema, const std::string& name,
                             const c10::Scalar& default_value) {
  const auto& schema_args = schema.arguments();
  for (size_t i = 0; i < args.size() && i < schema_args.size(); ++i) {
    if (schema_args[i].name() != name) continue;
    return args[i].isScalar() ? args[i].toScalar() : default_value;
  }
  return default_value;
}

torch::Tensor ExtractTensorArg(const std::vector<c10::IValue>& args,
                               const c10::FunctionSchema& schema, const std::string& name) {
  const auto& schema_args = schema.arguments();
  for (size_t i = 0; i < args.size() && i < schema_args.size(); ++i) {
    if (schema_args[i].name() != name) continue;
    return args[i].isTensor() ? args[i].toTensor() : torch::Tensor();
  }
  return torch::Tensor();
}

const IndexOpConfig* GetIndexOpConfig(const std::string& op_name) {
  // Mirrors Python config.py: IndexOps.INDEX_OPS
  // index_argnum = position of the index tensor, input_argnum = position of self/input
  // When input is non-empty and index is empty → return input unchanged
  // source_argnum = position of the argument written from; scatter's `.value`
  // overloads pass a Scalar there rather than a tensor
  static const std::unordered_map<std::string, IndexOpConfig> kConfigs = {
      {"scatter", {2, 0, 3}},
      {"scatter_add", {2, 0, 3}},
      {"_index_put_impl_", {1, 0, 2}},
  };
  auto base = ExtractBaseOpName(op_name);
  auto it = kConfigs.find(base);
  return (it != kConfigs.end()) ? &it->second : nullptr;
}

// ═══════════════════════════════════════════════════════════════
//  Meta dispatch utilities — used by HandleEmptyTensors.
// ═══════════════════════════════════════════════════════════════

at::Tensor TensorToMeta(const at::Tensor& t) {
  if (!t.defined()) return t;
  // Wrapped numbers are CPU scalar adapters, not ordinary tensors.  Keep
  // them on CPU so Python-registered Meta kernels can convert them back to
  // the original Python number.  Moving one to Meta and retaining the bit
  // violates pybind_utils' wrapped-number invariant.
  if (t.unsafeGetTensorImpl()->is_wrapped_number()) {
    return t;
  }
  auto meta = at::detail::empty_strided_meta(t.sizes(), t.strides(), t.scalar_type());
  return meta;
}

c10::IValue IValueToMeta(const c10::IValue& val) {
  if (val.isTensor()) {
    return c10::IValue(TensorToMeta(val.toTensor()));
  }

  if (val.isTensorList()) {
    auto tl = val.toTensorList();
    c10::List<at::Tensor> meta_list;
    meta_list.reserve(tl.size());
    for (size_t i = 0; i < tl.size(); ++i) {
      at::Tensor t = tl[i];
      meta_list.push_back(TensorToMeta(t));
    }
    return c10::IValue(std::move(meta_list));
  }

  if (val.isOptionalTensorList()) {
    c10::List<c10::optional<at::Tensor>> meta_list;
    for (const auto& opt_t_ref : val.toOptionalTensorList()) {
      c10::optional<at::Tensor> opt_t = opt_t_ref;
      if (opt_t.has_value() && opt_t->defined()) {
        meta_list.push_back(TensorToMeta(*opt_t));
      } else {
        meta_list.push_back(opt_t);
      }
    }
    return c10::IValue(std::move(meta_list));
  }

  if (val.isList()) {
    // GenericList — covers Tensor?[] (e.g. _index_put_impl_ indices).
    auto list = val.toList();
    c10::impl::GenericList meta_list(list.elementType());
    meta_list.reserve(list.size());
    for (size_t j = 0; j < list.size(); ++j) {
      auto elem = list.get(j);
      if (elem.isTensor()) {
        meta_list.push_back(c10::IValue(TensorToMeta(elem.toTensor())));
      } else {
        meta_list.push_back(elem);
      }
    }
    return c10::IValue(std::move(meta_list));
  }

  // Scalars, ints, bools, None, Device, ScalarType, MemoryFormat — pass through
  return val;
}

torch::jit::Stack RunMetaDispatch(const c10::OperatorHandle& op,
                                  const std::vector<c10::IValue>& args) {
  torch::jit::Stack meta_stack;
  meta_stack.reserve(args.size());
  for (const auto& arg : args) {
    meta_stack.push_back(IValueToMeta(arg));
  }

  // Canonical pattern: FunctionalTensorWrapper.cpp:873.
  op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::Meta), &meta_stack);
  return meta_stack;
}

std::vector<MetaOutputSpec> ExtractMetaOutputSpecs(torch::jit::Stack& stack) {
  std::vector<MetaOutputSpec> specs;
  for (const auto& out : stack) {
    if (out.isTensor()) {
      auto t = out.toTensor();
      if (t.defined()) {
        specs.push_back({t.sizes().vec(), t.scalar_type(), false});
      } else {
        specs.push_back({{}, c10::ScalarType::Undefined, true});
      }
    } else if (out.isTensorList()) {
      // Tensor[] return — flatten into individual specs.
      // PushOutputsToStack re-packs based on schema return type.
      for (const auto& t_ref : out.toTensorList()) {
        at::Tensor t = t_ref;
        if (t.defined()) {
          specs.push_back({t.sizes().vec(), t.scalar_type(), false});
        } else {
          specs.push_back({{}, c10::ScalarType::Undefined, true});
        }
      }
    }
    // Non-tensor returns (int, bool) are skipped for MetaOutputSpec.
    // Callers that need non-tensor returns should use RunMetaDispatch directly
    // and inspect the stack themselves.
  }
  return specs;
}

std::vector<MetaOutputSpec> InferOutputsViaMeta(const c10::OperatorHandle& op,
                                                const std::vector<c10::IValue>& args) {
  auto stack = RunMetaDispatch(op, args);
  return ExtractMetaOutputSpecs(stack);
}

bool NormalizeWrappedIntegerScalarViaMeta(const c10::OperatorHandle& op,
                                          std::vector<c10::IValue>& args,
                                          const c10::FunctionSchema& schema) {
  // The output dtype of a writable overload is constrained by its destination
  // and cannot prove the arithmetic operand dtype. Keep out=/in-place behavior
  // on its existing path.
  for (const auto& schema_arg : schema.arguments()) {
    const auto& alias_info = schema_arg.alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) return false;
  }

  if (schema.returns().size() != 1 ||
      schema.returns()[0].type()->kind() != c10::TypeKind::TensorType) {
    return false;
  }

  c10::optional<size_t> wrapped_index;
  c10::optional<c10::ScalarType> ordinary_dtype;
  size_t defined_tensor_count = 0;

  for (size_t i = 0; i < args.size(); ++i) {
    if (!args[i].isTensor()) continue;
    const auto& tensor = args[i].toTensor();
    if (!tensor.defined()) continue;
    ++defined_tensor_count;

    const bool is_integral = c10::isIntegralType(tensor.scalar_type(), /*includeBool=*/false);
    const bool is_wrapped = tensor.unsafeGetTensorImpl()->is_wrapped_number();
    if (is_wrapped) {
      if (wrapped_index.has_value() || tensor.dim() != 0 || !is_integral) return false;
      wrapped_index = i;
      continue;
    }

    // This proposal is intentionally scoped to the failing reference route:
    // one real rank-0 integer tensor plus one wrapped Python integer.
    if (ordinary_dtype.has_value() || tensor.dim() != 0 || !is_integral) return false;
    ordinary_dtype = tensor.scalar_type();
  }

  if (defined_tensor_count != 2 || !wrapped_index.has_value() || !ordinary_dtype.has_value()) {
    return false;
  }

  // Result dtype equality is meaningful only when these are the conventional
  // binary promoting operands. Reject tensor roles such as index, mask, or
  // destination even if their operator happens to return the other tensor's
  // dtype.
  const auto& schema_args = schema.arguments();
  if (schema_args.size() < 2 || args.size() < 2 || schema_args[0].name() != "self" ||
      schema_args[1].name() != "other" || (*wrapped_index != 0 && *wrapped_index != 1)) {
    return false;
  }

  std::vector<MetaOutputSpec> output_specs;
  try {
    output_specs = InferOutputsViaMeta(op, args);
  } catch (const c10::Error& e) {
    TORCH_NEURONX_DISPATCH_DEBUG("[NormalizeWrappedIntegerScalarViaMeta] Meta dispatch failed",
                                 " op=", c10::toString(schema.operator_name()),
                                 " error=", e.what_without_backtrace());
    return false;
  } catch (const std::exception& e) {
    TORCH_NEURONX_DISPATCH_DEBUG("[NormalizeWrappedIntegerScalarViaMeta] Meta dispatch failed",
                                 " op=", c10::toString(schema.operator_name()),
                                 " error=", e.what());
    return false;
  }

  if (output_specs.size() != 1 || output_specs[0].is_undefined ||
      output_specs[0].dtype != *ordinary_dtype) {
    return false;
  }

  auto wrapped = args[*wrapped_index].toTensor();
  auto normalized = wrapped.to(*ordinary_dtype, /*non_blocking=*/false, /*copy=*/true);
  normalized.unsafeGetTensorImpl()->set_wrapped_number(false);
  args[*wrapped_index] = c10::IValue(std::move(normalized));
  TORCH_NEURONX_DISPATCH_DEBUG("[NormalizeWrappedIntegerScalarViaMeta] normalized",
                               " op=", c10::toString(schema.operator_name()),
                               " wrapped_dtype=", wrapped.scalar_type(),
                               " meta_dtype=", *ordinary_dtype);
  return true;
}

// ═══════════════════════════════════════════════════════════════
//  IValue → Python conversion
// ═══════════════════════════════════════════════════════════════

PyArgsKwargs IValuesToPyArgsKwargs(const std::vector<c10::IValue>& args,
                                   const c10::FunctionSchema& schema) {
  const auto& schema_args = schema.arguments();
  std::vector<py::object> positional;
  py::dict py_kwargs;

  for (size_t i = 0; i < args.size() && i < schema_args.size(); ++i) {
    py::object py_obj;
    const auto& arg = args[i];
    const auto& arg_info = schema_args[i];

    if (arg.isTensor()) {
      auto t = arg.toTensor();
      py_obj = py::reinterpret_steal<py::object>(THPVariable_Wrap(t));
    } else {
      py_obj = utils::IValueToPyObjectWithSchema(arg, arg_info.real_type());
    }

    if (arg_info.kwarg_only()) {
      py_kwargs[py::str(arg_info.name())] = py_obj;
    } else {
      positional.push_back(py_obj);
    }
  }

  py::tuple py_args(positional.size());
  for (size_t i = 0; i < positional.size(); ++i) {
    py_args[i] = positional[i];
  }

  return {std::move(py_args), std::move(py_kwargs)};
}

// ═══════════════════════════════════════════════════════════════
//  Static vs runtime argument classification
// ═══════════════════════════════════════════════════════════════

bool IsScalarListType(const c10::Argument& arg) {
  auto type = arg.type();
  if (type->kind() != c10::TypeKind::ListType) return false;
  auto elem_type = type->cast<c10::ListType>()->getElementType();
  return elem_type->isSubtypeOf(*c10::NumberType::get());
}

bool IsStaticArg(const c10::FunctionSchema& schema, size_t i,
                 const std::unordered_set<size_t>& static_argnums,
                 const std::unordered_set<std::string>& static_argnames) {
  if (static_argnums.count(i) > 0) return true;
  const auto& schema_args = schema.arguments();
  if (i >= schema_args.size()) return false;
  return static_argnames.count(schema_args[i].name()) > 0;
}

bool IsRuntimeScalarList(const c10::FunctionSchema& schema, size_t i, const c10::IValue& arg,
                         const std::unordered_set<size_t>& static_argnums,
                         const std::unordered_set<std::string>& static_argnames) {
  const auto& schema_args = schema.arguments();
  if (i >= schema_args.size()) return false;
  if (schema_args[i].kwarg_only()) return false;
  if (!IsScalarListType(schema_args[i])) return false;
  if (!(arg.isIntList() || arg.isDoubleList() || arg.isList())) return false;
  return !IsStaticArg(schema, i, static_argnums, static_argnames);
}

}  // namespace dispatch
}  // namespace torch_neuronx

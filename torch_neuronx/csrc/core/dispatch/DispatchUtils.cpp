#include "torch_neuronx/csrc/core/dispatch/DispatchUtils.h"

#include <ATen/EmptyTensor.h>
#include <torch/csrc/autograd/python_variable.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/dispatch/DtypeUtils.h"
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

// ═══════════════════════════════════════════════════════════════
//  64-bit Dtype Detection and Downcasting
// ═══════════════════════════════════════════════════════════════

// Neuron-supported dtypes for MLIR lowering. Matches Python copy.py / to_copy.py.
// Dtypes NOT in this set are handled via CPU bounce in DispatchCopy/DispatchToCopy.
bool IsSupportedDtype(c10::ScalarType dt) {
  if (dt == c10::ScalarType::Long) return IsNativeInt64Enabled();
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
                                         std::vector<at::neuron::TensorContext>& tensor_metadata,
                                         std::vector<nrt_tensor_t*>& tensor_data_ptrs) {
  tensor_metadata.push_back(at::neuron::TensorContext::FromTensor(tensor));
  tensor_data_ptrs.push_back(
      c10_neuron::NeuronCachingAllocator::findTensor(const_cast<void*>(tensor.storage().data())));
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
        ExtractTensorInfoForFallback(tensor, ctx.tensor_metadata, ctx.tensor_data_ptrs);
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
        ExtractTensorInfoForFallback(t, ctx.tensor_metadata, ctx.tensor_data_ptrs);
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
            ExtractTensorInfoForFallback(t, ctx.tensor_metadata, ctx.tensor_data_ptrs);
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

const IndexOpConfig* GetIndexOpConfig(const std::string& op_name) {
  // Mirrors Python config.py: IndexOps.INDEX_OPS
  // index_argnum = position of the index tensor, input_argnum = position of self/input
  // When input is non-empty and index is empty → return input unchanged
  static const std::unordered_map<std::string, IndexOpConfig> kConfigs = {
      {"scatter", {2, 0}},
      {"scatter_add", {2, 0}},
      {"_index_put_impl_", {1, 0}},
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
  return at::detail::empty_strided_meta(t.sizes(), t.strides(), t.scalar_type());
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

}  // namespace dispatch
}  // namespace torch_neuronx

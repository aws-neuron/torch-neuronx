#include "torch_neuronx/csrc/core/cache/CacheKeyGenerator.h"

#include <c10/core/ScalarType.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/autograd/python_variable.h>

#include <algorithm>

#include "torch_neuronx/csrc/core/dispatch/DispatchUtils.h"

namespace at::neuron {

// ═══════════════════════════════════════════════════════════════
//  Filtered kwargs — from Python CompilationConfig.PYTORCH_SPECIFIC_PARAMS
// ═══════════════════════════════════════════════════════════════

const std::unordered_set<std::string>& CacheKeyGenerator::FilteredKwargs() {
  static const std::unordered_set<std::string> kFiltered = {
      "out", "device", "pin_memory", "memory_format", "layout", "requires_grad",
  };
  return kFiltered;
}

// ═══════════════════════════════════════════════════════════════
//  Shared type formatters
//
//  Each type has ONE formatting function. Both AppendIValue (C++ path)
//  and AppendPyObject (Python path) extract the C++ value and call
//  the same formatter, guaranteeing identical output.
// ═══════════════════════════════════════════════════════════════

void CacheKeyGenerator::AppendNone(std::ostringstream& key) { key << "None"; }

void CacheKeyGenerator::AppendBool(std::ostringstream& key, bool val) {
  key << "b" << (val ? 1 : 0);
}

void CacheKeyGenerator::AppendInt(std::ostringstream& key, int64_t val, bool is_static) {
  if (is_static) {
    key << "i" << val;
  } else {
    key << "int";
  }
}

void CacheKeyGenerator::AppendDouble(std::ostringstream& key, double val, bool is_static) {
  if (is_static) {
    key << "d" << val;
  } else {
    key << "float";
  }
}

void CacheKeyGenerator::AppendComplexDouble(std::ostringstream& key, c10::complex<double> val,
                                            bool is_static) {
  if (is_static) {
    key << "cd:" << val.real() << "+" << val.imag() << "i";
  } else {
    key << "complex";
  }
}

void CacheKeyGenerator::AppendString(std::ostringstream& key, std::string_view val) {
  key << "s:" << val;
}

void CacheKeyGenerator::AppendScalarType(std::ostringstream& key, c10::ScalarType st) {
  key << c10::toString(st);
}

void CacheKeyGenerator::AppendDevice(std::ostringstream& key, const c10::Device& dev) {
  key << "dev:" << dev.str();
}

void CacheKeyGenerator::AppendTensor(std::ostringstream& key, const at::Tensor& t) {
  key << "[";
  for (int64_t d = 0; d < t.dim(); ++d) {
    if (d > 0) key << ",";
    key << t.size(d);
  }
  key << "]:";
  AppendScalarType(key, t.scalar_type());
  key << ":S(";
  if (t.is_contiguous()) {
    key << "C";
  } else {
    for (int64_t d = 0; d < t.dim(); ++d) {
      if (d > 0) key << ",";
      key << t.stride(d);
    }
  }
  key << ")";
}

void CacheKeyGenerator::AppendTensorOrUndef(std::ostringstream& key, const at::Tensor& t) {
  if (t.defined()) {
    AppendTensor(key, t);
  } else {
    key << "undef";
  }
}

namespace {
// Template helper — avoids std::function overhead on the hot dispatch path.
template <typename AppendElemFn>
void AppendListImpl(std::ostringstream& key, const char* tag, size_t n,
                    AppendElemFn&& append_elem_fn) {
  key << tag << "[";
  for (size_t j = 0; j < n; ++j) {
    if (j > 0) key << ",";
    append_elem_fn(j);
  }
  key << "]";
}
}  // namespace

// ═══════════════════════════════════════════════════════════════
//  C++ IValue path — thin type dispatch to shared formatters
// ═══════════════════════════════════════════════════════════════

void CacheKeyGenerator::AppendIValue(std::ostringstream& key, const c10::IValue& val,
                                     bool is_static) {
  if (val.isNone()) {
    AppendNone(key);
  } else if (val.isBool()) {
    AppendBool(key, val.toBool());
  } else if (val.isInt()) {
    AppendInt(key, val.toInt(), is_static);
  } else if (val.isDouble()) {
    AppendDouble(key, val.toDouble(), is_static);
  } else if (val.isComplexDouble()) {
    AppendComplexDouble(key, val.toComplexDouble(), is_static);
  } else if (val.isScalar()) {
    // Only reachable for symbolic scalars (SymInt/SymFloat/SymBool).
    // Regular scalars decompose to Tag::Int/Double/Bool in IValue constructor.
    if (is_static) {
      key << "sc:" << val.toScalar().toDouble();
    } else {
      auto s = val.toScalar();
      if (s.isIntegral(/*includeBool=*/false)) {
        key << "scalar:int";
      } else if (s.isFloatingPoint()) {
        key << "scalar:float";
      } else if (s.isBoolean()) {
        key << "scalar:bool";
      } else if (s.isComplex()) {
        key << "scalar:complex";
      } else {
        key << "scalar";
      }
    }
  } else if (val.isString()) {
    AppendString(key, val.toStringRef());
  } else if (val.isDevice()) {
    AppendDevice(key, val.toDevice());
  } else if (val.isGenerator()) {
    key << "gen";
  } else if (val.isTensor()) {
    AppendTensorOrUndef(key, val.toTensor());
  } else if (val.isTensorList()) {
    const auto& tl = val.toTensorList();
    AppendListImpl(key, "TL", tl.size(),
                   [&](size_t j) { AppendTensorOrUndef(key, at::Tensor(tl[j])); });
  } else if (val.isIntList()) {
    const auto& il = val.toIntList();
    AppendListImpl(key, "IL", il.size(),
                   [&](size_t j) { AppendInt(key, il[j], /*is_static=*/true); });
  } else if (val.isBoolList()) {
    const auto& bl = val.toBoolList();
    AppendListImpl(key, "BL", bl.size(), [&](size_t j) { AppendBool(key, bl[j]); });
  } else if (val.isDoubleList()) {
    const auto& dl = val.toDoubleList();
    AppendListImpl(key, "DL", dl.size(),
                   [&](size_t j) { AppendDouble(key, dl[j], /*is_static=*/true); });
  } else if (val.isList()) {
    const auto& gl = val.toListRef();
    AppendListImpl(key, "GL", gl.size(),
                   [&](size_t j) { AppendIValue(key, gl[j], /*is_static=*/true); });
  } else {
    TORCH_WARN_ONCE("CacheKeyGenerator: unhandled IValue tag '", val.tagKind(),
                    "' — cache key may not be unique for different values of this type");
    key << val.tagKind();
  }
}

std::string CacheKeyGenerator::Generate(const std::string& op_name,
                                        const c10::FunctionSchema& schema,
                                        const std::vector<c10::IValue>& args,
                                        const std::unordered_set<size_t>& static_argnums,
                                        const std::unordered_set<std::string>& static_argnames) {
  std::ostringstream key;
  key << op_name << "--";
  // Int64 mode affects lowering — different modes must not share cache entries.
  static const auto int64_mode = torch_neuronx::dispatch::GetInt64Mode();
  if (int64_mode == torch_neuronx::dispatch::Int64Mode::kPacked) {
    key << "p64--";
  } else if (int64_mode == torch_neuronx::dispatch::Int64Mode::kNative) {
    key << "n64--";
  }

  const auto& schema_args = schema.arguments();
  const auto& filtered = FilteredKwargs();

  for (size_t i = 0; i < args.size(); ++i) {
    const auto& arg = args[i];

    bool is_kwarg_only = (i < schema_args.size() && schema_args[i].kwarg_only());

    // Determine if this arg is static (must happen BEFORE filtering).
    bool is_static = static_argnums.count(i) > 0;
    if (is_kwarg_only && i < schema_args.size()) {
      is_static = is_static || static_argnames.count(schema_args[i].name()) > 0;
    }
    // dtype args always encode their value — different dtypes produce
    // different HLO and must never share a cache entry.
    // Name-based check required because PyTorch lowers ScalarType? to int?
    // in the runtime schema, so the type-based check alone misses it.
    if (!is_static && i < schema_args.size()) {
      if (schema_args[i].name() == "dtype") {
        is_static = true;
      }
      if (!is_static) {
        auto type_kind = schema_args[i].type()->kind();
        if (type_kind == c10::TypeKind::ScalarTypeType) {
          is_static = true;
        } else if (type_kind == c10::TypeKind::OptionalType) {
          auto elem = schema_args[i].type()->cast<c10::OptionalType>()->getElementType();
          if (elem->kind() == c10::TypeKind::ScalarTypeType) {
            is_static = true;
          }
        }
      }
    }

    // Filter kwargs that don't affect compilation.
    // IMPORTANT: static args are NEVER filtered — they affect HLO by definition.
    // This ensures e.g. memory_format is included when registered as static
    // for aten::contiguous, even though it's in the filter set.
    // Exception: tensor "out" kwargs always included (affect output shape/dtype).
    if (is_kwarg_only && !is_static && i < schema_args.size()) {
      const auto& name = schema_args[i].name();
      if (filtered.count(name) && !(name == "out" && arg.isTensor())) {
        continue;
      }
    }

    key << "_";

    if (is_kwarg_only && i < schema_args.size()) {
      key << "kw:" << schema_args[i].name() << "=";
    }

    AppendIValue(key, arg, is_static);
  }

  return key.str();
}

// ═══════════════════════════════════════════════════════════════
//  Python path — thin type dispatch to shared formatters
// ═══════════════════════════════════════════════════════════════

void CacheKeyGenerator::AppendPyObject(std::ostringstream& key, const py::handle& obj,
                                       bool is_static) {
  // Check bool BEFORE int (Python bool is subclass of int).
  if (obj.is_none()) {
    AppendNone(key);
  } else if (py::isinstance<py::bool_>(obj)) {
    AppendBool(key, obj.cast<bool>());
  } else if (THPVariable_Check(obj.ptr())) {
    AppendTensorOrUndef(key, THPVariable_Unpack(obj.ptr()));
  } else if (py::isinstance<py::int_>(obj)) {
    AppendInt(key, obj.cast<int64_t>(), is_static);
  } else if (py::isinstance<py::float_>(obj)) {
    AppendDouble(key, obj.cast<double>(), is_static);
  } else if (py::isinstance<py::str>(obj)) {
    AppendString(key, obj.cast<std::string>());
  } else if (PyComplex_Check(obj.ptr())) {
    auto c = obj.cast<std::complex<double>>();
    AppendComplexDouble(key, c10::complex<double>(c.real(), c.imag()), is_static);
  } else if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
    auto seq = py::reinterpret_borrow<py::sequence>(obj);
    if (seq.empty()) {
      AppendListImpl(key, "GL", 0, [](size_t) {});
    } else {
      // Infer typed tag from first element.
      auto first = seq[0];
      const char* tag = "GL";
      if (THPVariable_Check(first.ptr()))
        tag = "TL";
      else if (py::isinstance<py::int_>(first) && !py::isinstance<py::bool_>(first))
        tag = "IL";
      else if (py::isinstance<py::bool_>(first))
        tag = "BL";
      else if (py::isinstance<py::float_>(first))
        tag = "DL";
      AppendListImpl(key, tag, seq.size(),
                     [&](size_t j) { AppendPyObject(key, seq[j], /*is_static=*/true); });
    }
  } else if (THPDtype_Check(obj.ptr())) {
    key << "dtype:";
    AppendScalarType(key, reinterpret_cast<THPDtype*>(obj.ptr())->scalar_type);
  } else if (THPDevice_Check(obj.ptr())) {
    AppendDevice(key, reinterpret_cast<THPDevice*>(obj.ptr())->device);
  } else {
    key << py::str(obj.get_type()).cast<std::string>();
  }
}

std::string CacheKeyGenerator::GenerateFromPython(
    const std::string& op_name, const py::tuple& positional_args, const py::dict& keyword_args,
    const std::unordered_set<size_t>& static_argnums,
    const std::unordered_set<std::string>& static_argnames) {
  std::ostringstream key;
  key << op_name << "--";
  static const auto int64_mode = torch_neuronx::dispatch::GetInt64Mode();
  if (int64_mode == torch_neuronx::dispatch::Int64Mode::kPacked) {
    key << "p64--";
  } else if (int64_mode == torch_neuronx::dispatch::Int64Mode::kNative) {
    key << "n64--";
  }

  const auto& filtered = FilteredKwargs();

  // Positional args
  for (size_t i = 0; i < positional_args.size(); ++i) {
    key << "_";
    bool is_static = static_argnums.count(i) > 0;
    AppendPyObject(key, positional_args[i], is_static);
  }

  // Keyword args (sorted by name for deterministic ordering).
  // We store (name, is_static) pairs so the static determination happens once.
  if (keyword_args.size() > 0) {
    std::vector<std::pair<std::string, bool>> kw_entries;
    for (auto& [k, v] : keyword_args) {
      auto name = k.cast<std::string>();
      bool is_static = static_argnames.count(name) > 0;
      // dtype kwargs always encode their value — different dtypes produce
      // different HLO and must never share a cache entry. Auto-detect by
      // name or by checking if the value is a torch.dtype object.
      if (!is_static && (name == "dtype" || THPDtype_Check(v.ptr()))) {
        is_static = true;
      }
      // Filter kwargs that don't affect compilation.
      // IMPORTANT: static args are NEVER filtered.
      if (!is_static) {
        bool is_out_tensor = (name == "out" && THPVariable_Check(v.ptr()));
        if (filtered.count(name) && !is_out_tensor) {
          continue;
        }
      }
      kw_entries.emplace_back(std::move(name), is_static);
    }
    std::sort(kw_entries.begin(), kw_entries.end());
    for (const auto& [name, is_static] : kw_entries) {
      key << "_kw:" << name << "=";
      AppendPyObject(key, keyword_args[py::str(name)], is_static);
    }
  }

  return key.str();
}

void CacheKeyGenerator::AppendIntList(std::ostringstream& key, const std::vector<int>& vals) {
  key << "IL[";
  for (size_t i = 0; i < vals.size(); ++i) {
    if (i > 0) key << ",";
    AppendInt(key, static_cast<int64_t>(vals[i]), /*is_static=*/true);
  }
  key << "]";
}

std::string CacheKeyGenerator::GenerateCollective(
    const std::string& op_type, const std::vector<at::Tensor>& inputs,
    const std::vector<std::vector<int>>& replica_groups, int reduce_op, bool slice_output) {
  std::ostringstream key;
  key << "coll|" << op_type;
  for (const auto& t : inputs) {
    key << "|";
    AppendTensor(key, t);
  }
  key << "|rg[";
  for (size_t i = 0; i < replica_groups.size(); ++i) {
    if (i > 0) key << ",";
    AppendIntList(key, replica_groups[i]);
  }
  key << "]";
  AppendInt(key, reduce_op, /*is_static=*/true);
  if (slice_output) AppendBool(key, true);
  return key.str();
}

std::string CacheKeyGenerator::GenerateDynamicCollectiveKey(
    const std::string& op_type, int64_t trace_numel, int64_t large_tile, int64_t medium_tile,
    int64_t small_tile, const std::vector<std::vector<int>>& replica_groups, c10::ScalarType dtype,
    int64_t unroll, int reduce_op) {
  std::ostringstream key;
  key << "coll|" << op_type << "|dynamic|trace_numel=" << trace_numel << "|tiles=" << large_tile
      << "," << medium_tile << "," << small_tile << "|dt=";
  AppendScalarType(key, dtype);
  key << "|rg[";
  for (size_t i = 0; i < replica_groups.size(); ++i) {
    if (i > 0) key << ",";
    AppendIntList(key, replica_groups[i]);
  }
  key << "]";
  if (reduce_op >= 0) {
    key << "|red=" << reduce_op;
  }
  // distinct NEFF per unroll variant
  key << "|u=" << unroll;
  return key.str();
}

}  // namespace at::neuron

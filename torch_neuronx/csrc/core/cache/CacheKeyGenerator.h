#pragma once

#include <torch/extension.h>

#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace at::neuron {

/**
 * Single source of truth for dispatch cache key generation.
 *
 * Combines features from both prior implementations:
 *   From C++ (NeuronDispatcher): strides, storage_offset, typed lists,
 *       device info, c10::toString(ScalarType) for dtype
 *   From Python (base.py/kernel.py): static/non-static scalar distinction,
 *       kwarg filtering (PYTORCH_SPECIFIC_PARAMS), version hash support
 *
 * Two entry points produce identical format:
 *   Generate()           — called from C++ dispatch (IValue-based args)
 *   GenerateFromPython() — called from Python via pybind (Python objects)
 */
class CacheKeyGenerator {
 public:
  /// Set of kwarg names to exclude from cache key (PyTorch-specific params
  /// that don't affect HLO compilation).  A kwarg in this set is still
  /// included if it is also registered as a static_argname.
  static const std::unordered_set<std::string>& FilteredKwargs();

  // ── C++ dispatch path ──
  // Called from NeuronDispatcher::GenerateCacheKey.
  // Uses FunctionSchema for kwarg identification and static_argnums for
  // static vs. non-static scalar distinction.
  static std::string Generate(const std::string& op_name, const c10::FunctionSchema& schema,
                              const std::vector<c10::IValue>& args,
                              const std::unordered_set<size_t>& static_argnums = {},
                              const std::unordered_set<std::string>& static_argnames = {});

  // ── Python pybind path ──
  // Called from Python via _C._generate_cache_key().
  // Positional args are in order; keyword args sorted by name.
  // static_argnums and static_argnames control scalar encoding.
  static std::string GenerateFromPython(
      const std::string& op_name, const py::tuple& positional_args, const py::dict& keyword_args,
      const std::unordered_set<size_t>& static_argnums = {},
      const std::unordered_set<std::string>& static_argnames = {});

 public:
  // Encode a tensor's shape, dtype, and strides into a cache key fragment.
  // Public so NkiDispatcher can reuse the same encoding format.
  static void AppendTensor(std::ostringstream& key, const at::Tensor& t);

 private:
  // ── Shared type formatters ──
  // Each type has exactly ONE formatting function. Both AppendIValue (C++ path)
  // and AppendPyObject (Python path) extract the C++ value and delegate here.
  // This guarantees identical cache key output regardless of entry point.
  static void AppendNone(std::ostringstream& key);
  static void AppendBool(std::ostringstream& key, bool val);
  static void AppendInt(std::ostringstream& key, int64_t val, bool is_static);
  static void AppendDouble(std::ostringstream& key, double val, bool is_static);
  static void AppendComplexDouble(std::ostringstream& key, c10::complex<double> val,
                                  bool is_static);
  static void AppendString(std::ostringstream& key, std::string_view val);
  static void AppendScalarType(std::ostringstream& key, c10::ScalarType st);
  static void AppendDevice(std::ostringstream& key, const c10::Device& dev);
  static void AppendTensorOrUndef(std::ostringstream& key, const at::Tensor& t);

  // AppendList is a private template in the .cpp file (avoids std::function overhead).

  // ── Type dispatch entry points (thin extractors that delegate to shared formatters) ──
  static void AppendIValue(std::ostringstream& key, const c10::IValue& val, bool is_static);
  static void AppendPyObject(std::ostringstream& key, const py::handle& obj, bool is_static);
};

}  // namespace at::neuron

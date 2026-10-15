#pragma once

#include <torch/extension.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace torch_neuronx {
namespace dispatch {

/**
 * Single source of truth for the lowering result schema.
 *
 * Defined in C++ and bound to Python via pybind11 (see NeuronBindings.cpp).
 * Python creates instances via torch_neuronx._C.LoweringResult(); C++ reads
 * them in ParseLoweringResult by casting directly — no manual attr() parsing,
 * no risk of field drift between the two languages.
 *
 * Fields that are only used within Python (input_specs, kw_input_specs) are
 * stored as py::object so Python can attach arbitrary data without requiring
 * a C++ type definition.
 */
struct LoweringResult {
  // StableHLO IR bytes for this (op, shape) combination
  py::bytes stablehlo_bytes;

  // Cache key for NEFF compilation cache
  std::string cache_key;

  // Which flattened inputs the HLO actually uses
  std::vector<int> kept_input_indices;

  // Number of inputs in the compiled IR. -1 preserves legacy filtering.
  int64_t ir_input_count = -1;

  // Which original outputs the HLO produces (for reconstruction)
  std::vector<int> kept_output_indices;

  // Output specifications: list of (shape_tuple, dtype_int) pairs.
  // dtype_int is a c10::ScalarType cast to int, or py::none() for undefined.
  py::list output_specs;

  // Original output specs (before any downcast, for upcast).  Same format.
  py::list original_output_specs;

  // Passthrough mappings: output_idx → input_idx
  std::unordered_map<int, int> passthrough_mappings;

  // Per-IR-input mapping from positional args (Python-only, not consumed by C++).
  py::list input_specs;

  // Per-IR-input mapping from keyword args (Python-only, not consumed by C++).
  py::list kw_input_specs;

  // Op metadata
  std::string op_name;
  bool is_inplace = false;
  bool has_collectives = false;
  bool is_empty_tensor = false;

  // Input-output aliasing: list of (ir_param_idx, output_idx) tuples.
  std::vector<std::pair<int, int>> input_output_aliases;

  // Inferred output dtypes at original (pre-downcast) precision.
  // Python passes torch.dtype objects (or None for non-tensor outputs).
  // ParseLoweringResult extracts c10::ScalarType via THPDtype.
  // Empty list means no dtype inference was performed (e.g., specialized ops).
  py::list inferred_output_dtypes;

  // Error state — if set, C++ dispatcher should fall back to CPU
  py::object error = py::none();

  // NKI DumpedConfig fields — attached by _lower_custom_nki on cache miss.
  // Transported to DispatchCacheEntry by ParseLoweringResult.
  py::list nki_return_types;            // list of (dtype_str, shape_tuple) pairs
  py::dict nki_operand_output_aliases;  // {input_idx: output_idx}
};

/**
 * Register the LoweringResult class with pybind11.
 * Called from PYBIND11_MODULE in NeuronBindings.cpp.
 */
inline void BindLoweringResult(py::module_& m) {
  py::class_<LoweringResult>(m, "LoweringResult")
      .def(py::init([](py::kwargs kwargs) {
        auto lr = std::make_unique<LoweringResult>();
        // Apply any keyword arguments to the corresponding fields.
        for (auto& [key, val] : kwargs) {
          auto name = key.cast<std::string>();
          if (name == "stablehlo_bytes")
            lr->stablehlo_bytes = val.cast<py::bytes>();
          else if (name == "cache_key")
            lr->cache_key = val.cast<std::string>();
          else if (name == "kept_input_indices")
            lr->kept_input_indices = val.cast<std::vector<int>>();
          else if (name == "ir_input_count")
            lr->ir_input_count = val.cast<int64_t>();
          else if (name == "kept_output_indices")
            lr->kept_output_indices = val.cast<std::vector<int>>();
          else if (name == "output_specs")
            lr->output_specs = val.cast<py::list>();
          else if (name == "original_output_specs")
            lr->original_output_specs = val.cast<py::list>();
          else if (name == "passthrough_mappings")
            lr->passthrough_mappings = val.cast<std::unordered_map<int, int>>();
          else if (name == "input_specs")
            lr->input_specs = val.cast<py::list>();
          else if (name == "kw_input_specs")
            lr->kw_input_specs = val.cast<py::list>();
          else if (name == "op_name")
            lr->op_name = val.cast<std::string>();
          else if (name == "is_inplace")
            lr->is_inplace = val.cast<bool>();
          else if (name == "has_collectives")
            lr->has_collectives = val.cast<bool>();
          else if (name == "is_empty_tensor")
            lr->is_empty_tensor = val.cast<bool>();
          else if (name == "input_output_aliases")
            lr->input_output_aliases = val.cast<std::vector<std::pair<int, int>>>();
          else if (name == "inferred_output_dtypes")
            lr->inferred_output_dtypes = val.cast<py::list>();
          else if (name == "error")
            lr->error = py::reinterpret_borrow<py::object>(val);
          else if (name == "nki_return_types")
            lr->nki_return_types = val.cast<py::list>();
          else if (name == "nki_operand_output_aliases")
            lr->nki_operand_output_aliases = val.cast<py::dict>();
          else
            throw py::type_error("LoweringResult: unknown field '" + name + "'");
        }
        return lr;
      }))
      .def_readwrite("stablehlo_bytes", &LoweringResult::stablehlo_bytes)
      .def_readwrite("cache_key", &LoweringResult::cache_key)
      .def_readwrite("kept_input_indices", &LoweringResult::kept_input_indices)
      .def_readwrite("ir_input_count", &LoweringResult::ir_input_count)
      .def_readwrite("kept_output_indices", &LoweringResult::kept_output_indices)
      .def_readwrite("output_specs", &LoweringResult::output_specs)
      .def_readwrite("original_output_specs", &LoweringResult::original_output_specs)
      .def_readwrite("passthrough_mappings", &LoweringResult::passthrough_mappings)
      .def_readwrite("input_specs", &LoweringResult::input_specs)
      .def_readwrite("kw_input_specs", &LoweringResult::kw_input_specs)
      .def_readwrite("op_name", &LoweringResult::op_name)
      .def_readwrite("is_inplace", &LoweringResult::is_inplace)
      .def_readwrite("has_collectives", &LoweringResult::has_collectives)
      .def_readwrite("is_empty_tensor", &LoweringResult::is_empty_tensor)
      .def_readwrite("input_output_aliases", &LoweringResult::input_output_aliases)
      .def_readwrite("inferred_output_dtypes", &LoweringResult::inferred_output_dtypes)
      .def_readwrite("error", &LoweringResult::error)
      .def_readwrite("nki_return_types", &LoweringResult::nki_return_types)
      .def_readwrite("nki_operand_output_aliases", &LoweringResult::nki_operand_output_aliases);
}

}  // namespace dispatch
}  // namespace torch_neuronx

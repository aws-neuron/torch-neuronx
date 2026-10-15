#include "torch_neuronx/csrc/core/dispatch/ParseLoweringResult.h"

#include <torch/csrc/Dtype.h>

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/NeuronOpsApi.h"
#include "torch_neuronx/csrc/core/dispatch/LoweringResult.h"

namespace torch_neuronx {
namespace dispatch {

// Helper: extract c10::ScalarType from either torch.dtype or raw int.
static c10::ScalarType ParseDtype(const py::handle& obj) {
  if (THPDtype_Check(obj.ptr())) {
    return reinterpret_cast<THPDtype*>(obj.ptr())->scalar_type;
  }
  return static_cast<c10::ScalarType>(obj.cast<int>());
}

// Helper: parse a py::list of (shape_tuple, dtype) tuples into OutputSpec vector.
// dtype may be None — mapped to ScalarType::Undefined.
static std::vector<DispatchCacheEntry::OutputSpec> ParseOutputSpecs(const py::list& specs,
                                                                    bool allow_none_dtype) {
  std::vector<DispatchCacheEntry::OutputSpec> result;
  for (auto item : specs) {
    auto tup = item.cast<py::tuple>();
    DispatchCacheEntry::OutputSpec spec;
    if (allow_none_dtype && tup[1].is_none()) {
      spec.dtype = c10::ScalarType::Undefined;
    } else {
      auto shape = tup[0].cast<py::tuple>();
      for (auto dim : shape) {
        spec.shape.push_back(dim.cast<int64_t>());
      }
      spec.dtype = ParseDtype(tup[1]);
    }
    // Parse optional target_strides (3rd element of tuple).
    // Format: ((shape...), dtype_int, (stride0, stride1, ...))
    // When present, the dispatch allocates a physical-shape buffer and fuses
    // a TRANSPOSE epilogue so the NEFF writes in the target stride layout.
    if (torch_neuronx::IsEpilogueEnabled() && tup.size() >= 3 && !tup[2].is_none()) {
      auto strides_tuple = tup[2].cast<py::tuple>();
      std::vector<int64_t> strides;
      strides.reserve(strides_tuple.size());
      for (auto s : strides_tuple) {
        strides.push_back(s.cast<int64_t>());
      }
      spec.target_strides = std::move(strides);
    }
    result.push_back(std::move(spec));
  }
  return result;
}

std::shared_ptr<DispatchCacheEntry> ParseLoweringResult(const py::object& result) {
  TORCH_NEURONX_DISPATCH_DEBUG("[ParseLoweringResult] parsing LoweringResult");

  // Cast to the pybind11-bound struct — single source of truth, no attr() parsing.
  const auto& lr = result.cast<const LoweringResult&>();

  // Check for error
  if (!lr.error.is_none()) {
    std::string error_msg = lr.error.cast<std::string>();
    TORCH_NEURONX_ERROR("[ParseLoweringResult] LoweringResult has error", " error=", error_msg);
    TORCH_CHECK(false, "ParseLoweringResult failed: ", error_msg);
    return nullptr;
  }

  auto entry = std::make_shared<DispatchCacheEntry>();

  // Direct field access — no string-based attr lookups.
  std::string_view bytes_view = lr.stablehlo_bytes.cast<std::string_view>();
  entry->stablehlo_bytes.assign(
      reinterpret_cast<const uint8_t*>(bytes_view.data()),
      reinterpret_cast<const uint8_t*>(bytes_view.data()) + bytes_view.size());

  entry->neff_cache_key = lr.cache_key;
  entry->kept_input_indices = lr.kept_input_indices;
  entry->ir_input_count = lr.ir_input_count;
  entry->kept_output_indices = lr.kept_output_indices;
  entry->output_specs = ParseOutputSpecs(lr.output_specs, /*allow_none_dtype=*/false);
  entry->original_output_specs =
      ParseOutputSpecs(lr.original_output_specs, /*allow_none_dtype=*/true);
  entry->passthrough_mappings = lr.passthrough_mappings;
  entry->input_output_aliases = lr.input_output_aliases;
  entry->op_name = lr.op_name;
  entry->is_inplace = lr.is_inplace;
  entry->has_collectives = lr.has_collectives;
  entry->is_empty_tensor = lr.is_empty_tensor;

  // Parse inferred output dtypes (torch.dtype objects → ScalarType).
  // None entries (non-tensor outputs) are mapped to ScalarType::Undefined.
  for (auto item : lr.inferred_output_dtypes) {
    if (item.is_none()) {
      entry->inferred_output_dtypes.push_back(c10::ScalarType::Undefined);
    } else {
      entry->inferred_output_dtypes.push_back(ParseDtype(item));
    }
  }

  // Parse NKI DumpedConfig fields (present only for custom NKI kernels).
  if (lr.nki_return_types.size() > 0) {
    for (auto item : lr.nki_return_types) {
      auto tup = item.cast<py::tuple>();
      DispatchCacheEntry::NkiOutputSpec rt;
      rt.dtype_str = tup[0].cast<std::string>();
      auto shape = tup[1].cast<py::tuple>();
      for (auto dim : shape) {
        rt.shape.push_back(dim.cast<int64_t>());
      }
      entry->nki_output_specs.push_back(std::move(rt));
    }
    for (auto& [k, v] : lr.nki_operand_output_aliases) {
      entry->nki_operand_output_aliases[k.cast<int>()] = v.cast<int>();
    }
    entry->has_nki_config = true;
  }

  TORCH_NEURONX_DISPATCH_DEBUG(
      "[ParseLoweringResult] parsed successfully", " op_name=", entry->op_name,
      " neff_cache_key=", entry->neff_cache_key, " stablehlo_bytes=", entry->stablehlo_bytes.size(),
      " kept_inputs=", entry->kept_input_indices.size(), " ir_input_count=", entry->ir_input_count,
      " kept_outputs=", entry->kept_output_indices.size(),
      " output_specs=", entry->output_specs.size(),
      " original_output_specs=", entry->original_output_specs.size(),
      " passthrough_mappings=", entry->passthrough_mappings.size(),
      " input_output_aliases=", entry->input_output_aliases.size(),
      " is_inplace=", entry->is_inplace, " has_collectives=", entry->has_collectives,
      " is_empty_tensor=", entry->is_empty_tensor,
      " inferred_output_dtypes=", entry->inferred_output_dtypes.size());

  return entry;
}

}  // namespace dispatch
}  // namespace torch_neuronx

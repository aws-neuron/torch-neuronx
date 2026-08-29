#pragma once

#include <ATen/core/function_schema.h>
#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <string>
#include <unordered_set>
#include <vector>

namespace torch_neuronx {
namespace dispatch {

/**
 * Preprocess raw IValue args in-place to mirror Python's
 * ArgumentProcessor.preprocess_inputs(scalar_to_tensor=True):
 *
 *   - int/float scalars → 0-dim int32/float32 tensors on Neuron device
 *   - int64 tensors → int32 (convert_for_neuron)
 *   - float64 tensors → float32 (convert_for_neuron)
 *   - Non-contiguous tensors → contiguous (when prologue disabled)
 *   - CPU 0-dim scalar tensors → moved to Neuron device
 *
 * Called before GenerateCacheKey so the cache key reflects preprocessed
 * shapes/dtypes, matching what Python's _get_or_compile_hlo sees.
 * Kwarg-only args (per schema) are preprocessed for kwargs scalars only.
 *
 * @param args             IValue args to preprocess in place.
 * @param schema           Op schema (to distinguish positional vs kwarg-only).
 * @param static_argnums   Positions that must NOT be converted scalar→tensor.
 */
void PreprocessArgs(std::vector<c10::IValue>& args, const c10::FunctionSchema& schema,
                    const std::unordered_set<size_t>& static_argnums = {},
                    bool skip_contiguous = false, const std::string& op_name = "");

}  // namespace dispatch
}  // namespace torch_neuronx

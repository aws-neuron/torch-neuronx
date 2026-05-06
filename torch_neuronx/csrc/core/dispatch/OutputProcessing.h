#pragma once

#include <ATen/core/function_schema.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/torch.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "torch_neuronx/csrc/core/dispatch/NeuronDispatcher.h"

namespace torch_neuronx {
namespace dispatch {

/**
 * Allocate execution output tensors from the (potentially downcasted)
 * output_specs.  For inplace ops the first output is the first input
 * tensor instead of a fresh allocation.
 *
 * Uses input_output_aliases from the cache entry to reuse IR input
 * tensors (from filtered_inputs) as output buffers when the StableHLO
 * module declares aliasing (mhlo.input_output_alias).
 */
std::vector<torch::Tensor> AllocateExecutionOutputs(
    const DispatchCacheEntry& entry, const std::vector<c10::IValue>& args,
    const std::vector<torch::Tensor>& filtered_inputs, const torch::Device& device,
    const c10::FunctionSchema& schema);

/**
 * Upcast execution tensors whose dtype was downcasted for HW execution
 * back to the original dtype that the user expects.
 *
 * Uses inferred_output_dtypes from the DispatchCacheEntry (populated by
 * lower_func's dtype inference on original-precision args) for exact
 * per-output upcasting.
 *
 * Example: original=float64, execution=float32 → result.to(float64)
 */
std::vector<torch::Tensor> UpcastExecutionOutputs(
    const std::vector<torch::Tensor>& exec_outputs, const DispatchCacheEntry& entry,
    const std::vector<std::pair<size_t, torch::Tensor>>& original_out_tensors = {});

/**
 * Reconstruct the full output structure that the caller expects.
 *
 * The HLO may have optimized away some outputs (e.g., empty pass-throughs).
 * This function maps the executed outputs back to their original positions
 * using kept_output_indices, fills passthrough slots from inputs, and
 * creates empty tensors for any remaining gaps.
 */
std::vector<torch::Tensor> ReconstructOutputs(const std::vector<torch::Tensor>& exec_outputs,
                                              const DispatchCacheEntry& entry,
                                              const std::vector<c10::IValue>& args,
                                              const torch::Device& device);

/**
 * Push output tensors onto the PyTorch JIT stack.
 * Handles single-return, multi-return, TensorList-return, and void-return schemas.
 */
void PushOutputsToStack(torch::jit::Stack* stack, const std::vector<torch::Tensor>& outputs,
                        const c10::FunctionSchema& schema);

/**
 * Extract flattened tensors from IValue args (handles TensorList, Optional).
 */
std::vector<torch::Tensor> ExtractTensorInputs(const std::vector<c10::IValue>& args);

}  // namespace dispatch
}  // namespace torch_neuronx

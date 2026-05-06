#pragma once

#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <string>
#include <vector>

namespace torch_neuronx {
namespace dispatch {

// ═══════════════════════════════════════════════════════════════
//  Neuron-safe dtype casting helpers
//
//  Neuron HW doesn't natively support int64 or float64. All dtype
//  conversions involving these types on a Neuron device must
//  round-trip through CPU:  device → CPU → cast → device.
//
//  These helpers centralise that pattern, eliminating 6+ copies
//  that were scattered across PreprocessArgs, UpcastExecutionOutputs,
//  and the copy-back section of Dispatch().
// ═══════════════════════════════════════════════════════════════

/**
 * Cast a tensor to @p target_dtype, routing through CPU when the
 * source or target is int64/float64 on a Neuron (PrivateUse1) device.
 *
 * If the tensor is already the target dtype, it is returned unchanged.
 *
 * @param t             Input tensor (any device).
 * @param target_dtype  Desired output dtype.
 * @return              Tensor with the requested dtype, on the same device as @p t.
 */
torch::Tensor NeuronSafeCast(const torch::Tensor& t, c10::ScalarType target_dtype);

/**
 * Copy @p result into @p orig, handling:
 *   - Shape mismatch (resize_ orig if needed)
 *   - Dtype mismatch (route through NeuronSafeCast when needed)
 *   - Same-storage short-circuit (no-op if data_ptr matches)
 *
 * This is the shared implementation of the "copy result back into
 * the caller's original out/inplace tensor" pattern that appeared
 * in Dispatch() and UpcastExecutionOutputs().
 *
 * @param orig           The caller's original tensor (mutated in place).
 * @param result         The execution result to copy from.
 * @param debug_context  Label for TORCH_NEURONX_DEBUG messages.
 */
void CopyBackToOriginal(torch::Tensor& orig, const torch::Tensor& result,
                        const std::string& debug_context);

/**
 * Resize @p out to match @p target_shape, emitting PyTorch's documented
 * "An output with one or more elements was resized…" UserWarning when
 * @p out is non-empty and its shape differs from @p target_shape.
 *
 * This preserves the behavior of at::native::resize_output_check for
 * out= paths that go through the Neuron C++ dispatcher (which bypasses
 * the framework's structured-kernel resize check).
 *
 * No-op if @p out is already the correct shape.
 *
 * @param out           The caller's out= tensor (mutated in place).
 * @param target_shape  Shape the output must have after the call.
 */
void ResizeOutputWithWarning(torch::Tensor& out, c10::IntArrayRef target_shape);

}  // namespace dispatch
}  // namespace torch_neuronx

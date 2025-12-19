#include "LazyTransformationManager.h"

#include <chrono>
#include <mutex>
#include <numeric>
#include <sstream>
#include <unordered_map>

#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/concatenation/OpConcatUtils.h"
#include "torch_neuronx/csrc/core/lazy_materialization/TypeUtils.h"

namespace c10_neuron {
namespace lazy {

namespace {

// MergedOperationCache: Cache for merged prologue + operation MLIR bytes
// This caches the final result of merging prologue with operation,
// avoiding expensive MLIR merge operations on every call.
static std::unordered_map<std::string, std::vector<uint8_t>> g_merged_operation_cache;
static std::mutex g_merged_operation_cache_mutex;

// Helper to apply cached MLIR bytes to kernel (in-place update)
void ApplyCachedMlirToKernel(at::neuron::XLACompilableKernelExecution* kernel,
                             const std::vector<uint8_t>& merged_mlir_bytes,
                             const std::string& merged_operation_cache_key) {
  // Update kernel with cached HLO bytes only - tensors are already correct
  kernel->UpdateHloBytes(merged_mlir_bytes);

  // Update kernel's cache key since MLIR has changed
  // Operations with different prologues must have different kernel cache keys
  kernel->UpdateCacheKey(merged_operation_cache_key);
}

}  // anonymous namespace

std::vector<uint8_t> LazyTransformationManager::ProcessPrologueWithMlirMerge(
    const PrologueResult& prologue_result, const std::string& op_name,
    at::neuron::XLACompilableKernelExecution* compilable_kernel) {
  // Get original operation's MLIR and pointer addresses
  const auto& original_hlo_bytes = compilable_kernel->GetHloBytes();
  std::string original_mlir_str(original_hlo_bytes.begin(), original_hlo_bytes.end());
  const auto& original_input_ptrs = compilable_kernel->GetInputPtrs();
  const auto& original_output_ptrs = compilable_kernel->GetOutputPtrs();

  // Generate cache key for the merged result (for kernel naming)
  std::string merged_operation_cache_key =
      prologue_result.prologue_cache_key + "_" + compilable_kernel->GetCacheKey();

  TORCH_NEURONX_DEBUG("ProcessPrologueWithMlirMerge: Merging prologue with operation",
                      "op=", op_name);

  /**
   * Brief: Dummy Addresses are used in place of real addresses to ensure robust merge of
   * modules in case inputs are repeated, which can cause dependency errors in
   * OpConcatUtils merge APIs.
   * **/

  // Define dummy address ranges to prevent collisions
  // Each range can hold up to MAX_ADDRESSES_PER_RANGE addresses
  const uintptr_t MAX_ADDRESSES_PER_RANGE = 0x10000;                  // 65536 addresses per range
  const uintptr_t PROLOGUE_INPUT_BASE = 0 * MAX_ADDRESSES_PER_RANGE;  // Range 0: [0x00000, 0x10000)
  const uintptr_t PROLOGUE_OUTPUT_BASE =
      1 * MAX_ADDRESSES_PER_RANGE;                                  // Range 1: [0x10000, 0x20000)
  const uintptr_t DIRECT_INPUT_BASE = 2 * MAX_ADDRESSES_PER_RANGE;  // Range 2: [0x20000, 0x30000)
  const uintptr_t OPERATION_OUTPUT_BASE =
      3 * MAX_ADDRESSES_PER_RANGE;  // Range 3: [0x30000, 0x40000)

  // Assertions to ensure we don't exceed address range limits
  TORCH_CHECK(original_input_ptrs.size() < MAX_ADDRESSES_PER_RANGE, "Too many operation inputs (",
              original_input_ptrs.size(), ") - exceeds maximum of ", MAX_ADDRESSES_PER_RANGE - 1,
              " for operation: ", op_name);

  TORCH_CHECK(original_output_ptrs.size() < MAX_ADDRESSES_PER_RANGE, "Too many operation outputs (",
              original_output_ptrs.size(), ") - exceeds maximum of ", MAX_ADDRESSES_PER_RANGE - 1,
              " for operation: ", op_name);

  // Build mapping from original input index to prologue index
  std::map<size_t, size_t> input_to_prologue_idx;
  for (size_t i = 0; i < prologue_result.tasks.size(); ++i) {
    size_t input_idx = prologue_result.tasks[i].input_index;
    input_to_prologue_idx[input_idx] = i;
  }

  // With order conservation, prologue inputs/outputs have same count as original inputs
  // Step 1: Prologue inputs - dummy addresses [0x0, 0x1, 0x2, ...]
  std::vector<void*> dummy_prologue_input_addrs;
  for (size_t i = 0; i < original_input_ptrs.size(); ++i) {
    dummy_prologue_input_addrs.push_back(reinterpret_cast<void*>(PROLOGUE_INPUT_BASE + i));
  }

  // Step 2: Prologue outputs - dummy addresses [0x10000, 0x10001, 0x10002, ...]
  // With order conservation, prologue has same number of outputs as inputs
  std::vector<void*> dummy_prologue_output_addrs;
  for (size_t i = 0; i < original_input_ptrs.size(); ++i) {
    dummy_prologue_output_addrs.push_back(reinterpret_cast<void*>(PROLOGUE_OUTPUT_BASE + i));
  }

  // Step 3: Operation inputs - hybrid dummy addresses
  // Transformed inputs use prologue outputs [0x10000+idx]
  // Non-transformed inputs use direct range [0x20000+i]
  std::vector<void*> dummy_operation_input_addrs;
  for (size_t i = 0; i < original_input_ptrs.size(); ++i) {
    auto it = input_to_prologue_idx.find(i);
    if (it != input_to_prologue_idx.end()) {
      // Transformed: consume prologue output
      size_t prologue_idx = it->second;
      dummy_operation_input_addrs.push_back(
          reinterpret_cast<void*>(PROLOGUE_OUTPUT_BASE + prologue_idx));
    } else {
      // Non-transformed: use separate direct range
      dummy_operation_input_addrs.push_back(reinterpret_cast<void*>(DIRECT_INPUT_BASE + i));
    }
  }

  // Step 4: Operation outputs - use dummy addresses [0x30000, 0x30001, 0x30002, ...]
  std::vector<void*> dummy_operation_output_addrs;
  for (size_t i = 0; i < original_output_ptrs.size(); ++i) {
    dummy_operation_output_addrs.push_back(reinterpret_cast<void*>(OPERATION_OUTPUT_BASE + i));
  }

  // Perform merge using explicit dummy addresses with mapping capture
  torch_neuronx::MergeMapping final_merge_mapping;
  std::string merged_mlir = torch_neuronx::mergeStableHLOModulesSkipIntermediates(
      prologue_result.prologue_mlir_str, original_mlir_str, dummy_prologue_input_addrs,
      dummy_prologue_output_addrs, dummy_operation_input_addrs, dummy_operation_output_addrs,
      &final_merge_mapping);

  TORCH_CHECK(!merged_mlir.empty(),
              "Failed to merge prologue with operation using dummy addresses for operation '",
              op_name, "'");

  // With NONE transformations, every input has a transformation task, so input order is conserved.
  // Assert this invariant and use original pointers directly.

  // Assert: Input count must match
  TORCH_CHECK(final_merge_mapping.total_inputs == original_input_ptrs.size(),
              "Input count mismatch after merge for operation '", op_name,
              "': ", "expected=", original_input_ptrs.size(),
              ", got=", final_merge_mapping.total_inputs);

  // Assert: Output count must match
  TORCH_CHECK(final_merge_mapping.total_outputs == original_output_ptrs.size(),
              "Output count mismatch after merge for operation '", op_name,
              "': ", "expected=", original_output_ptrs.size(),
              ", got=", final_merge_mapping.total_outputs);

  // Assert: Input order is conserved via prologue outputs
  // With NONE transformations, all inputs flow through prologue in order
  for (size_t i = 0; i < original_input_ptrs.size(); ++i) {
    auto it = final_merge_mapping.input_mapping.find(i);
    TORCH_CHECK(
        it != final_merge_mapping.input_mapping.end() && it->second.first == 1 &&
            it->second.second == i,
        "Input order not conserved at index ", i, " for operation '", op_name, "'. ",
        "Expected mapping to prologue output ", i, " (module 1). ",
        "Got module=", (it != final_merge_mapping.input_mapping.end() ? it->second.first : -1),
        ", index=", (it != final_merge_mapping.input_mapping.end() ? it->second.second : -1));
  }

  TORCH_NEURONX_DEBUG("Merged prologue with operation", "op=", op_name);

  // Return merged MLIR bytes
  return std::vector<uint8_t>(merged_mlir.begin(), merged_mlir.end());
}

void LazyTransformationManager::ProcessOperationInputs(
    at::neuron::OperationContext* operation_context) {
  // Early exit: Only process if kernel is compilable and has input transformations
  auto* xla_compilable = dynamic_cast<at::neuron::XLACompilableKernelExecution*>(
      operation_context->kernel_execution.get());
  if (!xla_compilable || !xla_compilable->HasInputTransformations()) {
    // No transformations needed, skip processing
    return;
  }

  // Get op_name and transformations from the compilable kernel
  const std::string& op_name = operation_context->GetOpName();
  std::vector<std::vector<TensorTransformation>> kernel_transforms =
      xla_compilable->GetInputTransformations();

  // Get input pointers and contexts from operation context
  const auto& input_ptrs = xla_compilable->GetInputPtrs();
  const auto& input_contexts = operation_context->input_tensors_context;

  // Compute prologue cache key using pointers and contexts
  std::string prologue_cache_key = OperationPrologue::ComputePrologueCacheKey(
      input_ptrs, input_contexts, op_name, kernel_transforms);

  // Generate merged operation cache key (prologue + operation)
  std::string merged_operation_cache_key = prologue_cache_key + "_" + xla_compilable->GetCacheKey();

  // Check MergedOperationCache FIRST - avoid prologue generation if cached result exists
  {
    std::lock_guard<std::mutex> lock(g_merged_operation_cache_mutex);
    auto it = g_merged_operation_cache.find(merged_operation_cache_key);
    if (it != g_merged_operation_cache.end()) {
      // Cache HIT - use cached merged MLIR directly
      TORCH_NEURONX_DEBUG("MergedOperationCache HIT", "op=", op_name,
                          "key=", merged_operation_cache_key);

      // Apply cached MLIR bytes to kernel (updates in-place)
      ApplyCachedMlirToKernel(xla_compilable, it->second, merged_operation_cache_key);
      return;
    }
  }

  // Cache MISS - generate prologue and merge
  TORCH_NEURONX_DEBUG("MergedOperationCache MISS", "op=", op_name,
                      "key=", merged_operation_cache_key);

  // Generate prologue using pointers and contexts, passing the pre-computed cache key
  auto prologue_result = OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, op_name,
                                                             kernel_transforms, prologue_cache_key);

  // Check for errors
  if (!prologue_result.success) {
    std::string error_message = "Failed to generate prologue: " + prologue_result.error_message;
    TORCH_NEURONX_ERROR("Prologue generation failed", "op=", op_name, "error=", error_message);
    throw std::runtime_error(error_message);
  }

  // Merge prologue with operation (throws on failure)
  auto merged_mlir_bytes = ProcessPrologueWithMlirMerge(prologue_result, op_name, xla_compilable);

  // Apply merged MLIR bytes to kernel
  ApplyCachedMlirToKernel(xla_compilable, merged_mlir_bytes, merged_operation_cache_key);

  // Cache the result for future use
  std::lock_guard<std::mutex> lock(g_merged_operation_cache_mutex);
  g_merged_operation_cache[merged_operation_cache_key] = merged_mlir_bytes;
  TORCH_NEURONX_DEBUG("Cached merged result", "op=", op_name);
}

void LazyTransformationManager::ClearMergedOperationCache() {
  std::lock_guard<std::mutex> lock(g_merged_operation_cache_mutex);
  g_merged_operation_cache.clear();
  TORCH_NEURONX_DEBUG("Cleared MergedOperationCache");
}

size_t LazyTransformationManager::GetMergedOperationCacheSize() {
  std::lock_guard<std::mutex> lock(g_merged_operation_cache_mutex);
  return g_merged_operation_cache.size();
}

}  // namespace lazy
}  // namespace c10_neuron

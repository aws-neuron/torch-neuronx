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
struct MergedOperationCacheEntry {
  std::vector<uint8_t> mlir_bytes;
  std::vector<size_t> input_dedup_map;  // empty if no dedup
  size_t deduped_input_count = 0;
  std::vector<std::vector<int64_t>> deduped_input_shapes;  // shape per deduped slot
};
static std::unordered_map<std::string, MergedOperationCacheEntry> g_merged_operation_cache;
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

// Helper to apply input deduplication to kernel src data
void ApplyInputDedup(at::neuron::XLACompilableKernelExecution* kernel,
                     const std::vector<size_t>& dedup_map, size_t deduped_count,
                     const std::vector<std::vector<int64_t>>& deduped_shapes,
                     const std::string& op_name) {
  if (deduped_count == 0 || deduped_count >= dedup_map.size()) return;

  const auto& orig_src_refs = kernel->GetSrcRefs();
  const auto& orig_src_ctxs = kernel->GetSrcTensorContexts();

  std::vector<at::neuron::TensorDataRef> deduped_refs(deduped_count);
  std::vector<at::neuron::TensorContext> deduped_ctxs(deduped_count);
  std::vector<bool> slot_filled(deduped_count, false);

  for (size_t i = 0; i < dedup_map.size(); ++i) {
    size_t slot = dedup_map[i];
    if (!slot_filled[slot]) {
      deduped_refs[slot] = at::neuron::TensorDataRef(orig_src_refs[i]);
      deduped_ctxs[slot] = orig_src_ctxs[i];
      deduped_ctxs[slot].storage_offset = 0;
      const auto& shape = deduped_shapes[slot];
      size_t full_numel = 1;
      for (auto d : shape) full_numel *= d;
      deduped_ctxs[slot].size_bytes = full_numel * deduped_ctxs[slot].element_size;
      deduped_ctxs[slot].numel = full_numel;
      deduped_ctxs[slot].shape = shape;
      slot_filled[slot] = true;
    }
  }

  kernel->UpdateSrcData(std::move(deduped_refs), deduped_ctxs);
  TORCH_NEURONX_DEBUG("Applied input dedup", "op=", op_name, "original=", dedup_map.size(),
                      "deduped=", deduped_count);
}

}  // anonymous namespace

std::vector<uint8_t> LazyTransformationManager::ProcessPrologueWithMlirMerge(
    const PrologueResult& prologue_result, const std::string& op_name,
    at::neuron::XLACompilableKernelExecution* compilable_kernel) {
  // Get original operation's MLIR and pointer addresses
  const auto& original_hlo_bytes = compilable_kernel->GetHloBytes();
  std::string original_mlir_str(original_hlo_bytes.begin(), original_hlo_bytes.end());
  const auto& original_input_ptrs = compilable_kernel->GetSrcPtrs();
  const auto& original_output_ptrs = compilable_kernel->GetDstPtrs();

  // Generate cache key for the merged result (for kernel naming)
  std::string merged_operation_cache_key =
      prologue_result.prologue_cache_key + "_" + compilable_kernel->GetCacheKey();

  TORCH_NEURONX_DEBUG("ProcessPrologueWithMlirMerge: Merging prologue with operation",
                      "op=", op_name);

  // Create slot vectors - each element's address serves as a unique identifier for merge API
  // Account for input deduplication from prologue
  size_t num_inputs = original_input_ptrs.size();
  size_t num_outputs = original_output_ptrs.size();
  bool has_dedup =
      prologue_result.deduped_input_count > 0 && prologue_result.deduped_input_count < num_inputs;
  size_t num_prologue_inputs = has_dedup ? prologue_result.deduped_input_count : num_inputs;

  std::vector<size_t> prologue_input_slots(num_prologue_inputs);
  std::vector<size_t> prologue_output_slots(num_inputs);
  std::vector<size_t> operation_input_slots(num_inputs);
  std::vector<size_t> operation_output_slots(num_outputs);

  // Build address vectors from slot objects
  std::vector<void*> prologue_input_addrs, prologue_output_addrs;
  std::vector<void*> operation_input_addrs, operation_output_addrs;

  for (size_t i = 0; i < num_prologue_inputs; ++i) {
    prologue_input_addrs.push_back(&prologue_input_slots[i]);
  }
  for (size_t i = 0; i < num_inputs; ++i) {
    prologue_output_addrs.push_back(&prologue_output_slots[i]);
    // Operation inputs connect to prologue outputs (same slot = data flows through)
    operation_input_addrs.push_back(&prologue_output_slots[i]);
  }
  for (size_t i = 0; i < num_outputs; ++i) {
    operation_output_addrs.push_back(&operation_output_slots[i]);
  }

  // Perform merge using slot addresses
  torch_neuronx::MergeMapping final_merge_mapping;
  std::string merged_mlir = torch_neuronx::mergeStableHLOModulesSkipIntermediates(
      prologue_result.prologue_mlir_str, original_mlir_str, prologue_input_addrs,
      prologue_output_addrs, operation_input_addrs, operation_output_addrs, &final_merge_mapping);

  TORCH_CHECK(!merged_mlir.empty(), "Failed to merge prologue with operation for '", op_name, "'");

  // With NONE transformations, every input has a transformation task, so input order is conserved.
  // Assert this invariant and use original pointers directly.

  // Assert: Input count must match (accounting for deduplication)
  TORCH_CHECK(final_merge_mapping.total_inputs == num_prologue_inputs,
              "Input count mismatch after merge for operation '", op_name,
              "': ", "expected=", num_prologue_inputs, ", got=", final_merge_mapping.total_inputs);

  // Assert: Output count must match
  TORCH_CHECK(final_merge_mapping.total_outputs == original_output_ptrs.size(),
              "Output count mismatch after merge for operation '", op_name,
              "': ", "expected=", original_output_ptrs.size(),
              ", got=", final_merge_mapping.total_outputs);

  // Assert: Input order is conserved via prologue outputs
  // With NONE transformations, all inputs flow through prologue in order
  // Skip strict mapping check when dedup is active (mapping changes)
  if (!has_dedup) {
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
  }

  TORCH_NEURONX_DEBUG("Merged prologue with operation", "op=", op_name);

  // Return merged MLIR bytes
  return std::vector<uint8_t>(merged_mlir.begin(), merged_mlir.end());
}

void LazyTransformationManager::ProcessOperationInputs(
    at::neuron::OperationContext* operation_context) {
  // Validate operation_context is not null
  TORCH_CHECK(operation_context != nullptr,
              "[ProcessOperationInputs] Received null operation_context");

  // Validate kernel_execution before dynamic_cast to catch use-after-free
  auto* kernel_ptr = operation_context->kernel_execution.get();
  TORCH_CHECK(kernel_ptr != nullptr,
              "[ProcessOperationInputs] Operation has null kernel_execution. ",
              "This indicates memory corruption or use-after-free. ",
              "stream=", static_cast<void*>(operation_context->stream),
              " pytorch_seq_nr=", operation_context->pytorch_sequence_nr);

  // Early exit: Only process if kernel is compilable and has input transformations
  auto* xla_compilable = dynamic_cast<at::neuron::XLACompilableKernelExecution*>(kernel_ptr);
  if (!xla_compilable || !xla_compilable->HasInputTransformations()) {
    // No transformations needed, skip processing
    return;
  }

  // Get op_name and transformations from the compilable kernel
  const std::string& op_name = operation_context->GetOpName();
  std::vector<std::vector<TensorTransformation>> kernel_transforms =
      xla_compilable->GetInputTransformations();

  // Get input pointers and contexts from kernel
  const auto& input_ptrs = xla_compilable->GetSrcPtrs();
  const auto& input_contexts = xla_compilable->GetInputContexts();

  // Compute prologue cache key using pointers and contexts
  std::string prologue_cache_key = OperationPrologue::ComputePrologueCacheKey(
      input_ptrs, input_contexts, op_name, kernel_transforms);

  // Compute input deduplication by comparing storage base pointers.
  // This must happen before cache lookup so the dedup signature is part of the key.
  std::unordered_map<uintptr_t, size_t> base_to_deduped_idx;
  std::vector<size_t> input_dedup_map(input_ptrs.size());
  size_t deduped_input_count = 0;

  for (size_t i = 0; i < input_ptrs.size(); ++i) {
    uintptr_t base_ptr =
        reinterpret_cast<uintptr_t>(input_ptrs[i]) -
        (input_contexts[i].get_storage_offset() * input_contexts[i].get_element_size());
    auto it = base_to_deduped_idx.find(base_ptr);
    if (it != base_to_deduped_idx.end()) {
      input_dedup_map[i] = it->second;
    } else {
      base_to_deduped_idx[base_ptr] = deduped_input_count;
      input_dedup_map[i] = deduped_input_count;
      deduped_input_count++;
    }
  }

  // Include dedup signature in cache key so different storage layouts get different entries
  std::string dedup_suffix;
  if (deduped_input_count < input_ptrs.size()) {
    std::stringstream ss;
    ss << "_dedup";
    for (size_t slot : input_dedup_map) {
      ss << "_" << slot;
    }
    dedup_suffix = ss.str();
  }

  // Generate merged operation cache key (prologue + dedup + operation)
  std::string merged_operation_cache_key =
      prologue_cache_key + dedup_suffix + "_" + xla_compilable->GetCacheKey();

  // Check MergedOperationCache FIRST - avoid prologue generation if cached result exists
  {
    std::lock_guard<std::mutex> lock(g_merged_operation_cache_mutex);
    auto it = g_merged_operation_cache.find(merged_operation_cache_key);
    if (it != g_merged_operation_cache.end()) {
      // Cache HIT - use cached merged MLIR directly
      TORCH_NEURONX_DEBUG("MergedOperationCache HIT", "op=", op_name,
                          "key=", merged_operation_cache_key);

      // Apply cached MLIR bytes to kernel (updates in-place)
      ApplyCachedMlirToKernel(xla_compilable, it->second.mlir_bytes, merged_operation_cache_key);
      // Apply input deduplication if cached entry has it
      ApplyInputDedup(xla_compilable, it->second.input_dedup_map, it->second.deduped_input_count,
                      it->second.deduped_input_shapes, op_name);
      return;
    }
  }

  // Cache MISS - generate prologue and merge
  TORCH_NEURONX_DEBUG("MergedOperationCache MISS", "op=", op_name,
                      "key=", merged_operation_cache_key);

  // Generate prologue using pointers and contexts, passing the pre-computed cache key and dedup
  auto prologue_result =
      OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, op_name, kernel_transforms,
                                          prologue_cache_key, input_dedup_map, deduped_input_count);

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

  // Build deduped input shapes for cache entry
  std::vector<std::vector<int64_t>> deduped_shapes;
  if (prologue_result.deduped_input_count > 0 &&
      prologue_result.deduped_input_count < input_ptrs.size()) {
    deduped_shapes.resize(prologue_result.deduped_input_count);
    std::vector<bool> filled(prologue_result.deduped_input_count, false);
    for (size_t i = 0; i < input_ptrs.size(); ++i) {
      size_t slot = prologue_result.input_dedup_map[i];
      if (!filled[slot]) {
        deduped_shapes[slot] = prologue_result.tasks[i].input_shape;
        filled[slot] = true;
      }
    }
    // Apply dedup to kernel src data
    ApplyInputDedup(xla_compilable, prologue_result.input_dedup_map,
                    prologue_result.deduped_input_count, deduped_shapes, op_name);
  }

  // Cache the result for future use
  std::lock_guard<std::mutex> lock(g_merged_operation_cache_mutex);
  g_merged_operation_cache[merged_operation_cache_key] =
      MergedOperationCacheEntry{merged_mlir_bytes, prologue_result.input_dedup_map,
                                prologue_result.deduped_input_count, deduped_shapes};
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

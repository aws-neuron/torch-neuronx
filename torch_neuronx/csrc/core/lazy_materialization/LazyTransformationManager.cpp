#include "LazyTransformationManager.h"

#include <chrono>
#include <mutex>
#include <numeric>
#include <set>
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
  // Epilogue extra-src wiring (dst→src): the merged epilogue MLIR has ONE extra
  // input (the flat base storage) per SCATTER_IOTA output. On a cache hit we must
  // re-add one extra src (the output buffer) per entry, in this same order, so
  // the kernel's src count matches the merged program's argument count.
  std::vector<size_t> scatter_iota_output_indices;
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
  kernel->UpdateCompileKey(merged_operation_cache_key);
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

  // The context offset is the prologue base offset. A base at offset 0 slices views from the
  // storage start, so keep offset 0 and cover the furthest view end. An anchored base (offset != 0)
  // reads from the view offset, so keep the offset and cover only the view.
  struct SlotAccum {
    size_t max_end_bytes = 0;
    size_t max_numel = 0;
    size_t offset = 0;
    bool offset_seen = false;
    bool offset_uniform = true;
  };
  std::vector<SlotAccum> slots(deduped_count);
  for (size_t i = 0; i < dedup_map.size(); ++i) {
    const auto& ctx = orig_src_ctxs[i];
    auto& acc = slots[dedup_map[i]];
    acc.max_end_bytes =
        std::max(acc.max_end_bytes, (ctx.storage_offset + ctx.numel) * ctx.element_size);
    acc.max_numel = std::max(acc.max_numel, ctx.numel);
    if (!acc.offset_seen) {
      acc.offset = ctx.storage_offset;
      acc.offset_seen = true;
    } else if (acc.offset != ctx.storage_offset) {
      acc.offset_uniform = false;
    }
  }

  for (size_t i = 0; i < dedup_map.size(); ++i) {
    size_t slot = dedup_map[i];
    if (!slot_filled[slot]) {
      deduped_refs[slot] = at::neuron::TensorDataRef(orig_src_refs[i]);
      deduped_ctxs[slot] = orig_src_ctxs[i];
      const auto& shape = deduped_shapes[slot];
      size_t full_numel = 1;
      for (auto d : shape) full_numel *= d;
      deduped_ctxs[slot].numel = full_numel;
      deduped_ctxs[slot].shape = shape;
      const auto& acc = slots[slot];
      // A deduped slot maps to ONE buffer with ONE base offset. Every view in the slot must
      // agree on that offset. ComputeInputDedupMap keys on (storage base, offset, shape), so
      // views at different offsets already fall in separate slots and this holds by
      // construction. Check it, so a later change to the dedup key cannot silently give an
      // anchored chain a base at offset 0 and read the wrong data. See PR #2100 review.
      TORCH_CHECK(acc.offset_uniform, "op '", op_name,
                  "' shares one input slot for views with different storage offsets. ",
                  "One slot gives one base offset. Keep these views in separate slots.");
      bool anchored = acc.offset != 0;
      if (anchored) {
        deduped_ctxs[slot].size_bytes = acc.max_numel * orig_src_ctxs[i].element_size;
      } else {
        deduped_ctxs[slot].storage_offset = 0;
        deduped_ctxs[slot].size_bytes = acc.max_end_bytes;
      }
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
      prologue_result.prologue_cache_key + "_" + compilable_kernel->GetCompileKey();

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
  // TODO(mengchiy): Remove usage of dynamic_cast
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

  // Compute input deduplication: inputs share a DMA slot only when they have the
  // same storage base pointer AND the same shape.
  // This must happen before cache lookup so the dedup signature is part of the key.
  auto [input_dedup_map, deduped_input_count] = ComputeInputDedupMap(input_ptrs, input_contexts);

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
      prologue_cache_key + dedup_suffix + "_" + xla_compilable->GetCompileKey();

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

// ============================================================================
// ProcessOperationOutputs — Epilogue integration
// ============================================================================

void LazyTransformationManager::ProcessOperationOutputs(
    at::neuron::OperationContext* operation_context) {
  TORCH_CHECK(operation_context != nullptr,
              "[ProcessOperationOutputs] Received null operation_context");

  auto* kernel_ptr = operation_context->kernel_execution.get();
  TORCH_CHECK(kernel_ptr != nullptr,
              "[ProcessOperationOutputs] Operation has null kernel_execution.");

  // Early exit: Only process if kernel is XLA compilable and has output transformations
  // TODO(mengchiy): Remove usage of dynamic_cast
  auto* xla_compilable = dynamic_cast<at::neuron::XLACompilableKernelExecution*>(kernel_ptr);
  if (!xla_compilable || !xla_compilable->HasOutputTransformations()) {
    return;
  }

  const std::string& op_name = operation_context->GetOpName();
  std::vector<std::vector<TensorTransformation>> kernel_transforms =
      xla_compilable->GetOutputTransformations();

  const auto& output_ptrs = xla_compilable->GetDstPtrs();
  const auto& output_contexts = xla_compilable->GetOutputContexts();

  // Compute epilogue cache key
  std::string epilogue_cache_key = OperationEpilogue::ComputeEpilogueCacheKey(
      output_ptrs, output_contexts, op_name, kernel_transforms);

  // Generate epilogue cache key for merged result
  std::string merged_epilogue_cache_key =
      epilogue_cache_key + "_" + xla_compilable->GetCompileKey();

  // Check MergedOperationCache for cached epilogue result
  {
    std::lock_guard<std::mutex> lock(g_merged_operation_cache_mutex);
    auto it = g_merged_operation_cache.find(merged_epilogue_cache_key);
    if (it != g_merged_operation_cache.end()) {
      TORCH_NEURONX_DEBUG("MergedOperationCache HIT (epilogue)", "op=", op_name);
      ApplyCachedMlirToKernel(xla_compilable, it->second.mlir_bytes, merged_epilogue_cache_key);

      // Re-wire extra src tensors for SCATTER_IOTA on cache hit.
      // The merged MLIR expects num_outputs + K inputs, where K is the number of
      // SCATTER_IOTA outputs (each adds an extra base/flat-storage input). The
      // kernel only has its original src tensors, so we must re-add one extra src
      // (the output buffer, which IS the flat storage) per scatter output — in the
      // SAME order the merged program's arguments were built (ascending output
      // index, matching scatter_iota_output_indices).
      if (!it->second.scatter_iota_output_indices.empty()) {
        const auto& dst_ptrs = xla_compilable->GetDstPtrs();
        const auto& dst_ctxs = xla_compilable->GetOutputContexts();
        const auto& dst_refs = xla_compilable->GetDstRefs();
        auto src_refs = xla_compilable->GetSrcRefs();
        auto src_ctxs = xla_compilable->GetSrcTensorContexts();
        for (size_t out_idx : it->second.scatter_iota_output_indices) {
          if (out_idx < dst_ptrs.size() && out_idx < dst_ctxs.size() && out_idx < dst_refs.size()) {
            src_refs.push_back(dst_refs[out_idx]);
            src_ctxs.push_back(dst_ctxs[out_idx]);
            TORCH_NEURONX_DEBUG("Epilogue cache HIT: re-wired extra src for SCATTER_IOTA",
                                "op=", op_name, "output_idx=", out_idx);
          }
        }
        xla_compilable->UpdateSrcData(std::move(src_refs), src_ctxs);
      }
      return;
    }
  }

  TORCH_NEURONX_DEBUG("MergedOperationCache MISS (epilogue)", "op=", op_name);

  // Generate epilogue
  auto epilogue_result = OperationEpilogue::GenerateEpilogue(output_ptrs, output_contexts, op_name,
                                                             kernel_transforms, epilogue_cache_key);

  if (!epilogue_result.success) {
    TORCH_NEURONX_WARN(
        "[ProcessOperationOutputs] Epilogue generation failed, falling back to "
        "CopyBackToOriginal. op=",
        op_name, " error=", epilogue_result.error_message);
    xla_compilable->SetOutputTransformations({});
    return;
  }

  if (!epilogue_result.has_transformations) {
    return;
  }

  // Merge epilogue with operation (appended after)
  auto merged_mlir_bytes = ProcessEpilogueWithMlirMerge(epilogue_result, op_name, xla_compilable);

  ApplyCachedMlirToKernel(xla_compilable, merged_mlir_bytes, merged_epilogue_cache_key);

  // Cache the result — record the SCATTER_IOTA output indices so a cache HIT can
  // re-wire one extra src (the output buffer / flat storage) per scatter output,
  // in the same order the merged program's extra arguments were built. Multiple
  // scatter outputs per op are fully supported (e.g. multi-output ops such as
  // max.dim / min.dim / _native_batch_norm_legit with a strided out= tensor).
  MergedOperationCacheEntry entry;
  entry.mlir_bytes = merged_mlir_bytes;
  entry.scatter_iota_output_indices = epilogue_result.scatter_iota_output_indices;

  std::lock_guard<std::mutex> lock(g_merged_operation_cache_mutex);
  g_merged_operation_cache[merged_epilogue_cache_key] = std::move(entry);
  TORCH_NEURONX_DEBUG("Cached merged epilogue result", "op=", op_name,
                      "num_scatter_outputs=", epilogue_result.scatter_iota_output_indices.size());
}

std::vector<uint8_t> LazyTransformationManager::ProcessEpilogueWithMlirMerge(
    const EpilogueResult& epilogue_result, const std::string& op_name,
    at::neuron::XLACompilableKernelExecution* compilable_kernel) {
  const auto& original_hlo_bytes = compilable_kernel->GetHloBytes();
  std::string original_mlir_str(original_hlo_bytes.begin(), original_hlo_bytes.end());
  const auto& original_output_ptrs = compilable_kernel->GetDstPtrs();

  size_t num_inputs = compilable_kernel->GetSrcPtrs().size();
  size_t num_outputs = original_output_ptrs.size();

  std::vector<size_t> operation_input_slots(num_inputs);
  std::vector<size_t> operation_output_slots(num_outputs);
  std::vector<size_t> epilogue_input_slots(num_outputs);
  std::vector<size_t> epilogue_output_slots(num_outputs);
  // One extra base-storage slot per output (only the scatter outputs' slots are
  // actually wired). Each must be a DISTINCT address so the merge creates a
  // separate function argument.
  const auto& scatter_indices = epilogue_result.scatter_iota_output_indices;
  std::set<size_t> scatter_set(scatter_indices.begin(), scatter_indices.end());
  std::vector<size_t> epilogue_base_slots(num_outputs);

  std::vector<void*> operation_input_addrs, operation_output_addrs;
  std::vector<void*> epilogue_input_addrs, epilogue_output_addrs;

  for (size_t i = 0; i < num_inputs; ++i) {
    operation_input_addrs.push_back(&operation_input_slots[i]);
  }
  for (size_t i = 0; i < num_outputs; ++i) {
    operation_output_addrs.push_back(&operation_output_slots[i]);
    epilogue_output_addrs.push_back(&epilogue_output_slots[i]);
  }

  // Build the epilogue module's input-argument signature in EXACTLY the order
  // GenerateEpilogue produced it: per output, the primary (update) input, then —
  // if that output has a SCATTER_IOTA — its extra base-storage input, immediately
  // after. This interleaved order (primary_0, base_0?, primary_1, base_1?, ...)
  // must match the merged epilogue MLIR's argument order, otherwise the operation
  // outputs and the scatter base storage get wired to the wrong arguments.
  for (size_t i = 0; i < num_outputs; ++i) {
    // Epilogue primary input connects to the operation output of the same index.
    epilogue_input_addrs.push_back(&operation_output_slots[i]);
    if (scatter_set.count(i)) {
      epilogue_input_addrs.push_back(&epilogue_base_slots[i]);
    }
  }
  if (!scatter_indices.empty()) {
    TORCH_NEURONX_DEBUG("ProcessEpilogueWithMlirMerge: adding extra base tensor inputs",
                        "op=", op_name, "num_scatter_outputs=", scatter_indices.size());
  }

  // Merge: operation + epilogue
  torch_neuronx::MergeMapping final_merge_mapping;
  std::string merged_mlir = torch_neuronx::mergeStableHLOModulesSkipIntermediates(
      original_mlir_str, epilogue_result.epilogue_mlir_str, operation_input_addrs,
      operation_output_addrs, epilogue_input_addrs, epilogue_output_addrs, &final_merge_mapping);

  TORCH_CHECK(!merged_mlir.empty(), "Failed to merge operation with epilogue for '", op_name, "'");

  // For each SCATTER_IOTA output the merged program has an extra input (the flat
  // storage). The output buffer IS that flat storage, so add one extra src per
  // scatter output, in ascending output-index order (the same order the extra
  // inputs were appended above and in GenerateEpilogue's per-output merge).
  if (!scatter_indices.empty()) {
    const auto& dst_ptrs = compilable_kernel->GetDstPtrs();
    const auto& dst_ctxs = compilable_kernel->GetOutputContexts();
    const auto& dst_refs = compilable_kernel->GetDstRefs();

    auto src_refs = compilable_kernel->GetSrcRefs();
    auto src_ctxs = compilable_kernel->GetSrcTensorContexts();

    for (size_t out_idx : scatter_indices) {
      if (out_idx < dst_ptrs.size() && out_idx < dst_ctxs.size() && out_idx < dst_refs.size()) {
        src_refs.push_back(dst_refs[out_idx]);
        src_ctxs.push_back(dst_ctxs[out_idx]);
        TORCH_NEURONX_DEBUG("Added flat storage as extra src input (SCATTER_IOTA)", "op=", op_name,
                            "output_idx=", out_idx);
      }
    }

    compilable_kernel->UpdateSrcData(std::move(src_refs), src_ctxs);
    TORCH_NEURONX_DEBUG(
        "After SCATTER_IOTA UpdateSrcData: num_src=", compilable_kernel->GetSrcPtrs().size(),
        " num_dst=", compilable_kernel->GetDstPtrs().size());
  }

  TORCH_NEURONX_DEBUG("Merged operation with epilogue", "op=", op_name,
                      "num_scatter_outputs=", scatter_indices.size());

  return std::vector<uint8_t>(merged_mlir.begin(), merged_mlir.end());
}

}  // namespace lazy
}  // namespace c10_neuron

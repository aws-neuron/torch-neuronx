#include "OperationTransformations.h"

#include <chrono>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_map>

#include "MlirGenerators.h"
#include "TransformationRegistry.h"
#include "TypeUtils.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/concatenation/OpConcatUtils.h"
#include "torch_neuronx/csrc/core/opbuilder/utility/StableHloUtils.h"
#include "torch_neuronx/csrc/core/utils/TensorContext.h"

namespace c10_neuron {
namespace lazy {

namespace {

// Cache for merged prologue MLIR strings
// Since order is always conserved (identity mapping), we only need to store the MLIR
static std::unordered_map<std::string, std::string> g_prologue_cache;
static std::mutex g_prologue_cache_mutex;

/**
 * @brief Convert TransformationType enum to human-readable string for cache keys
 */
std::string TransformationTypeToString(TransformationType type) {
  switch (type) {
    case TransformationType::TRANSPOSE:
      return "TRANSPOSE";
    case TransformationType::RESHAPE:
      return "RESHAPE";
    case TransformationType::BROADCAST:
      return "BROADCAST";
    case TransformationType::SLICE:
      return "SLICE";
    case TransformationType::SCATTER_IOTA:
      return "SCATTER_IOTA";
    case TransformationType::NONE:
      return "NONE";
    default:
      return "UNKNOWN";
  }
}

}  // anonymous namespace

// ============================================================================
// Transformation Utilities Implementation
// ============================================================================

namespace transformation_utils {

std::string GenerateMlir(const TransformationTask& task) {
  switch (task.type) {
    case TransformationType::TRANSPOSE:
      if (auto* params = std::get_if<TransposeParams>(&task.params)) {
        return mlir_generators::GenerateTranspose(params->current_perm, params->target_perm,
                                                  task.input_shape, task.element_type);
      }
      throw std::runtime_error("Invalid TransposeParams in TransformationTask");

    case TransformationType::RESHAPE:
      return mlir_generators::GenerateReshape(task.input_shape, task.output_shape,
                                              task.element_type);

    case TransformationType::SLICE:
      if (auto* params = std::get_if<SliceParams>(&task.params)) {
        // Use strided overload if any stride > 1, otherwise use simple overload
        bool has_non_unit_stride = std::any_of(params->strides.begin(), params->strides.end(),
                                               [](int64_t s) { return s > 1; });
        if (has_non_unit_stride) {
          return mlir_generators::GenerateSlice(task.input_shape, params->start_indices,
                                                params->end_indices, params->strides,
                                                task.element_type);
        }
        return mlir_generators::GenerateSlice(task.input_shape, params->start_indices,
                                              params->end_indices, task.element_type);
      }
      throw std::runtime_error("Invalid SliceParams in TransformationTask");

    case TransformationType::BROADCAST:
      if (auto* params = std::get_if<BroadcastParams>(&task.params)) {
        return mlir_generators::GenerateBroadcast(task.input_shape, task.output_shape,
                                                  params->broadcast_dims, task.element_type);
      }
      throw std::runtime_error("Invalid BroadcastParams in TransformationTask");

    case TransformationType::SCATTER_IOTA:
      if (auto* params = std::get_if<ScatterIotaParams>(&task.params)) {
        return mlir_generators::GenerateScatterIota(
            task.input_shape,        // view shape (update)
            params->view_strides,    // strides of destination view
            params->storage_offset,  // storage offset
            params->storage_span,    // total flat storage size
            task.element_type);
      }
      throw std::runtime_error("Invalid ScatterIotaParams in TransformationTask");

    case TransformationType::NONE:
      // Generate empty/identity MLIR for inputs with no transformations
      return mlir_generators::GenerateEmpty(task.input_shape, task.element_type);

    default:
      throw std::runtime_error("Unknown transformation type in GenerateMlir");
  }
}

}  // namespace transformation_utils

// ============================================================================
// OperationPrologue Implementation
// ============================================================================

PrologueResult OperationPrologue::GeneratePrologue(
    const std::vector<void*>& input_ptrs,
    const std::vector<at::neuron::TensorContext>& input_contexts, const std::string& op_name,
    const std::vector<std::vector<TensorTransformation>>& kernel_transforms,
    const std::string& prologue_cache_key, const std::vector<size_t>& input_dedup_map,
    size_t deduped_input_count) {
  // TODO: Add counter for prologue generation latency here

  // Step 1: Collect all transformation tasks from kernel transforms
  auto tasks = CollectTransformationTasks(input_ptrs, input_contexts, op_name, kernel_transforms);

  // Step 2: If no transformations, return early with empty cache key
  if (tasks.empty()) {
    PrologueResult result;
    result.success = true;
    result.has_transformations = false;
    result.prologue_cache_key = "";  // Empty cache key for no transformations
    return result;
  }

  TORCH_NEURONX_DEBUG("OperationPrologue: Collected transformation tasks", "op=", op_name,
                      "num_tasks=", tasks.size());

  // Step 3: Merge transformations into prologue using pre-computed cache key and dedup info
  auto result = MergeTransformations(tasks, op_name, input_ptrs.size(), prologue_cache_key,
                                     input_dedup_map, deduped_input_count);

  TORCH_NEURONX_DEBUG("OperationPrologue: Generated prologue", "op=", op_name,
                      "has_transformations=", result.has_transformations);

  return result;
}

std::string OperationPrologue::ComputePrologueCacheKey(
    const std::vector<void*>& input_ptrs,
    const std::vector<at::neuron::TensorContext>& input_contexts, const std::string& op_name,
    const std::vector<std::vector<TensorTransformation>>& kernel_transforms) {
  // Build cache key by directly iterating transformations
  // IMPORTANT: Must include shape/dtype for NONE transformations (inputs without kernel transforms)
  // to avoid cache collisions between different tensor configurations
  std::stringstream cache_key_ss;
  cache_key_ss << "prologue--";

  // Process each input
  for (size_t input_idx = 0; input_idx < input_ptrs.size(); ++input_idx) {
    // Check if this input has kernel transformations
    if (input_idx >= kernel_transforms.size() || kernel_transforms[input_idx].empty()) {
      // No kernel transformations - NONE will be created using input_context
      // Include shape and dtype in cache key
      const auto& input_ctx = input_contexts[input_idx];
      std::vector<int64_t> shape = input_ctx.get_shape();
      std::string element_type = ScalarTypeToElementTypeString(input_ctx.get_dtype());

      cache_key_ss << "i" << input_idx << "_";
      for (auto dim : shape) {
        cache_key_ss << dim << "x";
      }
      cache_key_ss << "_" << element_type << "_";
      continue;
    }

    // Process transformations for this input
    const auto& transforms = kernel_transforms[input_idx];
    for (const auto& transform : transforms) {
      cache_key_ss << "i" << input_idx << "_";
      // Add transformation type as human-readable string
      cache_key_ss << TransformationTypeToString(transform.type) << "_";

      // Add input shape
      for (auto dim : transform.input_shape) {
        cache_key_ss << dim << "x";
      }
      cache_key_ss << "_to_";

      // Add output shape
      for (auto dim : transform.output_shape) {
        cache_key_ss << dim << "x";
      }

      // Add type-specific parameters
      if (!transform.params.empty()) {
        cache_key_ss << "_params_";
        for (auto p : transform.params) {
          cache_key_ss << p << ",";
        }
      }

      // Add element type
      cache_key_ss << "_dtype_" << transform.element_type << "_";
    }
  }

  // Include input deduplication signature in the cache key.
  //
  // MergeTransformations (below) builds `input_slots(deduped_input_count)`
  // and wires per-input transformation chains into slots via
  // `input_dedup_map`. The resulting MLIR's main-function arity therefore
  // depends on the dedup state, not on the transformation shapes alone.
  // Two calls with identical transforms + dtypes + op_name but different
  // dedup state (e.g. identity-op self-comparison vs compute-op
  // self-comparison through an `aten::eq.Tensor_out` with two transposed
  // views) must NOT share a cached prologue — otherwise the cached MLIR
  // is returned with the wrong arity and
  // validateAndGetMainFunctions (OpConcatUtils.cpp) throws:
  //   "Module1 input address count (N) does not match function argument count (M)".
  //
  // Dedup is computed from storage base pointers, matching the
  // computation in LazyTransformationManager::ProcessOperationWithPrologue
  // so the key we emit here agrees with the `deduped_input_count` /
  // `input_dedup_map` passed to MergeTransformations.
  //
  // The suffix is only appended when dedup actually collapses inputs,
  // keeping the common no-dedup key exactly as before.
  {
    std::unordered_map<uintptr_t, size_t> base_to_slot;
    std::vector<size_t> dedup_map(input_ptrs.size());
    size_t deduped_count = 0;
    for (size_t i = 0; i < input_ptrs.size(); ++i) {
      uintptr_t base =
          reinterpret_cast<uintptr_t>(input_ptrs[i]) -
          (input_contexts[i].get_storage_offset() * input_contexts[i].get_element_size());
      auto it = base_to_slot.find(base);
      if (it != base_to_slot.end()) {
        dedup_map[i] = it->second;
      } else {
        base_to_slot[base] = deduped_count;
        dedup_map[i] = deduped_count;
        deduped_count++;
      }
    }
    if (deduped_count < input_ptrs.size()) {
      cache_key_ss << "dedup";
      for (size_t slot : dedup_map) {
        cache_key_ss << "_" << slot;
      }
      cache_key_ss << "_";
    }
  }

  cache_key_ss << "op_" << op_name;
  return cache_key_ss.str();
}

std::vector<TransformationTask> OperationPrologue::CollectTransformationTasks(
    const std::vector<void*>& input_ptrs,
    const std::vector<at::neuron::TensorContext>& input_contexts, const std::string& op_name,
    const std::vector<std::vector<TensorTransformation>>& kernel_transforms) {
  std::vector<TransformationTask> tasks;

  for (size_t i = 0; i < input_ptrs.size(); ++i) {
    const auto& input_ctx = input_contexts[i];

    // Check if this input has kernel transformations
    if (i >= kernel_transforms.size() || kernel_transforms[i].empty()) {
      // No transformations for this input - create a NONE/identity transformation task
      // using shape and dtype from input context
      std::vector<int64_t> shape = input_ctx.get_shape();
      std::string element_type = ScalarTypeToElementTypeString(input_ctx.get_dtype());

      TransformationTask none_task;
      none_task.input_index = i;
      none_task.input_shape = shape;
      none_task.output_shape = shape;
      none_task.element_type = element_type;
      none_task.type = TransformationType::NONE;
      none_task.params = std::monostate{};  // No params for NONE

      tasks.push_back(none_task);

      TORCH_NEURONX_DEBUG("Created NONE transformation task for input with no transformations",
                          "op=", op_name, "input_index=", i);
      continue;
    }

    const auto& transformations = kernel_transforms[i];

    TORCH_NEURONX_DEBUG("Processing kernel transformations", "op=", op_name, "input_index=", i,
                        "num_transforms=", transformations.size());

    // Process transformations in groups
    {
      // Initialize TransformationState for this input
      // Use the FIRST transformation's input_shape as the starting point
      // This follows the chaining invariant where transform[0].input_shape == current tensor shape
      std::vector<int64_t> initial_shape = transformations[0].input_shape;
      std::vector<int64_t> initial_perm = transformations[0].current_perm;

      // If current_perm is empty in first transformation, use identity
      if (initial_perm.empty()) {
        initial_perm.resize(initial_shape.size());
        std::iota(initial_perm.begin(), initial_perm.end(), 0);
      }

      // Get element type from input context
      std::string element_type = ScalarTypeToElementTypeString(input_ctx.get_dtype());

      // Create TransformationState to track changes across transformations
      TransformationState state(initial_shape, initial_perm, element_type);

      // Process transformations using handler-based dispatch
      auto& registry = TransformationRegistry::Get();
      size_t j = 0;
      while (j < transformations.size()) {
        // Get handler for this transformation type
        const TransformationHandler* handler = registry.GetHandler(transformations[j].type);

        if (handler) {
          // Find the end of the group for this transformation type
          size_t group_end = handler->FindGroupEnd(transformations, j, state);

          // Extract the group slice
          std::vector<TensorTransformation> group_transformations(
              transformations.begin() + j, transformations.begin() + group_end);

          // Process the group and create a task
          TransformationTask task = handler->ProcessGroup(i, state, group_transformations, op_name);
          tasks.push_back(task);

          // Move to the next unprocessed transformation
          j = group_end;
        } else {
          // Unknown transformation type - skip it
          TORCH_NEURONX_DEBUG("Skipping unknown transformation type", "op=", op_name,
                              "input_index=", i,
                              "type=", static_cast<int>(transformations[j].type));
          j++;
        }
      }
    }
  }

  return tasks;
}

PrologueResult OperationPrologue::MergeTransformations(const std::vector<TransformationTask>& tasks,
                                                       const std::string& op_name,
                                                       size_t num_inputs,
                                                       const std::string& prologue_cache_key,
                                                       const std::vector<size_t>& input_dedup_map,
                                                       size_t deduped_input_count) {
  PrologueResult result;
  result.success = true;
  result.has_transformations = true;
  result.tasks = tasks;

  // Initialize metadata
  result.input_was_transformed.resize(num_inputs, false);

  // STEP 1: Use provided cache key (computed once by caller)
  std::string merged_cache_key = prologue_cache_key;
  result.prologue_cache_key = merged_cache_key;

  // STEP 2: Check cache for complete merged prologue
  {
    std::lock_guard<std::mutex> lock(g_prologue_cache_mutex);
    auto it = g_prologue_cache.find(merged_cache_key);
    if (it != g_prologue_cache.end()) {
      // Cache hit - use cached MLIR directly
      // No need to reconstruct input/output lists since order is conserved
      result.prologue_mlir_str = it->second;

      TORCH_NEURONX_DEBUG("Prologue cache HIT", "cache_key=", merged_cache_key);

      return result;
    }
  }

  // STEP 3: Cache miss - perform merging
  TORCH_NEURONX_DEBUG("Prologue cache MISS", "cache_key=", merged_cache_key);

  // Assert: Every input has at least one transformation (including NONE).
  // Compound decompositions may produce multiple tasks per input (e.g., SLICE + TRANSPOSE),
  // so we check unique input indices rather than total task count.
  std::map<size_t, std::vector<TransformationTask>> tasks_by_input;
  for (const auto& task : tasks) {
    tasks_by_input[task.input_index].push_back(task);
    result.input_was_transformed[task.input_index] = true;
  }

  TORCH_CHECK(tasks_by_input.size() == num_inputs,
              "Expected one transformation chain per input (including NONE) for operation '",
              op_name, "'. Got chains for ", tasks_by_input.size(), " unique inputs but expected ",
              num_inputs, " inputs. ",
              "This violates the invariant that every input must have a transformation task.");

  // STEP 4: For each input, chain its transformations using slot-based addresses
  std::vector<std::string> per_input_transform_mlirs;

  // Slot objects for chaining - each slot's address serves as unique identifier
  size_t slot_input = 0, slot_intermediate = 1, slot_output = 2;

  for (const auto& [input_idx, input_tasks] : tasks_by_input) {
    std::string accumulated_transform_mlir;
    bool first_in_group = true;

    for (size_t task_idx = 0; task_idx < input_tasks.size(); ++task_idx) {
      const auto& task = input_tasks[task_idx];

      // Generate MLIR for this task
      std::string task_mlir = transformation_utils::GenerateMlir(task);

      if (first_in_group) {
        accumulated_transform_mlir = task_mlir;
        first_in_group = false;
      } else {
        TORCH_NEURONX_DEBUG("Chaining transformations using SkipIntermediates",
                            "input_idx=", input_idx, "task_idx=", task_idx);

        // Chain using slot addresses: accumulated(in→inter) + new(inter→out) = merged(in→out)
        std::string merged_mlir = torch_neuronx::mergeStableHLOModulesSkipIntermediates(
            accumulated_transform_mlir, task_mlir, {&slot_input}, {&slot_intermediate},
            {&slot_intermediate}, {&slot_output});

        if (merged_mlir.empty()) {
          result.success = false;
          result.error_message = "Failed to chain transformation in group for input " +
                                 std::to_string(input_idx) + " at task " + std::to_string(task_idx);
          return result;
        }
        accumulated_transform_mlir = merged_mlir;
      }
    }

    // Store the merged transformation for this input
    per_input_transform_mlirs.push_back(accumulated_transform_mlir);

    TORCH_NEURONX_DEBUG("Completed transformation chain for input", "op=", op_name,
                        "input_index=", input_idx,
                        "mlir_length=", accumulated_transform_mlir.length());
  }

  // STEP 5: Merge all per-input transformation chains together
  // Use deduplication info computed by caller to share input slots
  TORCH_NEURONX_DEBUG("Merging all transformation chains", "op=", op_name,
                      "num_chains=", per_input_transform_mlirs.size(),
                      "deduped_inputs=", deduped_input_count);

  // Create slot vectors - deduped_input_count unique input slots, one output slot per chain
  std::vector<size_t> input_slots(deduped_input_count);
  std::vector<size_t> output_slots(per_input_transform_mlirs.size());

  // Start with the first transformation chain
  std::string accumulated_mlir = per_input_transform_mlirs[0];
  std::vector<void*> accumulated_input_addrs = {&input_slots[input_dedup_map[0]]};
  std::vector<void*> accumulated_output_addrs = {&output_slots[0]};

  // Merge remaining chains as parallel operations
  for (size_t i = 1; i < per_input_transform_mlirs.size(); ++i) {
    std::vector<void*> chain_input_addrs = {&input_slots[input_dedup_map[i]]};
    std::vector<void*> chain_output_addrs = {&output_slots[i]};

    TORCH_NEURONX_DEBUG("Merging chain", "chain_index=", i);

    torch_neuronx::MergeMapping merge_mapping;
    std::string merged_mlir = torch_neuronx::mergeStableHLOModulesSkipIntermediates(
        accumulated_mlir, per_input_transform_mlirs[i], accumulated_input_addrs,
        accumulated_output_addrs, chain_input_addrs, chain_output_addrs, &merge_mapping);

    if (merged_mlir.empty()) {
      result.success = false;
      result.error_message = "Failed to merge transformation chain " + std::to_string(i);
      return result;
    }

    accumulated_mlir = merged_mlir;

    // Verify output count conservation (inputs may be deduped)
    TORCH_CHECK(merge_mapping.total_outputs == i + 1, "Expected ", i + 1,
                " outputs after merging chain ", i, " but got ", merge_mapping.total_outputs);

    // Append addresses for this chain
    // Only add input addr if this deduped slot hasn't been added yet
    void* input_addr = &input_slots[input_dedup_map[i]];
    if (std::find(accumulated_input_addrs.begin(), accumulated_input_addrs.end(), input_addr) ==
        accumulated_input_addrs.end()) {
      accumulated_input_addrs.push_back(input_addr);
    }
    accumulated_output_addrs.push_back(&output_slots[i]);

    TORCH_NEURONX_DEBUG("Merged chain", "chain_index=", i,
                        "total_inputs=", accumulated_input_addrs.size());
  }

  // With order conservation verified, just store the MLIR
  result.prologue_mlir_str = accumulated_mlir;
  result.input_dedup_map = input_dedup_map;
  result.deduped_input_count = deduped_input_count;

  // STEP 6: Cache the merged MLIR directly (no need to store indices since order is conserved)
  {
    std::lock_guard<std::mutex> lock(g_prologue_cache_mutex);
    g_prologue_cache[merged_cache_key] = result.prologue_mlir_str;

    TORCH_NEURONX_DEBUG("Cached merged prologue", "cache_key=", merged_cache_key);
  }

  return result;
}

// ============================================================================
// OperationEpilogue Implementation
//
// Reuses OperationPrologue's CollectTransformationTasks and
// transformation_utils::GenerateMlir. The epilogue is structurally identical
// to the prologue — the only difference is the transformations themselves
// (inverse permutations) and the cache prefix.
// ============================================================================

namespace {

// Separate cache for epilogue MLIR strings.
//
// Cache stores only the merged {epilogue_mlir_str}. The set of SCATTER_IOTA
// output indices (which drives how many extra base inputs
// ProcessEpilogueWithMlirMerge must wire) is reconstructed deterministically
// from the transformation `tasks` on both cache HIT and MISS — tasks are always
// available before the cache is consulted, so we don't need to store the index
// list. Reconstructing from tasks also keeps a single source of truth for the
// per-output scatter detection.
static std::unordered_map<std::string, std::string> g_epilogue_cache;
static std::mutex g_epilogue_cache_mutex;

// Reconstruct the SCATTER_IOTA output indices from the collected tasks.
// Returns the output (input_index) of every task whose type is SCATTER_IOTA,
// in ascending output-index order (matching the std::map iteration order used
// when the merged epilogue was built), so the extra-base-input wiring order in
// ProcessEpilogueWithMlirMerge agrees with the merged program's argument order.
static std::vector<size_t> CollectScatterIotaOutputIndices(
    const std::vector<TransformationTask>& tasks) {
  std::set<size_t> scatter_outputs;
  for (const auto& task : tasks) {
    if (task.type == TransformationType::SCATTER_IOTA) {
      scatter_outputs.insert(task.input_index);
    }
  }
  return std::vector<size_t>(scatter_outputs.begin(), scatter_outputs.end());
}

}  // anonymous namespace

std::string OperationEpilogue::ComputeEpilogueCacheKey(
    const std::vector<void*>& output_ptrs,
    const std::vector<at::neuron::TensorContext>& output_contexts, const std::string& op_name,
    const std::vector<std::vector<TensorTransformation>>& kernel_transforms) {
  // Reuse the same cache key format as prologue, just with "epilogue--" prefix
  std::stringstream cache_key_ss;
  cache_key_ss << "epilogue--";

  for (size_t idx = 0; idx < output_ptrs.size(); ++idx) {
    if (idx >= kernel_transforms.size() || kernel_transforms[idx].empty()) {
      const auto& ctx = output_contexts[idx];
      std::vector<int64_t> shape = ctx.get_shape();
      std::string element_type = ScalarTypeToElementTypeString(ctx.get_dtype());
      cache_key_ss << "o" << idx << "_";
      for (auto dim : shape) cache_key_ss << dim << "x";
      cache_key_ss << "_" << element_type << "_";
      continue;
    }

    for (const auto& transform : kernel_transforms[idx]) {
      cache_key_ss << "o" << idx << "_";
      cache_key_ss << TransformationTypeToString(transform.type) << "_";
      for (auto dim : transform.input_shape) cache_key_ss << dim << "x";
      cache_key_ss << "_to_";
      for (auto dim : transform.output_shape) cache_key_ss << dim << "x";
      if (!transform.params.empty()) {
        cache_key_ss << "_params_";
        for (auto p : transform.params) cache_key_ss << p << ",";
      }
      cache_key_ss << "_dtype_" << transform.element_type << "_";
    }
  }

  cache_key_ss << "op_" << op_name;
  return cache_key_ss.str();
}

EpilogueResult OperationEpilogue::GenerateEpilogue(
    const std::vector<void*>& output_ptrs,
    const std::vector<at::neuron::TensorContext>& output_contexts, const std::string& op_name,
    const std::vector<std::vector<TensorTransformation>>& kernel_transforms,
    const std::string& epilogue_cache_key) {
  // Reuse prologue's CollectTransformationTasks — it works identically for
  // both directions since it just converts TensorTransformation → TransformationTask
  // using the handler registry.
  auto tasks = OperationPrologue::CollectTransformationTasks(output_ptrs, output_contexts, op_name,
                                                             kernel_transforms);

  if (tasks.empty()) {
    EpilogueResult result;
    result.success = true;
    result.has_transformations = false;
    result.epilogue_cache_key = "";
    return result;
  }

  TORCH_NEURONX_DEBUG("OperationEpilogue: Collected transformation tasks", "op=", op_name,
                      "num_tasks=", tasks.size());

  // Build the epilogue result using the same merge logic as prologue
  // but with a separate cache and no deduplication (outputs don't share storage)
  EpilogueResult result;
  result.success = true;
  result.has_transformations = true;
  result.tasks = tasks;
  result.output_was_transformed.resize(output_ptrs.size(), false);
  result.epilogue_cache_key = epilogue_cache_key;

  // The set of scatter output indices is derived purely from the tasks and is
  // identical on cache HIT and MISS. Compute it once up front so both paths can
  // populate result.scatter_iota_output_indices consistently.
  result.scatter_iota_output_indices = CollectScatterIotaOutputIndices(tasks);

  // Check epilogue cache
  {
    std::lock_guard<std::mutex> lock(g_epilogue_cache_mutex);
    auto it = g_epilogue_cache.find(epilogue_cache_key);
    if (it != g_epilogue_cache.end()) {
      result.epilogue_mlir_str = it->second;
      // Restore output_was_transformed from tasks (tasks are available before cache check)
      for (const auto& task : tasks) {
        if (task.input_index < result.output_was_transformed.size()) {
          result.output_was_transformed[task.input_index] = true;
        }
      }
      TORCH_NEURONX_DEBUG("Epilogue cache HIT", "cache_key=", epilogue_cache_key,
                          "num_scatter_outputs=", result.scatter_iota_output_indices.size());
      return result;
    }
  }

  TORCH_NEURONX_DEBUG("Epilogue cache MISS", "cache_key=", epilogue_cache_key);

  // Group tasks by output index
  std::map<size_t, std::vector<TransformationTask>> tasks_by_output;
  for (const auto& task : tasks) {
    tasks_by_output[task.input_index].push_back(task);
    result.output_was_transformed[task.input_index] = true;
  }

  size_t num_outputs = output_ptrs.size();
  TORCH_CHECK(tasks_by_output.size() == num_outputs,
              "Expected one transformation chain per output for '", op_name, "'.");

  // Chain transformations per output.
  //
  // A SCATTER_IOTA task has TWO inputs — the update tensor and the flat base
  // storage. Each per-output chain that contains a scatter therefore exposes
  // ONE extra input (the base) in addition to its normal (update) input. We
  // must register that base as a distinct input slot so it survives into the
  // merged program as a real function argument; otherwise the merged MLIR gains
  // a 2nd argument with no corresponding input address and
  // validateAndGetMainFunctions throws:
  //   "Module1 input address count (N) does not match function argument count (M)".
  //
  // This applies whether the scatter is the FIRST task in the chain (the common
  // generic-scatter case — a lone SCATTER_IOTA per output) or a later task
  // (decomposed transpose ∘ slice, where the scatter follows a transpose).
  // Prior code only wired the base for non-first tasks, which silently dropped
  // it for the generic single-scatter case and for multi-output ops.
  std::vector<std::string> per_output_mlirs;
  // Per-output flag: does this output's chain expose an extra base input?
  std::vector<bool> per_output_has_base;
  size_t slot_input = 0, slot_intermediate = 1, slot_output = 2;
  // Distinct address used to denote "the extra base input" while building a
  // single output's chain. It is per-output-local; the final merge step assigns
  // each output's base a unique slot in the merged program.
  size_t slot_base = 3;

  for (const auto& [output_idx, output_tasks] : tasks_by_output) {
    std::string accumulated_mlir;
    bool first = true;
    bool chain_has_base = false;
    // Track the current chain's input signature so a first-task scatter keeps
    // BOTH its inputs (update + base) exposed on the chain's main function.
    std::vector<void*> chain_in_addrs;

    for (size_t ti = 0; ti < output_tasks.size(); ++ti) {
      std::string task_mlir = transformation_utils::GenerateMlir(output_tasks[ti]);
      bool is_scatter = (output_tasks[ti].type == TransformationType::SCATTER_IOTA);

      if (first) {
        accumulated_mlir = task_mlir;
        first = false;
        // The chain's initial input signature = this task's inputs.
        // For a scatter, that's {update(=slot_input), base(=slot_base)}.
        if (is_scatter) {
          chain_in_addrs = {&slot_input, &slot_base};
          chain_has_base = true;
        } else {
          chain_in_addrs = {&slot_input};
        }
      } else {
        // Connect this task's (first) input to the previous chain's output
        // (slot_intermediate). A scatter additionally consumes the base input
        // as a fresh unique input (slot_base).
        std::vector<void*> task_in_addrs;
        if (is_scatter) {
          task_in_addrs = {&slot_intermediate, &slot_base};
          chain_has_base = true;
        } else {
          task_in_addrs = {&slot_intermediate};
        }

        std::string merged = torch_neuronx::mergeStableHLOModulesSkipIntermediates(
            accumulated_mlir, task_mlir, chain_in_addrs, {&slot_intermediate}, task_in_addrs,
            {&slot_output});
        if (merged.empty()) {
          result.success = false;
          result.error_message =
              "Failed to chain epilogue transformation for output " + std::to_string(output_idx);
          return result;
        }
        accumulated_mlir = merged;
        // After merging, the chain's inputs are the original chain input plus
        // (for a scatter) the newly introduced base input.
        if (is_scatter) chain_in_addrs.push_back(&slot_base);
      }
    }
    per_output_mlirs.push_back(accumulated_mlir);
    per_output_has_base.push_back(chain_has_base);
  }

  // Merge all per-output chains together. Each chain contributes:
  //   - one primary input  (the update / contiguous result for that output), and
  //   - optionally one base input (if that output has a SCATTER_IOTA).
  //
  // We give every input its own unique slot address so the merged program keeps
  // one function argument per (update, base?) across all outputs, and record —
  // in output order — which outputs carry a base. LazyTransformationManager wires
  // one extra src per entry in scatter_iota_output_indices, in this same order.
  std::vector<size_t> input_slots(num_outputs);
  std::vector<size_t> base_slots(num_outputs);
  std::vector<size_t> out_slots(num_outputs);

  std::string accumulated_mlir;
  std::vector<void*> acc_in;
  std::vector<void*> acc_out;

  for (size_t i = 0; i < per_output_mlirs.size(); ++i) {
    // This chain's input signature: primary input, plus base input if present.
    std::vector<void*> chain_in = {&input_slots[i]};
    if (per_output_has_base[i]) chain_in.push_back(&base_slots[i]);
    std::vector<void*> chain_out = {&out_slots[i]};

    if (i == 0) {
      accumulated_mlir = per_output_mlirs[0];
      acc_in = chain_in;
      acc_out = chain_out;
      continue;
    }

    torch_neuronx::MergeMapping mm;
    std::string merged = torch_neuronx::mergeStableHLOModulesSkipIntermediates(
        accumulated_mlir, per_output_mlirs[i], acc_in, acc_out, chain_in, chain_out, &mm);

    if (merged.empty()) {
      result.success = false;
      result.error_message = "Failed to merge epilogue chain " + std::to_string(i);
      return result;
    }

    accumulated_mlir = merged;
    // Extend the accumulated input signature with this chain's inputs (primary
    // first, then base) so subsequent merges align argument order.
    acc_in.push_back(&input_slots[i]);
    if (per_output_has_base[i]) acc_in.push_back(&base_slots[i]);
    acc_out.push_back(&out_slots[i]);
  }

  result.epilogue_mlir_str = accumulated_mlir;

  // scatter_iota_output_indices was already populated up front from tasks.
  TORCH_NEURONX_DEBUG("Built merged epilogue", "op=", op_name, "num_outputs=", num_outputs,
                      "num_scatter_outputs=", result.scatter_iota_output_indices.size());

  // Cache the merged MLIR. The scatter output indices are reconstructed from
  // tasks on the next HIT, so only the MLIR needs to be stored.
  {
    std::lock_guard<std::mutex> lock(g_epilogue_cache_mutex);
    g_epilogue_cache[epilogue_cache_key] = result.epilogue_mlir_str;
    TORCH_NEURONX_DEBUG("Cached merged epilogue", "cache_key=", epilogue_cache_key,
                        "num_scatter_outputs=", result.scatter_iota_output_indices.size());
  }

  return result;
}

}  // namespace lazy
}  // namespace c10_neuron

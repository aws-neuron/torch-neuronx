#include "OperationPrologue.h"

#include <chrono>
#include <mutex>
#include <numeric>
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
        return mlir_generators::GenerateSlice(task.input_shape, params->start_indices,
                                              params->end_indices, task.element_type);
      }
      throw std::runtime_error("Invalid SliceParams in TransformationTask");

    case TransformationType::BROADCAST:
      // TODO: Implement broadcast MLIR generation
      throw std::runtime_error(
          "Broadcast MLIR generation not yet implemented - requires BroadcastOpBuilder");

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
    const std::string& prologue_cache_key) {
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

  // Step 3: Merge transformations into prologue using pre-computed cache key
  auto result = MergeTransformations(tasks, op_name, input_ptrs, prologue_cache_key);

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

  // Process each input
  for (size_t input_idx = 0; input_idx < input_ptrs.size(); ++input_idx) {
    // Check if this input has kernel transformations
    if (input_idx >= kernel_transforms.size() || kernel_transforms[input_idx].empty()) {
      // No kernel transformations - NONE will be created using input_context
      // Include shape and dtype in cache key
      const auto& input_ctx = input_contexts[input_idx];
      std::vector<int64_t> shape = input_ctx.get_shape();
      std::string element_type = ScalarTypeToElementTypeString(input_ctx.get_dtype());

      cache_key_ss << input_idx << "_NONE_";
      for (auto dim : shape) {
        cache_key_ss << dim << "x";
      }
      cache_key_ss << "_dtype_" << element_type << "_";
      continue;
    }

    // Process transformations for this input
    const auto& transforms = kernel_transforms[input_idx];
    for (const auto& transform : transforms) {
      cache_key_ss << input_idx << "_";
      // Add transformation type
      cache_key_ss << static_cast<int>(transform.type) << "_";

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
                                                       const std::vector<void*>& all_input_ptrs,
                                                       const std::string& prologue_cache_key) {
  PrologueResult result;
  result.success = true;
  result.has_transformations = true;
  result.tasks = tasks;

  // Initialize metadata
  result.input_was_transformed.resize(all_input_ptrs.size(), false);

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

  // Assert: Every input has at least one transformation (including NONE)
  // This invariant is critical for order conservation
  TORCH_CHECK(tasks.size() == all_input_ptrs.size(),
              "Expected one transformation per input (including NONE) for operation '", op_name,
              "'. Got ", tasks.size(), " tasks for ", all_input_ptrs.size(), " inputs. ",
              "This violates the invariant that every input must have a transformation task.");

  // Group tasks by input_index for verification
  std::map<size_t, std::vector<TransformationTask>> tasks_by_input;
  for (const auto& task : tasks) {
    tasks_by_input[task.input_index].push_back(task);
    result.input_was_transformed[task.input_index] = true;
  }

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
  // Each chain gets a unique slot; slot addresses serve as unique identifiers
  TORCH_NEURONX_DEBUG("Merging all transformation chains", "op=", op_name,
                      "num_chains=", per_input_transform_mlirs.size());

  // Create slot vectors - each element's address is unique
  std::vector<size_t> input_slots(per_input_transform_mlirs.size());
  std::vector<size_t> output_slots(per_input_transform_mlirs.size());

  // Start with the first transformation chain
  std::string accumulated_mlir = per_input_transform_mlirs[0];
  std::vector<void*> accumulated_input_addrs = {&input_slots[0]};
  std::vector<void*> accumulated_output_addrs = {&output_slots[0]};

  // Merge remaining chains as parallel operations
  for (size_t i = 1; i < per_input_transform_mlirs.size(); ++i) {
    std::vector<void*> chain_input_addrs = {&input_slots[i]};
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

    // Verify order conservation
    TORCH_CHECK(merge_mapping.total_inputs == i + 1, "Expected ", i + 1,
                " inputs after merging chain ", i, " but got ", merge_mapping.total_inputs);
    TORCH_CHECK(merge_mapping.total_outputs == i + 1, "Expected ", i + 1,
                " outputs after merging chain ", i, " but got ", merge_mapping.total_outputs);

    // Append addresses for this chain
    accumulated_input_addrs.push_back(&input_slots[i]);
    accumulated_output_addrs.push_back(&output_slots[i]);

    TORCH_NEURONX_DEBUG("Merged chain", "chain_index=", i,
                        "total_inputs=", accumulated_input_addrs.size());
  }

  // With order conservation verified, just store the MLIR
  result.prologue_mlir_str = accumulated_mlir;

  // STEP 6: Cache the merged MLIR directly (no need to store indices since order is conserved)
  {
    std::lock_guard<std::mutex> lock(g_prologue_cache_mutex);
    g_prologue_cache[merged_cache_key] = result.prologue_mlir_str;

    TORCH_NEURONX_DEBUG("Cached merged prologue", "cache_key=", merged_cache_key);
  }

  return result;
}

}  // namespace lazy
}  // namespace c10_neuron

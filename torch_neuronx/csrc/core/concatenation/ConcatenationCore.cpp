#include "ConcatenationCore.h"

#include <iomanip>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/concatenation/IrNodeManager.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"

namespace torch_neuronx {

namespace {

/**
 * @brief Helper to create a fallback result returning original operations
 *
 * Used when an exception occurs during concatenation processing to ensure
 * the original operations are returned for individual execution.
 *
 * @param op_contexts Original operation contexts to return
 * @param error_msg Error message describing what went wrong
 * @return ConcatenationResult with original operations and error info
 */
ConcatenationResult CreateFallbackResult(
    const std::list<at::neuron::OperationContext*>& op_contexts, const std::string& error_msg) {
  ConcatenationResult result;
  result.processed_operations =
      std::list<at::neuron::OperationContext*>(op_contexts.begin(), op_contexts.end());
  result.original_operations_consumed = op_contexts.size();

  TORCH_NEURONX_INFO("[CONCATENATION CORE] Falling back to original operations",
                     "reason:", error_msg, "op_count:", op_contexts.size());
  return result;
}

}  // namespace

ConcatenationCore::ConcatenationCore(
    std::vector<std::unique_ptr<AbstractIrConcatStrategy>>&& concat_strategies,
    std::unique_ptr<IrNodeManager>&& ir_node_manager)
    : concat_strategies(std::move(concat_strategies)),
      ir_node_manager(std::move(ir_node_manager)) {}

/**
 * @brief Process an existing (non-concatenated) IR node
 *
 * Handles the case where the IR node maps directly to an existing OperationContext.
 *
 * @param ir_node The IR node to process
 * @param op_context_results Output list to add the operation context to
 * @return Number of operations processed (always 1 for this case)
 */
int ConcatenationCore::ProcessExistingOperation(
    IrNode* ir_node, std::list<at::neuron::OperationContext*>& op_context_results) {
  auto op_context_it = ir_node_to_op_context_map.find(ir_node);
  if (op_context_it == ir_node_to_op_context_map.end()) {
    return 0;  // Not found - not an existing operation
  }

  op_context_results.push_back(op_context_it->second);
  ir_node_to_op_context_map.erase(ir_node);
  return 1;
}

/**
 * @brief Process an invalidated concatenated operation
 *
 * Handles the case where a concatenated operation's cache key was invalidated.
 * Returns the original operations instead of the concatenated one.
 *
 * @param ir_node The invalidated IR node
 * @param ir_concat_result The concatenation result containing original->concat mappings
 * @param op_context_results Output list to add the original operation contexts to
 * @return Number of original operations processed
 */
int ConcatenationCore::ProcessInvalidatedConcatenation(
    IrNode* ir_node, IrConcatResult& ir_concat_result,
    std::list<at::neuron::OperationContext*>& op_context_results) {
  if (invalidated_cache_keys_.find(ir_node->cache_key) == invalidated_cache_keys_.end()) {
    return -1;  // Not invalidated - caller should try other processing
  }

  // Get the original operations that would have been concatenated
  auto original_ir_nodes = ir_concat_result.concated_ir_to_original_irs[ir_node];
  int processed_count = 0;

  // Add the original operations to result list (no concatenation)
  for (auto original_ir_node : original_ir_nodes) {
    auto original_op_context_it = ir_node_to_op_context_map.find(original_ir_node);
    if (original_op_context_it != ir_node_to_op_context_map.end()) {
      op_context_results.push_back(original_op_context_it->second);
      processed_count++;
      ir_node_to_op_context_map.erase(original_ir_node);
    }
  }

  // Remove this invalidated entry from the map so success flag is calculated correctly
  ir_concat_result.concated_ir_to_original_irs.erase(ir_node);

  return processed_count;
}

/**
 * @brief Process a new concatenated operation
 *
 * Creates a fresh OperationContext for a concatenated IR node and sets up
 * the cascading relationships with the original operations.
 *
 * @param ir_node The concatenated IR node
 * @param ir_concat_result The concatenation result containing original->concat mappings
 * @param op_contexts Original operation contexts (for stream/device info)
 * @param op_context_results Output list to add the original operation contexts to
 * @return Number of original operations that were concatenated
 */
int ConcatenationCore::ProcessNewConcatenatedOperation(
    IrNode* ir_node, IrConcatResult& ir_concat_result,
    const std::list<at::neuron::OperationContext*>& op_contexts,
    std::list<at::neuron::OperationContext*>& op_context_results) {
  auto concat_op_context_unique =
      CreateOpContextFromIrNode(ir_node, op_contexts.front()->kernel_execution->GetDeviceId());
  auto* concat_op_context = concat_op_context_unique.get();

  // CRITICAL: Set the stream field from the original operation to avoid null pointer crash
  // The concatenated operation must execute on the same stream as the original operations
  concat_op_context->stream = op_contexts.front()->stream;

  // Set up cascading relationships for the original operations that were merged
  auto original_ir_nodes = ir_concat_result.concated_ir_to_original_irs[ir_node];
  int processed_count = original_ir_nodes.size();

  // First, collect all cascading operations before creating the immutable state
  std::vector<at::neuron::OperationContext*> cascading_ops;
  cascading_ops.reserve(original_ir_nodes.size());

  for (auto original_ir_node : original_ir_nodes) {
    if (!original_ir_node) {
      throw std::runtime_error("Null original IrNode in concated_ir_to_original_irs");
    }

    auto original_op_context_it = ir_node_to_op_context_map.find(original_ir_node);
    if (original_op_context_it != ir_node_to_op_context_map.end()) {
      auto* original_op = original_op_context_it->second;
      if (original_op) {
        cascading_ops.push_back(original_op);
      }
      ir_node_to_op_context_map.erase(original_ir_node);
    } else {
      throw std::runtime_error("Original IrNode not found in ir_node_to_op_context_map");
    }
  }

  // Create failure callback that will be invoked if compilation fails
  // This allows error handling without direct reference to ConcatenationEngine
  std::string cache_key_for_callback = ir_node->cache_key;
  at::neuron::ConcatenationFailureCallback failure_callback =
      [this, cache_key_for_callback](at::neuron::OperationContext* failed_op) {
        // Invalidate the cache entry to prevent retry for this concatenation
        InvalidateCacheEntry(cache_key_for_callback);

        // Clear the concatenation state for all cascading operations
        // so they can fall back to individual execution
        if (failed_op) {
          // Use GetConcatenationState() which handles both concatenated ops (raw ptr)
          // and cascading ops (shared_ptr) correctly
          auto* state = failed_op->GetConcatenationState();
          if (state) {
            for (auto* op : state->GetCascadingOperations()) {
              op->concatenation_state_ = nullptr;
            }
          }
        }
      };

  // Create shared concatenation state with all data upfront (immutable after construction)
  // ConcatenationState takes sole ownership of the concatenated operation via unique_ptr
  auto concat_state = std::make_shared<at::neuron::ConcatenationState>(
      std::move(concat_op_context_unique), std::move(cascading_ops), std::move(failure_callback));

  // Link the concatenated operation to the state using raw pointer (non-owning)
  // This avoids circular reference since ConcatenationState owns concat_op via unique_ptr
  concat_op_context->concatenation_state_raw_ = concat_state.get();

  // Link each cascading operation to the shared state and add to result list
  for (auto* original_op : concat_state->GetCascadingOperations()) {
    original_op->concatenation_state_ = concat_state;
    op_context_results.push_back(original_op);
  }

  return processed_count;
}

/**
 * @brief Post-process IR concat results back to OperationContext
 *
 * Converts the IR concatenation results back to OperationContext objects,
 * handling existing operations, invalidated concatenations, and new concatenations.
 *
 * @param ir_concat_result The concatenation result from try_concat
 * @param op_contexts Original operation contexts
 * @param op_context_results Output list for processed operation contexts
 * @return Total number of operations processed
 */
int ConcatenationCore::PostProcessConcatResults(
    IrConcatResult& ir_concat_result, const std::list<at::neuron::OperationContext*>& op_contexts,
    std::list<at::neuron::OperationContext*>& op_context_results) {
  int total_processed_op = 0;

  for (auto ir_node : ir_concat_result.compilable_irs) {
    // Case 1: Check if this is an existing (non-concatenated) operation
    int existing_count = ProcessExistingOperation(ir_node, op_context_results);
    if (existing_count > 0) {
      total_processed_op += existing_count;
      continue;
    }

    // Case 2: Check if this concatenated operation was invalidated
    int invalidated_count =
        ProcessInvalidatedConcatenation(ir_node, ir_concat_result, op_context_results);
    if (invalidated_count >= 0) {
      total_processed_op += invalidated_count;
      continue;  // Skip the concatenated operation - it won't be used
    }

    // Case 3: This is a new concatenated operation - create fresh OperationContext
    int concat_count =
        ProcessNewConcatenatedOperation(ir_node, ir_concat_result, op_contexts, op_context_results);
    total_processed_op += concat_count;
  }

  return total_processed_op;
}

/**
 * @brief Main entry point for processing buffered operations through concatenation
 */
ConcatenationResult ConcatenationCore::ProcessBufferedOperations(
    const std::list<at::neuron::OperationContext*>& op_contexts) {
  // Wrap entire processing in try-catch to ensure we return original operations on any error
  try {
    return ProcessBufferedOperationsImpl(op_contexts);
  } catch (const std::exception& e) {
    return CreateFallbackResult(op_contexts,
                                std::string("Exception during concatenation: ") + e.what());
  } catch (...) {
    return CreateFallbackResult(op_contexts, "Unknown exception during concatenation");
  }
}

/**
 * @brief Internal implementation of processBufferedOperations
 *
 * This method contains the actual concatenation logic, separated from the
 * public method to allow for clean exception handling.
 */
ConcatenationResult ConcatenationCore::ProcessBufferedOperationsImpl(
    const std::list<at::neuron::OperationContext*>& op_contexts) {
  // Create mixed IR node list: HLO nodes + DeallocHintNodes in execution order
  std::list<IrNode*> mixed_ir_nodes;
  std::list<std::unique_ptr<IrNode>> ir_nodes_owned;

  ir_node_to_op_context_map.clear();

  for (auto* op : op_contexts) {
    if (op->GetKernelType() == at::neuron::KernelTypeEnum::kHint) {
      auto* hint = dynamic_cast<at::neuron::HintDirectKernelExecution*>(op->kernel_execution.get());
      if (hint &&
          hint->GetHintType() == at::neuron::HintDirectKernelExecution::HintType::kDeallocation) {
        // Create DeallocHintNode
        auto hint_node = std::make_unique<DeallocHintNode>(hint->GetPtr());
        mixed_ir_nodes.push_back(hint_node.get());
        ir_nodes_owned.push_back(std::move(hint_node));
      } else if (hint && hint->GetHintType() ==
                             at::neuron::HintDirectKernelExecution::HintType::kAllocation) {
        // Create AllocHintNode - signals address (re)allocation to clear from pruned set
        auto hint_node = std::make_unique<AllocHintNode>(hint->GetPtr());
        mixed_ir_nodes.push_back(hint_node.get());
        ir_nodes_owned.push_back(std::move(hint_node));
      }
    } else {
      // Create IR node from HLO operation
      // TODO(mengchiy): Handle hint nodes also in CreateNodeFromOperationContext
      auto ir_node = ir_node_manager->CreateNodeFromOperationContext(op);
      if (ir_node) {
        ir_node_to_op_context_map[ir_node.get()] = op;
        mixed_ir_nodes.push_back(ir_node.get());
        ir_nodes_owned.push_back(std::move(ir_node));
      } else {
        // Failed to create IR node - throw exception which triggers fallback to op-by-op execution
        throw std::runtime_error("Failed to create IR node for operation: " + op->GetOpName());
      }
    }
  }

  // Pass mixed IR node list (HLO + hints) to IrNodeManager
  // IrNodeManager will process them with type-based dispatch
  IrConcatResult ir_concat_result = ir_node_manager->ConcatAll(mixed_ir_nodes);

  // Transfer ownership of created nodes to the result
  ir_concat_result.concat_ir_list.splice(ir_concat_result.concat_ir_list.end(), ir_nodes_owned);

  // Post-process: Convert IR nodes back to OperationContext
  std::list<at::neuron::OperationContext*> op_context_results;
  int total_processed_op =
      PostProcessConcatResults(ir_concat_result, op_contexts, op_context_results);

  // Build result - always return success per API contract
  bool concatenation_happened = ir_concat_result.concated_ir_to_original_irs.size() > 0;

  ConcatenationResult concat_res;

  if (concatenation_happened) {
    concat_res.processed_operations = std::move(op_context_results);
    concat_res.original_operations_consumed = total_processed_op;
  } else {
    // No concatenation - return all original operations
    concat_res.processed_operations =
        std::list<at::neuron::OperationContext*>(op_contexts.begin(), op_contexts.end());
    concat_res.original_operations_consumed = op_contexts.size();
  }

  return concat_res;
}

std::unique_ptr<at::neuron::OperationContext> ConcatenationCore::CreateOpContextFromIrNode(
    IrNode* ir_node, int device_id) {
  if (!ir_node) {
    TORCH_NEURONX_ERROR("Attempting to create OperationContext from null IrNode");
    throw std::invalid_argument("IrNode cannot be null");
  }

  if (ir_node->op_name.empty()) {
    TORCH_NEURONX_ERROR("IrNode validation failed - empty op_name",
                        "cache_key:", ir_node->cache_key,
                        "ir_type:", static_cast<int>(ir_node->ir_type));
    throw std::invalid_argument("IrNode must have valid op_name");
  }

  if (ir_node->cache_key.empty()) {
    TORCH_NEURONX_ERROR("IrNode validation failed - empty cache_key", "op_name:", ir_node->op_name,
                        "ir_type:", static_cast<int>(ir_node->ir_type));
    throw std::invalid_argument("IrNode must have valid cache_key");
  }

  // Use TensorDataRef and TensorContext from the IrNode
  // These are populated by IrNodeManager from original operations and propagated through merge
  // mapping
  std::vector<at::neuron::TensorDataRef> input_refs;
  std::vector<at::neuron::TensorDataRef> output_refs;
  std::vector<at::neuron::TensorContext> input_contexts;
  std::vector<at::neuron::TensorContext> output_contexts;

  // Validate TensorDataRef consistency with void* addresses
  // For concatenated operations, TensorDataRef MUST be populated to ensure proper NRT execution

  // Check for mismatch: if we have outputs but no output_data_refs, this is an error
  if (!ir_node->outputs.empty() && ir_node->output_data_refs.empty()) {
    TORCH_NEURONX_ERROR(
        "[CONCATENATION ERROR] IrNode has outputs but no output_data_refs! "
        "This indicates TensorDataRef was not properly propagated during concatenation.",
        "op_name:", ir_node->op_name, "cache_key:", ir_node->cache_key,
        "outputs_count:", ir_node->outputs.size(),
        "output_data_refs_count:", ir_node->output_data_refs.size());
    throw std::runtime_error(
        "IrNode has outputs but no output_data_refs - TensorDataRef not properly propagated "
        "during concatenation. op_name=" +
        ir_node->op_name + " outputs_count=" + std::to_string(ir_node->outputs.size()));
  }

  // Check for mismatch: if we have inputs but no input_data_refs, this is an error
  if (!ir_node->inputs.empty() && ir_node->input_data_refs.empty()) {
    TORCH_NEURONX_ERROR(
        "[CONCATENATION ERROR] IrNode has inputs but no input_data_refs! "
        "This indicates TensorDataRef was not properly propagated during concatenation.",
        "op_name:", ir_node->op_name, "cache_key:", ir_node->cache_key,
        "inputs_count:", ir_node->inputs.size(),
        "input_data_refs_count:", ir_node->input_data_refs.size());
    throw std::runtime_error(
        "IrNode has inputs but no input_data_refs - TensorDataRef not properly propagated "
        "during concatenation. op_name=" +
        ir_node->op_name + " inputs_count=" + std::to_string(ir_node->inputs.size()));
  }

  // Validate sizes match
  if (ir_node->input_data_refs.size() != ir_node->inputs.size()) {
    TORCH_NEURONX_ERROR("[CONCATENATION ERROR] IrNode input_data_refs size mismatch!",
                        "op_name:", ir_node->op_name, "cache_key:", ir_node->cache_key,
                        "inputs_count:", ir_node->inputs.size(),
                        "input_data_refs_count:", ir_node->input_data_refs.size());
    throw std::runtime_error(
        "IrNode input_data_refs size (" + std::to_string(ir_node->input_data_refs.size()) +
        ") does not match inputs size (" + std::to_string(ir_node->inputs.size()) +
        ") for op=" + ir_node->op_name);
  }

  if (ir_node->output_data_refs.size() != ir_node->outputs.size()) {
    TORCH_NEURONX_ERROR("[CONCATENATION ERROR] IrNode output_data_refs size mismatch!",
                        "op_name:", ir_node->op_name, "cache_key:", ir_node->cache_key,
                        "outputs_count:", ir_node->outputs.size(),
                        "output_data_refs_count:", ir_node->output_data_refs.size());
    throw std::runtime_error(
        "IrNode output_data_refs size (" + std::to_string(ir_node->output_data_refs.size()) +
        ") does not match outputs size (" + std::to_string(ir_node->outputs.size()) +
        ") for op=" + ir_node->op_name);
  }

  // Use TensorDataRef from node (contains proper NeuronTensorPtr for NRT execution)
  input_refs = ir_node->input_data_refs;
  output_refs = ir_node->output_data_refs;
  input_contexts = ir_node->input_contexts;
  output_contexts = ir_node->output_contexts;

  auto kernel = std::make_unique<at::neuron::XLACompilableKernelExecution>(
      ir_node->op_name, std::move(input_refs), std::move(output_refs), input_contexts,
      output_contexts, ir_node->cache_key, ir_node->ir_serialized, ir_node->has_collectives,
      device_id);

  // Assert that all outputs are unique (no duplicate pointers)
  {
    std::unordered_set<void*> unique_outputs(ir_node->outputs.begin(), ir_node->outputs.end());
    if (unique_outputs.size() != ir_node->outputs.size()) {
      TORCH_NEURONX_ERROR("Duplicate outputs detected when creating kernel",
                          "op_name:", ir_node->op_name, "cache_key:", ir_node->cache_key,
                          "outputs_count:", ir_node->outputs.size(),
                          "unique_outputs_count:", unique_outputs.size());

      // Print all input addresses
      std::string input_addresses_str = "Input addresses: [";
      for (size_t i = 0; i < ir_node->inputs.size(); ++i) {
        if (i > 0) input_addresses_str += ", ";
        std::ostringstream oss;
        oss << "0x" << std::hex << reinterpret_cast<uintptr_t>(ir_node->inputs[i]);
        input_addresses_str += oss.str();
      }
      input_addresses_str += "]";
      TORCH_NEURONX_ERROR(input_addresses_str);

      // Print all output addresses
      std::string output_addresses_str = "Output addresses: [";
      for (size_t i = 0; i < ir_node->outputs.size(); ++i) {
        if (i > 0) output_addresses_str += ", ";
        std::ostringstream oss;
        oss << "0x" << std::hex << reinterpret_cast<uintptr_t>(ir_node->outputs[i]);
        output_addresses_str += oss.str();
      }
      output_addresses_str += "]";
      TORCH_NEURONX_ERROR(output_addresses_str);

      throw std::runtime_error("All outputs must be unique when creating a kernel");
    }
  }

  // Override cache key back to base key (without compiler extension) to ensure
  // consistency between IrNodeManager cache and CompilationCache
  // This allows invalidation to work correctly across both caches
  kernel->UpdateCacheKey(ir_node->cache_key);

  return std::make_unique<at::neuron::OperationContext>(std::move(kernel));
}

std::unique_ptr<ConcatenationCore> ConcatenationCoreFactory::CreateInstance() {
  std::vector<std::unique_ptr<AbstractIrConcatStrategy>> strategies;

  auto all_strategy = std::make_unique<MatMulToMatMulStrategy>();
  if (!all_strategy) {
    throw std::runtime_error("Failed to create ConcatenateAllStrategy");
  }
  strategies.push_back(std::move(all_strategy));

  auto ir_node_manager = std::make_unique<IrNodeManager>();
  if (!ir_node_manager) {
    throw std::runtime_error("Failed to create IrNodeManager");
  }

  // Do this because std::make_unique doesn't work when using a private constructor
  // even if this is a friend class
  auto* coordinator = new ConcatenationCore(std::move(strategies), std::move(ir_node_manager));
  return std::unique_ptr<ConcatenationCore>(coordinator);
}

std::unique_ptr<ConcatenationCore> ConcatenationCoreFactory::CreateInstance(
    std::vector<std::unique_ptr<AbstractIrConcatStrategy>>&& strategies) {
  auto ir_node_manager = std::make_unique<IrNodeManager>();
  if (!ir_node_manager) {
    throw std::runtime_error("Failed to create IrNodeManager");
  }

  auto* core = new ConcatenationCore(std::move(strategies), std::move(ir_node_manager));
  return std::unique_ptr<ConcatenationCore>(core);
}

void ConcatenationCore::InvalidateCacheEntry(const std::string& key) {
  invalidated_cache_keys_.insert(key);
  // ir_node_manager->evict(key);
}

bool ConcatenationCore::IsFusibleBoundaryOperation(const std::string& op_name) {
  for (auto& concat_strategy : concat_strategies) {
    if (concat_strategy->IsFusibleBoundaryOperation(op_name)) {
      return true;
    }
  }

  return false;
}

}  // namespace torch_neuronx

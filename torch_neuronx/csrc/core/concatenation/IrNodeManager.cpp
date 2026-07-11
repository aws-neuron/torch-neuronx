#include "IrNodeManager.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "IrNode.h"
#include "llvm/Support/raw_ostream.h"
#include "stablehlo/dialect/ChloOps.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/concatenation/FunctionPruneUtils.h"
#include "torch_neuronx/csrc/core/concatenation/OpConcatUtils.h"
#include "torch_neuronx/csrc/core/metrics/NeuronMetrics.h"

namespace torch_neuronx {

IrNodeManager::IrNodeManager() : context(std::make_unique<mlir::MLIRContext>()) {
  TORCH_NEURONX_DEBUG("IrNodeManager created");
  context->getOrLoadDialect<mlir::func::FuncDialect>();
  context->getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
  context->getOrLoadDialect<mlir::chlo::ChloDialect>();

  // Load skip intermediate flag from environment variable
  // Default: enabled. Set TORCH_NEURONX_APPLY_SKIP_INTERMEDIATE=0 to disable
  const char* skip_intermediate_env = std::getenv("TORCH_NEURONX_APPLY_SKIP_INTERMEDIATE");
  if (skip_intermediate_env && std::string(skip_intermediate_env) == "0") {
    skip_intermediate_enabled_ = false;
    TORCH_NEURONX_DEBUG("Using TORCH_NEURONX_APPLY_SKIP_INTERMEDIATE=0 (disabled)");
  }

  // Load overlap check flag from environment variable
  // Default: enabled. Set TORCH_NEURONX_DISABLE_CONCAT_FOR_OVERLAP_IO_TENSOR=0 to disable
  const char* overlap_env = std::getenv("TORCH_NEURONX_DISABLE_CONCAT_FOR_OVERLAP_IO_TENSOR");
  if (overlap_env && std::string(overlap_env) == "0") {
    check_overlap_enabled_ = false;
    TORCH_NEURONX_DEBUG("Using TORCH_NEURONX_DISABLE_CONCAT_FOR_OVERLAP_IO_TENSOR=0 (disabled)");
  }

  // Load aliasing flag from environment variable
  // Default: disabled. Set TORCH_NEURONX_ENABLE_CONCATENATION_ALIASING=1 to enable
  const char* aliasing_env = std::getenv("TORCH_NEURONX_ENABLE_CONCATENATION_ALIASING");
  if (aliasing_env && std::string(aliasing_env) == "1") {
    aliasing_enabled_ = true;
    TORCH_NEURONX_DEBUG("Using TORCH_NEURONX_ENABLE_CONCATENATION_ALIASING=1 (enabled)");
  }
}

std::unique_ptr<IrNode> IrNodeManager::CreateConcatNode(IrNode* node1, IrNode* node2) {
  TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS("node1:", (node1 ? node1->op_name : "null"),
                                         "node2:", (node2 ? node2->op_name : "null"));

  if (!node1 || !node2) {
    TORCH_NEURONX_ERROR("Attempting to create concatenation node with null input",
                        "node1_valid:", (node1 != nullptr), "node2_valid:", (node2 != nullptr));
    throw std::invalid_argument("Both nodes must be non-null for concatenation");
  }

  // Check if both nodes are StableHLO type
  if (node1->ir_type == IrNodeType::STABLEHLO && node2->ir_type == IrNodeType::STABLEHLO) {
    TORCH_NEURONX_DEBUG("Validated nodes for StableHLO concatenation", "node1_op:", node1->op_name,
                        "node1_inputs:", node1->inputs.size(),
                        "node1_outputs:", node1->outputs.size(), "node2_op:", node2->op_name,
                        "node2_inputs:", node2->inputs.size(),
                        "node2_outputs:", node2->outputs.size());
    // Cast to StableHloNode and call StableHLO concat
    return CreateConcatNode(static_cast<StableHloNode*>(node1), static_cast<StableHloNode*>(node2));
  }

  TORCH_NEURONX_ERROR("Concatenation type mismatch - both nodes must be StableHLO type",
                      "node1_op:", node1->op_name, "node1_type:", static_cast<int>(node1->ir_type),
                      "node2_op:", node2->op_name, "node2_type:", static_cast<int>(node2->ir_type));
  throw std::runtime_error("Concat operation requires both nodes to be StableHLO type");
}

std::unique_ptr<StableHloNode> IrNodeManager::CreateConcatNode(StableHloNode* node1,
                                                               StableHloNode* node2) {
  if (!node1 || !node2) {
    TORCH_NEURONX_ERROR("Attempting StableHLO concatenation with null nodes",
                        "node1_valid:", (node1 != nullptr), "node2_valid:", (node2 != nullptr));
    throw std::invalid_argument("Both StableHloNodes must be non-null for concatenation");
  }

  TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS("node1:", node1->op_name, "node2:", node2->op_name);

  if (node1->cache_key.empty() || node2->cache_key.empty()) {
    TORCH_NEURONX_ERROR(
        "StableHLO concatenation validation failed - missing cache keys",
        "node1_op:", node1->op_name, "node1_cache_key_valid:", !node1->cache_key.empty(),
        "node2_op:", node2->op_name, "node2_cache_key_valid:", !node2->cache_key.empty());
    throw std::invalid_argument("Both nodes must have valid cache keys");
  }

  std::string cache_key = GenerateConcatenationCacheKey(node1, node2);
  TORCH_NEURONX_DEBUG("Generated concatenation cache key for StableHLO nodes",
                      "cache_key:", cache_key, "node1_op:", node1->op_name,
                      "node2_op:", node2->op_name);

  if (auto concat_op_it = ir_resource_cache_.find(cache_key);
      concat_op_it != ir_resource_cache_.end()) {
    TORCH_NEURONX_COUNTER("Concatenation.CacheHits", 1);
    TORCH_NEURONX_DEBUG("Found cached concatenation result - returning existing node",
                        "cache_key:", cache_key, "cached_node_op:", concat_op_it->second->op_name);
    auto ir_resource = concat_op_it->second.get();
    auto& mapping = ir_resource->merge_mapping;
    // Build tensor address arrays based on safe merge mapping (preserves all outputs)
    std::vector<void*> input_addrs;
    std::vector<void*> output_addrs;
    std::vector<at::neuron::TensorDataRef> input_data_refs;
    std::vector<at::neuron::TensorDataRef> output_data_refs;
    std::vector<at::neuron::TensorContext> input_contexts;
    std::vector<at::neuron::TensorContext> output_contexts;

    for (int i = 0; i < mapping->total_inputs; i++) {
      auto& mapping_entry = mapping->input_mapping[i];
      if (mapping_entry.first == 1) {
        // Input comes from node1
        input_addrs.push_back(node1->inputs[mapping_entry.second]);
        if (mapping_entry.second < node1->input_data_refs.size()) {
          input_data_refs.push_back(node1->input_data_refs[mapping_entry.second]);
        }
        if (mapping_entry.second < node1->input_contexts.size()) {
          input_contexts.push_back(node1->input_contexts[mapping_entry.second]);
        }
      } else {
        // Input comes from node2
        input_addrs.push_back(node2->inputs[mapping_entry.second]);
        if (mapping_entry.second < node2->input_data_refs.size()) {
          input_data_refs.push_back(node2->input_data_refs[mapping_entry.second]);
        }
        if (mapping_entry.second < node2->input_contexts.size()) {
          input_contexts.push_back(node2->input_contexts[mapping_entry.second]);
        }
      }
    }

    for (int i = 0; i < mapping->total_outputs; i++) {
      auto& mapping_entry = mapping->output_mapping[i];
      if (mapping_entry.first == 1) {
        // Output goes to node1's original output (all outputs preserved)
        output_addrs.push_back(node1->outputs[mapping_entry.second]);
        if (mapping_entry.second < node1->output_data_refs.size()) {
          output_data_refs.push_back(node1->output_data_refs[mapping_entry.second]);
        }
        if (mapping_entry.second < node1->output_contexts.size()) {
          output_contexts.push_back(node1->output_contexts[mapping_entry.second]);
        }
      } else {
        // Output goes to node2's original output (all outputs preserved)
        output_addrs.push_back(node2->outputs[mapping_entry.second]);
        if (mapping_entry.second < node2->output_data_refs.size()) {
          output_data_refs.push_back(node2->output_data_refs[mapping_entry.second]);
        }
        if (mapping_entry.second < node2->output_contexts.size()) {
          output_contexts.push_back(node2->output_contexts[mapping_entry.second]);
        }
      }
    }
    auto result_node = std::make_unique<StableHloNode>(
        ir_resource->op_name, std::move(cache_key), ir_resource->module, ir_resource->ir_serialized,
        std::move(input_addrs), std::move(output_addrs), ir_resource->has_collectives);
    result_node->input_data_refs = std::move(input_data_refs);
    result_node->output_data_refs = std::move(output_data_refs);
    result_node->input_contexts = std::move(input_contexts);
    result_node->output_contexts = std::move(output_contexts);
    return result_node;
  }

  TORCH_NEURONX_COUNTER("Concatenation.CacheMisses", 1);
  TORCH_NEURONX_DEBUG("cache miss", "cache_key:", cache_key);

  std::unique_ptr<StableHloNode> concat_node;
  try {
    // Call in-order merge function optimized for sequential execution (node1 before node2)
    concat_node = mergeStableHLOModulesWithTensorsInOrder(node1, node2, context.get());

    if (!concat_node) {
      TORCH_NEURONX_WARN("StableHLO module merge operation failed", "node1_op:", node1->op_name,
                         "node1_cache_key:", node1->cache_key, "node2_op:", node2->op_name,
                         "node2_cache_key:", node2->cache_key);
      return nullptr;
    }

    concat_node->cache_key = cache_key;
    concat_node->op_name = GenerateConcatenationOpName(node1, node2);

    TORCH_NEURONX_DEBUG(
        "Successfully created concatenated StableHLO node",
        "concatenated_op:", concat_node->op_name, "concatenated_cache_key:", cache_key,
        "original_node1:", node1->op_name, "original_node2:", node2->op_name,
        "inputs_count:", concat_node->inputs.size(), "outputs_count:", concat_node->outputs.size(),
        "has_collectives:", concat_node->has_collectives);

    CacheIrResource(concat_node.get(), cache_key, std::move(concat_node->merge_mapping));

    return concat_node;
  } catch (const std::exception& e) {
    TORCH_NEURONX_INFO("Exception occurred during StableHLO concatenation",
                       "error_message:", e.what(), "node1_op:", node1->op_name,
                       "node1_cache_key:", node1->cache_key, "node2_op:", node2->op_name,
                       "node2_cache_key:", node2->cache_key);
    return nullptr;
  }
}

/**
 * @brief Generate cache key for concatenated IR nodes with aliasing information
 *
 * Creates a unique cache key by combining node cache keys and aliasing patterns.
 * Different aliasing patterns produce different cache keys to ensure correct
 * cached module reuse.
 *
 * @param node1 First IR node (executes before node2)
 * @param node2 Second IR node (executes after node1)
 * @return Cache key format: "<node1_key>|<node2_key>|alias:<aliases>,"
 *         where aliases are: "1i<idx>->1o<idx>," (node1 self-aliasing),
 *                           "2i<idx>->2o<idx>," (node2 self-aliasing),
 *                           "1i<idx>->2o<idx>," (cross-node aliasing)
 *
 * Example: "key1|key2|alias:1i0->1o0,2i1->2o1,1i0->2o1,"
 */
std::string IrNodeManager::GenerateConcatenationCacheKey(const IrNode* node1, const IrNode* node2) {
  if (!node1 || !node2) {
    TORCH_NEURONX_ERROR("generate_concatenation_cache_key called with null nodes");
    throw std::invalid_argument("Both nodes must be non-null to generate cache key");
  }

  // Base key: concatenate individual node cache keys
  std::string cache_key = node1->cache_key + "|" + node2->cache_key;

  // Include aliasing flag in cache key to prevent reuse across flag changes
  cache_key += "|alias_enabled:" + std::string(aliasing_enabled_ ? "1" : "0");

  if (!aliasing_enabled_) {
    TORCH_NEURONX_DEBUG("Generated cache key (aliasing disabled):", cache_key);
    return cache_key;
  }

  // Append aliasing information to distinguish different memory sharing patterns
  std::string alias_suffix = "|alias:";

  // Node1 self-aliasing: detect when node1's input and output share same address
  // Format: "1i<input_idx>->1o<output_idx>,"
  for (size_t out_idx = 0; out_idx < node1->outputs.size(); ++out_idx) {
    void* out_addr = node1->outputs[out_idx];
    if (!out_addr) continue;
    for (size_t in_idx = 0; in_idx < node1->inputs.size(); ++in_idx) {
      if (node1->inputs[in_idx] == out_addr) {
        alias_suffix += "1i" + std::to_string(in_idx) + "->1o" + std::to_string(out_idx) + ",";
      }
    }
  }

  // Node2 self-aliasing: detect when node2's input and output share same address
  // Format: "2i<input_idx>->2o<output_idx>,"
  for (size_t out_idx = 0; out_idx < node2->outputs.size(); ++out_idx) {
    void* out_addr = node2->outputs[out_idx];
    if (!out_addr) continue;
    for (size_t in_idx = 0; in_idx < node2->inputs.size(); ++in_idx) {
      if (node2->inputs[in_idx] == out_addr) {
        alias_suffix += "2i" + std::to_string(in_idx) + "->2o" + std::to_string(out_idx) + ",";
      }
    }
  }

  // Cross-node aliasing: detect when node1's input shares address with node2's output
  // Format: "1i<input_idx>->2o<output_idx>,"
  // Note: Only check node1 input -> node2 output since if node1 output has the same address
  // with node2 input, it's either node2 depends on node1, in which case the node2 input won't
  // appear as top-level module input, or node1's output is deallocated, which will be removed from
  // top-level module output by skip intermediate handling.
  for (size_t out_idx = 0; out_idx < node2->outputs.size(); ++out_idx) {
    void* out_addr = node2->outputs[out_idx];
    if (!out_addr) continue;
    for (size_t in_idx = 0; in_idx < node1->inputs.size(); ++in_idx) {
      if (node1->inputs[in_idx] == out_addr) {
        alias_suffix += "1i" + std::to_string(in_idx) + "->2o" + std::to_string(out_idx) + ",";
      }
    }
  }

  cache_key += alias_suffix;
  TORCH_NEURONX_DEBUG("Generated cache key with aliasing:", cache_key);
  return cache_key;
}

std::string IrNodeManager::GenerateConcatenationOpName(const IrNode* node1, const IrNode* node2) {
  if (!node1 || !node2) {
    TORCH_NEURONX_ERROR("generate_concatenation_op_name called with null nodes");
    throw std::invalid_argument("Both nodes must be non-null to generate op name");
  }
  std::string op_name = node1->op_name + "|" + node2->op_name;
  TORCH_NEURONX_DEBUG("Generated concatenated op name:", op_name);
  return op_name;
}

std::string IrNodeManager::GeneratePruningCacheKey(
    const std::string& base_cache_key, const std::vector<size_t>& pruned_original_indices) {
  if (base_cache_key.empty()) {
    TORCH_NEURONX_ERROR("GeneratePruningCacheKey called with empty base cache key");
    throw std::invalid_argument("Base cache key cannot be empty");
  }

  // If no indices pruned, return base key unchanged
  if (pruned_original_indices.empty()) {
    return base_cache_key;
  }

  // Build suffix with sorted indices for consistent cache keys
  // Sort to ensure deterministic key regardless of pruning order
  std::vector<size_t> sorted_indices = pruned_original_indices;
  std::sort(sorted_indices.begin(), sorted_indices.end());

  std::string cache_key = base_cache_key + "|pruned";
  for (size_t idx : sorted_indices) {
    cache_key += "_" + std::to_string(idx);
  }

  TORCH_NEURONX_DEBUG("Generated pruning cache key", "base_key:", base_cache_key,
                      "pruned_indices_count:", pruned_original_indices.size(),
                      "result_key:", cache_key);
  return cache_key;
}

std::unique_ptr<IrNode> IrNodeManager::CreateNodeFromOperationContext(
    const at::neuron::OperationContext* operation_context) {
  TORCH_NEURONX_TRACE_FUNCTION();

  if (!operation_context) {
    TORCH_NEURONX_ERROR("Attempting to create IrNode from null OperationContext");
    throw std::invalid_argument("OperationContext cannot be null");
  }

  if (!operation_context->kernel_execution) {
    TORCH_NEURONX_ERROR("OperationContext validation failed - null kernel_execution",
                        "context_valid:", (context != nullptr));
    throw std::invalid_argument("OperationContext must have valid kernel_execution");
  }

  auto& kernel_execution = operation_context->kernel_execution;

  // Cast to XLACompilableKernelExecution to access XLA-specific methods
  auto* xla_kernel =
      dynamic_cast<at::neuron::XLACompilableKernelExecution*>(kernel_execution.get());
  if (!xla_kernel) {
    TORCH_NEURONX_ERROR("Kernel execution is not XLACompilableKernelExecution",
                        "op_name:", kernel_execution->GetOpName());
    throw std::runtime_error(
        "create_node_from_operation_context requires XLACompilableKernelExecution");
  }

  const std::string& cache_key = xla_kernel->GetCompileKey();

  TORCH_NEURONX_DEBUG("Creating IrNode from validated OperationContext",
                      "op_name:", xla_kernel->GetOpName(), "cache_key:", cache_key,
                      "ir_bytes_size:", xla_kernel->GetHloBytes().size(),
                      "has_collectives:", xla_kernel->HasCollectives());

  if (auto node = ir_resource_cache_.find(cache_key); node != ir_resource_cache_.end()) {
    TORCH_NEURONX_COUNTER("Concatenation.CacheHits", 1);
    TORCH_NEURONX_DEBUG("Found existing IrNode in cache - returning cached instance",
                        "cache_key:", cache_key, "cached_op:", node->second->op_name);
    auto ir_resource = node->second.get();
    std::vector<void*> input_data_ptr;
    for (const auto& input : xla_kernel->GetSrcPtrs()) {
      input_data_ptr.push_back(input);
    }

    std::vector<void*> output_data_ptr;
    for (const auto& output : xla_kernel->GetDstPtrs()) {
      output_data_ptr.push_back(output);
    }
    auto result_node = std::make_unique<StableHloNode>(
        ir_resource->op_name, std::move(cache_key), ir_resource->module, ir_resource->ir_serialized,
        std::move(input_data_ptr), std::move(output_data_ptr), ir_resource->has_collectives);

    // Copy TensorDataRef from the XLA kernel (preserves NeuronTensorPtr for NRT execution).
    result_node->input_data_refs = xla_kernel->GetSrcRefs();
    result_node->output_data_refs = xla_kernel->GetDstRefs();

    // Copy TensorContexts from the XLA kernel
    result_node->input_contexts = xla_kernel->GetInputContexts();
    result_node->output_contexts = xla_kernel->GetOutputContexts();

    return result_node;
  }

  // Do this because std::make_unique doesn't work when using a private constructor
  // even if this is a friend class
  auto node = std::unique_ptr<StableHloNode>(new StableHloNode(*operation_context, context.get()));

  // Copy TensorDataRef from the XLA kernel (preserves NeuronTensorPtr for NRT execution).
  node->input_data_refs = xla_kernel->GetSrcRefs();
  node->output_data_refs = xla_kernel->GetDstRefs();

  // Copy TensorContexts from the XLA kernel
  node->input_contexts = xla_kernel->GetInputContexts();
  node->output_contexts = xla_kernel->GetOutputContexts();

  CacheIrResource(node.get(), cache_key);

  TORCH_NEURONX_DEBUG("Cache miss. Created new StableHloNode and added to cache",
                      "op_name:", xla_kernel->GetOpName(), "cache_key:", cache_key,
                      "inputs_count:", node->inputs.size(), "outputs_count:", node->outputs.size(),
                      "has_collectives:", node->has_collectives,
                      "input_data_refs_count:", node->input_data_refs.size(),
                      "output_data_refs_count:", node->output_data_refs.size(),
                      "cache_size:", ir_resource_cache_.size());
  return node;
}

IrResource* IrNodeManager::GetByKey(std::string key) {
  if (key.empty()) {
    TORCH_NEURONX_WARN("Attempting to retrieve IrNode with empty cache key",
                       "cache_size:", ir_resource_cache_.size());
    return nullptr;
  }

  TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS("key:", key);

  if (auto it = ir_resource_cache_.find(key); it != ir_resource_cache_.end()) {
    TORCH_NEURONX_DEBUG("Successfully retrieved IrNode from cache", "cache_key:", key,
                        "found_op:", it->second->op_name);
    return it->second.get();
  }
  TORCH_NEURONX_DEBUG("IrNode not found in cache", "requested_key:", key,
                      "cache_size:", ir_resource_cache_.size());
  return nullptr;
}

size_t IrNodeManager::Evict(std::string key) {
  TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS("key:", key);

  size_t cache_size_before = ir_resource_cache_.size();
  size_t erased = ir_resource_cache_.erase(key);
  if (erased > 0) {
    TORCH_NEURONX_DEBUG("Successfully evicted IrNode from cache", "evicted_key:", key,
                        "cache_size_before:", cache_size_before,
                        "cache_size_after:", ir_resource_cache_.size());
  } else {
    TORCH_NEURONX_DEBUG("No IrNode found to evict from cache", "requested_key:", key,
                        "current_cache_size:", ir_resource_cache_.size());
  }
  return erased;
}

namespace {

/**
 * @brief Validates that HLO node addresses are not null and not in the pruned set
 *
 * This function ensures:
 * 1. No null input addresses
 * 2. No null output addresses
 * 3. No inputs reading from deallocated (pruned) addresses (use-after-free check)
 *
 * @param node The HLO node to validate
 * @param pruned_addresses Set of addresses that have been deallocated
 * @throws std::runtime_error if validation fails
 */
void ValidateHloNodeAddresses(const IrNode* node,
                              const std::unordered_set<void*>& pruned_addresses) {
  // Assert no null input addresses
  for (size_t i = 0; i < node->inputs.size(); ++i) {
    if (node->inputs[i] == nullptr) {
      TORCH_NEURONX_ERROR("[CONCAT_ASSERTION_FAILED] HLO node has null input address!",
                          "op_name:", node->op_name, "input_index:", i,
                          "total_inputs:", node->inputs.size());
      std::ostringstream oss;
      oss << "IrNodeManager::ConcatAll: HLO node '" << node->op_name
          << "' has null input address at index " << i << ". This indicates invalid tensor data.";
      throw std::runtime_error(oss.str());
    }
  }

  // Assert no null output addresses
  for (size_t i = 0; i < node->outputs.size(); ++i) {
    if (node->outputs[i] == nullptr) {
      TORCH_NEURONX_ERROR("[CONCAT_ASSERTION_FAILED] HLO node has null output address!",
                          "op_name:", node->op_name, "output_index:", i,
                          "total_outputs:", node->outputs.size());
      std::ostringstream oss;
      oss << "IrNodeManager::ConcatAll: HLO node '" << node->op_name
          << "' has null output address at index " << i << ". This indicates invalid tensor data.";
      throw std::runtime_error(oss.str());
    }
  }

  // Assert that none of the inputs are in pruned_addresses (use-after-free check)
  for (void* input_addr : node->inputs) {
    if (pruned_addresses.find(input_addr) != pruned_addresses.end()) {
      TORCH_NEURONX_ERROR("[CONCAT_ASSERTION_FAILED] HLO node input reads from pruned address!",
                          "op_name:", node->op_name, "pruned_input_addr:", input_addr,
                          "pruned_addresses_count:", pruned_addresses.size());
      std::ostringstream oss;
      oss << "IrNodeManager::ConcatAll: HLO node '" << node->op_name << "' has input at address "
          << input_addr << " which was deallocated (pruned) but not reallocated. "
          << "This indicates a tensor lifetime tracking error.";
      throw std::runtime_error(oss.str());
    }
  }
}

}  // anonymous namespace

std::vector<size_t> IrNodeManager::FindIndicesToPrune(const StableHloNode* node, void* address) {
  std::vector<size_t> pruned_indices;
  for (size_t i = 0; i < node->outputs.size(); ++i) {
    if (node->outputs[i] == address) {
      pruned_indices.push_back(i);
    }
  }
  return pruned_indices;
}

void IrNodeManager::UpdateNodeOutputVectors(StableHloNode* node,
                                            const std::unordered_set<size_t>& pruned_indices) {
  std::vector<void*> new_outputs;
  std::vector<at::neuron::TensorDataRef> new_output_data_refs;
  std::vector<at::neuron::TensorContext> new_output_contexts;

  for (size_t i = 0; i < node->outputs.size(); ++i) {
    if (pruned_indices.find(i) == pruned_indices.end()) {
      new_outputs.push_back(node->outputs[i]);
      if (i < node->output_data_refs.size()) {
        new_output_data_refs.push_back(node->output_data_refs[i]);
      }
      if (i < node->output_contexts.size()) {
        new_output_contexts.push_back(node->output_contexts[i]);
      }
    }
  }

  node->outputs = std::move(new_outputs);
  node->output_data_refs = std::move(new_output_data_refs);
  node->output_contexts = std::move(new_output_contexts);
}

void IrNodeManager::SerializeMlirModule(StableHloNode* node) {
  std::string pruned_mlir_string;
  llvm::raw_string_ostream mlir_stream(pruned_mlir_string);
  (*node->module)->print(mlir_stream);
  mlir_stream.flush();
  node->ir_serialized.assign(pruned_mlir_string.begin(), pruned_mlir_string.end());
}

void IrNodeManager::CacheIrResource(StableHloNode* node, const std::string& cache_key,
                                    std::unique_ptr<MergeMapping> merge_mapping) {
  // Invariant: callers must explicitly provide merge_mapping for concatenated nodes.
  if (!merge_mapping && node->merge_mapping) {
    throw std::runtime_error(
        "CacheIrResource: merge_mapping not provided but node has one. "
        "Callers must explicitly pass merge_mapping for concatenated nodes. "
        "cache_key=" +
        cache_key);
  }

  auto ir_resource = std::make_unique<IrResource>(
      node->module, std::move(merge_mapping), node->op_name, node->cache_key, node->ir_serialized,
      node->ir_type, node->has_collectives);
  ir_resource_cache_[cache_key] = std::move(ir_resource);
}

bool IrNodeManager::PruneMlirFunction(StableHloNode* node, void* dealloc_address) {
  // Clone the module so in-place pruning does not mutate shared cached entries.
  auto original_module = node->module;
  node->module = std::make_shared<mlir::OwningOpRef<mlir::ModuleOp>>((*node->module)->clone());

  // Build indices to keep (all except those matching dealloc_address)
  std::unordered_set<size_t> indicesToKeep;
  for (size_t i = 0; i < node->outputs.size(); ++i) {
    if (node->outputs[i] != dealloc_address) {
      indicesToKeep.insert(i);
    }
  }

  // Find the 'main' function in the MLIR module
  auto funcOp = (*node->module)->lookupSymbol<mlir::func::FuncOp>("main");
  if (!funcOp) {
    node->module = std::move(original_module);
    TORCH_NEURONX_WARN("Could not find 'main' function in module during MLIR prune");
    return false;
  }

  // Call the low-level index-based pruning API
  try {
    pruneFunctionResultsByIndices(funcOp, indicesToKeep);
  } catch (const std::exception& e) {
    node->module = std::move(original_module);
    TORCH_NEURONX_WARN("Failed to prune MLIR function", "error:", e.what());
    return false;
  }

  // Remap mhlo.input_output_alias output indices after pruning. Drop entries for
  // pruned outputs. Update both FuncOp and ModuleOp (alias can be on either).
  auto remapAliasAttr = [&](mlir::Operation* op) {
    if (!op->hasAttr("mhlo.input_output_alias")) return;

    // Build old→new index mapping for kept outputs
    std::unordered_map<size_t, size_t> old_to_new;
    size_t new_idx = 0;
    for (size_t i = 0; i < node->outputs.size(); i++) {
      if (indicesToKeep.count(i)) {
        old_to_new[i] = new_idx++;
      }
    }

    mlir::OpBuilder builder(op->getContext());
    auto oldAliasAttr = op->getAttrOfType<mlir::ArrayAttr>("mhlo.input_output_alias");
    llvm::SmallVector<mlir::Attribute> newAliasList;

    for (auto attr : oldAliasAttr) {
      auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(attr);
      if (!dictAttr) continue;

      auto outputIdxAttr = dictAttr.getAs<mlir::DenseI64ArrayAttr>("output_index");
      if (!outputIdxAttr || outputIdxAttr.empty()) continue;

      size_t old_out_idx = static_cast<size_t>(outputIdxAttr[0]);

      // If this aliased output was pruned, drop the entry
      if (old_to_new.find(old_out_idx) == old_to_new.end()) {
        TORCH_NEURONX_DEBUG("Removing alias entry for pruned output[", old_out_idx, "]");
        continue;
      }

      // Remap to new index
      size_t new_out_idx = old_to_new[old_out_idx];
      auto aliasInner = dictAttr.getAs<mlir::DictionaryAttr>("alias");
      auto newOutputIdxAttr = builder.getDenseI64ArrayAttr({static_cast<int64_t>(new_out_idx)});
      auto newEntry =
          builder.getDictionaryAttr({builder.getNamedAttr("alias", aliasInner),
                                     builder.getNamedAttr("output_index", newOutputIdxAttr)});
      newAliasList.push_back(newEntry);
    }

    if (newAliasList.empty()) {
      op->removeAttr("mhlo.input_output_alias");
    } else {
      op->setAttr("mhlo.input_output_alias", builder.getArrayAttr(newAliasList));
    }
  };

  // Remap alias on the FuncOp (set by ConcatAll overlap detection)
  remapAliasAttr(funcOp.getOperation());
  // Remap alias on the ModuleOp (set by OpConcatUtils during merge)
  remapAliasAttr((*node->module)->getOperation());

  return true;
}

std::unique_ptr<MergeMapping> IrNodeManager::RebuildMergeMappingAfterPrune(
    const MergeMapping& old_mapping, const std::unordered_set<size_t>& pruned_set) {
  // Start from a full copy — input_mapping and other fields are unchanged by pruning.
  auto new_mapping = std::make_unique<MergeMapping>(old_mapping);

  // Rebuild output_mapping: skip pruned indices, re-index remaining sequentially
  new_mapping->output_mapping.clear();
  size_t expected_remaining = old_mapping.total_outputs - pruned_set.size();
  new_mapping->output_mapping.reserve(expected_remaining);
  size_t new_idx = 0;
  for (size_t old_idx = 0; old_idx < old_mapping.total_outputs; ++old_idx) {
    if (pruned_set.count(old_idx)) {
      continue;
    }
    auto it = old_mapping.output_mapping.find(old_idx);
    if (it == old_mapping.output_mapping.end()) {
      throw std::runtime_error(
          "RebuildMergeMappingAfterPrune: Missing output_mapping for non-pruned index " +
          std::to_string(old_idx));
    }
    new_mapping->output_mapping[new_idx] = it->second;
    ++new_idx;
  }
  new_mapping->total_outputs = new_idx;
  return new_mapping;
}

void IrNodeManager::ValidatePostPruneConsistency(const StableHloNode* node) {
  auto funcOp = (*node->module)->lookupSymbol<mlir::func::FuncOp>("main");
  if (!funcOp) {
    throw std::runtime_error(
        "Post-prune validation: Cannot find 'main' function in pruned module, cache_key=" +
        node->cache_key);
  }
  size_t mlir_results = funcOp.getNumResults();
  size_t output_count = node->outputs.size();
  if (mlir_results != output_count) {
    throw std::runtime_error("Post-prune validation: MLIR result count (" +
                             std::to_string(mlir_results) + ") != outputs size (" +
                             std::to_string(output_count) + "), cache_key=" + node->cache_key);
  }
  if (!node->output_data_refs.empty() && node->output_data_refs.size() != output_count) {
    throw std::runtime_error("Post-prune validation: output_data_refs size (" +
                             std::to_string(node->output_data_refs.size()) + ") != outputs size (" +
                             std::to_string(output_count) + "), cache_key=" + node->cache_key);
  }
  if (!node->output_contexts.empty() && node->output_contexts.size() != output_count) {
    throw std::runtime_error("Post-prune validation: output_contexts size (" +
                             std::to_string(node->output_contexts.size()) + ") != outputs size (" +
                             std::to_string(output_count) + "), cache_key=" + node->cache_key);
  }
}

bool IrNodeManager::PruneSingleOutput(StableHloNode* node, void* dealloc_address,
                                      PruneSource source) {
  if (!node || !node->module) {
    TORCH_NEURONX_WARN("Invalid node or module for pruning");
    return false;
  }

  auto pruned_indices = FindIndicesToPrune(node, dealloc_address);

  if (pruned_indices.empty()) {
    return true;
  }

  if (pruned_indices.size() >= node->outputs.size()) {
    TORCH_NEURONX_INFO("Would prune all outputs — aborting concatenation",
                       "dealloc_addr:", dealloc_address, "op_name:", node->op_name,
                       "cache_key:", node->cache_key, "outputs_count:", node->outputs.size(),
                       "pruned_count:", pruned_indices.size());
    return false;
  }

  // Save the original cache key before any mutations for cache lookup
  std::string original_cache_key = node->cache_key;
  // Differentiate prune source in the key to prevent collisions between
  // ConcatAll skip-intermediates and double-buffer lookahead pruning.
  std::string source_prefix = (source == PruneSource::kDoubleBuffer) ? "|db_pruned" : "|pruned";
  std::vector<size_t> sorted_indices = pruned_indices;
  std::sort(sorted_indices.begin(), sorted_indices.end());
  std::string new_cache_key = original_cache_key + source_prefix;
  for (size_t idx : sorted_indices) {
    new_cache_key += "_" + std::to_string(idx);
  }
  std::unordered_set<size_t> pruned_set(pruned_indices.begin(), pruned_indices.end());

  // Check cache for already-pruned MLIR
  if (auto* cached = GetByKey(new_cache_key)) {
    node->module = cached->module;
    node->ir_serialized = cached->ir_serialized;
    node->cache_key = new_cache_key;
    UpdateNodeOutputVectors(node, pruned_set);

    auto funcOp_hit = (*node->module)->lookupSymbol<mlir::func::FuncOp>("main");
    if (funcOp_hit && funcOp_hit.getNumResults() != node->outputs.size()) {
      throw std::runtime_error("PruneSingleOutput cache-hit mismatch: mlir_results=" +
                               std::to_string(funcOp_hit.getNumResults()) +
                               " outputs=" + std::to_string(node->outputs.size()) +
                               " cache_key=" + new_cache_key);
    }
    return true;
  }

  // Clone + prune MLIR (PruneMlirFunction handles clone internally)
  if (!PruneMlirFunction(node, dealloc_address)) {
    return false;
  }

  // Rebuild MergeMapping from the original (pre-prune) cache entry
  std::unique_ptr<MergeMapping> new_merge_mapping;
  if (auto* original_cached = GetByKey(original_cache_key)) {
    if (original_cached->merge_mapping) {
      new_merge_mapping =
          RebuildMergeMappingAfterPrune(*original_cached->merge_mapping, pruned_set);
    }
  }

  UpdateNodeOutputVectors(node, pruned_set);
  SerializeMlirModule(node);
  ValidatePostPruneConsistency(node);

  node->cache_key = new_cache_key;
  CacheIrResource(node, new_cache_key, std::move(new_merge_mapping));

  return true;
}

/**
 * @brief Validate that aliasing attributes match actual input-output address relationships
 *
 * Checks if the concatenated StableHLO module has correct aliasing attributes when
 * inputs and outputs share the same address.
 *
 * TODO: Move to the util class for debugging purpose
 */
static void validateAliasingAttributes(IrNode* node) {
  static bool debug_enabled = []() {
    const char* env = std::getenv("TORCH_NEURONX_DEBUG_CONCATENATION");
    return env && std::string(env) == "1";
  }();
  if (!debug_enabled) return;

  if (!node) return;

  auto* stablehlo_node = dynamic_cast<StableHloNode*>(node);
  if (!stablehlo_node || !stablehlo_node->module) return;

  // Find input-output address matches
  std::vector<std::pair<size_t, size_t>> expected_aliases;  // (input_idx, output_idx)
  for (size_t out_idx = 0; out_idx < node->outputs.size(); ++out_idx) {
    void* out_addr = node->outputs[out_idx];
    if (!out_addr) continue;

    for (size_t in_idx = 0; in_idx < node->inputs.size(); ++in_idx) {
      void* in_addr = node->inputs[in_idx];
      if (in_addr == out_addr) {
        expected_aliases.emplace_back(in_idx, out_idx);
      }
    }
  }

  if (expected_aliases.empty()) {
    return;  // No aliasing expected, validation passes
  }

  // Parse aliasing attributes from module
  auto& module = *(stablehlo_node->module);
  if (!module) return;

  auto aliasAttr = (*module)->getAttr("mhlo.input_output_alias");

  if (!aliasAttr) {
    TORCH_NEURONX_ERROR(
        "ALIASING VALIDATION FAILED: Expected aliasing but no mhlo.input_output_alias attribute "
        "found");
    TORCH_NEURONX_ERROR("  Expected ", expected_aliases.size(), " alias relationships:");
    for (const auto& [in_idx, out_idx] : expected_aliases) {
      TORCH_NEURONX_ERROR("    input[", in_idx, "] -> output[", out_idx,
                          "] addr=", node->inputs[in_idx]);
    }
    TORCH_NEURONX_ERROR("  StableHLO module:");
    std::string module_str;
    llvm::raw_string_ostream stream(module_str);
    module->print(stream);
    TORCH_NEURONX_ERROR(stream.str());
    return;
  }

  // Parse actual aliasing pairs from attribute
  auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(aliasAttr);
  if (!arrayAttr) {
    TORCH_NEURONX_ERROR("ALIASING VALIDATION FAILED: mhlo.input_output_alias is not an ArrayAttr");
    return;
  }

  std::set<std::pair<size_t, size_t>> actual_aliases;
  for (auto attr : arrayAttr) {
    auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(attr);
    if (!dictAttr) continue;

    // Get "alias" inner dict and "output_index" array
    auto aliasInnerAttr = dictAttr.get("alias");
    auto outputIndexAttr = dictAttr.get("output_index");

    if (!aliasInnerAttr || !outputIndexAttr) continue;

    auto aliasInnerDict = mlir::dyn_cast<mlir::DictionaryAttr>(aliasInnerAttr);
    auto outputIndexArray = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(outputIndexAttr);

    if (!aliasInnerDict || !outputIndexArray) continue;

    // Get parameter_index from inner dict
    auto paramNumberAttr = aliasInnerDict.get("parameter_number");
    if (!paramNumberAttr) continue;

    auto paramNumberInt = mlir::dyn_cast<mlir::IntegerAttr>(paramNumberAttr);
    if (!paramNumberInt) continue;

    size_t input_idx = paramNumberInt.getInt();
    size_t output_idx = outputIndexArray[0];

    actual_aliases.insert({input_idx, output_idx});
  }

  // Compare expected vs actual
  std::set<std::pair<size_t, size_t>> expected_set(expected_aliases.begin(),
                                                   expected_aliases.end());

  if (actual_aliases != expected_set) {
    TORCH_NEURONX_ERROR("ALIASING VALIDATION FAILED: Aliasing pairs mismatch");
    TORCH_NEURONX_ERROR("  Expected ", expected_set.size(), " aliases:");
    for (const auto& [in_idx, out_idx] : expected_set) {
      TORCH_NEURONX_ERROR("    input[", in_idx, "] -> output[", out_idx,
                          "] addr=", node->inputs[in_idx]);
    }
    TORCH_NEURONX_ERROR("  Actual ", actual_aliases.size(), " aliases:");
    for (const auto& [in_idx, out_idx] : actual_aliases) {
      TORCH_NEURONX_ERROR("    input[", in_idx, "] -> output[", out_idx, "]");
    }
    TORCH_NEURONX_ERROR("  StableHLO module:");
    std::string module_str;
    llvm::raw_string_ostream stream(module_str);
    module->print(stream);
    TORCH_NEURONX_ERROR(stream.str());
    return;
  }

  TORCH_NEURONX_DEBUG("Aliasing validation PASSED: ", expected_aliases.size(), " aliases match");
}

void IrNodeManager::DiscardMergedNode(ConcatState& state) {
  if (state.merged) {
    Evict(state.merged->cache_key);
  }
  state.merged = nullptr;
  state.merged_owned.reset();
}

void IrNodeManager::ProcessIrNode(IrNode* node, ConcatState& state) {
  if (node->ir_type == IrNodeType::ALLOC_HINT) {
    // ALLOC: Remove address from pruned set (it's been reallocated)
    auto* hint_node = static_cast<AllocHintNode*>(node);
    state.pruned_addresses.erase(hint_node->alloc_address);
    TORCH_NEURONX_DEBUG("[TYPE_DISPATCH] Processed alloc hint - removed from pruned set",
                        "addr:", hint_node->alloc_address,
                        "pruned_addresses_remaining:", state.pruned_addresses.size());

  } else if (node->ir_type == IrNodeType::DEALLOC_HINT) {
    // DEALLOC: Add address to pruned set and prune merged node immediately
    auto* hint_node = static_cast<DeallocHintNode*>(node);
    state.pruned_addresses.insert(hint_node->dealloc_address);
    TORCH_NEURONX_DEBUG("[TYPE_DISPATCH] Processed dealloc hint - added to pruned set",
                        "addr:", hint_node->dealloc_address,
                        "pruned_addresses_count:", state.pruned_addresses.size());

    // Prune the merged node immediately if we have one and skip_intermediate is enabled
    if (skip_intermediate_enabled_ && state.merged != nullptr) {
      bool prune_success =
          PruneSingleOutput(static_cast<StableHloNode*>(state.merged), hint_node->dealloc_address);
      if (!prune_success) {
        DiscardMergedNode(state);
        throw std::runtime_error("Pruning failed for dealloc_address");
      }
    }

  } else {
    // kHLO: Validate inputs then concat
    ValidateHloNodeAddresses(node, state.pruned_addresses);

    if (state.merged == nullptr) {
      // First kHLO - use as starter (no concat yet)
      state.merged = node;
      TORCH_NEURONX_DEBUG("[TYPE_DISPATCH] First kHLO - using as starter",
                          "op_name:", node->op_name);
    } else {
      // Concat immediately with previous merged result
      TORCH_NEURONX_DEBUG("[TYPE_DISPATCH] Concatenating kHLO", "current_op:", node->op_name,
                          "merged_op:", state.merged->op_name);

      state.merged_owned = CreateConcatNode(state.merged, node);
      if (!state.merged_owned) {
        // Concatenation failed - throw exception to trigger fallback in ConcatenationCore
        throw std::runtime_error("Concatenation failed for op: " + node->op_name);
      }
      state.merged = state.merged_owned.get();
    }

    state.original_hlo_nodes.push_back(node);
  }
}

IrConcatResult IrNodeManager::ConcatAll(const std::list<IrNode*>& input_ir_list) {
  TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS("input_ir_list_size:", input_ir_list.size());

  IrConcatResult res;

  // Handle empty input
  if (input_ir_list.empty()) {
    TORCH_NEURONX_DEBUG("Empty input list - nothing to concatenate");
    return res;
  }

  // Initialize state for incremental concatenation
  ConcatState state;

  // Process each node with type-based dispatch
  for (auto* node : input_ir_list) {
    ProcessIrNode(node, state);
  }

  // Build result
  if (state.merged_owned) {
    // Annotate remaining input-output address overlap with mhlo.input_output_alias
    // on the ModuleOp so the compiler knows these buffers are shared.
    bool has_overlap = false;
    if (check_overlap_enabled_) {
      auto* hlo_node = static_cast<StableHloNode*>(state.merged);
      std::unordered_set<void*> input_addrs;
      for (void* addr : hlo_node->inputs) {
        if (addr) input_addrs.insert(addr);
      }

      std::vector<std::pair<size_t, size_t>> overlap_pairs;  // (output_idx, input_idx)
      for (size_t out_idx = 0; out_idx < hlo_node->outputs.size(); out_idx++) {
        void* out_addr = hlo_node->outputs[out_idx];
        if (!out_addr) continue;
        if (input_addrs.count(out_addr)) {
          // Find the matching input index
          for (size_t in_idx = 0; in_idx < hlo_node->inputs.size(); in_idx++) {
            if (hlo_node->inputs[in_idx] == out_addr) {
              overlap_pairs.emplace_back(out_idx, in_idx);
              break;
            }
          }
          has_overlap = true;
        }
      }

      if (has_overlap && aliasing_enabled_) {
        // Add mhlo.input_output_alias for overlapping pairs on the ModuleOp.
        // Convention: alias attribute lives on the ModuleOp (matching OpConcatUtils
        // addAliasingForConcatenatedModule and what neuronx-cc reads).
        mlir::OpBuilder builder((*hlo_node->module)->getContext());
        llvm::SmallVector<mlir::Attribute> aliasList;

        for (auto& [out_idx, in_idx] : overlap_pairs) {
          auto paramNumAttr = builder.getI64IntegerAttr(static_cast<int64_t>(in_idx));
          auto paramIdxAttr = builder.getDenseI64ArrayAttr({});
          auto kindAttr = builder.getStringAttr("must_alias");
          auto aliasInner =
              builder.getDictionaryAttr({builder.getNamedAttr("kind", kindAttr),
                                         builder.getNamedAttr("parameter_number", paramNumAttr),
                                         builder.getNamedAttr("parameter_index", paramIdxAttr)});
          auto outputIdxAttr = builder.getDenseI64ArrayAttr({static_cast<int64_t>(out_idx)});
          auto aliasOuter =
              builder.getDictionaryAttr({builder.getNamedAttr("alias", aliasInner),
                                         builder.getNamedAttr("output_index", outputIdxAttr)});
          aliasList.push_back(aliasOuter);

          TORCH_NEURONX_DEBUG("Added input_output_alias: output[", out_idx, "] -> input[", in_idx,
                              "] addr=", hlo_node->outputs[out_idx]);
        }

        if (!aliasList.empty()) {
          auto moduleOp = (*hlo_node->module)->getOperation();
          moduleOp->setAttr(builder.getStringAttr("mhlo.input_output_alias"),
                            builder.getArrayAttr(aliasList));
        }
        has_overlap = false;  // Handled via aliasing — don't discard
      }
    }

    if (has_overlap) {
      TORCH_NEURONX_DEBUG("Discarding concatenated IR due to input-output overlap. op names: ",
                          state.merged->op_name);
      res.compilable_irs.insert(res.compilable_irs.end(), state.original_hlo_nodes.begin(),
                                state.original_hlo_nodes.end());
    } else {
      // We have a concatenated result
      res.compilable_irs.push_back(state.merged);
      for (auto* orig : state.original_hlo_nodes) {
        res.concated_ir_to_original_irs[state.merged].push_back(orig);
      }
      res.concat_ir_list.push_back(std::move(state.merged_owned));
      TORCH_NEURONX_DEBUG("ConcatAll completed with concatenation",
                          "compilable_irs:", res.compilable_irs.size(),
                          "original_hlo_count:", state.original_hlo_nodes.size());

      if (aliasing_enabled_) {
        validateAliasingAttributes(state.merged);
      }
    }

  } else if (state.merged) {
    // Single HLO node, no concatenation occurred
    res.compilable_irs.push_back(state.merged);
    TORCH_NEURONX_DEBUG("ConcatAll completed - single HLO node, no concatenation",
                        "op_name:", state.merged->op_name);
  } else {
    // No HLO nodes found (only hints)
    TORCH_NEURONX_DEBUG("ConcatAll completed - no HLO nodes found");
  }

  return res;
}

}  // namespace torch_neuronx

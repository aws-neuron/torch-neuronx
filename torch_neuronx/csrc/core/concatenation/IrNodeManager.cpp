#include "IrNodeManager.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "IrNode.h"
#include "llvm/Support/raw_ostream.h"
#include "stablehlo/dialect/ChloOps.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/concatenation/FunctionPruneUtils.h"
#include "torch_neuronx/csrc/core/concatenation/OpConcatUtils.h"

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

std::string IrNodeManager::GenerateConcatenationCacheKey(const IrNode* node1, const IrNode* node2) {
  if (!node1 || !node2) {
    TORCH_NEURONX_ERROR("generate_concatenation_cache_key called with null nodes");
    throw std::invalid_argument("Both nodes must be non-null to generate cache key");
  }
  std::string cache_key = node1->cache_key + "|" + node2->cache_key;
  TORCH_NEURONX_DEBUG("Generated cache key:", cache_key);
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

  const std::string& cache_key = xla_kernel->GetCacheKey();

  TORCH_NEURONX_DEBUG("Creating IrNode from validated OperationContext",
                      "op_name:", xla_kernel->GetOpName(), "cache_key:", cache_key,
                      "ir_bytes_size:", xla_kernel->GetHloBytes().size(),
                      "has_collectives:", xla_kernel->HasCollectives());

  if (auto node = ir_resource_cache_.find(cache_key); node != ir_resource_cache_.end()) {
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

    // Populate TensorDataRef from the XLA kernel (preserves NeuronTensorPtr for NRT execution)
    auto src_ptrs = xla_kernel->GetSrcDataPtrs();
    result_node->input_data_refs.reserve(src_ptrs.size());
    for (size_t i = 0; i < src_ptrs.size(); ++i) {
      // Create TensorDataRef with the nrt_tensor_t pointer using a no-op deleter
      // since ownership is retained by the original kernel
      result_node->input_data_refs.emplace_back(
          c10_neuron::NeuronCachingAllocator::TensorPtr(src_ptrs[i], [](nrt_tensor_t*) {}),
          xla_kernel->GetSrcPtrs()[i]);
    }

    auto dst_ptrs = xla_kernel->GetDstDataPtrs();
    result_node->output_data_refs.reserve(dst_ptrs.size());
    for (size_t i = 0; i < dst_ptrs.size(); ++i) {
      result_node->output_data_refs.emplace_back(
          c10_neuron::NeuronCachingAllocator::TensorPtr(dst_ptrs[i], [](nrt_tensor_t*) {}),
          xla_kernel->GetDstPtrs()[i]);
    }

    // Copy TensorContexts from the XLA kernel
    result_node->input_contexts = xla_kernel->GetInputContexts();
    result_node->output_contexts = xla_kernel->GetOutputContexts();

    return result_node;
  }

  // Do this because std::make_unique doesn't work when using a private constructor
  // even if this is a friend class
  auto node = std::unique_ptr<StableHloNode>(new StableHloNode(*operation_context, context.get()));

  // Populate TensorDataRef for the new node (cache miss case)
  auto src_ptrs = xla_kernel->GetSrcDataPtrs();
  node->input_data_refs.reserve(src_ptrs.size());
  for (size_t i = 0; i < src_ptrs.size(); ++i) {
    node->input_data_refs.emplace_back(
        c10_neuron::NeuronCachingAllocator::TensorPtr(src_ptrs[i], [](nrt_tensor_t*) {}),
        xla_kernel->GetSrcPtrs()[i]);
  }

  auto dst_ptrs = xla_kernel->GetDstDataPtrs();
  node->output_data_refs.reserve(dst_ptrs.size());
  for (size_t i = 0; i < dst_ptrs.size(); ++i) {
    node->output_data_refs.emplace_back(
        c10_neuron::NeuronCachingAllocator::TensorPtr(dst_ptrs[i], [](nrt_tensor_t*) {}),
        xla_kernel->GetDstPtrs()[i]);
  }

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
  auto ir_resource = std::make_unique<IrResource>(
      node->module, std::move(merge_mapping), node->op_name, node->cache_key, node->ir_serialized,
      node->ir_type, node->has_collectives);
  ir_resource_cache_[cache_key] = std::move(ir_resource);
}

bool IrNodeManager::PruneMlirFunction(StableHloNode* node, void* dealloc_address) {
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
    TORCH_NEURONX_WARN("[PRUNE_MLIR] Could not find 'main' function in module");
    return false;
  }

  // Call the low-level index-based pruning API
  try {
    pruneFunctionResultsByIndices(funcOp, indicesToKeep);
  } catch (const std::exception& e) {
    TORCH_NEURONX_WARN("[PRUNE_MLIR] Failed to prune MLIR function", "error:", e.what());
    return false;
  }

  return true;
}

bool IrNodeManager::PruneSingleOutput(StableHloNode* node, void* dealloc_address) {
  if (!node || !node->module) {
    TORCH_NEURONX_WARN("[PRUNE_SINGLE] Invalid node or module for pruning");
    return false;
  }

  // Find indices to prune (all matching dealloc_address in CURRENT outputs)
  auto pruned_indices = FindIndicesToPrune(node, dealloc_address);

  // If nothing to prune (address not in outputs), return success
  if (pruned_indices.empty()) {
    TORCH_NEURONX_DEBUG("[PRUNE_SINGLE] Address not found in outputs - nothing to prune",
                        "addr:", dealloc_address, "outputs_count:", node->outputs.size());
    return true;
  }

  // Generate new cache key by appending pruning suffix to current key
  std::string new_cache_key = GeneratePruningCacheKey(node->cache_key, pruned_indices);
  std::unordered_set<size_t> pruned_set(pruned_indices.begin(), pruned_indices.end());

  TORCH_NEURONX_DEBUG("[PRUNE_SINGLE] Pruning outputs at address", "addr:", dealloc_address,
                      "pruned_indices_count:", pruned_indices.size(),
                      "outputs_before:", node->outputs.size(), "old_cache_key:", node->cache_key,
                      "new_cache_key:", new_cache_key);

  // Check cache for already-pruned MLIR
  if (auto* cached = GetByKey(new_cache_key)) {
    TORCH_NEURONX_DEBUG("[PRUNE_SINGLE] Cache HIT - restoring from cache",
                        "cache_key:", new_cache_key);
    node->module = cached->module;
    node->ir_serialized = cached->ir_serialized;
    node->cache_key = new_cache_key;
    UpdateNodeOutputVectors(node, pruned_set);
    return true;
  }

  TORCH_NEURONX_DEBUG("[PRUNE_SINGLE] Cache MISS - performing actual pruning",
                      "cache_key:", new_cache_key);

  // Perform actual MLIR function pruning
  if (!PruneMlirFunction(node, dealloc_address)) {
    return false;
  }

  // Update node state
  UpdateNodeOutputVectors(node, pruned_set);
  SerializeMlirModule(node);
  node->cache_key = new_cache_key;
  CacheIrResource(node, new_cache_key);

  TORCH_NEURONX_DEBUG("[PRUNE_SINGLE] Successfully pruned, re-serialized, and cached MLIR",
                      "outputs_after:", node->outputs.size(),
                      "ir_serialized_size:", node->ir_serialized.size(),
                      "cache_key:", new_cache_key, "cache_size:", ir_resource_cache_.size());

  return true;
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
        // Pruning failed - throw exception to trigger fallback in ConcatenationCore
        TORCH_NEURONX_WARN("[TYPE_DISPATCH] Pruning failed - falling back to individual ops",
                           "dealloc_addr:", hint_node->dealloc_address);
        std::ostringstream oss;
        oss << "Pruning failed for dealloc_address " << hint_node->dealloc_address;
        throw std::runtime_error(oss.str());
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
    // We have a concatenated result
    res.compilable_irs.push_back(state.merged);
    for (auto* orig : state.original_hlo_nodes) {
      res.concated_ir_to_original_irs[state.merged].push_back(orig);
    }
    res.concat_ir_list.push_back(std::move(state.merged_owned));
    TORCH_NEURONX_DEBUG("ConcatAll completed with concatenation",
                        "compilable_irs:", res.compilable_irs.size(),
                        "original_hlo_count:", state.original_hlo_nodes.size());
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

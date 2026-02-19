#ifndef IR_CONCAT_HELPER_H
#define IR_CONCAT_HELPER_H

#include <string>
#include <unordered_map>
#include <vector>

#include "IrNode.h"
#include "mlir/IR/OwningOpRef.h"
#include "torch_neuronx/csrc/core/concatenation/OpConcatUtils.h"

namespace torch_neuronx {

/**
 * @brief Result structure for IR concatenation operations
 *
 * IrConcatResult encapsulates the outcome of applying a concatenation strategy
 * to a sequence of IR nodes. It provides a structured way to separate nodes
 * into different categories based on their readiness for compilation and
 * maintains traceability between concatenated and original nodes.
 *
 * ## Design Purpose:
 *
 * ### Pipeline Processing:
 * - Enables multi-stage concatenation where different strategies can be applied
 * - Separates "ready" nodes from those needing further analysis
 * - Supports incremental optimization across multiple strategy passes
 *
 * ### Traceability:
 * - Maps concatenated nodes back to their constituent original nodes
 * - Enables debugging and performance analysis of concatenation decisions
 * - Supports rollback scenarios if concatenation proves suboptimal
 *
 * ### Memory Management:
 * - Owns concatenated nodes via unique_ptr for automatic cleanup
 * - Uses raw pointers for existing nodes to avoid ownership conflicts
 * - Provides move semantics for efficient result passing
 *
 * ## Usage in Concatenation Pipeline:
 *
 * ```cpp
 * // Strategy processes input nodes
 * IrConcatResult result = strategy.IsFusibleBoundaryOperation(input_nodes, manager,
 * prev_analysis);
 *
 * ```
 *
 * @note This structure enables flexible concatenation pipelines while maintaining
 *       clear ownership semantics and operation traceability
 */
struct IrConcatResult {
  /// IR nodes ready for immediate compilation (individual or successfully concatenated)
  std::list<IrNode*> compilable_irs;

  /// IR nodes requiring further processing by subsequent concatenation strategies
  std::list<IrNode*> remain_irs;

  /// Traceability map: concatenated nodes -> their constituent original nodes
  /// Enables debugging, performance analysis, and potential rollback scenarios
  std::unordered_map<IrNode*, std::vector<IrNode*>> concated_ir_to_original_irs;

  /// Ownership container for newly created concatenated IR nodes
  /// Ensures proper memory management and automatic cleanup
  std::list<std::unique_ptr<IrNode>> concat_ir_list;

  /// Default constructor creates empty result
  IrConcatResult() = default;

  /// Constructor for cases where no concatenation occurred
  /// @param remain_irs All input nodes remain for further processing
  IrConcatResult(std::list<IrNode*>&& remain_irs) : remain_irs(std::move(remain_irs)) {}

  /**
   * @brief Complete constructor for full concatenation results
   *
   * @param compilable_irs Nodes ready for immediate compilation
   * @param remain_irs Nodes requiring further strategy processing
   * @param concated_ir_to_original_irs Traceability mapping for concatenated nodes
   * @param concat_ir_list Ownership container for newly created nodes
   */
  IrConcatResult(std::list<IrNode*>&& compilable_irs, std::list<IrNode*>&& remain_irs,
                 std::unordered_map<IrNode*, std::vector<IrNode*>>&& concated_ir_to_original_irs,
                 std::list<std::unique_ptr<IrNode>>&& concat_ir_list)
      : compilable_irs(std::move(compilable_irs)),
        remain_irs(std::move(remain_irs)),
        concated_ir_to_original_irs(std::move(concated_ir_to_original_irs)),
        concat_ir_list(std::move(concat_ir_list)) {}
};

/**
 * @brief Cached resource container for concatenated IR operations
 *
 * IrResource holds all necessary data for a cached concatenated operation,
 * including the parsed MLIR module, tensor mappings, and metadata.
 * Used internally by IrNodeManager to store and retrieve concatenation results.
 *
 * ## Memory Management:
 * - MLIR module is shared via shared_ptr to allow multiple references
 * - MergeMapping is uniquely owned and contains tensor address mappings
 * - Serialized IR is stored for potential re-parsing or debugging
 *
 * ## Lifecycle:
 * - Created during concatenation operations
 * - Cached in IrNodeManager for reuse
 * - Destroyed when evicted from cache or manager is destroyed
 *
 * @note This struct is not intended for direct instantiation by users
 * @see IrNodeManager for public API to access cached resources
 */
struct IrResource {
  /// Shared MLIR module containing concatenated operations
  std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>> module;

  /// Tensor address mapping from original nodes to concatenated result
  std::unique_ptr<MergeMapping> merge_mapping;

  /// Human-readable operation name for debugging
  std::string op_name;

  /// Unique cache identifier for lookup
  std::string cache_key;

  /// Serialized IR representation for re-parsing if needed
  std::vector<std::uint8_t> ir_serialized;

  /// Type of IR (e.g., STABLEHLO)
  IrNodeType ir_type;

  /// Whether operation uses collective communication
  bool has_collectives;

  /**
   * @brief Construct IrResource with all required components
   *
   * @param module Shared MLIR module containing concatenated operations
   * @param merge_mapping Tensor address mapping (ownership transferred)
   * @param op_name Operation name for identification
   * @param cache_key Unique cache identifier
   * @param ir_serialized Serialized IR bytes
   * @param ir_type Type of IR representation
   * @param has_collectives Whether operation uses collective communication
   */
  IrResource(std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>>& module,
             std::unique_ptr<MergeMapping>&& merge_mapping, std::string& op_name,
             std::string& cache_key, std::vector<std::uint8_t>& ir_serialized, IrNodeType ir_type,
             bool has_collectives)
      : module(module),
        merge_mapping(std::move(merge_mapping)),
        op_name(op_name),
        cache_key(cache_key),
        ir_serialized(ir_serialized),
        ir_type(ir_type),
        has_collectives(has_collectives) {}
};

/**
 * @brief Manager for IR node creation, concatenation, and caching
 *
 * IrNodeManager serves as the central factory and cache for IR nodes,
 * providing functionality to create individual nodes from operation contexts
 * and to concatenate multiple nodes into optimized composite operations.
 *
 * ## Key Responsibilities:
 * - Create IR nodes from operation contexts
 * - Concatenate compatible IR nodes for optimization
 * - Cache concatenated MLIR modules to avoid redundant work
 * - Manage the lifecycle of cached MLIR module objects
 * - Generate unique identifiers for caching and debugging
 *
 * ## Memory Ownership:
 * - IrNodeManager owns the MLIR context and all cached resources
 * - Returned nodes share ownership of MLIR modules via shared_ptr
 * - Cache eviction invalidates all references to evicted resources
 *
 * ## Usage Examples:
 *
 * ### Basic Node Creation:
 * ```cpp
 * IrNodeManager manager;
 * auto node = manager.create_node_from_operation_context(op_context);
 * if (node) {
 *   // Use the node...
 * }
 * ```
 *
 * ### Node Concatenation:
 * ```cpp
 * IrNodeManager manager;
 * auto node1 = manager.create_node_from_operation_context(ctx1);
 * auto node2 = manager.create_node_from_operation_context(ctx2);
 *
 * // Concatenate two nodes for optimization
 * auto concat_node = manager.CreateConcatNode(node1.get(), node2.get());
 * if (concat_node) {
 *   // Use concatenated node which combines both operations
 * }
 * ```
 *
 * ### Cache Management:
 * ```cpp
 * IrNodeManager manager;
 *
 * // Check if concatenation already exists
 * std::string cache_key = manager.generate_concatenation_cache_key(node1, node2);
 * if (auto cached = manager.get_by_key(cache_key)) {
 *   // Use cached result
 * } else {
 *   // Create new concatenation
 *   auto new_concat = manager.CreateConcatNode(node1, node2);
 * }
 *
 * // Evict from cache when memory pressure is high
 * manager.evict(cache_key);
 * ```
 *
 * ## Thread Safety:
 * - Not thread-safe; use from single thread or with external synchronization
 * - MLIR context is not thread-safe and shared across all operations
 *
 * ## Performance Considerations:
 * - Concatenation is expensive; results are cached automatically
 * - Cache keys are generated from node properties for deterministic lookup
 * - Memory usage grows with cache size; use evict() for cleanup
 */
class IrNodeManager {
 public:
  IrNodeManager();

  IrNodeManager(const IrNodeManager&) = delete;

  IrNodeManager& operator=(const IrNodeManager&) = delete;

  IrNodeManager(IrNodeManager&&) = default;

  IrNodeManager& operator=(IrNodeManager&&) = default;

  /**
   * @brief Create a concatenated node from two generic IR nodes
   *
   * Attempts to concatenate two IR nodes into a single optimized operation.
   * The concatenation process depends on the specific IR types and may involve
   * MLIR module merging for StableHLO nodes.
   *
   * @param node1 First IR node to concatenate (must be non-null)
   * @param node2 Second IR node to concatenate (must be non-null)
   * @return Unique pointer to concatenated node, or nullptr if concatenation fails
   *
   * @throws std::invalid_argument if either node is null
   * @throws std::runtime_error if nodes have incompatible IR types
   *
   * @note Currently only supports StableHLO node concatenation
   * @note Pruning is now handled incrementally in ConcatAll, not during CreateConcatNode
   *
   * @see CreateConcatNode(StableHloNode*, StableHloNode*) for type-safe version
   */
  std::unique_ptr<IrNode> CreateConcatNode(IrNode* node1, IrNode* node2);

  /**
   * @brief Create a concatenated StableHloNode from two StableHLO nodes
   *
   * Specialized concatenation for StableHLO nodes that performs MLIR-level
   * module merging and optimization. This version provides better type safety
   * and optimization opportunities compared to the generic version.
   *
   * @param node1 First StableHloNode to concatenate (must be non-null)
   * @param node2 Second StableHloNode to concatenate (must be non-null)
   * @return Unique pointer to concatenated StableHloNode, or nullptr if concatenation fails
   *
   * @throws std::invalid_argument if either node is null or has empty cache key
   *
   * ## Concatenation Process:
   * 1. Generate cache key from input nodes
   * 2. Perform MLIR module merging (preserves all outputs)
   * 3. Build tensor address mappings
   *
   * @note Pruning is now handled incrementally in ConcatAll via PruneSingleOutput
   * @note Handles tensor address remapping automatically
   * @note Result includes merge mapping for tensor address translation
   */
  std::unique_ptr<StableHloNode> CreateConcatNode(StableHloNode* node1, StableHloNode* node2);

  /**
   * @brief Create an IR node from a PyTorch operation context
   *
   * Factory method that creates appropriate IR node types based on the
   * operation context's IR format and metadata. Handles the conversion
   * from operation context to the internal IR node format.
   *
   * @param context PyTorch operation context containing IR and metadata
   * @return Pointer to created IR node, or nullptr if creation fails
   *
   * @note The returned node type depends on the IR format in the context
   * @note Created nodes are not automatically cached; use concatenation for caching
   */
  std::unique_ptr<IrNode> CreateNodeFromOperationContext(
      const at::neuron::OperationContext* context);

  /**
   * @brief Retrieve a cached concatenated node by key
   *
   * Looks up a previously concatenated node in the cache using its unique key.
   * Useful for avoiding redundant concatenation operations.
   *
   * @param key Cache key for the desired node
   * @return Pointer to cached node, or nullptr if not found
   *
   * @note Cache keys are generated deterministically from input node properties
   */
  IrResource* GetByKey(std::string key);

  /**
   * @brief Remove a cached node and free its resources
   *
   * Evicts a node from the concatenation cache and deallocates its memory.
   * Used for cache management and memory pressure relief.
   *
   * @param key Cache key of the node to evict
   * @return Number of nodes evicted (0 or 1)
   *
   * @note Evicted nodes become invalid and should not be accessed
   */
  size_t Evict(std::string key);

  /**
   * @brief Generate a unique cache key for two nodes
   */
  std::string GenerateConcatenationCacheKey(const IrNode* node1, const IrNode* node2);

  /**
   * @brief Generate a descriptive operation name for concatenated nodes
   */
  std::string GenerateConcatenationOpName(const IrNode* node1, const IrNode* node2);

  /**
   * @brief Generate cache key for pruned MLIR module
   *
   * Appends pruning suffix to base cache key using CURRENT output indices.
   * The indices are the positions in the current output array being pruned.
   * Indices are sorted to ensure deterministic cache keys regardless of pruning order.
   *
   * @param base_cache_key The current cache key of the node (may include prior pruning suffixes)
   * @param pruned_indices Current indices being pruned (sorted internally)
   * @return Cache key with pruning suffix appended (e.g., "HLO1|HLO2|pruned_1|pruned_2")
   *
   * ## Example:
   * - Base: "HLO1|HLO2", prune current index 1 → "HLO1|HLO2|pruned_1"
   * - Then: "HLO1|HLO2|pruned_1", prune current index 2 → "HLO1|HLO2|pruned_1|pruned_2"
   *
   * @note If pruned_indices is empty, returns base_cache_key unchanged
   */
  std::string GeneratePruningCacheKey(const std::string& base_cache_key,
                                      const std::vector<size_t>& pruned_indices);

  /**
   * @brief Prune outputs from a StableHloNode based on deallocation address
   *
   * Removes all outputs matching the given dealloc_address from the node's MLIR
   * function and updates the node's output vectors accordingly. The same address may
   * appear at multiple output indices (e.g., when a tensor is returned multiple times).
   *
   * Features caching: generates a cache key based on the current cache_key + pruned indices,
   * checks cache before doing actual pruning, and stores result in cache on miss.
   *
   * @param node The StableHloNode to prune (must be a concatenated/merged node)
   * @param dealloc_address The address of the output(s) to remove
   * @return true if pruning succeeded (or nothing to prune), false if pruning failed
   *
   * ## Cache Key Evolution:
   * - After merge: cache_key = "HLO1|HLO2"
   * - After prune index 1: cache_key = "HLO1|HLO2|pruned_1"
   * - After prune index 2: cache_key = "HLO1|HLO2|pruned_1|pruned_2"
   *
   * @note Updates node->cache_key, node->module, node->ir_serialized, and output vectors
   */
  bool PruneSingleOutput(StableHloNode* node, void* dealloc_address);

  mlir::MLIRContext* GetContext() const { return context.get(); }

  /**
   * @brief Concatenate IR nodes with type-based processing (HLO + DeallocHint)
   *
   * Processes a mixed list of IR nodes (HLO operations + deallocation hint markers)
   * with ordering-aware pruning:
   * 1. Merges consecutive HLO nodes
   * 2. When encountering DeallocHintNode(s), collects addresses to prune
   * 3. Prunes those addresses from the next merge
   * 4. Caches the pruned result
   *
   * @param input_ir_list Mixed list of HLO and DeallocHintNode in execution order
   * @return IrConcatResult with merged compilable operations
   *
   * ## Type-Based Dispatch:
   * - StableHloNode: Merge with accumulated nodes
   * - DeallocHintNode: Collect address for pruning
   *
   * ## Example:
   * Input: [node1, node2, node3, dealloc(a), node4, dealloc(b)]
   * - Merge node1+node2+node3, encounter dealloc(a), prune 'a'
   * - Merge with node4, encounter dealloc(b), prune 'b'
   */
  IrConcatResult ConcatAll(const std::list<IrNode*>& input_ir_list);

  /**
   * @brief State container for incremental concatenation in ConcatAll
   *
   * Holds all mutable state needed during the concatenation loop, allowing
   * ProcessIrNode to modify state without passing many individual parameters.
   */
  struct ConcatState {
    /// Addresses that have been deallocated but not yet reallocated
    std::unordered_set<void*> pruned_addresses;

    /// Current merged node (raw pointer, may point to owned or external node)
    IrNode* merged = nullptr;

    /// Ownership of merged node if we created it via concatenation
    std::unique_ptr<IrNode> merged_owned;

    /// All original HLO nodes that have been merged (for traceability)
    std::vector<IrNode*> original_hlo_nodes;
  };

  /**
   * @brief Process a single IR node during concatenation
   *
   * Type-based dispatch that handles each node type:
   * - ALLOC_HINT: Removes address from pruned set
   * - DEALLOC_HINT: Adds address to pruned set and prunes merged node
   * - STABLEHLO: Validates and concatenates with current merged node
   *
   * @param node The IR node to process
   * @param state Mutable state for the concatenation operation
   * @throws std::runtime_error if pruning or concatenation fails
   */
  void ProcessIrNode(IrNode* node, ConcatState& state);

 private:
  /**
   * @brief Find indices of outputs matching the given address
   *
   * @param node The node to search outputs in
   * @param address The address to match
   * @return Vector of indices where outputs match the address
   */
  std::vector<size_t> FindIndicesToPrune(const StableHloNode* node, void* address);

  /**
   * @brief Update node output vectors by removing pruned indices
   *
   * @param node The node to update
   * @param pruned_indices Set of indices to remove
   */
  void UpdateNodeOutputVectors(StableHloNode* node,
                               const std::unordered_set<size_t>& pruned_indices);

  /**
   * @brief Re-serialize MLIR module to ir_serialized bytes
   *
   * @param node The node containing the module to serialize
   */
  void SerializeMlirModule(StableHloNode* node);

  /**
   * @brief Store IR resource in cache for future reuse
   *
   * @param node The node to cache (can be pruned, concatenated, or single op)
   * @param cache_key The cache key for storage
   * @param merge_mapping Optional merge mapping for concatenated nodes (nullptr for single/pruned
   * ops)
   */
  void CacheIrResource(StableHloNode* node, const std::string& cache_key,
                       std::unique_ptr<MergeMapping> merge_mapping = nullptr);

  /**
   * @brief Perform actual MLIR function pruning
   *
   * @param node The node containing the MLIR module
   * @param dealloc_address The address to prune from outputs
   * @return true if pruning succeeded, false otherwise
   */
  bool PruneMlirFunction(StableHloNode* node, void* dealloc_address);

  /// MLIR context managing the module's lifetime and resources
  std::unique_ptr<mlir::MLIRContext> context;

  /// Cache mapping concatenation keys to concatenated IR nodes
  std::unordered_map<std::string, std::unique_ptr<IrResource>> ir_resource_cache_;

  /// Feature flag for skip intermediate optimization
  /// Controlled by TORCH_NEURONX_APPLY_SKIP_INTERMEDIATE environment variable
  /// Default: enabled. Set TORCH_NEURONX_APPLY_SKIP_INTERMEDIATE=0 to disable
  bool skip_intermediate_enabled_ = true;
};

}  // namespace torch_neuronx

#endif

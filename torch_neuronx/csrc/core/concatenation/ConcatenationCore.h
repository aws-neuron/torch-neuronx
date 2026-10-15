#ifndef _OPERATION_CONTEXT_COORDINATOR_H_
#define _OPERATION_CONTEXT_COORDINATOR_H_

#include <list>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/concatenation/IrConcatStrategy.h"
#include "torch_neuronx/csrc/core/concatenation/IrNodeManager.h"

namespace torch_neuronx {

/**
 * @brief Result structure for concatenation processing operations
 *
 * ConcatenationResult encapsulates the outcome of processing a batch of operations
 * through the concatenation pipeline. It provides comprehensive information about
 * what was processed, what's ready for execution, and any issues encountered.
 *
 * ## Design Purpose:
 *
 * ### Execution Readiness:
 * - Contains operations ready for immediate compilation and execution
 * - May include both individual operations and concatenated fusion groups
 * - Maintains operation ordering and dependency relationships
 *
 * ### Analysis Continuity:
 * - Carries forward analysis state for subsequent processing batches
 *
 * ## Usage in Processing Pipeline:
 *
 * ```cpp
 * // Process a batch of operations
 * ConcatenationResult result = core.processBufferedOperations(input_ops);
 *
 * // Always safe to use - on error, original operations are returned as fallback
 * for (auto* op : result.processed_operations) {
 *    submitOperationForCompilation(op);
 * }
 * ```
 *
 * @note This structure enables robust pipeline processing with graceful error handling
 */
struct ConcatenationResult {
  /// Operations ready for immediate compilation and execution
  /// May include individual operations or concatenated fusion groups
  /// Note: Always contains valid operations - on error, returns original ops as fallback
  std::list<at::neuron::OperationContext*> processed_operations;

  /// Number of input operations consumed during processing
  /// Used for pipeline flow control and progress tracking
  size_t original_operations_consumed = 0;
};

/**
 * @brief Central orchestrator for operation optimization through concatenation
 *
 * ConcatenationCore serves as the main coordination point for applying concatenation
 * strategies to optimize sequences of operations. It manages the entire pipeline from
 * raw operations to optimized, compilation-ready concatenated operation.
 *
 * ## Architecture Overview:
 *
 * ### Multi-Stage Pipeline:
 * 1. **Preprocessing**: Converts OperationContext objects to IR nodes
 * 2. **Strategy Application**: Applies concatenation strategies in sequence
 * 3. **Postprocessing**: Converts optimized IR back to OperationContext
 * 4. **Result Assembly**: Packages results with metadata and analysis state
 *
 * ### Strategy Coordination:
 * - Manages multiple concatenation strategies (MatMulToMatMul, etc.)
 * - Applies strategies in configured order for maximum optimization
 * - Each strategy processes remaining operations from previous strategy
 * - Maintains traceability between original and concatenated operations
 *
 * ### Resource Management:
 * - Owns IrNodeManager for IR node creation and caching
 * - Manages strategy lifecycle and configuration
 * - Handles memory cleanup for temporary IR representations
 * - Maintains operation context mappings for result reconstruction
 *
 * ## Usage Patterns:
 *
 * ### Basic Operation Processing:
 * ```cpp
 * auto core = ConcatenationCoreFactory::create_instance();
 *
 * // Process a batch of operations
 * std::list<OperationContext*> ops = get_pending_operations();
 * auto result = core->processBufferedOperations(ops);
 *
 * // Submit optimized operations for execution
 * for (auto* op : result.processed_operations) {
 *   compilation_engine.submit(op);
 * }
 * ```
 *
 * ### Custom Strategy Configuration:
 * ```cpp
 * std::vector<std::unique_ptr<AbstractIrConcatStrategy>> strategies;
 * strategies.push_back(std::make_unique<MatMulToMatMulConcatStrategy>());
 * strategies.push_back(std::make_unique<CustomOptimizationStrategy>());
 *
 * auto core = ConcatenationCoreFactory::create_instance(std::move(strategies));
 * ```
 *
 * ## Error Handling:
 * - Robust error handling with graceful degradation
 * - Failed concatenations fall back to individual operations
 * - No operation loss - all input operations are preserved in output
 *
 * ## Performance Characteristics:
 * - Optimized for batch processing of operation sequences
 * - Caching reduces redundant IR parsing and analysis
 * - Strategy ordering affects optimization effectiveness
 * - Memory usage scales with batch size and IR complexity
 *
 * @note Thread Safety: Not thread-safe. Each thread should use separate instances.
 */
class ConcatenationCore {
 public:
  ConcatenationCore(const ConcatenationCore&) = delete;

  ConcatenationCore& operator=(const ConcatenationCore&) = delete;

  ConcatenationCore(ConcatenationCore&&) = delete;

  ConcatenationCore& operator=(ConcatenationCore&&) = delete;

  /**
   * @brief Process a batch of operations through the concatenation pipeline
   *
   * This is the main entry point for operation optimization. It processes a batch
   * of operations through the complete concatenation pipeline, applying all
   * configured strategies to identify and create optimized operation groups.
   *
   * @param buffered_ops List of operations to process and optimize
   * @return ConcatenationResult containing optimized operations and processing metadata
   *
   * ## Processing Pipeline:
   * 1. **Validation**: Ensures input operations are valid for processing
   * 2. **Preprocessing**: Converts OperationContext to IR nodes via IrNodeManager
   * 3. **Strategy Application**: Applies concatenation strategies in sequence
   * 4. **Postprocessing**: Converts optimized IR nodes back to OperationContext
   * 5. **Result Assembly**: Creates ConcatenationResult with metadata
   *
   * ## Error Handling:
   * - Graceful degradation: Failed concatenations return original operations
   * - No operation loss: All input operations appear in result.processed_operations
   *
   * @note ProcessBufferedOperations
   *  should only take a sequence of ops that only have 1 boundary op and ends with
   *  the boundary op. Use IsFusibleBoundaryOperation to find out the boundary ops.
   * @note Operations in result may be individual or concatenated groups
   * @note Maintains operation ordering and dependency relationships
   * @note Thread-safe within single thread; not safe across threads
   */
  ConcatenationResult ProcessBufferedOperations(
      const std::list<at::neuron::OperationContext*>& buffered_ops);

  /**
   * @brief Get the list of configured concatenation strategies
   *
   * Returns a vector of pointers to the concatenation strategies currently
   * configured for this coordinator. Useful for debugging and introspection.
   *
   * @return Vector of strategy pointers (non-owning)
   *
   * @note Returned pointers are valid only while the coordinator exists
   */
  const std::vector<AbstractIrConcatStrategy*> GetConcatStrategies() const {
    std::vector<AbstractIrConcatStrategy*> concat_strategies;
    for (auto& strategy : this->concat_strategies) {
      concat_strategies.push_back(strategy.get());
    }
    return concat_strategies;
  }

  /**
   * @brief Clear the concatenation cache
   */
  void ClearCache() {}

  /**
   * @brief Get the current size of the concatenation cache
   * @return Number of cached concatenation results
   */
  size_t GetCacheSize() const { return 0; }

  /**
   * @brief Invalidate a cache entry for the given key
   *
   * This marks the cache entry as invalid so it will not be retrieved or used
   * even if the same key is encountered again. The entry itself remains in the
   * cache but is effectively disabled.
   *
   * @param key The cache key to invalidate
   */
  void InvalidateCacheEntry(const std::string& key);

  /**
   * @brief Check if an operation is a boundary op that indicates end of fusion boundary
   *
   *
   * @param op_name Name of the operation to check
   * @return true if the operation can be fused at a boundary, false otherwise
   *
   * @note This should be used before ProcessBufferedOperations. ProcessBufferedOperations
   *  should only take a sequence of ops that only have 1 boundary op and ends with
   *  the boundary op.
   * @note This affects concatenation decisions at fusion group boundaries
   */
  bool IsFusibleBoundaryOperation(const std::string& op_name);

  /**
   * @brief Create an OperationContext from a concatenated IR node
   *
   * Converts a concatenated IR node back into an operation context
   * suitable for execution. This method performs validation on TensorDataRef
   * to ensure proper tensor mapping between IR representation and execution.
   *
   * ## Validation Rules:
   * - If inputs exist, input_data_refs must be populated and match size
   * - If outputs exist, output_data_refs must be populated and match size
   * - Throws std::runtime_error on validation failure
   *
   * @param ir_node IR node to convert (typically a concatenated node)
   * @param device_id Device ID for the kernel execution
   * @return Unique pointer to created operation context
   *
   * @throws std::runtime_error if TensorDataRef validation fails
   * @throws std::invalid_argument if ir_node is null or has empty op_name/cache_key
   */
  std::unique_ptr<at::neuron::OperationContext> CreateOpContextFromIrNode(IrNode* ir_node,
                                                                          int device_id);

 private:
  /**
   * @brief Private constructor used by factory
   *
   * Constructs a coordinator with the specified concatenation strategies
   * and IR node manager.
   *
   * @param concat_strategies Vector of concatenation strategies to use
   * @param ir_node_manager Manager for IR node creation and caching
   */
  ConcatenationCore(std::vector<std::unique_ptr<AbstractIrConcatStrategy>>&& concat_strategies,
                    std::unique_ptr<IrNodeManager>&& ir_node_manager);

  /**
   * @brief Internal implementation of processBufferedOperations
   *
   * Separated from the public method to allow for clean exception handling.
   * If any exception occurs, the caller returns original operations.
   */
  ConcatenationResult ProcessBufferedOperationsImpl(
      const std::list<at::neuron::OperationContext*>& op_contexts);

  /**
   * @brief Process an existing (non-concatenated) IR node
   * @return Number of operations processed (0 if not found, 1 if found)
   */
  int ProcessExistingOperation(IrNode* ir_node,
                               std::list<at::neuron::OperationContext*>& op_context_results);

  /**
   * @brief Process an invalidated concatenated operation
   * @return Number of original operations processed, or -1 if not invalidated
   */
  int ProcessInvalidatedConcatenation(IrNode* ir_node, IrConcatResult& ir_concat_result,
                                      std::list<at::neuron::OperationContext*>& op_context_results);

  /**
   * @brief Process a new concatenated operation
   * @return Number of original operations that were concatenated
   */
  int ProcessNewConcatenatedOperation(IrNode* ir_node, IrConcatResult& ir_concat_result,
                                      const std::list<at::neuron::OperationContext*>& op_contexts,
                                      std::list<at::neuron::OperationContext*>& op_context_results);

  /**
   * @brief Post-process IR concat results back to OperationContext
   * @return Total number of operations processed
   */
  int PostProcessConcatResults(IrConcatResult& ir_concat_result,
                               const std::list<at::neuron::OperationContext*>& op_contexts,
                               std::list<at::neuron::OperationContext*>& op_context_results);

  /// Mapping from IR nodes to their original operation contexts
  std::unordered_map<IrNode*, at::neuron::OperationContext*> ir_node_to_op_context_map;

  /// Vector of concatenation strategies to apply in sequence
  std::vector<std::unique_ptr<AbstractIrConcatStrategy>> concat_strategies;

  /// Manager for creating and caching IR nodes
  std::unique_ptr<IrNodeManager> ir_node_manager;

  std::unordered_set<std::string> invalidated_cache_keys_;

  /// Factory has access to private constructor
  friend class ConcatenationCoreFactory;
};

/**
 * @brief Factory for creating optimally configured ConcatenationCore instances
 *
 * ConcatenationCoreFactory provides a centralized way to create ConcatenationCore
 * instances with appropriate default configurations or custom strategy sets.
 * It handles the complex initialization of strategies and resource managers.
 *
 * ## Design Benefits:
 * - Encapsulates complex initialization logic
 * - Provides sensible defaults for common use cases
 * - Enables easy customization for specialized optimization needs
 * - Ensures proper resource management and strategy coordination
 *
 * ## Usage Patterns:
 *
 * ### Default Configuration:
 * ```cpp
 * auto core = ConcatenationCoreFactory::create_instance();
 * // Uses default strategies optimized for common workloads
 * ```
 *
 * ### Custom Strategy Configuration:
 * ```cpp
 * std::vector<std::unique_ptr<AbstractIrConcatStrategy>> strategies;
 * strategies.push_back(std::make_unique<MatMulToMatMulConcatStrategy>());
 * strategies.push_back(std::make_unique<ConvolutionFusionStrategy>());
 *
 * auto core = ConcatenationCoreFactory::create_instance(std::move(strategies));
 * ```
 */
class ConcatenationCoreFactory {
 public:
  /**
   * @brief Create a coordinator with default concatenation strategies
   *
   * Default strategies typically include:
   * - TransposeMatmulConcatStrategy for transpose+matmul optimization
   * - NullOpStrategy as a fallback for remaining operations
   *
   * @return Unique pointer to configured coordinator
   *
   * @throws std::runtime_error if strategy or manager creation fails
   */
  static std::unique_ptr<ConcatenationCore> CreateInstance();

  /**
   * @brief Create a coordinator with custom concatenation strategies
   *
   */
  static std::unique_ptr<ConcatenationCore> CreateInstance(
      std::vector<std::unique_ptr<AbstractIrConcatStrategy>>&& strategies);
};

}  // namespace torch_neuronx

#endif

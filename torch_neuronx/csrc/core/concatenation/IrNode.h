#ifndef TORCH_NEURONX_IRNODE_H_
#define TORCH_NEURONX_IRNODE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "llvm/IR/Function.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/utils/TensorContext.h"

namespace torch_neuronx {

// Forward declaration
class IrNodeManager;
struct MergeMapping;

/**
 * @brief Execution status of an IR node
 *
 * Tracks the lifecycle state of an IR node from creation through execution.
 * Used for debugging, monitoring, and ensuring proper execution ordering.
 */
enum class IrNodeStatus {
  CREATED = 0,                  ///< Node created but not yet submitted
  SUBMITTED_FOR_EXECUTION = 1,  ///< Node submitted to execution queue
  EXECUTING = 2,                ///< Node currently being executed
  EXECUTED = 3,                 ///< Node execution completed successfully
  FAILED = 4                    ///< Node execution failed
};

/**
 * @brief Type of intermediate representation used by the node
 *
 * Distinguishes between different IR formats to enable type-specific
 * processing and optimization strategies.
 */
enum class IrNodeType {
  STABLEHLO = 0,     ///< StableHLO intermediate representation
  HLO = 1,           ///< Legacy HLO intermediate representation
  DEALLOC_HINT = 2,  ///< Deallocation hint marker (not executable IR)
  ALLOC_HINT = 3,    ///< Allocation hint marker (not executable IR)
  OTHERS = 100       ///< Other/custom intermediate representations
};

/**
 * @brief Base class for all IR node representations in concatenation system
 *
 * IrNode serves as the foundational abstraction for intermediate representation
 * nodes in the concatenation pipeline. It provides the minimal
 * interface required for IR manipulation while maintaining loose coupling with
 * other system components.
 *
 * ## Design Philosophy:
 *
 * ### Minimal Interface for Maximum Flexibility:
 * - Contains only essential data needed for concatenation operations
 * - Avoids dependencies on heavy PyTorch or MLIR types in the base interface
 * - Enables polymorphic handling of different IR formats (StableHLO, HLO, etc.)
 *
 * ### Decoupling Strategy:
 * - Separates IR representation from execution context (OperationContext)
 * - Uses void* pointers for tensor data to avoid PyTorch tensor dependencies
 * - Allow more flexible lifetime than OperationContext
 *
 * ### Concatenation-Optimized Design:
 * - Input/output tensor addresses enable dependency analysis between nodes
 * - Type information enables specialized concatenation strategies
 *
 *
 * ## Usage in Concatenation Pipeline:
 *
 * ```cpp
 * // Create nodes from different sources
 * auto node1 = manager.create_node_from_operation_context(ctx1);
 * auto node2 = manager.create_node_from_operation_context(ctx2);
 * ```
 *
 * @note This design enables the concatenation system to work with minimal
 *       dependencies while supporting rich IR-specific optimizations
 */
struct IrNode {
  /**
   * @brief Construct an IR node with essential concatenation data
   *
   * Creates an IrNode with the minimal data required for concatenation operations.
   * This constructor embodies the decoupling principle by accepting only the
   * essential information needed for IR manipulation and analysis.
   *
   * @param op_name Human-readable operation name for debugging and identification
   * @param cache_key Unique identifier enabling efficient caching and lookup
   * @param ir_serialized Serialized IR bytes for storage and re-parsing
   * @param ir_type IR format type enabling specialized processing strategies
   * @param inputs Input tensor data pointers for dependency analysis
   * @param outputs Output tensor data pointers for dependency analysis
   * @param has_collectives Collective communication flag for execution planning
   *
   * @note Uses void* for tensor data to avoid PyTorch dependencies in base class
   * @note Serialized IR enables storage without maintaining parsed representations
   */
  IrNode(const std::string& op_name, const std::string& cache_key,
         const std::vector<uint8_t>& ir_serialized, const IrNodeType ir_type,
         std::vector<void*>&& inputs, std::vector<void*>&& outputs, const bool has_collectives);

  IrNode(const std::string& op_name, const std::string&& cache_key,
         std::vector<uint8_t>& ir_serialized, const IrNodeType ir_type, std::vector<void*>&& inputs,
         std::vector<void*>&& outputs, const bool has_collectives);

  std::string op_name;                      ///< Operation identifier for debugging and logging
  std::string cache_key;                    ///< Unique key for caching and result lookup
  std::vector<std::uint8_t> ir_serialized;  ///< Serialized IR for storage and re-parsing
  IrNodeType ir_type;                       ///< IR format type for specialized processing
  std::vector<void*> inputs;                ///< Input tensor addresses for dependency analysis
  std::vector<void*> outputs;               ///< Output tensor addresses for dependency analysis
  IrNodeStatus status;                      ///< Execution lifecycle state
  bool has_collectives;                     ///< Collective communication requirement flag

  /// Input TensorDataRef for runtime execution (contains NeuronTensorPtr for NRT)
  /// These are populated from original operations and propagated through concatenation
  std::vector<at::neuron::TensorDataRef> input_data_refs;

  /// Output TensorDataRef for runtime execution (contains NeuronTensorPtr for NRT)
  /// These are populated from original operations and propagated through concatenation
  std::vector<at::neuron::TensorDataRef> output_data_refs;

  /// Input TensorContext for slicing metadata (needed for contiguous tensor handling)
  std::vector<at::neuron::TensorContext> input_contexts;

  /// Output TensorContext for slicing metadata
  std::vector<at::neuron::TensorContext> output_contexts;

  virtual ~IrNode() = default;

 protected:
  IrNode() = default;
};

/**
 * @brief Specialized IR node for StableHLO operations
 *
 * Extends IrNode to provide StableHLO-specific functionality, including
 * direct access to parsed MLIR modules and contexts.
 *
 * Key Features:
 * - Direct MLIR module access for advanced analysis
 * - Managed MLIR context for proper memory handling
 * - Support for both serialized and parsed representations
 * - Integration with StableHLO dialect operations
 * - Optimized for concatenation and fusion operations
 *
 * Usage:
 * - Created from OperationContext by IrNodeManager
 * - Used in concatenation strategies for optimization
 * - Enables MLIR-level transformations and analysis
 */
struct StableHloNode : public IrNode {
  /**
   * @brief Construct StableHloNode from serialized representation
   *
   * Creates a StableHloNode with serialized IR that will be parsed
   * on-demand when MLIR module access is required.
   *
   * @param op_name Operation name for identification
   * @param cache_key Unique cache identifier
   * @param ir_serialized Serialized StableHLO IR
   * @param inputs Input tensor data pointers
   * @param outputs Output tensor data pointers
   * @param has_collectives Whether operation uses collective communication
   */
  StableHloNode(const std::string& op_name, const std::string& cache_key,
                std::vector<uint8_t>&& ir_serialized, std::vector<void*>&& inputs,
                std::vector<void*>&& outputs, const bool has_collectives,
                mlir::MLIRContext* context);

  /**
   * @brief Construct StableHloNode from parsed MLIR module
   *
   * Creates a StableHloNode with an already-parsed MLIR module,
   * typically used for concatenated or transformed operations.
   *
   * @param op_name Operation name for identification
   * @param cache_key Unique cache identifier
   * @param module Parsed MLIR module containing StableHLO operations
   * @param context MLIR context managing the module
   * @param inputs Input tensor data pointers
   * @param outputs Output tensor data pointers
   * @param has_collectives Whether operation uses collective communication
   */
  StableHloNode(const std::string& op_name, const std::string& cache_key,
                std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>>& module,
                std::vector<void*>&& inputs, std::vector<void*>&& outputs,
                const bool has_collectives);

  StableHloNode(const std::string& op_name, const std::string&& cache_key,
                std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>>& module,
                std::vector<uint8_t>& ir_serialized, std::vector<void*>&& inputs,
                std::vector<void*>&& outputs, const bool has_collectives);

  void SetMergeMapping(std::unique_ptr<MergeMapping>&& mapping);
  const MergeMapping* GetMergeMapping() const;

  virtual ~StableHloNode() = default;

  /// Parsed MLIR module containing StableHLO operations
  std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>> module;

 private:
  /**
   * @brief Private constructor from OperationContext
   *
   * @note Making this constructor private because each OperationContext is a unique resource,
   * the conversion between a StableHloNode and OperationContext should be handled properly with
   * a saved mapping, instead of creating new object every time.
   * TODO: Used internally by IrNodeManager to create StableHloNode instances
   * from existing OperationContext objects.
   *
   * @param operation_context Source operation context
   */
  StableHloNode(const at::neuron::OperationContext& operation_context, mlir::MLIRContext* context);

  /// IrNodeManager has access to private constructor
  friend class IrNodeManager;

  // Only exist when the node is a concatenated node.
  std::unique_ptr<MergeMapping> merge_mapping;
};

/**
 * @brief Deallocation hint marker node for ordering-aware pruning
 *
 * Represents a deallocation hint in the IR timeline. Not an executable operation,
 * but a marker indicating when an address can be pruned from merged outputs.
 * Enables cache-safe, order-aware pruning without pointer matching.
 */
struct DeallocHintNode : public IrNode {
  /// Address being deallocated
  void* dealloc_address;

  /**
   * @brief Construct a deallocation hint node
   * @param addr Address to be deallocated
   */
  DeallocHintNode(void* addr)
      : IrNode("dealloc_hint", "", {}, IrNodeType::DEALLOC_HINT, {}, {}, false),
        dealloc_address(addr) {}

  virtual ~DeallocHintNode() = default;
};

/**
 * @brief Allocation hint marker node for ordering-aware tracking
 *
 * Represents an allocation hint in the IR timeline. Not an executable operation,
 * but a marker indicating when an address has been (re)allocated.
 * Used with DeallocHintNode to ensure tensors are not read after being pruned
 * unless they have been reallocated.
 */
struct AllocHintNode : public IrNode {
  /// Address being allocated
  void* alloc_address;

  /**
   * @brief Construct an allocation hint node
   * @param addr Address that was allocated
   */
  AllocHintNode(void* addr)
      : IrNode("alloc_hint", "", {}, IrNodeType::ALLOC_HINT, {}, {}, false), alloc_address(addr) {}

  virtual ~AllocHintNode() = default;
};

/**
 * @brief Legacy HLO node for backward compatibility
 *
 * Maintains compatibility with existing collective graph operations
 * that use the legacy HLO intermediate representation format.
 * New code should prefer StableHloNode for better optimization support.
 *
 * @deprecated Use StableHloNode for new implementations
 */
struct HloNode : public IrNode {
  /// Virtual destructor for proper cleanup
  virtual ~HloNode() = default;
};

}  // namespace torch_neuronx

#endif  // TORCH_NEURONX_IRNODE_H_

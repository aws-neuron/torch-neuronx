#ifndef TORCH_NEURONX_IR_CONCAT_STRATEGY_H_
#define TORCH_NEURONX_IR_CONCAT_STRATEGY_H_

#include <list>
#include <unordered_map>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "torch_neuronx/csrc/core/concatenation/IrNode.h"
#include "torch_neuronx/csrc/core/concatenation/IrNodeManager.h"
#include "torch_neuronx/csrc/core/concatenation/OpConcatUtils.h"

namespace torch_neuronx {

/**
 * @brief Abstract base class for IR concatenation strategies
 *
 * Defines the interface for implementing different concatenation optimization strategies.
 * Each strategy can analyze a list of IR nodes and determine which ones can be
 * concatenated together for improved compilation efficiency.
 *
 * The strategy pattern allows for pluggable concatenation algorithms that can be
 * tailored to specific operation patterns (e.g., transpose+matmul, convolution chains).
 */
class AbstractIrConcatStrategy {
 public:
  /**
   * @brief Check if operation can be a boundary for concatenation
   *
   * Determines if the given operation is a boundary node.
   * Boundary operations are those that defines a concatenation boundary. Operations between
   * 2 boundaries can be concatenated together.
   *
   * @param op_name Name of the operation to check
   * @return true if operation can be a fusion boundary, false otherwise
   */
  virtual bool IsFusibleBoundaryOperation(const std::string& op_name) = 0;

  /// Virtual destructor for proper cleanup of derived classes
  virtual ~AbstractIrConcatStrategy() = default;
};

class MatMulToMatMulStrategy : public AbstractIrConcatStrategy {
 public:
  /**
   * @brief Check if operation is a fusible boundary operation for MatMul chains
   *
   * Determines if the given operation can be part of a MatMul-to-MatMul fusion chain.
   * Only MatMul operations are considered fusible boundaries.
   *
   * @param op_name Name of the operation to check
   * @return true if operation is a MatMul, false otherwise
   */
  bool IsFusibleBoundaryOperation(const std::string& op_name) override;

  /// Virtual destructor for proper cleanup
  virtual ~MatMulToMatMulStrategy() = default;
};

}  // namespace torch_neuronx
#endif  // TORCH_NEURONX_IR_CONCAT_STRATEGY_H_

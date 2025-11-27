#ifndef TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_OP_MODULE_BUILDER_H_
#define TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_OP_MODULE_BUILDER_H_

#include <memory>
#include <string>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace torch_neuronx {

/**
 * @brief Base class for building MLIR modules with StableHLO operations
 *
 * This class provides common utilities and infrastructure for creating MLIR modules
 * containing StableHLO operations. It manages MLIR context, handles module creation,
 * and provides optional verification support.
 *
 * Subclasses should implement buildOperation() to create specific StableHLO operations.
 */
class OpModuleBuilder {
 public:
  /**
   * @brief Constructor
   * @param enable_verification If true, verifies the generated MLIR module (default: false)
   */
  explicit OpModuleBuilder(bool enable_verification = false);

  virtual ~OpModuleBuilder() = default;

  /**
   * @brief Build and return the MLIR module
   * @return MLIR module containing the operation
   * @throws std::runtime_error if module generation or verification fails
   */
  mlir::OwningOpRef<mlir::ModuleOp> build();

 protected:
  /**
   * @brief Get MLIR context
   * @return Reference to MLIR context
   */
  mlir::MLIRContext& getContext();

  /**
   * @brief Get MLIR builder
   * @return Reference to OpBuilder
   */
  mlir::OpBuilder& getBuilder();

  /**
   * @brief Create a function with the given name and type
   * @param name Function name
   * @param func_type Function type (inputs and outputs)
   * @return Created function operation
   */
  mlir::func::FuncOp createFunction(const std::string& name, mlir::FunctionType func_type);

  /**
   * @brief Pure virtual method to build the specific operation
   *
   * Subclasses must implement this to create their specific StableHLO operation
   * within the provided function. The function body and argument have already been
   * created, and the builder is positioned at the start of the function body.
   *
   * @param func The function to build the operation in
   * @param builder The OpBuilder positioned at function body start
   * @return The result value of the operation (to be returned by the function)
   */
  virtual mlir::Value buildOperation(mlir::func::FuncOp func, mlir::OpBuilder& builder) = 0;

  /**
   * @brief Get input tensor type (to be overridden by subclasses)
   * @return Input tensor type
   */
  virtual mlir::RankedTensorType getInputType() = 0;

  /**
   * @brief Get output tensor type (to be overridden by subclasses)
   * @return Output tensor type
   */
  virtual mlir::RankedTensorType getOutputType() = 0;

 private:
  bool enable_verification_;

  // Shared context (keeps context alive for returned modules)
  std::shared_ptr<mlir::MLIRContext> shared_context_;

  // Owned builder (unique to this instance)
  std::unique_ptr<mlir::OpBuilder> owned_builder_;

  // Raw pointers for convenient access (non-owning)
  mlir::MLIRContext* context_;
  mlir::OpBuilder* builder_;

  /**
   * @brief Initialize MLIR context and load required dialects
   */
  void initializeContext();

  /**
   * @brief Verify the generated module
   * @param module Module to verify
   * @throws std::runtime_error if verification fails
   */
  void verifyModule(mlir::ModuleOp module);
};

}  // namespace torch_neuronx

#endif  // TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_OP_MODULE_BUILDER_H_

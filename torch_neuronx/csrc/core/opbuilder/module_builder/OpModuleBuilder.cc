#include "OpModuleBuilder.h"

#include <memory>
#include <stdexcept>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"

namespace torch_neuronx {

// Shared context pool to keep contexts alive for returned modules
// Using shared_ptr to ensure context outlives the module
static std::shared_ptr<mlir::MLIRContext> getOrCreateSharedContext() {
  // Thread-local shared context to avoid cross-thread issues
  static thread_local std::shared_ptr<mlir::MLIRContext> tls_shared_context;

  if (!tls_shared_context) {
    tls_shared_context = std::make_shared<mlir::MLIRContext>();
    tls_shared_context->getOrLoadDialect<mlir::func::FuncDialect>();
    tls_shared_context->getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
  }

  return tls_shared_context;
}

OpModuleBuilder::OpModuleBuilder(bool enable_verification)
    : enable_verification_(enable_verification),
      shared_context_(getOrCreateSharedContext()),
      owned_builder_(nullptr),
      context_(shared_context_.get()),
      builder_(nullptr) {
  initializeContext();
}

void OpModuleBuilder::initializeContext() {
  // Context already initialized in getOrCreateSharedContext
  // Just create the builder
  owned_builder_ = std::make_unique<mlir::OpBuilder>(context_);
  builder_ = owned_builder_.get();
}

mlir::MLIRContext& OpModuleBuilder::getContext() { return *context_; }

mlir::OpBuilder& OpModuleBuilder::getBuilder() { return *builder_; }

mlir::func::FuncOp OpModuleBuilder::createFunction(const std::string& name,
                                                   mlir::FunctionType func_type) {
  auto func = builder_->create<mlir::func::FuncOp>(builder_->getUnknownLoc(), name, func_type);
  func.setPublic();
  return func;
}

void OpModuleBuilder::verifyModule(mlir::ModuleOp module) {
  if (enable_verification_) {
    TORCH_NEURONX_DEBUG("OpModuleBuilder: Verifying MLIR module");
    if (mlir::failed(mlir::verify(module))) {
      throw std::runtime_error("Generated MLIR module failed verification");
    }
    TORCH_NEURONX_DEBUG("OpModuleBuilder: Module verification passed");
  }
}

mlir::OwningOpRef<mlir::ModuleOp> OpModuleBuilder::build() {
  TORCH_NEURONX_DEBUG("OpModuleBuilder: Starting module build");

  // Create module
  auto module = builder_->create<mlir::ModuleOp>(builder_->getUnknownLoc());
  auto& moduleBody = module.getBodyRegion().front();
  builder_->setInsertionPointToEnd(&moduleBody);

  // Get input and output types
  auto inputType = getInputType();
  auto outputType = getOutputType();

  // Create main function
  auto funcType = builder_->getFunctionType({inputType}, {outputType});
  auto mainFunc = createFunction("main", funcType);

  // Create function body
  auto& funcBody = mainFunc.getBody().emplaceBlock();
  funcBody.addArgument(inputType, builder_->getUnknownLoc());
  builder_->setInsertionPointToStart(&funcBody);

  // Build the operation (implemented by subclass)
  mlir::Value result = buildOperation(mainFunc, *builder_);

  // Create return operation
  builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), result);

  // Verify if enabled
  verifyModule(module);

  TORCH_NEURONX_DEBUG("OpModuleBuilder: Successfully built module");

  return mlir::OwningOpRef<mlir::ModuleOp>(module);
}

}  // namespace torch_neuronx

#include "StableHloUtils.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"

namespace torch_neuronx {
namespace stablehlo_utils {

std::string moduleToString(mlir::ModuleOp module) {
  TORCH_NEURONX_DEBUG("moduleToString: Converting MLIR module to string");

  std::string result;
  llvm::raw_string_ostream stream(result);

  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(false);
  flags.printGenericOpForm(false);

  module.print(stream, flags);

  TORCH_NEURONX_DEBUG("moduleToString: Successfully converted module to string");

  return result;
}

std::string moduleToBytecode(mlir::ModuleOp module) {
  TORCH_NEURONX_DEBUG("moduleToBytecode: Converting MLIR module to bytecode");

  std::string result;
  llvm::raw_string_ostream stream(result);

  // Serialize to bytecode (binary format)
  if (mlir::failed(mlir::writeBytecodeToFile(module, stream))) {
    TORCH_NEURONX_DEBUG("ERROR: moduleToBytecode: Failed to convert module to bytecode");
    return "";
  }

  TORCH_NEURONX_DEBUG("moduleToBytecode: Successfully converted module to bytecode");
  return result;
}

mlir::OwningOpRef<mlir::ModuleOp> parseModuleFromBytecode(const std::string& bytecode,
                                                          mlir::MLIRContext* context) {
  TORCH_NEURONX_DEBUG("parseModuleFromBytecode: Parsing MLIR module from bytecode");

  if (context == nullptr) {
    TORCH_NEURONX_ERROR("parseModuleFromBytecode: context cannot be null");
    return nullptr;
  }

  llvm::StringRef data(bytecode);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(data, context);

  if (!module) {
    TORCH_NEURONX_DEBUG("ERROR: parseModuleFromBytecode: Failed to parse module from bytecode");
  } else {
    TORCH_NEURONX_DEBUG("parseModuleFromBytecode: Successfully parsed module from bytecode");
  }

  return module;
}

}  // namespace stablehlo_utils
}  // namespace torch_neuronx

#include "StableHloUtils.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
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

}  // namespace stablehlo_utils
}  // namespace torch_neuronx

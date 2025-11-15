#ifndef TORCH_NEURONX_CSRC_CORE_STABLEHLO_UTILITY_STABLEHLO_UTILS_H_
#define TORCH_NEURONX_CSRC_CORE_STABLEHLO_UTILITY_STABLEHLO_UTILS_H_

#include <string>

#include "mlir/IR/BuiltinOps.h"

namespace torch_neuronx {
namespace stablehlo_utils {

/**
 * @brief Convert MLIR module to string representation
 *
 * Converts an MLIR module to its string representation with proper formatting.
 * This operation is relatively expensive (~1900 μs for typical modules).
 *
 * @param module MLIR module to convert
 * @return MLIR module as formatted string
 *
 * @example
 * auto module = createTransposeModule(...);
 * std::string mlir_str = moduleToString(module.get());
 */
std::string moduleToString(mlir::ModuleOp module);

}  // namespace stablehlo_utils
}  // namespace torch_neuronx

#endif  // TORCH_NEURONX_CSRC_CORE_STABLEHLO_UTILITY_STABLEHLO_UTILS_H_

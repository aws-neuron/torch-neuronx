#ifndef TORCH_NEURONX_CSRC_CORE_STABLEHLO_UTILITY_STABLEHLO_UTILS_H_
#define TORCH_NEURONX_CSRC_CORE_STABLEHLO_UTILITY_STABLEHLO_UTILS_H_

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

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

/**
 * @brief Convert MLIR module to bytecode format
 *
 * Serializes an MLIR module to bytecode (binary) format.
 *
 * @param module MLIR module to convert
 * @return Bytecode representation as string, empty on failure
 */
std::string moduleToBytecode(mlir::ModuleOp module);

/**
 * @brief Parse MLIR module from bytecode format
 *
 * Deserializes an MLIR module from bytecode (binary) format.
 *
 * @param bytecode Bytecode string to parse
 * @param context MLIR context to use for parsing
 * @return OwningOpRef to parsed module, nullptr on failure
 */
mlir::OwningOpRef<mlir::ModuleOp> parseModuleFromBytecode(const std::string& bytecode,
                                                          mlir::MLIRContext* context);

}  // namespace stablehlo_utils
}  // namespace torch_neuronx

#endif  // TORCH_NEURONX_CSRC_CORE_STABLEHLO_UTILITY_STABLEHLO_UTILS_H_

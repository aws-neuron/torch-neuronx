#ifndef TORCH_NEURONX_CSRC_CORE_STABLEHLO_UTILITY_TYPE_UTILS_H_
#define TORCH_NEURONX_CSRC_CORE_STABLEHLO_UTILITY_TYPE_UTILS_H_

#include <string>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace torch_neuronx {
namespace type_utils {

/**
 * @brief Convert element type string to MLIR type
 *
 * Converts a string representation of a type (e.g., "f32", "i32") to the
 * corresponding MLIR type using the provided builder.
 *
 * @param builder MLIR OpBuilder to use for type creation
 * @param element_type Element type string
 *
 * Supported types:
 * - Floating point: "f16", "bf16", "f32", "f64"
 * - Signed integers: "i8", "i16", "i32", "i64"
 * - Unsigned integers: "ui8", "ui16", "ui32", "ui64"
 *
 * @return MLIR type corresponding to the string
 * @throws std::invalid_argument if element type is unsupported
 *
 * @example
 * mlir::OpBuilder builder(&context);
 * mlir::Type float_type = stringToMlirType(builder, "f32");
 * mlir::Type int_type = stringToMlirType(builder, "i32");
 */
mlir::Type stringToMlirType(mlir::OpBuilder& builder, const std::string& element_type);

}  // namespace type_utils
}  // namespace torch_neuronx

#endif  // TORCH_NEURONX_CSRC_CORE_STABLEHLO_UTILITY_TYPE_UTILS_H_

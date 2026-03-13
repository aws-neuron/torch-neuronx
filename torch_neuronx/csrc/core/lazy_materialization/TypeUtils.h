#ifndef TORCH_NEURONX_CSRC_CORE_LAZY_MATERIALIZATION_TYPE_UTILS_H_
#define TORCH_NEURONX_CSRC_CORE_LAZY_MATERIALIZATION_TYPE_UTILS_H_

#include <c10/core/ScalarType.h>

#include <string>

namespace c10_neuron {
namespace lazy {

/**
 * @brief Convert PyTorch scalar type to StableHLO element type string
 *
 * Converts c10::ScalarType to the corresponding StableHLO element type string
 * representation used in MLIR (e.g., Float -> "f32", Half -> "f16").
 *
 * @param scalar_type PyTorch scalar type to convert
 * @return StableHLO element type string ("f32", "f16", "bf16", "i32", "i64", "bool")
 * @throws std::runtime_error if scalar type is not supported
 *
 * @example
 * std::string type_str = ScalarTypeToElementTypeString(c10::ScalarType::Float);
 * // Returns "f32"
 */
std::string ScalarTypeToElementTypeString(c10::ScalarType scalar_type);

}  // namespace lazy
}  // namespace c10_neuron

#endif  // TORCH_NEURONX_CSRC_CORE_LAZY_MATERIALIZATION_TYPE_UTILS_H_

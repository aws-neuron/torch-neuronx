#pragma once

#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"

namespace c10_neuron {
namespace lazy {
namespace mlir_generators {

// ============================================================================
// TRANSPOSE - StableHLO transpose operation generation
// ============================================================================

/**
 * @brief Generate StableHLO MLIR for transpose operation
 *
 * Creates a transpose operation that permutes tensor dimensions according to
 * the specified permutation. The permutation is computed from source to
 * destination permutation states.
 *
 * @param source_perm Source permutation state (dimension ordering before)
 * @param dest_perm Destination permutation state (dimension ordering after)
 * @param input_shape Shape of the input tensor
 * @param element_type Element type as string (e.g., "f32", "bf16")
 * @return MLIR string for transpose operation
 *
 * @example
 * // Transpose from [0,1,2,3] to [0,2,1,3] (swap dims 1 and 2)
 * auto mlir = GenerateTranspose({0,1,2,3}, {0,2,1,3}, {2,3,4,5}, "bf16");
 */
std::string GenerateTranspose(const std::vector<int64_t>& source_perm,
                              const std::vector<int64_t>& dest_perm,
                              const std::vector<int64_t>& input_shape,
                              const std::string& element_type);

/**
 * @brief Generate StableHLO MLIR for transpose operation (MLIR type)
 *
 * Same as above but accepts mlir::Type directly instead of string.
 *
 * @param source_perm Source permutation state
 * @param dest_perm Destination permutation state
 * @param input_shape Shape of the input tensor
 * @param element_type Element type as mlir::Type
 * @return MLIR string for transpose operation
 */
std::string GenerateTranspose(const std::vector<int64_t>& source_perm,
                              const std::vector<int64_t>& dest_perm,
                              const std::vector<int64_t>& input_shape, mlir::Type element_type);

// ============================================================================
// RESHAPE - StableHLO reshape operation generation
// ============================================================================

/**
 * @brief Generate StableHLO MLIR for reshape operation
 *
 * Creates a reshape operation that changes tensor shape while preserving
 * element count. Input and output must have the same number of elements.
 *
 * @param input_shape Shape of the input tensor
 * @param output_shape Desired output shape
 * @param element_type Element type as string (e.g., "f32", "bf16")
 * @return MLIR string for reshape operation
 *
 * @throws std::invalid_argument if element counts don't match
 *
 * @example
 * // Reshape from [2,3,4] to [6,4]
 * auto mlir = GenerateReshape({2,3,4}, {6,4}, "f32");
 */
std::string GenerateReshape(const std::vector<int64_t>& input_shape,
                            const std::vector<int64_t>& output_shape,
                            const std::string& element_type);

/**
 * @brief Generate StableHLO MLIR for reshape operation (MLIR type)
 *
 * Same as above but accepts mlir::Type directly instead of string.
 *
 * @param input_shape Shape of the input tensor
 * @param output_shape Desired output shape
 * @param element_type Element type as mlir::Type
 * @return MLIR string for reshape operation
 */
std::string GenerateReshape(const std::vector<int64_t>& input_shape,
                            const std::vector<int64_t>& output_shape, mlir::Type element_type);

// ============================================================================
// SLICE - StableHLO slice operation generation
// ============================================================================

/**
 * @brief Generate StableHLO MLIR for slice operation
 *
 * Creates a slice operation that extracts a contiguous region from a tensor.
 * Start and end indices define the slice bounds for each dimension.
 *
 * @param input_shape Shape of the input tensor
 * @param start_indices Starting indices for each dimension
 * @param end_indices Ending indices (exclusive) for each dimension
 * @param element_type Element type as string (e.g., "f32", "bf16")
 * @return MLIR string for slice operation
 *
 * @throws std::invalid_argument if indices are invalid or out of bounds
 *
 * @example
 * // Slice [4,6,8] to get [2:4, 1:5, 0:8] -> output shape [2,4,8]
 * auto mlir = GenerateSlice({4,6,8}, {2,1,0}, {4,5,8}, "f32");
 */
std::string GenerateSlice(const std::vector<int64_t>& input_shape,
                          const std::vector<int64_t>& start_indices,
                          const std::vector<int64_t>& end_indices, const std::string& element_type);

/**
 * @brief Generate StableHLO MLIR for slice operation (MLIR type)
 *
 * Same as above but accepts mlir::Type directly instead of string.
 *
 * @param input_shape Shape of the input tensor
 * @param start_indices Starting indices for each dimension
 * @param end_indices Ending indices (exclusive) for each dimension
 * @param element_type Element type as mlir::Type
 * @return MLIR string for slice operation
 */
std::string GenerateSlice(const std::vector<int64_t>& input_shape,
                          const std::vector<int64_t>& start_indices,
                          const std::vector<int64_t>& end_indices, mlir::Type element_type);

// ============================================================================
// EMPTY - StableHLO empty/identity operation generation
// ============================================================================

/**
 * @brief Generate StableHLO MLIR for empty/identity operation
 *
 * Creates an empty operation that simply passes the input to the output
 * without any transformation. This is useful for inputs that require no
 * transformations but need a StableHLO module representation.
 *
 * @param shape Shape of the input/output tensor
 * @param element_type Element type as string (e.g., "f32", "bf16")
 * @return MLIR string for empty/identity operation
 *
 * @example
 * // Create identity for tensor of shape [2,3,4]
 * auto mlir = GenerateEmpty({2,3,4}, "f32");
 */
std::string GenerateEmpty(const std::vector<int64_t>& shape, const std::string& element_type);

/**
 * @brief Generate StableHLO MLIR for empty/identity operation (MLIR type)
 *
 * Same as above but accepts mlir::Type directly instead of string.
 *
 * @param shape Shape of the input/output tensor
 * @param element_type Element type as mlir::Type
 * @return MLIR string for empty/identity operation
 */
std::string GenerateEmpty(const std::vector<int64_t>& shape, mlir::Type element_type);

}  // namespace mlir_generators
}  // namespace lazy
}  // namespace c10_neuron

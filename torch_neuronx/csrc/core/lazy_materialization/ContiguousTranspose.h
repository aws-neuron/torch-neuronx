#ifndef TORCH_NEURONX_CSRC_CORE_LAZY_MATERIALIZATION_CONTIGUOUS_TRANSPOSE_H_
#define TORCH_NEURONX_CSRC_CORE_LAZY_MATERIALIZATION_CONTIGUOUS_TRANSPOSE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"

namespace torch_neuronx {

/**
 * @brief Compute transpose permutation from source to destination ordering
 *
 * Given a source permutation and destination permutation, computes the
 * transpose permutation needed to transform from source to destination.
 *
 * @param source_perm Source dimension ordering (e.g., [0, 1, 2, 3])
 * @param dest_perm Destination dimension ordering (e.g., [0, 2, 1, 3])
 * @return Transpose permutation array for StableHLO transpose operation
 *
 * @throws std::invalid_argument if permutations are invalid or incompatible
 *
 * @example
 * source_perm = [0, 1, 2, 3]
 * dest_perm = [0, 2, 1, 3]
 * result = [0, 2, 1, 3]  // Swap dimensions 1 and 2
 */
std::vector<int64_t> computeTransposePermutation(const std::vector<int64_t>& source_perm,
                                                 const std::vector<int64_t>& dest_perm);

/**
 * @brief Create StableHLO MLIR module with transpose operation
 * @tparam ElementType Either std::string or mlir::Type
 */
template <typename ElementType>
mlir::OwningOpRef<mlir::ModuleOp> createContiguousTransposeModule(
    const std::vector<int64_t>& source_perm, const std::vector<int64_t>& dest_perm,
    const std::vector<int64_t>& input_shape, ElementType element_type,
    bool enable_verification = false);

/**
 * @brief Generate StableHLO MLIR string with transpose operation
 * @tparam ElementType Either std::string or mlir::Type
 */
template <typename ElementType>
std::string generateContiguousTransposeMlir(const std::vector<int64_t>& source_perm,
                                            const std::vector<int64_t>& dest_perm,
                                            const std::vector<int64_t>& input_shape,
                                            ElementType element_type,
                                            bool enable_verification = false);

}  // namespace torch_neuronx

#endif  // TORCH_NEURONX_CSRC_CORE_LAZY_MATERIALIZATION_CONTIGUOUS_TRANSPOSE_H_

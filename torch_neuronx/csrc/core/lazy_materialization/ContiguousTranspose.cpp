#include "ContiguousTranspose.h"

#include <algorithm>
#include <stdexcept>

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/opbuilder/module_builder/TransposeOpBuilder.h"
#include "torch_neuronx/csrc/core/opbuilder/utility/StableHloUtils.h"

namespace torch_neuronx {

std::vector<int64_t> computeTransposePermutation(const std::vector<int64_t>& source_perm,
                                                 const std::vector<int64_t>& dest_perm) {
  TORCH_NEURONX_DEBUG("computeTransposePermutation: Computing transpose permutation");

  // Validate inputs
  if (source_perm.size() != dest_perm.size()) {
    throw std::invalid_argument("source_perm and dest_perm must have the same size (got " +
                                std::to_string(source_perm.size()) + " and " +
                                std::to_string(dest_perm.size()) + ")");
  }

  size_t rank = source_perm.size();

  // Create inverse mapping of source_perm: source_perm[i] -> i
  std::vector<int64_t> source_inverse(rank);
  for (size_t i = 0; i < rank; ++i) {
    source_inverse[source_perm[i]] = i;
  }

  // Compute transpose permutation: for each position in dest_perm,
  // find which position in source_perm has that dimension
  std::vector<int64_t> transpose_perm(rank);
  for (size_t i = 0; i < rank; ++i) {
    transpose_perm[i] = source_inverse[dest_perm[i]];
  }

  TORCH_NEURONX_DEBUG("computeTransposePermutation: Computed permutation");

  return transpose_perm;
}

// Template implementation
template <typename ElementType>
mlir::OwningOpRef<mlir::ModuleOp> createContiguousTransposeModule(
    const std::vector<int64_t>& source_perm, const std::vector<int64_t>& dest_perm,
    const std::vector<int64_t>& input_shape, ElementType element_type, bool enable_verification) {
  TORCH_NEURONX_DEBUG("createContiguousTransposeModule: Starting module creation");

  // Compute transpose permutation
  std::vector<int64_t> transpose_perm = computeTransposePermutation(source_perm, dest_perm);

  // Create builder and build module
  TransposeOpBuilder builder(transpose_perm, input_shape, element_type, enable_verification);
  auto module = builder.build();

  TORCH_NEURONX_DEBUG("createContiguousTransposeModule: Successfully created module");

  return module;
}

template <typename ElementType>
std::string generateContiguousTransposeMlir(const std::vector<int64_t>& source_perm,
                                            const std::vector<int64_t>& dest_perm,
                                            const std::vector<int64_t>& input_shape,
                                            ElementType element_type, bool enable_verification) {
  TORCH_NEURONX_DEBUG("generateContiguousTransposeMlir: Generating MLIR string");

  // Create the module
  auto module = createContiguousTransposeModule(source_perm, dest_perm, input_shape, element_type,
                                                enable_verification);

  // Convert to string
  std::string result = stablehlo_utils::moduleToString(module.get());

  TORCH_NEURONX_DEBUG("generateContiguousTransposeMlir: Successfully generated MLIR");

  return result;
}

// Explicit template instantiations
template mlir::OwningOpRef<mlir::ModuleOp> createContiguousTransposeModule<std::string>(
    const std::vector<int64_t>&, const std::vector<int64_t>&, const std::vector<int64_t>&,
    std::string, bool);

template mlir::OwningOpRef<mlir::ModuleOp> createContiguousTransposeModule<const char*>(
    const std::vector<int64_t>&, const std::vector<int64_t>&, const std::vector<int64_t>&,
    const char*, bool);

template mlir::OwningOpRef<mlir::ModuleOp> createContiguousTransposeModule<mlir::Type>(
    const std::vector<int64_t>&, const std::vector<int64_t>&, const std::vector<int64_t>&,
    mlir::Type, bool);

template std::string generateContiguousTransposeMlir<std::string>(const std::vector<int64_t>&,
                                                                  const std::vector<int64_t>&,
                                                                  const std::vector<int64_t>&,
                                                                  std::string, bool);

template std::string generateContiguousTransposeMlir<const char*>(const std::vector<int64_t>&,
                                                                  const std::vector<int64_t>&,
                                                                  const std::vector<int64_t>&,
                                                                  const char*, bool);

template std::string generateContiguousTransposeMlir<mlir::Type>(const std::vector<int64_t>&,
                                                                 const std::vector<int64_t>&,
                                                                 const std::vector<int64_t>&,
                                                                 mlir::Type, bool);

}  // namespace torch_neuronx

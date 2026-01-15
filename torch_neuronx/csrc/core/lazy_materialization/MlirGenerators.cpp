#include "torch_neuronx/csrc/core/lazy_materialization/MlirGenerators.h"

#include <algorithm>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/lazy_materialization/ContiguousTranspose.h"
#include "torch_neuronx/csrc/core/opbuilder/module_builder/EmptyOpBuilder.h"
#include "torch_neuronx/csrc/core/opbuilder/module_builder/ReshapeOpBuilder.h"
#include "torch_neuronx/csrc/core/opbuilder/module_builder/SliceOpBuilder.h"
#include "torch_neuronx/csrc/core/opbuilder/module_builder/TransposeOpBuilder.h"
#include "torch_neuronx/csrc/core/opbuilder/utility/StableHloUtils.h"

namespace c10_neuron {
namespace lazy {
namespace mlir_generators {

namespace {

// Static cache to store generated MLIR strings
// Maps cache_key -> mlir_str to avoid regenerating the same MLIR
static std::unordered_map<std::string, std::string> g_mlir_cache;
static std::mutex g_mlir_cache_mutex;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Calculate total number of elements in a shape
 */
int64_t CalculateNumElements(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
}

// ============================================================================
// Cache Key Generation Helpers
// ============================================================================

/**
 * @brief Generate cache key for transpose transformation
 */
std::string GenerateTransposeCacheKey(const std::vector<int64_t>& source_perm,
                                      const std::vector<int64_t>& dest_perm,
                                      const std::vector<int64_t>& input_shape,
                                      const std::string& element_type) {
  std::stringstream ss;
  ss << "transpose_";
  for (auto p : source_perm) ss << p << "_";
  ss << "to_";
  for (auto p : dest_perm) ss << p << "_";
  for (auto s : input_shape) ss << s << "x";
  ss << "_" << element_type;
  return ss.str();
}

/**
 * @brief Generate cache key for reshape transformation
 */
std::string GenerateReshapeCacheKey(const std::vector<int64_t>& input_shape,
                                    const std::vector<int64_t>& output_shape,
                                    const std::string& element_type) {
  std::stringstream ss;
  ss << "reshape_";
  for (auto s : input_shape) ss << s << "x";
  ss << "_to_";
  for (auto s : output_shape) ss << s << "x";
  ss << "_" << element_type;
  return ss.str();
}

/**
 * @brief Generate cache key for slice transformation
 */
std::string GenerateSliceCacheKey(const std::vector<int64_t>& input_shape,
                                  const std::vector<int64_t>& start_indices,
                                  const std::vector<int64_t>& end_indices,
                                  const std::string& element_type) {
  std::stringstream ss;
  ss << "slice_";
  for (auto s : input_shape) ss << s << "x";
  ss << "_start_";
  for (auto idx : start_indices) ss << idx << "_";
  ss << "end_";
  for (auto idx : end_indices) ss << idx << "_";
  ss << element_type;
  return ss.str();
}

/**
 * @brief Generate cache key for empty/identity transformation
 */
std::string GenerateEmptyCacheKey(const std::vector<int64_t>& shape,
                                  const std::string& element_type) {
  std::stringstream ss;
  ss << "empty_";
  for (auto s : shape) ss << s << "x";
  ss << "_" << element_type;
  return ss.str();
}

}  // anonymous namespace

// ============================================================================
// TRANSPOSE Implementation
// ============================================================================

std::string GenerateTranspose(const std::vector<int64_t>& source_perm,
                              const std::vector<int64_t>& dest_perm,
                              const std::vector<int64_t>& input_shape,
                              const std::string& element_type) {
  TORCH_NEURONX_DEBUG("MlirGenerators::GenerateTranspose", "input_shape_size=", input_shape.size(),
                      "element_type=", element_type);

  // Generate cache key
  std::string cache_key =
      GenerateTransposeCacheKey(source_perm, dest_perm, input_shape, element_type);

  // Check cache
  {
    std::lock_guard<std::mutex> lock(g_mlir_cache_mutex);
    auto it = g_mlir_cache.find(cache_key);
    if (it != g_mlir_cache.end()) {
      TORCH_NEURONX_DEBUG("MlirGenerators::GenerateTranspose: Cache HIT", "cache_key=", cache_key);
      return it->second;
    }
  }

  TORCH_NEURONX_DEBUG("MlirGenerators::GenerateTranspose: Cache MISS", "cache_key=", cache_key);

  // Compute transpose permutation using existing function
  std::vector<int64_t> transpose_perm =
      torch_neuronx::computeTransposePermutation(source_perm, dest_perm);

  // Create builder and build module
  torch_neuronx::TransposeOpBuilder builder(transpose_perm, input_shape, element_type,
                                            true /* enable_verification */);
  auto module = builder.build();

  // Convert to string
  std::string result = torch_neuronx::stablehlo_utils::moduleToString(module.get());

  // Store in cache
  {
    std::lock_guard<std::mutex> lock(g_mlir_cache_mutex);
    g_mlir_cache[cache_key] = result;
    TORCH_NEURONX_DEBUG("MlirGenerators::GenerateTranspose: Cached MLIR", "cache_key=", cache_key,
                        "cache_size=", g_mlir_cache.size());
  }

  TORCH_NEURONX_DEBUG("MlirGenerators::GenerateTranspose: Successfully generated MLIR");

  return result;
}

// ============================================================================
// RESHAPE Implementation
// ============================================================================

std::string GenerateReshape(const std::vector<int64_t>& input_shape,
                            const std::vector<int64_t>& output_shape,
                            const std::string& element_type) {
  TORCH_NEURONX_DEBUG("MlirGenerators::GenerateReshape", "input_shape_size=", input_shape.size(),
                      "output_shape_size=", output_shape.size(), "element_type=", element_type);

  // Generate cache key
  std::string cache_key = GenerateReshapeCacheKey(input_shape, output_shape, element_type);

  // Check cache
  {
    std::lock_guard<std::mutex> lock(g_mlir_cache_mutex);
    auto it = g_mlir_cache.find(cache_key);
    if (it != g_mlir_cache.end()) {
      TORCH_NEURONX_DEBUG("MlirGenerators::GenerateReshape: Cache HIT", "cache_key=", cache_key);
      return it->second;
    }
  }

  TORCH_NEURONX_DEBUG("MlirGenerators::GenerateReshape: Cache MISS", "cache_key=", cache_key);

  // Validate element counts match
  int64_t input_elements = CalculateNumElements(input_shape);
  int64_t output_elements = CalculateNumElements(output_shape);

  if (input_elements != output_elements) {
    throw std::invalid_argument("Reshape: element count mismatch. Input has " +
                                std::to_string(input_elements) + " elements, output has " +
                                std::to_string(output_elements) + " elements");
  }

  try {
    // Create ReshapeOpBuilder to generate MLIR
    torch_neuronx::ReshapeOpBuilder builder(input_shape, output_shape, element_type,
                                            true /* enable_verification */);

    // Build and get MLIR module
    auto mlir_module = builder.build();

    // Convert MLIR module to string
    std::string result = torch_neuronx::stablehlo_utils::moduleToString(mlir_module.get());

    // Store in cache
    {
      std::lock_guard<std::mutex> lock(g_mlir_cache_mutex);
      g_mlir_cache[cache_key] = result;
      TORCH_NEURONX_DEBUG("MlirGenerators::GenerateReshape: Cached MLIR", "cache_key=", cache_key,
                          "cache_size=", g_mlir_cache.size());
    }

    TORCH_NEURONX_DEBUG("MlirGenerators::GenerateReshape: Successfully generated MLIR");

    return result;
  } catch (const std::exception& e) {
    TORCH_NEURONX_DEBUG("Failed to generate reshape MLIR", "error=", e.what());
    throw;
  }
}

// ============================================================================
// SLICE Implementation
// ============================================================================

std::string GenerateSlice(const std::vector<int64_t>& input_shape,
                          const std::vector<int64_t>& start_indices,
                          const std::vector<int64_t>& end_indices,
                          const std::string& element_type) {
  TORCH_NEURONX_DEBUG("MlirGenerators::GenerateSlice", "input_shape_size=", input_shape.size(),
                      "element_type=", element_type);

  // Generate cache key
  std::string cache_key =
      GenerateSliceCacheKey(input_shape, start_indices, end_indices, element_type);

  // Check cache
  {
    std::lock_guard<std::mutex> lock(g_mlir_cache_mutex);
    auto it = g_mlir_cache.find(cache_key);
    if (it != g_mlir_cache.end()) {
      TORCH_NEURONX_DEBUG("MlirGenerators::GenerateSlice: Cache HIT", "cache_key=", cache_key);
      return it->second;
    }
  }

  TORCH_NEURONX_DEBUG("MlirGenerators::GenerateSlice: Cache MISS", "cache_key=", cache_key);

  // Validate inputs
  if (start_indices.size() != input_shape.size() || end_indices.size() != input_shape.size()) {
    throw std::invalid_argument(
        "Slice: start_indices and end_indices must have same size as input_shape");
  }

  // Validate each dimension
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (start_indices[i] < 0 || start_indices[i] > input_shape[i]) {
      throw std::invalid_argument("Slice: start_indices[" + std::to_string(i) +
                                  "] = " + std::to_string(start_indices[i]) +
                                  " is out of range [0, " + std::to_string(input_shape[i]) + "]");
    }
    if (end_indices[i] < start_indices[i] || end_indices[i] > input_shape[i]) {
      throw std::invalid_argument("Slice: end_indices[" + std::to_string(i) +
                                  "] = " + std::to_string(end_indices[i]) +
                                  " is invalid (must be in [" + std::to_string(start_indices[i]) +
                                  ", " + std::to_string(input_shape[i]) + "])");
    }
  }

  try {
    // Create strides vector (all 1s for simple slice without striding)
    std::vector<int64_t> strides(input_shape.size(), 1);

    // Create SliceOpBuilder to generate MLIR
    torch_neuronx::SliceOpBuilder builder(input_shape, start_indices, end_indices, strides,
                                          element_type, true /* enable_verification */);

    // Build and get MLIR module
    auto mlir_module = builder.build();

    // Convert MLIR module to string
    std::string result = torch_neuronx::stablehlo_utils::moduleToString(mlir_module.get());

    // Store in cache
    {
      std::lock_guard<std::mutex> lock(g_mlir_cache_mutex);
      g_mlir_cache[cache_key] = result;
      TORCH_NEURONX_DEBUG("MlirGenerators::GenerateSlice: Cached MLIR", "cache_key=", cache_key,
                          "cache_size=", g_mlir_cache.size());
    }

    TORCH_NEURONX_DEBUG("MlirGenerators::GenerateSlice: Successfully generated MLIR");

    return result;
  } catch (const std::exception& e) {
    TORCH_NEURONX_DEBUG("Failed to generate slice MLIR", "error=", e.what());
    throw;
  }
}

// ============================================================================
// EMPTY Implementation
// ============================================================================

std::string GenerateEmpty(const std::vector<int64_t>& shape, const std::string& element_type) {
  TORCH_NEURONX_DEBUG("MlirGenerators::GenerateEmpty", "shape_size=", shape.size(),
                      "element_type=", element_type);

  // Generate cache key
  std::string cache_key = GenerateEmptyCacheKey(shape, element_type);

  // Check cache
  {
    std::lock_guard<std::mutex> lock(g_mlir_cache_mutex);
    auto it = g_mlir_cache.find(cache_key);
    if (it != g_mlir_cache.end()) {
      TORCH_NEURONX_DEBUG("MlirGenerators::GenerateEmpty: Cache HIT", "cache_key=", cache_key);
      return it->second;
    }
  }

  TORCH_NEURONX_DEBUG("MlirGenerators::GenerateEmpty: Cache MISS", "cache_key=", cache_key);

  try {
    // Create EmptyOpBuilder to generate MLIR
    torch_neuronx::EmptyOpBuilder builder(shape, element_type, true /* enable_verification */);

    // Build and get MLIR module
    auto mlir_module = builder.build();

    // Convert MLIR module to string
    std::string result = torch_neuronx::stablehlo_utils::moduleToString(mlir_module.get());

    // Store in cache
    {
      std::lock_guard<std::mutex> lock(g_mlir_cache_mutex);
      g_mlir_cache[cache_key] = result;
      TORCH_NEURONX_DEBUG("MlirGenerators::GenerateEmpty: Cached MLIR", "cache_key=", cache_key,
                          "cache_size=", g_mlir_cache.size());
    }

    TORCH_NEURONX_DEBUG("MlirGenerators::GenerateEmpty: Successfully generated MLIR");

    return result;
  } catch (const std::exception& e) {
    TORCH_NEURONX_DEBUG("Failed to generate empty MLIR", "error=", e.what());
    throw;
  }
}

}  // namespace mlir_generators
}  // namespace lazy
}  // namespace c10_neuron

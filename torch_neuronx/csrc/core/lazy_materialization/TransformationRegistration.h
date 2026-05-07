#pragma once

namespace c10_neuron {
namespace lazy {

/**
 * @brief Register all supported transformations with the TransformationRegistry
 *
 * This function registers TRANSPOSE and SLICE transformations with the global
 * TransformationRegistry. Each registration includes:
 * - Creator: Detects transformation pattern from tensor properties
 * - Handler: Groups consecutive transformations for optimization
 * - Materializer: Generates MLIR and metadata for the transformation
 *
 * This function is automatically called via static initialization when the
 * module is loaded. It should only be called once.
 *
 * Registered transformations:
 * - TRANSPOSE (priority 100): Detects and materializes transpose patterns
 * - SLICE (priority 90): Detects and materializes slice patterns
 *
 * Note: RESHAPE is intentionally excluded as it cannot be detected from
 * tensor properties alone - reshapes are created explicitly by operations.
 */
void RegisterTransformations();

}  // namespace lazy
}  // namespace c10_neuron

#pragma once

#include <ATen/Tensor.h>

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "torch_neuronx/csrc/core/lazy_materialization/TransformationTypes.h"

namespace c10_neuron {
namespace lazy {

// ============================================================================
// Type Aliases for Clarity
// ============================================================================

// Creator function signature: Takes tensor and returns optional result
using TransformationCreator = std::function<std::optional<TransformationCreationResult>(
    const at::Tensor& input, const std::string& op_name, size_t input_index)>;

// ============================================================================
// TransformationDescriptor - Complete specification for a transformation
// ============================================================================

/**
 * @brief Complete descriptor for a registered transformation type
 *
 * This structure contains all components needed to support a transformation:
 * - Creator: Detects the transformation pattern from tensor properties
 * - Handler: Groups consecutive transformations and generates tasks
 *
 * Both components are required for registration - this enforces
 * complete transformation support.
 *
 * Note: MLIR generation is handled separately by TransformationMlirGenerator,
 * which takes TransformationTask objects from handlers.
 */
struct TransformationDescriptor {
  // Function to detect transformation pattern from tensor
  TransformationCreator creator;

  // Handler for grouping and processing transformations
  std::unique_ptr<TransformationHandler> handler;

  // Priority for detection (higher = checked first)
  // Useful when multiple transformations could match
  int priority;

  // Human-readable name for logging and debugging
  std::string name;

  // Validate that all required components are present
  bool IsComplete() const { return creator && handler && !name.empty(); }
};

// ============================================================================
// TransformationRegistry - Central registry for all transformations
// ============================================================================

/**
 * @brief Singleton registry managing all transformation types
 *
 * This registry acts as the single source of truth for transformation support.
 * It enforces that each transformation type has complete implementation
 * (creator, handler, and materializer) before it can be used.
 *
 * Key responsibilities:
 * - Register transformation types with their implementations
 * - Lookup transformation components by type
 * - Provide automatic discovery of all registered transformations
 * - Enforce consistency across all APIs
 *
 * Usage:
 *   // Register a transformation
 *   TransformationRegistry::Get().Register(
 *       TransformationType::TRANSPOSE,
 *       TransformationDescriptor{...});
 *
 *   // Lookup a transformation
 *   auto* desc = TransformationRegistry::Get().Lookup(
 *       TransformationType::TRANSPOSE);
 *
 *   // Try to detect any transformation
 *   auto result = TransformationRegistry::Get()
 *                   .TryCreateAnyTransformation(tensor, ...);
 */
class TransformationRegistry {
 public:
  /**
   * @brief Get the singleton instance
   *
   * The registry is initialized on first access and persists for the
   * lifetime of the process.
   */
  static TransformationRegistry& Get();

  /**
   * @brief Register a transformation type with its complete implementation
   *
   * This method registers all components needed to support a transformation.
   * Registration will fail (throw exception) if:
   * - The type is already registered
   * - The descriptor is incomplete (missing creator, handler, or materializer)
   *
   * @param type The transformation type to register
   * @param descriptor Complete descriptor with all components
   * @throws std::invalid_argument if type already registered or descriptor incomplete
   */
  void Register(TransformationType type, TransformationDescriptor descriptor);

  /**
   * @brief Lookup transformation descriptor by type
   *
   * Returns nullptr if the transformation type is not registered.
   * This is the primary way for callers to access transformation components.
   *
   * @param type The transformation type to look up
   * @return Pointer to descriptor if registered, nullptr otherwise
   */
  const TransformationDescriptor* Lookup(TransformationType type) const;

  /**
   * @brief Get handler for a transformation type
   *
   * Convenience method that looks up the descriptor and returns the handler.
   * Returns nullptr if the type is not registered.
   *
   * @param type The transformation type
   * @return Pointer to handler if registered, nullptr otherwise
   */
  const TransformationHandler* GetHandler(TransformationType type) const;

  /**
   * @brief Get all registered transformation types
   *
   * Returns types in priority order (highest priority first).
   * Useful for debugging and validation.
   *
   * @return Vector of registered transformation types, ordered by priority
   */
  std::vector<TransformationType> GetAllTypes() const;

  /**
   * @brief Try to create transformation from tensor using any registered creator
   *
   * This method tries all registered transformation creators in priority order
   * until one successfully detects a pattern. This provides automatic discovery
   * - when new transformations are registered, they're automatically tried.
   *
   * @param input The tensor to analyze for transformation patterns
   * @param op_name Name of the operation (for logging)
   * @param input_index Index of this input in the operation (for logging)
   * @return TransformationCreationResult if pattern detected, std::nullopt otherwise
   *
   * @example
   * auto result = TransformationRegistry::Get()
   *                 .TryCreateAnyTransformation(input, "aten::matmul", 0);
   * if (result.has_value()) {
   *   // Transformation detected! Process it.
   *   transformations.push_back(result->transformation);
   * }
   */
  std::optional<TransformationCreationResult> TryCreateAnyTransformation(const at::Tensor& input,
                                                                         const std::string& op_name,
                                                                         size_t input_index) const;

  /**
   * @brief Check if a transformation type is registered
   *
   * @param type The transformation type to check
   * @return true if registered, false otherwise
   */
  bool IsRegistered(TransformationType type) const;

  /**
   * @brief Get count of registered transformations
   *
   * @return Number of registered transformation types
   */
  size_t GetRegistrationCount() const { return registry_.size(); }

  // Prevent copying and moving
  TransformationRegistry(const TransformationRegistry&) = delete;
  TransformationRegistry& operator=(const TransformationRegistry&) = delete;
  TransformationRegistry(TransformationRegistry&&) = delete;
  TransformationRegistry& operator=(TransformationRegistry&&) = delete;

 private:
  // Private constructor for singleton
  TransformationRegistry();

  // Registry storage: type -> descriptor
  // Using std::map to maintain ordered iteration by type
  std::map<TransformationType, TransformationDescriptor> registry_;

  // Priority-sorted list of types for TryCreateAnyTransformation
  // Updated when registrations occur
  mutable std::vector<TransformationType> priority_sorted_types_;
  mutable bool priority_sort_dirty_ = true;

  // Update the priority-sorted list if needed
  void UpdatePrioritySortedTypes() const;
};

}  // namespace lazy
}  // namespace c10_neuron

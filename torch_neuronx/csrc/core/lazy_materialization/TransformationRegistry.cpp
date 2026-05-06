#include "torch_neuronx/csrc/core/lazy_materialization/TransformationRegistry.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "torch_neuronx/csrc/core/NeuronLogging.h"

namespace c10_neuron {
namespace lazy {

// ============================================================================
// TransformationRegistry Implementation
// ============================================================================

TransformationRegistry::TransformationRegistry() {
  TORCH_NEURONX_DEBUG("TransformationRegistry: Initialized (registrations will follow)");
}

TransformationRegistry& TransformationRegistry::Get() {
  static TransformationRegistry instance;
  return instance;
}

void TransformationRegistry::Register(TransformationType type,
                                      TransformationDescriptor descriptor) {
  // Validate descriptor completeness
  if (!descriptor.IsComplete()) {
    std::ostringstream oss;
    oss << "TransformationRegistry::Register: Incomplete descriptor for type "
        << static_cast<int>(type);
    if (!descriptor.creator) oss << " (missing creator)";
    if (!descriptor.handler) oss << " (missing handler)";
    if (descriptor.name.empty()) oss << " (missing name)";
    throw std::invalid_argument(oss.str());
  }

  // Check if already registered
  if (registry_.count(type) > 0) {
    throw std::invalid_argument("TransformationRegistry::Register: Type " + descriptor.name + " (" +
                                std::to_string(static_cast<int>(type)) + ") is already registered");
  }

  TORCH_NEURONX_DEBUG("TransformationRegistry: Registering", "type=", descriptor.name,
                      "priority=", descriptor.priority);

  // Insert into registry
  registry_[type] = std::move(descriptor);

  // Mark priority sort as dirty
  priority_sort_dirty_ = true;

  TORCH_NEURONX_DEBUG("TransformationRegistry: Registration complete",
                      "total_registered=", registry_.size());
}

const TransformationDescriptor* TransformationRegistry::Lookup(TransformationType type) const {
  auto it = registry_.find(type);
  if (it == registry_.end()) {
    return nullptr;
  }
  return &it->second;
}

const TransformationHandler* TransformationRegistry::GetHandler(TransformationType type) const {
  auto* descriptor = Lookup(type);
  if (!descriptor) {
    return nullptr;
  }
  return descriptor->handler.get();
}

std::vector<TransformationType> TransformationRegistry::GetAllTypes() const {
  UpdatePrioritySortedTypes();
  return priority_sorted_types_;
}

std::optional<TransformationCreationResult> TransformationRegistry::TryCreateAnyTransformation(
    const at::Tensor& input, const std::string& op_name, size_t input_index) const {
  // Ensure priority sorted list is up to date
  UpdatePrioritySortedTypes();

  // Try each creator in priority order
  for (const auto& type : priority_sorted_types_) {
    auto it = registry_.find(type);
    if (it == registry_.end()) {
      // This should never happen - priority_sorted_types_ should only contain registered types
      TORCH_CHECK(false,
                  "TransformationRegistry: Type in priority_sorted_types_ not found in registry. "
                  "This indicates a bug in the registry implementation. Type: ",
                  static_cast<int>(type));
    }

    const auto& descriptor = it->second;

    TORCH_NEURONX_DEBUG("TransformationRegistry: Trying creator", "type=", descriptor.name,
                        "op=", op_name, "input_idx=", input_index);

    // Try this creator
    auto result = descriptor.creator(input, op_name, input_index);

    if (result.has_value()) {
      TORCH_NEURONX_DEBUG("TransformationRegistry: Pattern detected", "type=", descriptor.name,
                          "pattern=", result->pattern_name);
      return result;
    }
  }

  // No transformation detected
  TORCH_NEURONX_DEBUG("TransformationRegistry: No transformation pattern detected", "op=", op_name,
                      "input_idx=", input_index);
  return std::nullopt;
}

bool TransformationRegistry::IsRegistered(TransformationType type) const {
  return registry_.count(type) > 0;
}

void TransformationRegistry::UpdatePrioritySortedTypes() const {
  if (!priority_sort_dirty_) {
    return;
  }

  // Build list of types with their priorities
  std::vector<std::pair<TransformationType, int>> type_priority_pairs;
  type_priority_pairs.reserve(registry_.size());

  for (const auto& [type, descriptor] : registry_) {
    type_priority_pairs.emplace_back(type, descriptor.priority);
  }

  // Sort by priority (highest first)
  std::sort(type_priority_pairs.begin(), type_priority_pairs.end(),
            [](const auto& a, const auto& b) {
              // Higher priority first
              if (a.second != b.second) {
                return a.second > b.second;
              }
              // If same priority, use type enum value for stable ordering
              return static_cast<int>(a.first) < static_cast<int>(b.first);
            });

  // Extract just the types
  priority_sorted_types_.clear();
  priority_sorted_types_.reserve(type_priority_pairs.size());
  for (const auto& [type, priority] : type_priority_pairs) {
    priority_sorted_types_.push_back(type);
  }

  priority_sort_dirty_ = false;

  TORCH_NEURONX_DEBUG("TransformationRegistry: Priority sort updated",
                      "count=", priority_sorted_types_.size());
}

}  // namespace lazy
}  // namespace c10_neuron

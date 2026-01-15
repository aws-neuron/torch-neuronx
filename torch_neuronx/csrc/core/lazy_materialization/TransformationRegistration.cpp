#include <sstream>

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/lazy_materialization/MlirGenerators.h"
#include "torch_neuronx/csrc/core/lazy_materialization/TransformationRegistry.h"
#include "torch_neuronx/csrc/core/lazy_materialization/Transformations.h"

namespace c10_neuron {
namespace lazy {

// ============================================================================
// TRANSPOSE Registration
// ============================================================================

namespace {

// Transpose creator wrapper - detects transpose pattern from tensor
std::optional<TransformationCreationResult> CreateTransposeTransformation(
    const at::Tensor& input, const std::string& op_name, size_t input_index) {
  TORCH_NEURONX_DEBUG("CreateTransposeTransformation called", "op=", op_name,
                      "input_index=", input_index);
  auto result = Creators::TryCreateTranspose(input, op_name, input_index);
  TORCH_NEURONX_DEBUG("CreateTransposeTransformation returning", "op=", op_name,
                      "has_value=", result.has_value());
  return result;
}

}  // anonymous namespace

// ============================================================================
// SLICE Registration
// ============================================================================

namespace {

// Slice creator wrapper - detects slice pattern from tensor
std::optional<TransformationCreationResult> CreateSliceTransformation(const at::Tensor& input,
                                                                      const std::string& op_name,
                                                                      size_t input_index) {
  return Creators::TryCreateSlice(input, op_name, input_index);
}

}  // anonymous namespace

// ============================================================================
// Registration Function
// ============================================================================

void RegisterTransformations() {
  // Make this idempotent - only register once
  static bool registered = false;
  if (registered) {
    TORCH_NEURONX_DEBUG("RegisterTransformations: Already registered, skipping");
    return;
  }
  registered = true;

  auto& registry = TransformationRegistry::Get();

  TORCH_NEURONX_DEBUG("RegisterTransformations: Starting registration");

  // Register TRANSPOSE
  {
    TransformationDescriptor transpose_desc;
    transpose_desc.creator = CreateTransposeTransformation;
    transpose_desc.handler = std::make_unique<Handlers::TransposeHandler>();
    transpose_desc.priority = 100;  // High priority - transpose is common
    transpose_desc.name = "TRANSPOSE";

    registry.Register(TransformationType::TRANSPOSE, std::move(transpose_desc));
    TORCH_NEURONX_DEBUG("RegisterTransformations: Registered TRANSPOSE");
  }

  // Register SLICE
  {
    TransformationDescriptor slice_desc;
    slice_desc.creator = CreateSliceTransformation;
    slice_desc.handler = std::make_unique<Handlers::SliceHandler>();
    slice_desc.priority = 90;  // Slightly lower than transpose
    slice_desc.name = "SLICE";

    registry.Register(TransformationType::SLICE, std::move(slice_desc));
    TORCH_NEURONX_DEBUG("RegisterTransformations: Registered SLICE");
  }

  TORCH_NEURONX_DEBUG("RegisterTransformations: Complete",
                      "total_registered=", registry.GetRegistrationCount());
}

// Static initializer to auto-register on startup
namespace {
struct TransformationRegistrar {
  TransformationRegistrar() { RegisterTransformations(); }
};
static TransformationRegistrar g_transformation_registrar;
}  // anonymous namespace

}  // namespace lazy
}  // namespace c10_neuron

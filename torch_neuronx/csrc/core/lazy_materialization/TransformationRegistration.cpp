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
// BROADCAST Registration
// ============================================================================

namespace {

// Broadcast creator wrapper - detects broadcast pattern from tensor
std::optional<TransformationCreationResult> CreateBroadcastTransformation(
    const at::Tensor& input, const std::string& op_name, size_t input_index) {
  return Creators::TryCreateBroadcast(input, op_name, input_index);
}

}  // anonymous namespace

// ============================================================================
// RESHAPE Registration
// ============================================================================

namespace {

// Reshape creator — RESHAPE transformations are emitted internally by TryCreateCompound
// (to insert size-1 dims before BROADCAST). There is no standalone tensor pattern that
// maps to RESHAPE, so this creator always returns nullopt.
std::optional<TransformationCreationResult> CreateReshapeTransformation(
    const at::Tensor& /*input*/, const std::string& /*op_name*/, size_t /*input_index*/) {
  return std::nullopt;  // RESHAPE is only emitted internally, never detected from tensor strides
}

// Reshape handler: passes the RESHAPE transformation through to GenerateMlir.
// No grouping needed — each RESHAPE is processed individually.
class ReshapeHandler : public TransformationHandler {
 public:
  size_t FindGroupEnd(const std::vector<TensorTransformation>& /*transformations*/,
                      size_t start_index, const TransformationState& /*state*/) const override {
    return start_index + 1;
  }

  TransformationTask ProcessGroup(size_t input_index, TransformationState& state,
                                  const std::vector<TensorTransformation>& group,
                                  const std::string& op_name) const override {
    TORCH_NEURONX_DEBUG("Processing RESHAPE", "op=", op_name, "input_index=", input_index);
    const auto& transform = group[0];
    state.current_shape = transform.output_shape;

    return TransformationTask{.input_index = input_index,
                              .type = TransformationType::RESHAPE,
                              .input_shape = transform.input_shape,
                              .output_shape = transform.output_shape,
                              .element_type = transform.element_type,
                              .current_perm = state.current_perm,
                              .params = std::monostate{}};
  }
};

}  // anonymous namespace

// ============================================================================
// COMPOUND Registration
// ============================================================================

namespace {

// Compound creator wrapper - general stride decomposition
std::optional<TransformationCreationResult> CreateCompoundTransformation(const at::Tensor& input,
                                                                         const std::string& op_name,
                                                                         size_t input_index) {
  return Creators::TryCreateCompound(input, op_name, input_index);
}

// Compound handler: the compound creator emits SLICE and TRANSPOSE transformations,
// so the existing SliceHandler and TransposeHandler process them in CollectTransformationTasks.
// This handler is a safety placeholder required by the registry but should not be reached
// because compound never emits TensorTransformation with type=COMPOUND.
class CompoundHandler : public TransformationHandler {
 public:
  size_t FindGroupEnd(const std::vector<TensorTransformation>& transformations, size_t start_index,
                      const TransformationState& /*state*/) const override {
    return start_index + 1;
  }

  TransformationTask ProcessGroup(size_t input_index, TransformationState& state,
                                  const std::vector<TensorTransformation>& group,
                                  const std::string& op_name) const override {
    // This should never be called — compound produces SLICE/TRANSPOSE types, not COMPOUND.
    TORCH_CHECK(false,
                "CompoundHandler::ProcessGroup should never be called. "
                "Compound creator emits SLICE and TRANSPOSE transformations, "
                "not COMPOUND type. op=",
                op_name, " input_index=", input_index);
    // Unreachable, but silence compiler warnings
    return TransformationTask{};
  }
};

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
  // TODO(mengchiy): Remove TryCreateTranspose and let TryCreateCompound handle all cases.
  // Then fix the base tensor shape inconsistency in compound by always computing the base
  // shape as the full storage extent from offset 0, ensuring all views of the same storage
  // get the same base shape.
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

  // Register BROADCAST
  {
    TransformationDescriptor broadcast_desc;
    broadcast_desc.creator = CreateBroadcastTransformation;
    broadcast_desc.handler = std::make_unique<Handlers::BroadcastHandler>();
    broadcast_desc.priority = 85;  // Lower than slice
    broadcast_desc.name = "BROADCAST";

    registry.Register(TransformationType::BROADCAST, std::move(broadcast_desc));
    TORCH_NEURONX_DEBUG("RegisterTransformations: Registered BROADCAST");
  }

  // Register RESHAPE
  // RESHAPE is emitted internally by TryCreateCompound (to insert size-1 dims before BROADCAST).
  // The creator always returns nullopt (no standalone detection from tensor strides).
  {
    TransformationDescriptor reshape_desc;
    reshape_desc.creator = CreateReshapeTransformation;
    reshape_desc.handler = std::make_unique<ReshapeHandler>();
    reshape_desc.priority = 80;  // Between BROADCAST and COMPOUND
    reshape_desc.name = "RESHAPE";

    registry.Register(TransformationType::RESHAPE, std::move(reshape_desc));
    TORCH_NEURONX_DEBUG("RegisterTransformations: Registered RESHAPE");
  }

  // Register COMPOUND (general stride decomposition)
  // Runs after TRANSPOSE and SLICE — handles compound patterns they cannot.
  // The creator emits SLICE and TRANSPOSE TensorTransformations which are
  // processed by the existing handlers in CollectTransformationTasks.
  {
    TransformationDescriptor compound_desc;
    compound_desc.creator = CreateCompoundTransformation;
    compound_desc.handler = std::make_unique<CompoundHandler>();
    compound_desc.priority = 50;  // Lower priority - only for compound patterns
    compound_desc.name = "COMPOUND";

    registry.Register(TransformationType::COMPOUND, std::move(compound_desc));
    TORCH_NEURONX_DEBUG("RegisterTransformations: Registered COMPOUND");
  }

  // Register COMPOUND_FOLDED (folded decomposition for narrow→view→permute patterns)
  // Runs after COMPOUND — handles patterns where GCD check fails due to coprime strides.
  // The creator emits SLICE, RESHAPE, and TRANSPOSE TensorTransformations which are
  // processed by the existing handlers in CollectTransformationTasks.
  {
    TransformationDescriptor folded_desc;
    folded_desc.creator = [](const at::Tensor& input, const std::string& op_name,
                             size_t input_index) {
      return Creators::TryCreateFoldedDecomposition(input, op_name, input_index);
    };
    folded_desc.handler = std::make_unique<CompoundHandler>();
    folded_desc.priority = 45;  // After COMPOUND at 50
    folded_desc.name = "COMPOUND_FOLDED";

    registry.Register(TransformationType::COMPOUND_FOLDED, std::move(folded_desc));
    TORCH_NEURONX_DEBUG("RegisterTransformations: Registered COMPOUND_FOLDED");
  }

  // Register SCATTER_IOTA (epilogue only — scatter with iota-computed flat indices)
  // The creator always returns nullopt (detection is done by TryCreateEpilogue).
  // The handler converts TensorTransformation → TransformationTask for MLIR generation.
  {
    TransformationDescriptor scatter_iota_desc;
    scatter_iota_desc.creator = [](const at::Tensor&, const std::string&,
                                   size_t) -> std::optional<TransformationCreationResult> {
      return std::nullopt;
    };
    scatter_iota_desc.handler = std::make_unique<Handlers::ScatterIotaHandler>();
    scatter_iota_desc.priority = 0;  // Never used for detection — epilogue only
    scatter_iota_desc.name = "SCATTER_IOTA";

    registry.Register(TransformationType::SCATTER_IOTA, std::move(scatter_iota_desc));
    TORCH_NEURONX_DEBUG("RegisterTransformations: Registered SCATTER_IOTA");
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

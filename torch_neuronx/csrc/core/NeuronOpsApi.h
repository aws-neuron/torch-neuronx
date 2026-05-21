// NeuronOpsApi.h — Shared declarations for operation submission helpers.
//
// These functions and types are used by both NeuronBindings.cpp (Python bindings)
// and NeuronDispatcher.cpp (C++ dispatch path). Extracting them into a common
// header avoids ODR violations from duplicate definitions and eliminates the
// need for fragile forward declarations.
//
// The *implementations* live in NeuronBindings.cpp.

#pragma once

#include <torch/torch.h>

#include <string>
#include <utility>
#include <vector>

#include "torch_neuronx/csrc/core/KernelExecution.h"                           // TensorDataRef
#include "torch_neuronx/csrc/core/lazy_materialization/TransformationTypes.h"  // TensorTransformation
#include "torch_neuronx/csrc/core/utils/TensorContext.h"                       // TensorContext

namespace torch_neuronx {

// ═══════════════════════════════════════════════════════════════
//  InputTransformationResult
//
//  Holds the output of ProcessInputTransformations():
//    - final_inputs:        tensors after contiguous / transformation processing
//    - kernel_transforms:   per-input transformation metadata for prologue
//    - has_input_transforms: quick flag — true if any kernel_transforms[i] is non-empty
// ═══════════════════════════════════════════════════════════════

struct InputTransformationResult {
  std::vector<torch::Tensor> final_inputs;
  std::vector<std::vector<c10_neuron::lazy::TensorTransformation>> kernel_transforms;
  bool has_input_transforms = false;
};

// ═══════════════════════════════════════════════════════════════
//  ProcessInputTransformations
//
//  Walks each input tensor:
//    - If contiguous → pass through.
//    - If non-contiguous + collective → make contiguous via NRT strided copy.
//    - If non-contiguous + non-collective → try lazy transformation; fallback
//      to NRT strided copy if unsupported.
//
//  Defined in NeuronBindings.cpp.
// ═══════════════════════════════════════════════════════════════

InputTransformationResult ProcessInputTransformations(const std::string& op_name,
                                                      const std::vector<torch::Tensor>& inputs,
                                                      bool has_collectives);

// ═══════════════════════════════════════════════════════════════
//  IsEpilogueEnabled
//
//  Check if the epilogue optimization is enabled via environment variable.
//  Cached on first call — env var is read once at startup.
//  When enabled, non-contiguous output tensors are handled by fusing an
//  inverse transformation into the operation MLIR, eliminating the separate
//  CopyBackToOriginal / .copy_() step.
// ═══════════════════════════════════════════════════════════════

inline bool IsEpilogueEnabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("TORCH_NEURONX_ENABLE_EPILOGUE");
    return env == nullptr || std::string(env) != "0";
  }();
  return enabled;
}

// ═══════════════════════════════════════════════════════════════
//  OutputTransformationResult
//
//  Holds the output of ProcessOutputTransformations():
//    - final_outputs:        tensors after epilogue processing (base tensors or contiguous temps)
//    - kernel_transforms:    per-output transformation metadata for epilogue
//    - has_output_transforms: quick flag — true if any kernel_transforms[i] is non-empty
// ═══════════════════════════════════════════════════════════════

struct OutputTransformationResult {
  std::vector<torch::Tensor> final_outputs;
  std::vector<std::vector<c10_neuron::lazy::TensorTransformation>> kernel_transforms;
  bool has_output_transforms = false;
};

// ═══════════════════════════════════════════════════════════════
//  ProcessOutputTransformations
//
//  Walks each output tensor:
//    - If contiguous or epilogue disabled → pass through.
//    - If non-contiguous + epilogue enabled → try epilogue transformation;
//      fallback to contiguous temp if unsupported.
//
//  Defined in NeuronBindings.cpp.
// ═══════════════════════════════════════════════════════════════

OutputTransformationResult ProcessOutputTransformations(const std::string& op_name,
                                                        const std::vector<torch::Tensor>& outputs,
                                                        bool has_collectives);

// ═══════════════════════════════════════════════════════════════
//  PrepareKernelTensors
//
//  Converts a vector of torch::Tensors into the NRT-level
//  (TensorDataRef, TensorContext) pairs required by kernel execution.
//
//  By default, throws if any tensor is non-contiguous.  Callers that
//  know the kernel accounts for non-contiguous layouts (e.g.
//  aten::contiguous lowering, prologue transformations) can pass
//  allow_non_contiguous=true to opt out of the check.  In that case
//  TensorContext.size_bytes is computed from the physical storage byte
//  range rather than numel * element_size.
//
//  Defined in NeuronBindings.cpp.
// ═══════════════════════════════════════════════════════════════

std::pair<std::vector<at::neuron::TensorDataRef>, std::vector<at::neuron::TensorContext>>
PrepareKernelTensors(const std::vector<torch::Tensor>& tensors, bool allow_non_contiguous = false);

}  // namespace torch_neuronx

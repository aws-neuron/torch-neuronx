#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>

#include <string>

#include "torch_neuronx/csrc/core/KernelExecution.h"

namespace at::neuron {

// Submit an async kRead*N → kHost → kWrite*M chain on the current Neuron stream.
// Caller owns: contiguity of every tensor, same-device, and any op-specific
// precondition / stream (hostCC) gating. Pinned-buffer lifetime is handled
// internally.
bool SubmitHostChain(const std::string& op_name, c10::ArrayRef<at::Tensor> inputs,
                     c10::ArrayRef<at::Tensor> outputs, HostKernelExecution::HostFn fn);

}  // namespace at::neuron

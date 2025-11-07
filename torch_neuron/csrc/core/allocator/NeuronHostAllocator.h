#pragma once

#include <c10/core/Allocator.h>

namespace torch_neuron {

// Get the singleton instance of NeuronHostAllocator
at::Allocator* GetNeuronHostAllocator();

}  // namespace torch_neuron

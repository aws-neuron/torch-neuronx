// Mock implementations of Python utility functions for testing

#include <string>

#include "torch/torch.h"

namespace torch_neuronx {

namespace ops {

// Mock contiguous operation - only this one is missing from aten_neuron
at::Tensor call_python_contiguous_op(const at::Tensor& tensor, c10::MemoryFormat memory_format) {
  // Simple mock - just return contiguous tensor
  return tensor.contiguous(memory_format);
}

}  // namespace ops

}  // namespace torch_neuronx

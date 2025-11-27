#pragma once

#include <ATen/Tensor.h>

namespace torch_neuronx {
namespace ops {

// Internal contiguous operation that can be called from C++ without going through dispatcher
// This is used internally to avoid recursion when copy operations need to make tensors contiguous
at::Tensor contiguous_internal(const at::Tensor& self,
                               c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous);

}  // namespace ops
}  // namespace torch_neuronx

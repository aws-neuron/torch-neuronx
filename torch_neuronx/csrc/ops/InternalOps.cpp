#include "InternalOps.h"

#include <torch/extension.h>

#include "../utils/CopyUtils.h"

namespace torch_neuronx {
namespace ops {

// Forward declaration of Python binding function
at::Tensor call_python_contiguous_op(const at::Tensor& self, c10::MemoryFormat memory_format);

at::Tensor contiguous_internal(const at::Tensor& self, c10::MemoryFormat memory_format) {
  // If already contiguous, return self
  if (self.is_contiguous(memory_format)) {
    return self;
  }

  // Call the Python contiguous operation directly without going through dispatcher
  return call_python_contiguous_op(self, memory_format);
}

}  // namespace ops
}  // namespace torch_neuronx

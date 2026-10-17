#include "InternalOps.h"

#include <ATen/ops/contiguous_native.h>
#include <ATen/ops/contiguous_ops.h>

#include "torch_neuronx/csrc/core/utils/CopyUtils.h"

namespace torch_neuronx {
namespace ops {

at::Tensor contiguous_internal(const at::Tensor& self, c10::MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }

  auto dispatch_key_set = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  return at::_ops::contiguous::redispatch(dispatch_key_set, self, memory_format);
}

}  // namespace ops
}  // namespace torch_neuronx

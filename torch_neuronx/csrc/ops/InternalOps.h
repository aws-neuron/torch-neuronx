#pragma once

#include <ATen/Tensor.h>

namespace torch_neuronx {
namespace ops {

// Internal contiguous operation that can be called from C++ without going through dispatcher.
// This is used internally to avoid recursion when copy operations need to make tensors contiguous.
//
// When TORCH_NEURONX_CPP_DISPATCH=1: redispatches through PrivateUse1 → DispatchContiguousImpl (C++
// path) When TORCH_NEURONX_CPP_DISPATCH=0: calls the Python contiguous_internal via the registered
// callback
at::Tensor contiguous_internal(const at::Tensor& self,
                               c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous);

// Set/get the Python contiguous callback. Called from NeuronBindings.cpp
// via _set_python_contiguous_op. Stored as a raw PyObject* to avoid
// destructor ordering issues during Python finalization.
void SetPythonContiguousOp(PyObject* op);
PyObject* GetPythonContiguousOp();

}  // namespace ops
}  // namespace torch_neuronx

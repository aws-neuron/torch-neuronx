// Mock implementations of Python utility functions for testing

#include <string>

#include "torch/torch.h"

namespace torch_neuronx {

namespace ops {

// Mock stubs for the Python contiguous op mechanism (InternalOps.h)
void SetPythonContiguousOp(PyObject* /*op*/) {}
PyObject* GetPythonContiguousOp() { return nullptr; }

}  // namespace ops

}  // namespace torch_neuronx

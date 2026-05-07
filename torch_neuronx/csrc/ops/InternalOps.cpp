#include "InternalOps.h"

#include <ATen/ops/contiguous_native.h>
#include <torch/extension.h>

#include "../utils/CopyUtils.h"

namespace torch_neuronx {
namespace ops {

// ── Python contiguous callback (raw PyObject* for finalization safety) ──

static PyObject* python_contiguous_op = nullptr;

void SetPythonContiguousOp(PyObject* op) {
  if (python_contiguous_op) {
    Py_XDECREF(python_contiguous_op);
  }
  python_contiguous_op = op;
  Py_XINCREF(python_contiguous_op);
}

PyObject* GetPythonContiguousOp() { return python_contiguous_op; }

// ── Dispatch routing ──

static bool UseCppDispatch() {
  static const bool enabled = []() {
    const char* env = std::getenv("TORCH_NEURONX_CPP_DISPATCH");
    // Default on — only disabled when explicitly set to "0"
    return env == nullptr || std::string(env) != "0";
  }();
  return enabled;
}

static at::Tensor CallPythonContiguousOp(const at::Tensor& self, c10::MemoryFormat memory_format) {
  TORCH_CHECK(python_contiguous_op, "Python contiguous operation not registered");
  py::gil_scoped_acquire gil;
  py::object op = py::reinterpret_borrow<py::object>(python_contiguous_op);
  py::object result = op(self, memory_format);
  return result.cast<at::Tensor>();
}

at::Tensor contiguous_internal(const at::Tensor& self, c10::MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }

  if (UseCppDispatch()) {
    // C++ dispatch: redispatch through PrivateUse1 → DispatchContiguousImpl
    auto dispatch_key_set = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
    return at::_ops::contiguous::redispatch(dispatch_key_set, self, memory_format);
  }

  // Python dispatch: call the registered Python contiguous operation
  return CallPythonContiguousOp(self, memory_format);
}

}  // namespace ops
}  // namespace torch_neuronx

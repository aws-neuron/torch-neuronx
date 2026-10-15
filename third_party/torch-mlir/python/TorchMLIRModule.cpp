//===-- TorchMLIRModule.cpp - Nanobind module for torch-mlir ----*- C++ -*-===//
//
// Uses the MLIR C API capsule protocol to accept Context objects.
// Does NOT use MLIR_PYTHON_PACKAGE_PREFIX — uses raw capsule names
// to match the upstream LLVM _mlir.so which uses "mlir." prefix.
//===----------------------------------------------------------------------===//

#include "nanobind/nanobind.h"
#include "mlir-c/IR.h"
#include "torch-mlir-c/Dialects.h"
#include "torch-mlir-c/Registration.h"

namespace nb = nanobind;

// Use the same capsule name as upstream LLVM (no torch_mlir. prefix)
static const char *CAPSULE_CONTEXT = "torch_mlir.ir.Context._CAPIPtr";

NB_MODULE(_torchMlir, m) {
  torchMlirRegisterAllPasses();

  m.doc() = "torch-mlir main python extension";

  m.def(
      "register_dialect",
      [](nb::object contextObj, bool load) {
        nb::object capsule = nb::getattr(contextObj, "_CAPIPtr");
        MlirContext context;
        context.ptr = PyCapsule_GetPointer(capsule.ptr(), CAPSULE_CONTEXT);
        if (!context.ptr) {
          throw nb::value_error("Failed to extract MlirContext from capsule");
        }
        MlirDialectHandle handle = mlirGetDialectHandle__torch__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  m.def("get_int64_max", []() { return INT64_MAX; });

  m.def("get_int64_min", []() { return INT64_MIN; });
}

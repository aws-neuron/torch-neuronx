# Python shim for _mlirRegisterEverything.
# Calls MLIR C API functions from _mlir.so via ctypes.
import ctypes
import os

_lib = None

def _get_lib():
    global _lib
    if _lib is None:
        mlir_libs_dir = os.path.dirname(__file__)
        _lib = ctypes.CDLL(os.path.join(mlir_libs_dir, "_mlir.so"))
    return _lib

def _get_capsule_ptr(obj, capsule_name):
    capsule = obj._CAPIPtr
    ptr = ctypes.pythonapi.PyCapsule_GetPointer(
        ctypes.py_object(capsule), capsule_name.encode()
    )
    return ptr

def register_dialects(registry):
    """Register all upstream MLIR dialects."""
    lib = _get_lib()
    lib.mlirRegisterAllDialects.argtypes = [ctypes.c_void_p]
    lib.mlirRegisterAllDialects.restype = None
    ptr = _get_capsule_ptr(registry, "torch_mlir.ir.DialectRegistry._CAPIPtr")
    lib.mlirRegisterAllDialects(ptr)

def register_llvm_translations(context):
    """Register all LLVM translations."""
    lib = _get_lib()
    lib.mlirRegisterAllLLVMTranslations.argtypes = [ctypes.c_void_p]
    lib.mlirRegisterAllLLVMTranslations.restype = None
    ptr = _get_capsule_ptr(context, "torch_mlir.ir.Context._CAPIPtr")
    lib.mlirRegisterAllLLVMTranslations(ptr)

def context_init_hook(context):
    """Post-init hook: register torch-mlir dialects (including StableHLO/CHLO)."""
    lib = _get_lib()
    lib.torchMlirRegisterAllDialects.argtypes = [ctypes.c_void_p]
    lib.torchMlirRegisterAllDialects.restype = None
    ptr = _get_capsule_ptr(context, "torch_mlir.ir.Context._CAPIPtr")
    lib.torchMlirRegisterAllDialects(ptr)

# Register all passes on import (matches C extension behavior)
_lib = _get_lib()
_lib.mlirRegisterAllPasses()
_lib.torchMlirRegisterAllPasses()

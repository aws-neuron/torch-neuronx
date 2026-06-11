torch_mlir_enable_werror()

add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)

set(LinkedLibs
  MLIRComplexDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRSupport

  # Dialects.
  TorchMLIRTorchDialect
  TorchMLIRTorchConversionDialect

  # Dialect passes.
  TorchMLIRTorchConversionPasses
  TorchMLIRTorchPasses

  # Conversion passes.
  TorchMLIRConversionPasses
)

if(TORCH_MLIR_ENABLE_STABLEHLO)
  list(APPEND LinkedLibs StablehloLinalgTransforms StablehloPasses)
endif()

add_mlir_library(TorchMLIRInitAll
  InitAll.cpp

  LINK_LIBS PUBLIC
  ${LinkedLibs}
)

torch_mlir_target_includes(TorchMLIRInitAll)

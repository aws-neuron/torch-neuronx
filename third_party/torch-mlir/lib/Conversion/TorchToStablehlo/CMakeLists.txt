add_mlir_conversion_library(TorchMLIRTorchToStablehlo
  TorchToStablehlo.cpp
  TorchToStablehloCustomLowering.cpp
  ChloToStablehloCustomLowering.cpp
  StablehloCustomLegalization.cpp
  StablehloLegalizeUtils.cpp
  Basic.cpp
  GatherScatter.cpp
  Linear.cpp
  Norm.cpp
  ViewLike.cpp
  Reduction.cpp
  Rng.cpp
  Pooling.cpp
  Uncategorized.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToStablehlo

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRComplexDialect
  ChloOps
  StablehloOps
  TorchMLIRTorchDialect
  TorchMLIRConversionUtils
)

torch_mlir_target_includes(TorchMLIRTorchToStablehlo)

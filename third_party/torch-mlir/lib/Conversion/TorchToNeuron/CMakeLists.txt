add_mlir_conversion_library(TorchMLIRTorchToNeuron
  TorchToNeuron.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToNeuron

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  StablehloOps
  TorchMLIRConversionUtils
  TorchMLIRTorchDialect
  TorchMLIRTorchConversionDialect
)

if(TORCH_MLIR_ENABLE_STABLEHLO)
  add_subdirectory(TorchToStablehlo)
endif()
add_subdirectory(TorchToNeuron)
add_subdirectory(Utils)

# TODO: Automate this with add_torch_mlir_conversion_library.
set(linked_libs TorchMLIRTorchToNeuron
                TorchMLIRConversionUtils)
if(TORCH_MLIR_ENABLE_STABLEHLO)
  list(APPEND linked_libs TorchMLIRTorchToStablehlo)
endif()

add_mlir_library(TorchMLIRConversionPasses
  Passes.cpp

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  ${linked_libs}
)

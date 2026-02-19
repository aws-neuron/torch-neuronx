#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "torch_neuronx/csrc/core/compilation/NeuronCompiler.h"

namespace at::neuron::debug {

// Dumps IR (HLO/MLIR) details to disk for debugging purposes.
// Controlled by TORCH_NEURONX_DUMP_HLO_PB environment variable.
//
// When enabled (TORCH_NEURONX_DUMP_HLO_PB=1), creates a directory structure:
//   <dump_dir>/tn_op_dumps/<op_name>/<kernel_class_name>-<cache_key>/
//     - module.pb: The HLO/MLIR bytes
//     - compiler.txt: The neuronx-cc command line
//     - meta_data.txt: Metadata (op_name, cache_key, compiler args)
//
// Args:
//   hlo_bytes: The HLO/MLIR module bytes to dump
//   config: Compilation configuration
//   op_name: Name of the operation (optional)
//   cache_key: Cache key for the compilation (optional)
//   kernel_class_name: Name of the kernel class (optional, defaults to "UnknownKernel")
void DumpIRDetails(const std::vector<uint8_t>& hlo_bytes, const CompilationConfig& config,
                   const std::string& op_name = "", const std::string& cache_key = "",
                   const std::string& kernel_class_name = "UnknownKernel");

}  // namespace at::neuron::debug

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace at::neuron {

// TensorizerCompiler provides functionality to compile Linalg IR to NEFF format
// via the tensorizer pipeline. This class handles the compilation pipeline
// Linalg -> NISA -> BIR -> NEFF as an alternative to the standard neuronx-cc
// HLO path, using neuron-opt and nki-opt for intermediate transformations.
class TensorizerCompiler {
 public:
  // Compiles Linalg IR bytes to NEFF format through the tensorizer pipeline.
  //
  // Pipeline: Linalg -> neuron-opt -> NISA -> nki-opt -> BIR -> neuronx-cc -> NEFF
  //
  // Args:
  //   linalg_bytes: Linalg IR as UTF-8 encoded bytes
  //   platform_target: Target platform (e.g., "trn1", "trn2")
  //
  // Returns:
  //   The compiled NEFF file as a byte vector
  //
  // Throws:
  //   std::runtime_error if any compilation step fails
  static std::vector<uint8_t> CompileLinalgToNeff(const std::vector<uint8_t>& linalg_bytes,
                                                  const std::string& platform_target);

 private:
  // Compiles Linalg IR to NISA using neuron-opt.
  static void CompileLinalgToNisa(const std::string& linalg_path, const std::string& nisa_path);

  // Compiles NISA to BIR using nki-opt.
  static void CompileNisaToBir(const std::string& nisa_path, const std::string& bir_path);

  // Renames BIR tensor names to match executor naming convention.
  // Converts: in_tensor_X -> inputX, out_tensor_X -> outputX
  static void RenameBirTensors(const std::string& bir_path);

  // Compiles BIR to NEFF using NeuronCompiler::CompileHloToNeff.
  static std::vector<uint8_t> CompileBirToNeff(const std::string& bir_path,
                                               const std::string& platform_target);

  // Gets the path to neuron-opt (from NEURON_OPT_PATH env var or default).
  static std::string GetNeuronOptPath();

  // Gets the path to nki-opt (from NKI_OPT_PATH env var or default).
  static std::string GetNkiOptPath();

  friend class TensorizerCompilerTest;
};

}  // namespace at::neuron

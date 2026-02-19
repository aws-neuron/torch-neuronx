#include "CompilerDebugUtils.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/utils/StringUtils.h"

namespace at::neuron::debug {

namespace {

constexpr const char* kDumpHloPbEnvVar = "TORCH_NEURONX_DUMP_HLO_PB";
constexpr const char* kDumpDirEnvVar = "TORCH_NEURONX_DUMP";

}  // namespace

void DumpIRDetails(const std::vector<uint8_t>& hlo_bytes, const CompilationConfig& config,
                   const std::string& op_name, const std::string& cache_key,
                   const std::string& kernel_class_name) {
  if (hlo_bytes.empty()) {
    TORCH_NEURONX_DEBUG("DumpIRDetails: Skipping dump - empty hlo_bytes");
    return;
  }
  const char* dump_hlo_pb = std::getenv(kDumpHloPbEnvVar);
  TORCH_NEURONX_DEBUG("DumpIRDetails: TORCH_NEURONX_DUMP_HLO_PB=",
                      dump_hlo_pb ? dump_hlo_pb : "<not set>");

  if (!dump_hlo_pb || std::string(dump_hlo_pb) != "1") {
    return;
  }

  const char* dump_dir_env = std::getenv(kDumpDirEnvVar);
  std::string dump_dir = dump_dir_env ? dump_dir_env : "/tmp";
  dump_dir += "/tn_op_dumps";

  std::string simplified_op = utils::SimplifyForFilename(op_name.empty() ? "unknown_op" : op_name);
  std::string simplified_cache =
      utils::SimplifyForFilename(cache_key.empty() ? "unknown_key" : cache_key);
  std::string folder_path =
      dump_dir + "/" + simplified_op + "/" + kernel_class_name + "-" + simplified_cache;

  TORCH_NEURONX_DEBUG("DumpIRDetails: Creating directory:", folder_path);

  std::error_code ec;
  std::filesystem::create_directories(folder_path, ec);
  if (ec) {
    TORCH_NEURONX_WARN("DumpIRDetails: Failed to create directories:", ec.message());
    return;
  }

  std::string module_path = folder_path + "/module.pb";
  TORCH_NEURONX_DEBUG("DumpIRDetails: Writing module.pb (", hlo_bytes.size(), " bytes)");
  std::ofstream module_file(module_path, std::ios::binary);
  if (module_file) {
    module_file.write(reinterpret_cast<const char*>(hlo_bytes.data()), hlo_bytes.size());
  } else {
    TORCH_NEURONX_WARN("DumpIRDetails: Failed to open module.pb for writing");
  }

  // TODO: automate compile args extraction later on from what compiler dumps
  std::string compiler_args = "neuronx-cc compile module.pb --framework " + config.framework +
                              " --target " + config.platform_target + " --lnc " +
                              config.logical_neuron_cores + " " + config.optimization_level;
  if (!config.additional_args.empty()) {
    compiler_args += " " + config.additional_args;
  }
  compiler_args += " --output output.neff";

  std::ofstream compiler_file(folder_path + "/compiler.txt");
  if (compiler_file) {
    compiler_file << compiler_args;
  }

  std::ofstream meta_file(folder_path + "/meta_data.txt");
  if (meta_file) {
    meta_file << op_name << "\n" << cache_key << "\n" << compiler_args << "\n";
  }

  TORCH_NEURONX_DEBUG("DumpIRDetails: Dump completed for", op_name);
}

}  // namespace at::neuron::debug

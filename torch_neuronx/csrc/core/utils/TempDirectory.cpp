#include "torch_neuronx/csrc/core/utils/TempDirectory.h"

#include <fstream>

namespace at::neuron {

void TempDirectory::WriteFile(const std::string& filename,
                              const std::vector<uint8_t>& bytes) const {
  std::string filepath = path_ + "/" + filename;
  std::ofstream out(filepath, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Failed to create file: " + filepath);
  }
  out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

std::vector<uint8_t> TempDirectory::ReadFile(const std::string& filename) const {
  std::string filepath = path_ + "/" + filename;
  std::ifstream in(filepath, std::ios::binary | std::ios::ate);
  if (!in) {
    throw std::runtime_error("Failed to read file: " + filepath);
  }
  std::streamsize size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(size);
  if (!in.read(reinterpret_cast<char*>(bytes.data()), size)) {
    throw std::runtime_error("Failed to read data from: " + filepath);
  }
  return bytes;
}

}  // namespace at::neuron

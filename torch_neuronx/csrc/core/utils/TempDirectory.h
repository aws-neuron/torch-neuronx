#pragma once

#include <ftw.h>
#include <sys/stat.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace at::neuron {

// RAII wrapper for temporary directory management.
// Automatically creates a temporary directory on construction and removes it on destruction.
// Uses POSIX nftw instead of std::filesystem::remove_all to avoid bugs in some libstdc++ versions.
class TempDirectory {
 public:
  // Creates a temporary directory with the given prefix.
  // Default prefix is "neuron_temp_".
  explicit TempDirectory(const std::string& prefix = "neuron_temp_") {
    std::string temp_template = "/tmp/" + prefix + "XXXXXX";
    std::vector<char> temp_buf(temp_template.begin(), temp_template.end());
    temp_buf.push_back('\0');

    char* temp_dir = mkdtemp(temp_buf.data());
    if (!temp_dir) {
      throw std::runtime_error("Failed to create temporary directory: " +
                               std::string(strerror(errno)));
    }
    path_ = temp_dir;
  }

  ~TempDirectory() { Cleanup(); }

  // Non-copyable
  TempDirectory(const TempDirectory&) = delete;
  TempDirectory& operator=(const TempDirectory&) = delete;

  // Movable
  TempDirectory(TempDirectory&& other) noexcept : path_(std::move(other.path_)) {
    other.path_.clear();
  }

  TempDirectory& operator=(TempDirectory&& other) noexcept {
    if (this != &other) {
      Cleanup();
      path_ = std::move(other.path_);
      other.path_.clear();
    }
    return *this;
  }

  const std::string& path() const { return path_; }

  // Returns the path as a std::filesystem::path
  std::filesystem::path fs_path() const { return std::filesystem::path(path_); }

 private:
  void Cleanup() {
    if (!path_.empty()) {
      // Use POSIX nftw instead of std::filesystem::remove_all to avoid
      // segfault bugs in some libstdc++ versions (e.g., PyTorch 2.9 environment)
      nftw(
          path_.c_str(),
          [](const char* fpath, const struct stat*, int, struct FTW*) { return remove(fpath); }, 64,
          FTW_DEPTH | FTW_PHYS);
      path_.clear();
    }
  }

  std::string path_;
};

}  // namespace at::neuron

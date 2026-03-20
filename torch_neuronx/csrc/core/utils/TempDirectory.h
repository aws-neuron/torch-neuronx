#pragma once

#include <ftw.h>
#include <sys/stat.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace at::neuron {

// RAII wrapper for temporary directory management.
// Automatically creates a temporary directory on construction and removes it on destruction.
// Uses POSIX nftw instead of std::filesystem::remove_all to avoid bugs in some libstdc++ versions.
class TempDirectory {
 public:
  // Creates or uses a directory.
  // If is_full_path is false (default): path_or_prefix is treated as a prefix for a temp directory
  //   under /tmp (e.g., "neuron_temp_" -> "/tmp/neuron_temp_XXXXXX").
  // If is_full_path is true: path_or_prefix is the exact directory path to use/create.
  // If preserve is true: the directory will NOT be cleaned up on destruction.
  explicit TempDirectory(const std::string& path_or_prefix = "neuron_temp_",
                         bool is_full_path = false, bool preserve = false)
      : preserve_(preserve) {
    if (is_full_path) {
      // Use the exact path provided, create if it doesn't exist
      if (!std::filesystem::exists(path_or_prefix)) {
        std::filesystem::create_directories(path_or_prefix);
      }
      path_ = path_or_prefix;
    } else {
      // Create a temp directory with the given prefix under /tmp
      std::string temp_template = "/tmp/" + path_or_prefix + "XXXXXX";
      std::vector<char> temp_buf(temp_template.begin(), temp_template.end());
      temp_buf.push_back('\0');

      char* temp_dir = mkdtemp(temp_buf.data());
      if (!temp_dir) {
        throw std::runtime_error("Failed to create temporary directory: " +
                                 std::string(strerror(errno)));
      }
      path_ = temp_dir;
    }
  }

  ~TempDirectory() { Cleanup(); }

  // Non-copyable
  TempDirectory(const TempDirectory&) = delete;
  TempDirectory& operator=(const TempDirectory&) = delete;

  // Movable
  TempDirectory(TempDirectory&& other) noexcept
      : path_(std::move(other.path_)), preserve_(other.preserve_) {
    other.path_.clear();
  }

  TempDirectory& operator=(TempDirectory&& other) noexcept {
    if (this != &other) {
      Cleanup();
      path_ = std::move(other.path_);
      preserve_ = other.preserve_;
      other.path_.clear();
    }
    return *this;
  }

  const std::string& path() const { return path_; }

  // Returns the path as a std::filesystem::path
  std::filesystem::path fs_path() const { return std::filesystem::path(path_); }

  // Writes bytes to a file within this temp directory.
  void WriteFile(const std::string& filename, const std::vector<uint8_t>& bytes) const;

  // Reads a file within this temp directory as bytes.
  std::vector<uint8_t> ReadFile(const std::string& filename) const;

  // Returns whether this directory will be preserved on destruction
  bool preserve() const { return preserve_; }

 private:
  void Cleanup() {
    if (!path_.empty() && !preserve_) {
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
  bool preserve_ = false;
};

}  // namespace at::neuron

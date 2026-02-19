#include "PersistentCacheBackend.h"

#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/cache/CacheUtils.h"

namespace at::neuron {

namespace {

// Check if environment variable is set to "1".
bool IsEnvEnabled(const char* value) { return value && std::string(value) == "1"; }

// Gets the cache directory from environment variable.
// Uses TORCH_NEURONX_NEFF_CACHE_DIR or defaults to /tmp/neff_cache
std::filesystem::path GetCacheDirFromEnv() {
  const char* neuron_cache_dir = std::getenv("TORCH_NEURONX_NEFF_CACHE_DIR");
  if (neuron_cache_dir && neuron_cache_dir[0] != '\0') {
    return std::filesystem::path(neuron_cache_dir);
  }
  return std::filesystem::path("/tmp/neff_cache");
}

}  // namespace

PersistentCacheBackend::PersistentCacheBackend(const std::string& extension)
    : cache_dir_(GetCacheDirFromEnv()), extension_(extension) {
  // Create cache directory if it doesn't exist
  std::error_code ec;
  std::filesystem::create_directories(cache_dir_, ec);
  if (ec) {
    throw std::runtime_error("Failed to create persistent cache directory: " + cache_dir_.string() +
                             " (" + ec.message() + ")");
  }
}

PersistentCacheBackend::PersistentCacheBackend(const std::filesystem::path& cache_dir,
                                               const std::string& extension)
    : cache_dir_(cache_dir), extension_(extension) {
  // Create cache directory if it doesn't exist
  std::error_code ec;
  std::filesystem::create_directories(cache_dir_, ec);
  if (ec) {
    throw std::runtime_error("Failed to create persistent cache directory: " + cache_dir_.string() +
                             " (" + ec.message() + ")");
  }
}

bool PersistentCacheBackend::IsCachingDisabled() {
  // Check Neuron neff cache disable
  const char* neuron_disable = std::getenv("TORCH_NEURONX_NEFF_DISABLE_CACHE");
  return IsEnvEnabled(neuron_disable);
}

std::filesystem::path PersistentCacheBackend::GetShardedPath(const std::string& key) const {
  // 2-level directory sharding for scalability
  // key "abcd1234..." -> cache_dir/ab/cd/abcd1234....neff
  return cache_dir_ / key.substr(0, 2) / key.substr(2, 2) / (key + extension_);
}

bool PersistentCacheBackend::Exists(const std::string& key) const {
  std::filesystem::path path = GetShardedPath(key);
  std::error_code ec;
  bool exists = std::filesystem::exists(path, ec);
  if (ec) {
    TORCH_NEURONX_DEBUG("Cache existence check failed", "key=", key, "path=", path.string(),
                        "error=", ec.message());
    return false;
  }
  return exists;
}

std::optional<std::vector<uint8_t>> PersistentCacheBackend::Get(const std::string& key) const {
  std::filesystem::path path = GetShardedPath(key);

  // Open file
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    TORCH_NEURONX_DEBUG("Cache miss", "key=", key, "path=", path.string());
    return std::nullopt;
  }

  // Get file size
  std::streamsize size = file.tellg();
  if (size <= 0) {
    TORCH_NEURONX_DEBUG("Cache file empty or error", "key=", key, "size=", size);
    return std::nullopt;
  }
  file.seekg(0, std::ios::beg);

  // Read data
  std::vector<uint8_t> data(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
    TORCH_NEURONX_WARN("Failed to read cache file", "key=", key, "path=", path.string());
    return std::nullopt;
  }

  TORCH_NEURONX_DEBUG("Cache hit", "key=", key, "size=", data.size());
  return data;
}

void PersistentCacheBackend::AtomicWrite(const std::filesystem::path& path,
                                         const std::vector<uint8_t>& data) {
  // Create parent directories
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  if (ec) {
    throw std::runtime_error("Failed to create directory: " + path.parent_path().string() + " (" +
                             ec.message() + ")");
  }

  // Create temporary file atomically using mkstemp (POSIX)
  // mkstemp creates a unique file and returns an open fd - race-condition free
  std::string temp_template = (path.parent_path() / ".tmp.XXXXXX").string();
  std::vector<char> temp_path_buf(temp_template.begin(), temp_template.end());
  temp_path_buf.push_back('\0');

  int fd = mkstemp(temp_path_buf.data());
  if (fd == -1) {
    throw std::runtime_error("Failed to create temporary file: " + std::string(strerror(errno)));
  }

  std::filesystem::path temp_path(temp_path_buf.data());

  // Write data directly using POSIX write
  const uint8_t* write_ptr = data.data();
  size_t remaining = data.size();
  while (remaining > 0) {
    ssize_t written = write(fd, write_ptr, remaining);
    if (written == -1) {
      int err = errno;
      close(fd);
      std::filesystem::remove(temp_path, ec);
      throw std::runtime_error("Failed to write to temporary file: " + std::string(strerror(err)));
    }
    write_ptr += written;
    remaining -= static_cast<size_t>(written);
  }

  if (close(fd) == -1) {
    int err = errno;
    std::filesystem::remove(temp_path, ec);
    throw std::runtime_error("Failed to close temporary file: " + std::string(strerror(err)));
  }

  // Atomic rename
  std::filesystem::rename(temp_path, path, ec);
  if (ec) {
    // Clean up temp file on rename failure
    std::filesystem::remove(temp_path);
    throw std::runtime_error("Failed to rename temporary file: " + temp_path.string() + " -> " +
                             path.string() + " (" + ec.message() + ")");
  }
}

bool PersistentCacheBackend::Put(const std::string& key, const std::vector<uint8_t>& data) {
  std::filesystem::path path = GetShardedPath(key);

  try {
    AtomicWrite(path, data);
    TORCH_NEURONX_DEBUG("Cache put", "key=", key, "size=", data.size(), "path=", path.string());
    return true;
  } catch (const std::exception& e) {
    TORCH_NEURONX_WARN("Cache put failed", "key=", key, "error=", e.what());
    return false;
  }
}

bool PersistentCacheBackend::Remove(const std::string& key) {
  std::filesystem::path path = GetShardedPath(key);

  std::error_code ec;
  bool removed = std::filesystem::remove(path, ec);
  if (ec) {
    TORCH_NEURONX_DEBUG("Cache remove failed", "key=", key, "error=", ec.message());
    return false;
  }

  if (removed) {
    TORCH_NEURONX_DEBUG("Cache remove", "key=", key);
  }
  return true;  // Return true even if file didn't exist
}

void PersistentCacheBackend::Clear() {
  std::error_code ec;

  if (!std::filesystem::exists(cache_dir_, ec)) {
    TORCH_NEURONX_DEBUG("Cache clear: directory does not exist", "path=", cache_dir_.string());
    return;
  }

  size_t files_removed = 0;
  size_t dirs_removed = 0;

  // First, collect all files to remove to avoid iterator invalidation issues.
  // The recursive_directory_iterator can throw an error if
  // directories are modified during iteration (e.g., by another process).
  std::vector<std::filesystem::path> files_to_remove;
  try {
    for (auto it = std::filesystem::recursive_directory_iterator(
             cache_dir_, std::filesystem::directory_options::skip_permission_denied, ec);
         it != std::filesystem::recursive_directory_iterator();) {
      if (ec) {
        TORCH_NEURONX_DEBUG("Error iterating cache directory", "path=", cache_dir_.string(),
                            "error=", ec.message());
        ec.clear();
        // Try to continue iteration
        try {
          ++it;
        } catch (const std::filesystem::filesystem_error& e) {
          TORCH_NEURONX_DEBUG("Iterator increment failed, stopping iteration", "error=", e.what());
          break;
        }
        continue;
      }

      if (it->is_regular_file(ec) && !ec) {
        const auto& path = it->path();
        if (path.extension() == extension_) {
          files_to_remove.push_back(path);
        }
      }

      // Increment iterator with error handling
      try {
        ++it;
      } catch (const std::filesystem::filesystem_error& e) {
        TORCH_NEURONX_DEBUG("Iterator increment failed", "error=", e.what());
        break;
      }
    }
  } catch (const std::filesystem::filesystem_error& e) {
    TORCH_NEURONX_DEBUG("Error during directory iteration", "error=", e.what());
  }

  // Remove the collected files
  for (const auto& path : files_to_remove) {
    if (std::filesystem::remove(path, ec)) {
      files_removed++;
    } else if (ec) {
      TORCH_NEURONX_DEBUG("Failed to remove cache file", "path=", path.string(),
                          "error=", ec.message());
      ec.clear();
    }
  }

  // Clean up empty shard directories (2-level deep)
  // Use error handling for each directory operation
  try {
    for (auto level1_it = std::filesystem::directory_iterator(cache_dir_, ec);
         level1_it != std::filesystem::directory_iterator(); ++level1_it) {
      if (ec) {
        ec.clear();
        continue;
      }
      if (!level1_it->is_directory(ec) || ec) {
        ec.clear();
        continue;
      }

      std::filesystem::path level1_path = level1_it->path();

      // Collect level2 directories to clean up
      std::vector<std::filesystem::path> level2_dirs_to_remove;
      try {
        for (auto level2_it = std::filesystem::directory_iterator(level1_path, ec);
             level2_it != std::filesystem::directory_iterator(); ++level2_it) {
          if (ec) {
            ec.clear();
            continue;
          }
          if (!level2_it->is_directory(ec) || ec) {
            ec.clear();
            continue;
          }

          std::filesystem::path level2_path = level2_it->path();
          if (std::filesystem::is_empty(level2_path, ec) && !ec) {
            level2_dirs_to_remove.push_back(level2_path);
          }
          ec.clear();
        }
      } catch (const std::filesystem::filesystem_error& e) {
        TORCH_NEURONX_DEBUG("Error iterating level2 directory", "error=", e.what());
      }

      // Remove empty level2 directories
      for (const auto& dir : level2_dirs_to_remove) {
        if (std::filesystem::remove(dir, ec)) {
          dirs_removed++;
        }
        ec.clear();
      }

      // Remove empty level1 directory
      if (std::filesystem::is_empty(level1_path, ec) && !ec) {
        if (std::filesystem::remove(level1_path, ec)) {
          dirs_removed++;
        }
      }
      ec.clear();
    }
  } catch (const std::filesystem::filesystem_error& e) {
    TORCH_NEURONX_DEBUG("Error during directory cleanup", "error=", e.what());
  }

  TORCH_NEURONX_DEBUG("Cache cleared", "files_removed=", files_removed,
                      "dirs_removed=", dirs_removed, "path=", cache_dir_.string());
}

}  // namespace at::neuron

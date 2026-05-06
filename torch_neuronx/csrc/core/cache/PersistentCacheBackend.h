#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace at::neuron {

// NFS-compatible persistent cache backend for compiled NEFFs.
//
// This class provides atomic read/write operations to a shared filesystem
// (NFS or local disk) with 2-level directory sharding for scalability.
// It uses atomic write operations (temp file + rename) that are safe for
// concurrent access across processes and nodes.
//
// Directory structure:
//   cache_dir/
//   ├── ab/
//   │   └── cd/
//   │       └── abcd1234...neff
//
// Environment variables:
// - TORCH_NEURONX_NEFF_CACHE_DIR: Neuron-specific override
// - TORCH_NEURONX_NEFF_DISABLE_CACHE: Disable NEFF caching specifically
class PersistentCacheBackend {
 public:
  // File extension for cached NEFF files.
  static constexpr const char* kDefaultExtension = ".neff";

  // Constructs a PersistentCacheBackend with the default cache directory from environment variable.
  //
  // The cache directory is read from TORCH_NEURONX_NEFF_CACHE_DIR environment variable,
  // falling back to /tmp/neff_cache if not set.
  //
  // Args:
  //   extension: File extension for cached files (default: ".neff")
  explicit PersistentCacheBackend(const std::string& extension = kDefaultExtension);

  // Constructs a PersistentCacheBackend with the given cache directory.
  //
  // Args:
  //   cache_dir: Root directory for cache storage. Will be created if it doesn't exist.
  //   extension: File extension for cached files (default: ".neff")
  PersistentCacheBackend(const std::filesystem::path& cache_dir,
                         const std::string& extension = kDefaultExtension);

  // Default destructor.
  ~PersistentCacheBackend() = default;

  // PersistentCacheBackend is copyable and movable.
  PersistentCacheBackend(const PersistentCacheBackend&) = default;
  PersistentCacheBackend& operator=(const PersistentCacheBackend&) = default;
  PersistentCacheBackend(PersistentCacheBackend&&) = default;
  PersistentCacheBackend& operator=(PersistentCacheBackend&&) = default;

  // Checks if a cache entry exists for the given key.
  //
  // Args:
  //   key: Cache key (typically a hash string)
  //
  // Returns:
  //   true if the cache entry exists, false otherwise
  bool Exists(const std::string& key) const;

  // Retrieves a cache entry for the given key.
  //
  // Args:
  //   key: Cache key (typically a hash string)
  //
  // Returns:
  //   The cached bytes if found, std::nullopt otherwise
  std::optional<std::vector<uint8_t>> Get(const std::string& key) const;

  // Stores data in the cache with the given key.
  //
  // Uses atomic write (temp file + rename) to ensure consistency on Persistent.
  //
  // Args:
  //   key: Cache key (typically a hash string)
  //   data: Bytes to store
  //
  // Returns:
  //   true if the data was successfully stored, false on error
  bool Put(const std::string& key, const std::vector<uint8_t>& data);

  // Removes a cache entry for the given key.
  //
  // Args:
  //   key: Cache key to remove
  //
  // Returns:
  //   true if the entry was removed (or didn't exist), false on error
  bool Remove(const std::string& key);

  // Clears all entries from the cache.
  //
  // Removes all files with the configured extension from the cache directory.
  // Empty shard directories are also removed.
  void Clear();

  // Returns the cache directory path.
  const std::filesystem::path& GetCacheDir() const { return cache_dir_; }

  // Checks if caching is disabled via environment variables.
  //
  // Checks:
  // - TORCH_NEURONX_NEFF_DISABLE_CACHE=1
  static bool IsCachingDisabled();

 private:
  // Gets the sharded file path for a cache key.
  //
  // Implements 2-level directory sharding:
  //   key "abcd1234" -> cache_dir/ab/cd/abcd1234.neff
  std::filesystem::path GetShardedPath(const std::string& key) const;

  // Atomically writes bytes to a file using temp file + rename.
  //
  // This is Persistent-safe: writes to a temp file in the same directory,
  // then atomically renames to the target path.
  static void AtomicWrite(const std::filesystem::path& path, const std::vector<uint8_t>& data);

  std::filesystem::path cache_dir_;
  std::string extension_;
};

}  // namespace at::neuron

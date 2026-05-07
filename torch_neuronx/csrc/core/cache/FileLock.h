#pragma once

#include <chrono>
#include <filesystem>
#include <string>

namespace at::neuron {

// RAII file lock for cross-process synchronization.
//
// This class provides exclusive file locking using POSIX flock() for
// coordinating compilation across multiple processes.
class FileLock {
 public:
  // Default lock timeout
  static constexpr int kDefaultTimeoutSeconds = 1200;

  // Constructs a FileLock for the given path.
  //
  // Args:
  //   lock_path: Path to the lock file (will be created if it doesn't exist)
  //   timeout: Maximum time to wait for lock acquisition
  explicit FileLock(const std::filesystem::path& lock_path,
                    std::chrono::seconds timeout = std::chrono::seconds(kDefaultTimeoutSeconds));

  // Destructor releases the lock if held.
  ~FileLock();

  // FileLock is not copyable (file descriptors can't be safely copied).
  FileLock(const FileLock&) = delete;
  FileLock& operator=(const FileLock&) = delete;

  // FileLock is movable.
  FileLock(FileLock&& other) noexcept;
  FileLock& operator=(FileLock&& other) noexcept;

  // Acquires the lock, blocking until acquired or timeout.
  //
  // Returns:
  //   true if the lock was acquired successfully
  //   false if the lock could not be acquired within the timeout
  //
  // Throws:
  //   std::runtime_error if there's an I/O error (not timeout)
  bool Acquire();

  // Releases the lock if held.
  void Release();

  // Returns true if this instance currently holds the lock.
  bool IsHeld() const { return is_held_; }

  // Returns the lock file path.
  const std::filesystem::path& GetLockPath() const { return lock_path_; }

  // Gets the lock timeout from environment variable.
  // Uses TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT or defaults to kDefaultTimeoutSeconds.
  static std::chrono::seconds GetLockTimeoutFromEnv();

 private:
  // Opens or creates the lock file.
  void OpenLockFile();

  // Closes the lock file descriptor.
  void CloseLockFile();

  // Validates that fd_ points to the same inode as lock_path_ on disk.
  //
  // This check is necessary to detect orphaned lock scenarios where our flock()
  // succeeds on a deleted file. flock() locks are per-inode, not per-path, so if
  // another process deleted our lock file and created a new one, we may hold a
  // lock on the old (orphaned) inode while the true lock file has a different
  // inode. This could cause multiple processes to think they hold exclusive locks.
  //
  // Returns:
  //   true if fd_ and lock_path_ have the same inode (lock is valid)
  //   false if inodes differ or stat fails (lock is on orphaned inode)
  bool ValidateLockInode() const;

  std::filesystem::path lock_path_;
  std::chrono::seconds timeout_;
  int fd_{-1};
  bool is_held_{false};
};

// RAII guard for automatic lock acquisition and release.
class FileLockGuard {
 public:
  // Default maximum number of retries for lock acquisition.
  static constexpr int kDefaultMaxRetries = 10;

  // Constructs a guard and acquires the lock.
  //
  // Args:
  //   lock_path: Path to the lock file
  //   timeout: Maximum time to wait for lock acquisition
  //   max_retries: Maximum number of acquisition retries
  //
  // Throws:
  //   std::runtime_error if lock cannot be acquired within timeout or max retries
  explicit FileLockGuard(const std::filesystem::path& lock_path,
                         std::chrono::seconds timeout = FileLock::GetLockTimeoutFromEnv(),
                         int max_retries = kDefaultMaxRetries);

  // Destructor releases the lock.
  ~FileLockGuard() = default;

  // FileLockGuard is not copyable or movable.
  FileLockGuard(const FileLockGuard&) = delete;
  FileLockGuard& operator=(const FileLockGuard&) = delete;
  FileLockGuard(FileLockGuard&&) = delete;
  FileLockGuard& operator=(FileLockGuard&&) = delete;

  // Returns true if lock is held.
  bool IsHeld() const { return lock_.IsHeld(); }

 private:
  FileLock lock_;
};

}  // namespace at::neuron

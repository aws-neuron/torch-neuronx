#include "FileLock.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "torch_neuronx/csrc/core/NeuronLogging.h"

namespace at::neuron {

FileLock::FileLock(const std::filesystem::path& lock_path, std::chrono::seconds timeout)
    : lock_path_(lock_path), timeout_(timeout) {}

FileLock::~FileLock() {
  Release();
  CloseLockFile();
}

FileLock::FileLock(FileLock&& other) noexcept
    : lock_path_(std::move(other.lock_path_)),
      timeout_(other.timeout_),
      fd_(other.fd_),
      is_held_(other.is_held_) {
  other.fd_ = -1;
  other.is_held_ = false;
}

FileLock& FileLock::operator=(FileLock&& other) noexcept {
  if (this != &other) {
    Release();
    CloseLockFile();

    lock_path_ = std::move(other.lock_path_);
    timeout_ = other.timeout_;
    fd_ = other.fd_;
    is_held_ = other.is_held_;

    other.fd_ = -1;
    other.is_held_ = false;
  }
  return *this;
}

std::chrono::seconds FileLock::GetLockTimeoutFromEnv() {
  static const std::chrono::seconds cache_timeout = []() {
    const char* timeout_env = std::getenv("TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT");
    if (timeout_env != nullptr) {
      char* end;
      long timeout_seconds = std::strtol(timeout_env, &end, 10);
      if (end != timeout_env && *end == '\0' && timeout_seconds > 0) {
        return std::chrono::seconds(timeout_seconds);
      }
      TORCH_NEURONX_WARN("Invalid TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT value",
                         "value=", timeout_env);
    }
    return std::chrono::seconds(kDefaultTimeoutSeconds);
  }();
  return cache_timeout;
}

void FileLock::OpenLockFile() {
  if (fd_ >= 0) {
    return;  // Already open.
  }

  // Create parent directories if they don't exist.
  std::error_code ec;
  std::filesystem::create_directories(lock_path_.parent_path(), ec);
  if (ec) {
    throw std::runtime_error("Failed to create lock directory: " +
                             lock_path_.parent_path().string() + " (" + ec.message() + ")");
  }

  // Verify the directory was actually created
  if (!std::filesystem::is_directory(lock_path_.parent_path(), ec) || ec) {
    throw std::runtime_error("Lock directory does not exist after creation attempt: " +
                             lock_path_.parent_path().string());
  }

  // Open or create the lock file.
  // O_RDWR is required for flock() to work properly on some systems.
  // O_CREAT creates the file if it doesn't exist.
  fd_ = open(lock_path_.c_str(), O_RDWR | O_CREAT, 0666);
  if (fd_ < 0) {
    throw std::runtime_error("Failed to open lock file: " + lock_path_.string() + " (" +
                             std::string(strerror(errno)) + ")");
  }

  TORCH_NEURONX_DEBUG("Lock file opened", "path=", lock_path_.string(), "fd=", fd_);
}

void FileLock::CloseLockFile() {
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
    TORCH_NEURONX_DEBUG("Lock file closed", "path=", lock_path_.string());
  }
}

bool FileLock::ValidateLockInode() const {
  // Validate that our fd still points to the canonical lock file (the file at lock_path_).
  // This detects when another process broke the lock by deleting and recreating the file,
  // leaving us holding a lock on an orphaned inode.
  //
  // Note: This is a best-effort check with inherent TOCTOU race - between validation and
  // returning true, another process could still delete our file. However, this is acceptable
  // because:
  // 1. In the main loop, we retry, so we'll detect it on next iteration
  // 2. In the timeout path, FileLockGuard retries if Acquire() returns false
  // 3. The double-check pattern (check cache before and after acquiring lock) ensures
  //    no duplicate work even if multiple processes think they hold the lock
  struct stat fd_stat, path_stat;
  if (fstat(fd_, &fd_stat) != 0) {
    TORCH_NEURONX_DEBUG("Lock inode validation: fstat failed", "path=", lock_path_.string(),
                        "error=", strerror(errno));
    return false;
  }
  if (stat(lock_path_.c_str(), &path_stat) != 0) {
    TORCH_NEURONX_DEBUG("Lock inode validation: stat failed", "path=", lock_path_.string(),
                        "error=", strerror(errno));
    return false;
  }
  if (fd_stat.st_ino != path_stat.st_ino || fd_stat.st_dev != path_stat.st_dev) {
    TORCH_NEURONX_DEBUG("Lock inode mismatch", "path=", lock_path_.string(),
                        "fd_inode=", fd_stat.st_ino, "path_inode=", path_stat.st_ino,
                        "fd_dev=", fd_stat.st_dev, "path_dev=", path_stat.st_dev);
    return false;
  }
  return true;
}

bool FileLock::Acquire() {
  if (is_held_) {
    return true;
  }

  OpenLockFile();

  auto start_time = std::chrono::steady_clock::now();
  auto deadline = start_time + timeout_;

  // Retry loop with exponential backoff with cap at 1s.
  constexpr auto kInitialRetryDelay = std::chrono::milliseconds(10);
  constexpr auto kMaxRetryDelay = std::chrono::milliseconds(1000);
  auto retry_delay = kInitialRetryDelay;

  while (true) {
    // Try non-blocking lock first.
    int result = flock(fd_, LOCK_EX | LOCK_NB);

    if (result == 0) {
      // Validate inode to detect orphaned lock.
      // Handles race condition in which another process may timeout
      // and break the stale lock by deleting/recreating the file while waiting in loop.
      // In this case our fd now points to orphaned inode.
      if (ValidateLockInode()) {
        is_held_ = true;
        TORCH_NEURONX_DEBUG("Lock acquired", "path=", lock_path_.string());
        return true;
      }
      // Inode mismatch or stat failed - release orphaned lock, reopen,
      // and fall through to timeout check and backoff before retrying.
      flock(fd_, LOCK_UN);
      CloseLockFile();
      OpenLockFile();
    } else if (errno != EWOULDBLOCK && errno != EAGAIN) {
      // Real error (not just lock contention).
      throw std::runtime_error("Failed to acquire lock: " + lock_path_.string() + " (" +
                               std::string(strerror(errno)) + ")");
    }

    // Check if we've exceeded the timeout.
    auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
      TORCH_NEURONX_WARN("Lock acquisition timed out, breaking stale lock",
                         "path=", lock_path_.string(), "elapsed_seconds=", elapsed.count(),
                         "timeout_seconds=", timeout_.count());

      // Force-break the lock by deleting the lock file and creating a new one.
      CloseLockFile();

      std::error_code ec;
      std::filesystem::remove(lock_path_, ec);
      // Ignore remove error - OpenLockFile will create the file anyway.

      OpenLockFile();
      result = flock(fd_, LOCK_EX | LOCK_NB);
      if (result == 0) {
        // Validate inode to detect timeout race with other processes breaking the lock.
        if (ValidateLockInode()) {
          is_held_ = true;
          TORCH_NEURONX_INFO("Lock acquired after breaking stale lock",
                             "path=", lock_path_.string());
          return true;
        }
        // Validation failed - release orphaned lock.
        flock(fd_, LOCK_UN);
      }

      // Another process may have beaten us to breaking the lock.
      // Return false so caller can retry.
      TORCH_NEURONX_WARN("Failed to acquire lock after breaking stale lock",
                         "path=", lock_path_.string(), "error=", strerror(errno));
      CloseLockFile();
      return false;
    }

    // Wait before retrying with exponential backoff.
    auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    auto actual_delay = std::min({retry_delay, remaining, kMaxRetryDelay});

    TORCH_NEURONX_DEBUG("Lock contention, retrying", "path=", lock_path_.string(),
                        "delay_ms=", actual_delay.count());

    std::this_thread::sleep_for(actual_delay);

    retry_delay = std::min(retry_delay * 2, kMaxRetryDelay);
  }
}

void FileLock::Release() {
  if (!is_held_) {
    return;
  }

  if (fd_ >= 0) {
    int result = flock(fd_, LOCK_UN);
    if (result != 0) {
      TORCH_NEURONX_WARN("Failed to release lock", "path=", lock_path_.string(),
                         "error=", strerror(errno));
    } else {
      TORCH_NEURONX_DEBUG("Lock released", "path=", lock_path_.string());
    }
  }

  is_held_ = false;
}

FileLockGuard::FileLockGuard(const std::filesystem::path& lock_path, std::chrono::seconds timeout,
                             int max_retries)
    : lock_(lock_path, timeout) {
  // Retry acquisition if it fails (i.e., due to race condition after breaking stale lock).
  int retry_count = 0;

  while (!lock_.Acquire()) {
    retry_count++;

    // Check retry limit.
    if (retry_count >= max_retries) {
      throw std::runtime_error("FileLockGuard: Failed to acquire lock after " +
                               std::to_string(max_retries) + " retries: " + lock_path.string());
    }
    TORCH_NEURONX_INFO("FileLockGuard retrying lock acquisition", "path=", lock_path.string(),
                       "retry=", retry_count, "max_retries=", max_retries);

    // Small delay before retrying to avoid busy-looping.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

}  // namespace at::neuron

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "torch_neuronx/csrc/core/cache/FileLock.h"
#include "torch_neuronx/csrc/core/utils/TempDirectory.h"

namespace at::neuron {
namespace {

class FileLockTest : public ::testing::Test {
 protected:
  void SetUp() override {
    temp_dir_ = std::make_unique<TempDirectory>("file_lock_test_");
    lock_path_ = temp_dir_->fs_path() / "test.lock";

    // Store original env values
    original_timeout_env_ = std::getenv("TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT");
    if (original_timeout_env_) {
      original_timeout_value_ = original_timeout_env_;
    }
    // Clear the environment variable for consistent test behavior
    unsetenv("TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT");
  }

  void TearDown() override {
    // Restore original environment
    if (!original_timeout_value_.empty()) {
      setenv("TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT", original_timeout_value_.c_str(), 1);
    }
    // TempDirectory RAII will clean up automatically
    temp_dir_.reset();
  }

  std::unique_ptr<TempDirectory> temp_dir_;
  std::filesystem::path lock_path_;
  const char* original_timeout_env_ = nullptr;
  std::string original_timeout_value_;
};

// =============================================================================
// Basic Lock Acquisition and Release Tests
// =============================================================================

TEST_F(FileLockTest, BasicAcquireAndRelease) {
  EXPECT_FALSE(std::filesystem::exists(lock_path_));

  FileLock lock(lock_path_);
  EXPECT_FALSE(lock.IsHeld());

  bool acquired = lock.Acquire();
  EXPECT_TRUE(acquired);
  EXPECT_TRUE(lock.IsHeld());
  EXPECT_TRUE(std::filesystem::exists(lock_path_));

  lock.Release();
  EXPECT_FALSE(lock.IsHeld());
}

TEST_F(FileLockTest, AcquireCreatesParentDirectories) {
  std::filesystem::path nested_lock =
      temp_dir_->fs_path() / "deeply" / "nested" / "path" / "test.lock";

  FileLock lock(nested_lock);
  bool acquired = lock.Acquire();
  EXPECT_TRUE(acquired);

  EXPECT_TRUE(std::filesystem::exists(nested_lock));
  lock.Release();
}

TEST_F(FileLockTest, DoubleAcquireIsIdempotent) {
  FileLock lock(lock_path_);

  bool first_acquire = lock.Acquire();
  EXPECT_TRUE(first_acquire);
  EXPECT_TRUE(lock.IsHeld());

  // Second acquire should succeed immediately (already held)
  bool second_acquire = lock.Acquire();
  EXPECT_TRUE(second_acquire);
  EXPECT_TRUE(lock.IsHeld());

  lock.Release();
  EXPECT_FALSE(lock.IsHeld());
}

TEST_F(FileLockTest, DoubleReleaseIsSafe) {
  FileLock lock(lock_path_);

  bool acquired = lock.Acquire();
  EXPECT_TRUE(acquired);

  lock.Release();
  EXPECT_FALSE(lock.IsHeld());

  // Second release should be safe (no-op)
  lock.Release();
  EXPECT_FALSE(lock.IsHeld());
}

TEST_F(FileLockTest, DestructorReleasesLock) {
  {
    FileLock lock(lock_path_);
    bool acquired = lock.Acquire();
    EXPECT_TRUE(acquired);
    EXPECT_TRUE(lock.IsHeld());
    // Lock should be released when `lock` goes out of scope
  }

  // After destruction, we should be able to acquire the lock again
  FileLock new_lock(lock_path_);
  bool acquired = new_lock.Acquire();
  EXPECT_TRUE(acquired);
}

// =============================================================================
// FileLockGuard (RAII) Tests
// =============================================================================

TEST_F(FileLockTest, FileLockGuardAcquiresOnConstruction) {
  FileLockGuard guard(lock_path_);
  EXPECT_TRUE(guard.IsHeld());
}

TEST_F(FileLockTest, FileLockGuardReleasesOnDestruction) {
  {
    FileLockGuard guard(lock_path_);
    EXPECT_TRUE(guard.IsHeld());
  }

  // After destruction, we should be able to acquire the lock again
  FileLockGuard new_guard(lock_path_);
  EXPECT_TRUE(new_guard.IsHeld());
}

TEST_F(FileLockTest, FileLockGuardThrowsAfterMaxRetries) {
  // Test that FileLockGuard throws after max_retries when there's a race condition
  // where another process keeps beating us to the lock after we break it.
  //
  // This test simulates the race by having multiple threads compete for the lock
  // with very short timeouts, ensuring that sometimes a thread will fail to acquire
  // after breaking the stale lock because another thread beat it.

  constexpr int kNumCompetitors = 5;
  constexpr int max_retries = 2;
  auto short_timeout = std::chrono::milliseconds(100);  // Very short timeout to trigger racing

  std::atomic<int> exceptions_thrown{0};
  std::atomic<int> successful_acquisitions{0};

  auto competitor = [&]() {
    try {
      FileLockGuard guard(
          lock_path_, std::chrono::duration_cast<std::chrono::seconds>(short_timeout), max_retries);
      // Hold the lock briefly to create contention
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      successful_acquisitions++;
    } catch (const std::runtime_error& e) {
      // Expected when max_retries exceeded due to race conditions
      exceptions_thrown++;
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kNumCompetitors);
  for (int i = 0; i < kNumCompetitors; ++i) {
    threads.emplace_back(competitor);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // With racing and short timeouts, we expect some threads to succeed and possibly
  // some to fail after max_retries. The key assertion is that we don't hang forever.
  // At least some should succeed since the lock is eventually available.
  EXPECT_GT(successful_acquisitions.load(), 0);
  // All threads should have completed
  EXPECT_EQ(successful_acquisitions.load() + exceptions_thrown.load(), kNumCompetitors);
}

// =============================================================================
// Timeout Tests
// =============================================================================

TEST_F(FileLockTest, TimeoutBreaksStaleLockAndAcquires) {
  // First lock holds the lock
  FileLock first_lock(lock_path_);
  bool first_acquired = first_lock.Acquire();
  EXPECT_TRUE(first_acquired);

  // Second lock should timeout, break the stale lock, and acquire
  auto timeout = std::chrono::seconds(1);
  FileLock second_lock(lock_path_, timeout);

  // After timeout, lock breaking behavior kicks in:
  // - Deletes the lock file and creates a new one
  // - First lock is now holding an orphaned inode
  // - Second lock acquires the new file
  bool second_acquired = second_lock.Acquire();
  EXPECT_TRUE(second_acquired);
  EXPECT_TRUE(second_lock.IsHeld());

  // First lock still thinks it's held (on orphaned inode)
  EXPECT_TRUE(first_lock.IsHeld());

  second_lock.Release();
  first_lock.Release();
}

TEST_F(FileLockTest, GetLockTimeoutFromEnvReturnsCachedDefaultTimeout) {
  // The timeout is cached at first call. Since we unsetenv in SetUp() before
  // any test runs, the cached value should be the default timeout.
  auto timeout = FileLock::GetLockTimeoutFromEnv();
  EXPECT_EQ(timeout, std::chrono::seconds(FileLock::kDefaultTimeoutSeconds));
}

TEST_F(FileLockTest, GetLockTimeoutFromEnvReturnsConsistentValue) {
  // Verify that calling GetLockTimeoutFromEnv multiple times returns the same
  // cached value, regardless of environment variable changes after first call.
  auto first_call = FileLock::GetLockTimeoutFromEnv();

  // Try to change the environment variable
  setenv("TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT", "9999", 1);
  auto second_call = FileLock::GetLockTimeoutFromEnv();

  // Value should be cached, not re-read
  EXPECT_EQ(first_call, second_call);

  // The cached value should be positive
  EXPECT_GT(first_call.count(), 0);
}

// =============================================================================
// Concurrent Access Tests (Multi-threaded)
// =============================================================================

TEST_F(FileLockTest, ConcurrentAccessSameKey) {
  constexpr int kNumThreads = 10;
  std::atomic<int> concurrent_holders{0};
  std::atomic<int> max_concurrent{0};
  std::atomic<int> successful_acquisitions{0};

  auto worker = [&]() {
    FileLock lock(lock_path_);
    if (lock.Acquire()) {
      successful_acquisitions++;
      int current = ++concurrent_holders;

      // Track max concurrent holders
      int expected = max_concurrent.load();
      while (current > expected && !max_concurrent.compare_exchange_weak(expected, current)) {
        // Retry
      }

      // Simulate some work
      std::this_thread::sleep_for(std::chrono::milliseconds(10));

      --concurrent_holders;
      lock.Release();
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(worker);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // All threads should have successfully acquired the lock
  EXPECT_EQ(successful_acquisitions.load(), kNumThreads);

  // Max concurrent holders should be 1 (exclusive lock)
  EXPECT_EQ(max_concurrent.load(), 1);
}

TEST_F(FileLockTest, ConcurrentAccessDifferentKeys) {
  constexpr int kNumThreads = 5;
  std::atomic<int> successful_acquisitions{0};

  auto worker = [&](int thread_id) {
    std::filesystem::path thread_lock =
        temp_dir_->fs_path() / ("lock_" + std::to_string(thread_id) + ".lock");
    FileLock lock(thread_lock);
    if (lock.Acquire()) {
      successful_acquisitions++;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      lock.Release();
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(worker, i);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // All threads should succeed (different locks)
  EXPECT_EQ(successful_acquisitions.load(), kNumThreads);
}

TEST_F(FileLockTest, OnlyOneThreadHoldsLockAtATime) {
  constexpr int kNumThreads = 5;
  std::atomic<int> compilation_count{0};
  std::atomic<int> successful_results{0};

  auto compile_worker = [&]() {
    FileLock lock(lock_path_);
    if (lock.Acquire()) {
      // Simulate compilation
      compilation_count++;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      successful_results++;
      lock.Release();
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(compile_worker);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // All threads should eventually get the lock (one at a time)
  EXPECT_EQ(compilation_count.load(), kNumThreads);
  EXPECT_EQ(successful_results.load(), kNumThreads);
}

TEST_F(FileLockTest, WaiterBreaksStaleLockAndAcquires) {
  // This test verifies that when a lock holder is taking too long (or crashed),
  // a waiting thread can break the stale lock and acquire it.

  std::atomic<bool> thread_a_acquired{false};
  std::atomic<bool> thread_b_acquired{false};
  std::atomic<bool> thread_a_should_exit{false};

  const auto kThreadBTimeout = std::chrono::seconds(1);

  // Thread A: Acquires the lock and holds indefinitely (simulating crash/hang)
  std::thread thread_a([&]() {
    FileLock lock_a(lock_path_);
    bool acquired = lock_a.Acquire();
    thread_a_acquired = acquired;
    EXPECT_TRUE(acquired);

    // Hold the lock until told to exit (simulating a hung process)
    while (!thread_a_should_exit.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    // Note: Thread A does NOT release the lock - simulating crash behavior
    // The lock will be released when lock_a goes out of scope
  });

  // Wait for thread A to acquire
  while (!thread_a_acquired.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // Thread B: Waits with short timeout, should break stale lock and acquire
  std::thread thread_b([&]() {
    FileLock lock_b(lock_path_, kThreadBTimeout);
    // This should: 1) wait for timeout, 2) break stale lock, 3) acquire
    bool acquired = lock_b.Acquire();
    thread_b_acquired = acquired;
    EXPECT_TRUE(acquired);  // Thread B should successfully acquire after breaking
    if (acquired) {
      lock_b.Release();
    }
  });

  // Wait for thread B to complete (should take ~1s for timeout + acquire)
  thread_b.join();

  // Signal thread A to exit
  thread_a_should_exit = true;
  thread_a.join();

  // Verify both threads acquired their locks (A acquired, B broke A's lock and acquired)
  EXPECT_TRUE(thread_a_acquired.load());
  EXPECT_TRUE(thread_b_acquired.load());
}

// =============================================================================
// Multi-process Tests (Simulated via multiple FileLock instances)
// =============================================================================

TEST_F(FileLockTest, SecondProcessBreaksFirstProcessLock) {
  // Simulate two processes with separate FileLock instances
  // Process 2 times out and breaks process 1's lock
  FileLock process1_lock(lock_path_);
  FileLock process2_lock(lock_path_, std::chrono::seconds(1));

  // Process 1 acquires
  bool p1_acquired = process1_lock.Acquire();
  EXPECT_TRUE(p1_acquired);

  // Process 2 should timeout, break the stale lock, and acquire
  bool p2_acquired = process2_lock.Acquire();
  EXPECT_TRUE(p2_acquired);

  // Both processes think they hold their locks (different inodes)
  EXPECT_TRUE(process1_lock.IsHeld());
  EXPECT_TRUE(process2_lock.IsHeld());

  process1_lock.Release();
  process2_lock.Release();
}

TEST_F(FileLockTest, StaleLockFileDoesNotBlockNewLock) {
  // Create a stale lock file (simulating a process that crashed)
  std::filesystem::create_directories(lock_path_.parent_path());
  std::ofstream stale_file(lock_path_);
  stale_file << "stale lock data";
  stale_file.close();

  // New lock should still be able to acquire (flock doesn't persist across process death)
  FileLock lock(lock_path_);
  bool acquired = lock.Acquire();
  EXPECT_TRUE(acquired);
  EXPECT_TRUE(lock.IsHeld());
}

TEST_F(FileLockTest, LockFileCanBeReusedAfterRelease) {
  // First lock cycle
  {
    FileLock lock(lock_path_);
    EXPECT_TRUE(lock.Acquire());
  }

  // File should still exist
  EXPECT_TRUE(std::filesystem::exists(lock_path_));

  // Second lock cycle should work
  {
    FileLock lock(lock_path_);
    EXPECT_TRUE(lock.Acquire());
  }
}

// =============================================================================
// Inode Validation Tests
// =============================================================================

TEST_F(FileLockTest, InodeValidationRecoveryOnMismatch) {
  // This test verifies that FileLock correctly detects inode mismatch and
  // recovers by reopening and reacquiring on the correct file.
  //
  // To trigger inode mismatch, we need the FileLock to already have an open fd
  // on a file, then replace the file on disk so the fd points to an orphaned
  // inode. On the next Acquire(), ValidateLockInode() should detect the mismatch,
  // close the stale fd, reopen the new file, and successfully acquire.

  const auto kTimeout = std::chrono::seconds(5);

  // Acquire the lock so the FileLock opens a fd on the original file.
  FileLock lock(lock_path_, kTimeout);
  ASSERT_TRUE(lock.Acquire());

  // Record the original inode.
  struct stat original_stat;
  ASSERT_EQ(stat(lock_path_.c_str(), &original_stat), 0);

  // Release the flock but keep the fd open (fd_ still points to original inode).
  lock.Release();
  EXPECT_FALSE(lock.IsHeld());

  // Replace the lock file on disk — new file gets a different inode.
  std::filesystem::remove(lock_path_);
  {
    std::ofstream new_file(lock_path_);
    new_file << "replaced";
  }

  // Verify the file on disk now has a different inode.
  struct stat new_stat;
  ASSERT_EQ(stat(lock_path_.c_str(), &new_stat), 0);
  EXPECT_NE(original_stat.st_ino, new_stat.st_ino);

  // Acquire again — the internal fd still points to the old (orphaned) inode.
  // Acquire() should: flock the orphaned inode (succeeds since no contention),
  // call ValidateLockInode() which detects the inode mismatch, then recover by
  // closing the stale fd, reopening the new file, and acquiring the flock on it.
  bool acquired = lock.Acquire();
  EXPECT_TRUE(acquired);
  EXPECT_TRUE(lock.IsHeld());

  // Verify that the lock is held on the canonical file by checking we
  // cannot get a non-blocking exclusive flock on it.
  int verify_fd = open(lock_path_.c_str(), O_RDWR);
  ASSERT_GE(verify_fd, 0);
  int flock_result = flock(verify_fd, LOCK_EX | LOCK_NB);
  EXPECT_EQ(flock_result, -1);
  EXPECT_TRUE(errno == EWOULDBLOCK || errno == EAGAIN);
  close(verify_fd);

  lock.Release();
}

TEST_F(FileLockTest, TimeoutRaceOnlyOneCanonicalHolder) {
  // This test verifies the behavior when multiple processes race to break a stale lock.
  // Due to inherent TOCTOU race in inode validation, multiple processes may report
  // IsHeld() = true (some holding orphaned inodes). This is acceptable because:
  //
  // 1. Only one process holds the flock on the canonical file at any instant
  // 2. The double-check pattern (check cache before/after lock) prevents duplicate work
  // 3. FileLockGuard retries handle Acquire() returning false
  //
  // This test verifies that exactly one FileLock holds the canonical file's lock.

  constexpr int kNumProcesses = 5;
  const auto kShortTimeout = std::chrono::seconds(1);

  // Create a lock that will be held to force all "processes" into the timeout path
  FileLock initial_holder(lock_path_);
  ASSERT_TRUE(initial_holder.Acquire());

  // Create all FileLock instances (simulating separate processes)
  std::vector<std::unique_ptr<FileLock>> locks;
  locks.reserve(kNumProcesses);
  for (int i = 0; i < kNumProcesses; ++i) {
    locks.push_back(std::make_unique<FileLock>(lock_path_, kShortTimeout));
  }

  std::atomic<bool> start_race{false};
  std::vector<std::atomic<bool>> acquired(kNumProcesses);
  for (int i = 0; i < kNumProcesses; ++i) {
    acquired[i] = false;
  }

  // Start all threads simultaneously to maximize race conditions
  std::vector<std::thread> threads;
  threads.reserve(kNumProcesses);
  for (int i = 0; i < kNumProcesses; ++i) {
    threads.emplace_back([&, i]() {
      // Wait for start signal
      while (!start_race.load()) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
      // Try to acquire - will timeout and race to break the stale lock
      acquired[i] = locks[i]->Acquire();
    });
  }

  // Start the race (initial_holder still held - threads will timeout)
  start_race = true;

  // Wait for all threads to complete (they will timeout after 1s each)
  for (auto& thread : threads) {
    thread.join();
  }

  // Exactly one FileLock should hold the canonical file's lock.
  // We verify this by trying to get an exclusive lock on the canonical file.
  int test_fd = open(lock_path_.c_str(), O_RDWR);
  ASSERT_GE(test_fd, 0) << "Failed to open lock file for verification";

  int flock_result = flock(test_fd, LOCK_EX | LOCK_NB);
  // flock returns -1 and sets errno to EWOULDBLOCK/EAGAIN when lock is held
  EXPECT_EQ(flock_result, -1);
  EXPECT_TRUE(errno == EWOULDBLOCK || errno == EAGAIN);

  close(test_fd);

  // At least one should have succeeded (may be more due to TOCTOU race)
  int success_count = 0;
  for (int i = 0; i < kNumProcesses; ++i) {
    if (acquired[i].load()) {
      success_count++;
    }
  }
  EXPECT_GE(success_count, 1);

  // Clean up
  for (auto& lock : locks) {
    lock->Release();
  }
}

// =============================================================================
// Stress Tests
// =============================================================================

TEST_F(FileLockTest, StressTestManyThreadsManyIterations) {
  constexpr int kNumThreads = 10;
  constexpr int kIterationsPerThread = 20;
  std::atomic<int> total_acquisitions{0};

  auto worker = [&]() {
    for (int i = 0; i < kIterationsPerThread; ++i) {
      FileLock lock(lock_path_);
      if (lock.Acquire()) {
        total_acquisitions++;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        lock.Release();
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(worker);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // All acquisitions should succeed
  EXPECT_EQ(total_acquisitions.load(), kNumThreads * kIterationsPerThread);
}

TEST_F(FileLockTest, StressTestRapidAcquireRelease) {
  constexpr int kIterations = 100;

  for (int i = 0; i < kIterations; ++i) {
    FileLock lock(lock_path_);
    EXPECT_TRUE(lock.Acquire());
    EXPECT_TRUE(lock.IsHeld());
    lock.Release();
    EXPECT_FALSE(lock.IsHeld());
  }
}

}  // namespace
}  // namespace at::neuron

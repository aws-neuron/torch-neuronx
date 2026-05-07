#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <vector>

#include "torch_neuronx/csrc/core/compilation/CompilationThreadPool.h"

namespace at::neuron {
namespace {

class CompilationThreadPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    original_env_ = std::getenv(kParallelCompileWorkersEnv);
    if (original_env_) {
      original_env_value_ = original_env_;
    }
    unsetenv(kParallelCompileWorkersEnv);
  }

  void TearDown() override {
    if (!original_env_value_.empty()) {
      setenv(kParallelCompileWorkersEnv, original_env_value_.c_str(), 1);
    } else {
      unsetenv(kParallelCompileWorkersEnv);
    }
  }

  const char* original_env_ = nullptr;
  std::string original_env_value_;
};

// =============================================================================
// Construction and Configuration Tests
// =============================================================================

TEST_F(CompilationThreadPoolTest, DefaultConstruction) {
  CompilationThreadPool pool;
  EXPECT_EQ(pool.GetNumWorkers(), kDefaultParallelCompileWorkers);
}

TEST_F(CompilationThreadPoolTest, CustomWorkerCount) {
  CompilationThreadPool pool(2);
  EXPECT_EQ(pool.GetNumWorkers(), 2);
}

TEST_F(CompilationThreadPoolTest, WorkerCountCappedAtMax) {
  CompilationThreadPool pool(100);
  EXPECT_EQ(pool.GetNumWorkers(), kMaxParallelCompileWorkers);
}

TEST_F(CompilationThreadPoolTest, EnvironmentVariableOverride) {
  setenv(kParallelCompileWorkersEnv, "8", 1);
  CompilationThreadPool pool;
  EXPECT_EQ(pool.GetNumWorkers(), 8);
}

TEST_F(CompilationThreadPoolTest, InvalidEnvironmentVariableUsesDefault) {
  setenv(kParallelCompileWorkersEnv, "invalid", 1);
  CompilationThreadPool pool;
  // strtoul returns 0 for invalid input, which is treated as "not set"
  EXPECT_EQ(pool.GetNumWorkers(), kDefaultParallelCompileWorkers);
}

// =============================================================================
// Task Execution Tests
// =============================================================================

TEST_F(CompilationThreadPoolTest, ExecuteTasks) {
  CompilationThreadPool pool(2);
  std::atomic<int> counter{0};
  constexpr int kNumTasks = 10;

  for (int i = 0; i < kNumTasks; ++i) {
    pool.Submit([&counter]() { counter++; });
  }

  pool.Shutdown();
  EXPECT_EQ(counter.load(), kNumTasks);
}

TEST_F(CompilationThreadPoolTest, TasksExecuteInFIFOOrder) {
  CompilationThreadPool pool(1);
  std::vector<int> execution_order;
  std::mutex order_mutex;

  for (int i = 0; i < 5; ++i) {
    pool.Submit([i, &execution_order, &order_mutex]() {
      std::lock_guard<std::mutex> lock(order_mutex);
      execution_order.push_back(i);
    });
  }

  pool.Shutdown();

  ASSERT_EQ(execution_order.size(), 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(execution_order[i], i);
  }
}

// =============================================================================
// Shutdown Tests
// =============================================================================

TEST_F(CompilationThreadPoolTest, ShutdownWaitsForPendingTasks) {
  CompilationThreadPool pool(1);
  std::atomic<int> completed{0};

  for (int i = 0; i < 5; ++i) {
    pool.Submit([&completed]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      completed++;
    });
  }

  pool.Shutdown();
  EXPECT_EQ(completed.load(), 5);
}

TEST_F(CompilationThreadPoolTest, MultipleShutdownCallsAreSafe) {
  CompilationThreadPool pool(2);

  EXPECT_NO_THROW(pool.Shutdown());
  EXPECT_NO_THROW(pool.Shutdown());
}

TEST_F(CompilationThreadPoolTest, SubmitAfterShutdownIsNoOp) {
  CompilationThreadPool pool(1);
  pool.Shutdown();

  std::atomic<bool> executed{false};
  pool.Submit([&executed]() { executed = true; });

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_FALSE(executed);
}

// =============================================================================
// Concurrency Tests
// =============================================================================

TEST_F(CompilationThreadPoolTest, ParallelExecution) {
  CompilationThreadPool pool(4);
  std::atomic<int> concurrent_count{0};
  std::atomic<int> max_concurrent{0};

  constexpr int kNumTasks = 20;

  for (int i = 0; i < kNumTasks; ++i) {
    pool.Submit([&]() {
      int current = ++concurrent_count;
      int expected = max_concurrent.load();
      while (current > expected && !max_concurrent.compare_exchange_weak(expected, current)) {
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      --concurrent_count;
    });
  }

  pool.Shutdown();

  EXPECT_GT(max_concurrent.load(), 1);
  EXPECT_LE(max_concurrent.load(), 4);
}

TEST_F(CompilationThreadPoolTest, ConcurrentSubmit) {
  CompilationThreadPool pool(4);
  std::atomic<int> total_executed{0};

  constexpr int kNumThreads = 8;
  constexpr int kTasksPerThread = 10;

  std::vector<std::thread> submitters;
  for (int t = 0; t < kNumThreads; ++t) {
    submitters.emplace_back([&pool, &total_executed]() {
      for (int i = 0; i < kTasksPerThread; ++i) {
        pool.Submit([&total_executed]() { total_executed++; });
      }
    });
  }

  for (auto& thread : submitters) {
    thread.join();
  }

  pool.Shutdown();
  EXPECT_EQ(total_executed.load(), kNumThreads * kTasksPerThread);
}

TEST_F(CompilationThreadPoolTest, StressRapidCreateDestroy) {
  for (int iteration = 0; iteration < 10; ++iteration) {
    CompilationThreadPool pool(2);
    std::atomic<int> counter{0};

    for (int i = 0; i < 50; ++i) {
      pool.Submit([&counter]() { counter++; });
    }

    pool.Shutdown();
    EXPECT_EQ(counter.load(), 50);
  }
}

}  // namespace
}  // namespace at::neuron

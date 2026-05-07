#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace at::neuron {

constexpr size_t kDefaultParallelCompileWorkers = 16;
constexpr size_t kMaxParallelCompileWorkers = 64;
constexpr const char* kParallelCompileWorkersEnv = "NEURON_PARALLEL_COMPILE_WORKERS";

// Thread pool for parallel HLO compilation.
class CompilationThreadPool {
 public:
  using Task = std::function<void()>;

  // Constructs a thread pool with the specified number of workers.
  explicit CompilationThreadPool(size_t num_workers = kDefaultParallelCompileWorkers);
  ~CompilationThreadPool();

  CompilationThreadPool(const CompilationThreadPool&) = delete;
  CompilationThreadPool& operator=(const CompilationThreadPool&) = delete;

  // Submits a task to the pool.
  void Submit(Task task);

  // Shuts down and waits for pending tasks.
  void Shutdown();

  // Returns the number of worker threads.
  size_t GetNumWorkers() const { return workers_.size(); }

 private:
  void WorkerLoop();

  std::vector<std::thread> workers_;
  std::queue<Task> task_queue_;
  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::atomic<bool> shutdown_{false};
};

// Resolves worker count from env var or use default.
size_t ResolveCompilationWorkerCount(size_t requested = kDefaultParallelCompileWorkers);

}  // namespace at::neuron

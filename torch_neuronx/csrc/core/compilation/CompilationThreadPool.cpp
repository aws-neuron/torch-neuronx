#include "CompilationThreadPool.h"

#include <algorithm>
#include <cstdlib>

namespace at::neuron {

size_t ResolveCompilationWorkerCount(size_t requested) {
  size_t num_workers = requested;

  if (const char* env = std::getenv(kParallelCompileWorkersEnv)) {
    size_t env_workers = std::strtoul(env, nullptr, 10);
    if (env_workers > 0) {
      num_workers = env_workers;
    }
  }

  return std::min(num_workers, kMaxParallelCompileWorkers);
}

CompilationThreadPool::CompilationThreadPool(size_t num_workers) {
  num_workers = ResolveCompilationWorkerCount(num_workers);

  workers_.reserve(num_workers);
  for (size_t i = 0; i < num_workers; ++i) {
    workers_.emplace_back(&CompilationThreadPool::WorkerLoop, this);
  }
}

CompilationThreadPool::~CompilationThreadPool() { Shutdown(); }

void CompilationThreadPool::Submit(Task task) {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (shutdown_) {
      return;
    }
    task_queue_.push(std::move(task));
  }
  queue_cv_.notify_one();
}

void CompilationThreadPool::Shutdown() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (shutdown_) {
      return;
    }
    shutdown_ = true;
  }
  queue_cv_.notify_all();

  for (auto& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

void CompilationThreadPool::WorkerLoop() {
  while (true) {
    Task task;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this] { return shutdown_ || !task_queue_.empty(); });

      if (shutdown_ && task_queue_.empty()) {
        return;
      }

      task = std::move(task_queue_.front());
      task_queue_.pop();
    }

    task();
  }
}

}  // namespace at::neuron

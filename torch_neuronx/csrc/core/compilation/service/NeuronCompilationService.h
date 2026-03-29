#pragma once

#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "torch_neuronx/csrc/core/compilation/NeuronCompiler.h"

namespace at::neuron {

// Lightweight compilation service with bounded worker pool.
class NeuronCompilationService {
 public:
  NeuronCompilationService();
  ~NeuronCompilationService();

  NeuronCompilationService(const NeuronCompilationService&) = delete;
  NeuronCompilationService& operator=(const NeuronCompilationService&) = delete;

  void Start(const std::string& socket_path, size_t num_workers = 0);
  void Stop();
  bool IsRunning() const { return running_.load(); }

 private:
  std::string socket_path_;
  int listen_fd_{-1};
  std::atomic<bool> running_{false};

  std::thread acceptor_thread_;

  CompilationConfig compilation_config_;

  struct CompileResult {
    std::string neff_path{};
    std::string error_message{};
  };

  struct CompileJob {
    std::string hlo_file_path;
    std::string ir_type;
    std::string optimization_level;
    std::string additional_args;
    std::promise<CompileResult> result;
  };

  std::vector<std::thread> workers_;
  std::queue<std::shared_ptr<CompileJob>> job_queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;

  void AcceptorLoop();
  void HandleClient(int client_fd);
  void WorkerLoop();
};

}  // namespace at::neuron

#pragma once

#include <atomic>
#include <string>
#include <vector>

#include "torch_neuronx/csrc/core/compilation/service/ServiceProtocol.h"

namespace at::neuron {

// Client for the shared compilation service.
// Each call to CompileSync creates a fresh connection, sends the request,
// blocks until the response arrives, and closes the connection.
// This avoids socket sharing between pool threads — no interleaving, no mutex.
class CompilationServiceClient {
 public:
  CompilationServiceClient();
  ~CompilationServiceClient();

  CompilationServiceClient(const CompilationServiceClient&) = delete;
  CompilationServiceClient& operator=(const CompilationServiceClient&) = delete;

  // Ensure the service is running (lazy startup). Call once before CompileSync.
  bool EnsureServiceRunning(const std::string& socket_path);

  // Synchronous compile: connects, sends HLO, blocks until NEFF returned, disconnects.
  // Thread-safe — each call uses its own connection.
  std::vector<uint8_t> CompileSync(const std::vector<uint8_t>& hlo_bytes,
                                   const std::string& persistent_cache_key,
                                   const std::string& ir_type,
                                   const std::string& optimization_level,
                                   const std::string& additional_args);

 private:
  std::string socket_path_;
  std::atomic<bool> service_ready_{false};
  std::atomic<uint64_t> request_seq_{0};

  int ConnectToService();
  std::string WriteHloTempFile(const std::vector<uint8_t>& hlo_bytes, uint64_t unique_id);
};

}  // namespace at::neuron

#include "CompilationServiceClient.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/cache/FileLock.h"
#include "torch_neuronx/csrc/core/compilation/service/NeuronCompilationService.h"

namespace at::neuron {

static std::unique_ptr<NeuronCompilationService> g_service_instance;
static std::mutex g_service_mutex;

CompilationServiceClient::CompilationServiceClient() = default;
CompilationServiceClient::~CompilationServiceClient() = default;

int CompilationServiceClient::ConnectToService() {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return -1;

  struct sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

  if (connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
    close(fd);
    return -1;
  }
  return fd;
}

bool CompilationServiceClient::EnsureServiceRunning(const std::string& socket_path) {
  if (service_ready_.load()) return true;
  if (socket_path.empty()) return false;

  socket_path_ = socket_path;

  // Try connecting — service may already be running
  int fd = ConnectToService();
  if (fd >= 0) {
    close(fd);
    service_ready_ = true;
    return true;
  }

  // Lazy startup: first rank starts the service
  std::string lock_path = socket_path + ".startup_lock";
  FileLockGuard startup_guard(lock_path);

  // Double-check
  fd = ConnectToService();
  if (fd >= 0) {
    close(fd);
    service_ready_ = true;
    return true;
  }

  {
    std::lock_guard lock(g_service_mutex);
    g_service_instance = std::make_unique<NeuronCompilationService>();
    g_service_instance->Start(socket_path);
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  fd = ConnectToService();
  if (fd >= 0) {
    close(fd);
    service_ready_ = true;
    return true;
  }

  TORCH_NEURONX_ERROR("Failed to start compilation service");
  return false;
}

std::vector<uint8_t> CompilationServiceClient::CompileSync(const std::vector<uint8_t>& hlo_bytes,
                                                           const std::string& ir_type,
                                                           const std::string& optimization_level,
                                                           const std::string& additional_args,
                                                           std::string* out_error) {
  if (!service_ready_.load()) return {};

  // Each call gets its own connection — no sharing between pool threads
  int fd = ConnectToService();
  if (fd < 0) return {};

  uint64_t req_id = request_seq_.fetch_add(1);
  std::string hlo_path = WriteHloTempFile(hlo_bytes, req_id);
  if (hlo_path.empty()) {
    close(fd);
    return {};
  }

  CompileRequest req;
  req.req_id = req_id;
  req.hlo_file_path = hlo_path;
  req.ir_type = ir_type;
  req.optimization_level = optimization_level;
  req.additional_args = additional_args;
  auto payload = req.Serialize();

  std::vector<uint8_t> neff_bytes;

  if (SendMessage(fd, ServiceMsgType::kCompileRequest, payload)) {
    ServiceMsgType resp_type;
    std::vector<uint8_t> resp_payload;
    if (RecvMessage(fd, &resp_type, &resp_payload) &&
        resp_type == ServiceMsgType::kCompileResponse) {
      auto resp = CompileResponse::Deserialize(resp_payload.data(), resp_payload.size());
      if (resp.success && !resp.neff_cache_path.empty()) {
        std::ifstream ifs(resp.neff_cache_path, std::ios::binary);
        if (ifs) {
          neff_bytes.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
        }
        std::remove(resp.neff_cache_path.c_str());
      } else if (out_error) {
        *out_error = resp.error_message;
      }
    }
  }

  close(fd);
  std::remove(hlo_path.c_str());
  return neff_bytes;
}

std::string CompilationServiceClient::WriteHloTempFile(const std::vector<uint8_t>& hlo_bytes,
                                                       uint64_t unique_id) {
  std::string path =
      "/tmp/neuron_hlo_" + std::to_string(getpid()) + "_" + std::to_string(unique_id) + ".pb";
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) return "";
  ofs.write(reinterpret_cast<const char*>(hlo_bytes.data()),
            static_cast<std::streamsize>(hlo_bytes.size()));
  return path;
}

}  // namespace at::neuron

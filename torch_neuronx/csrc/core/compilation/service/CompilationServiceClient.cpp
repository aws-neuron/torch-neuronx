#include "CompilationServiceClient.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
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
  if (service_ready_.load()) {
    if (socket_path != socket_path_) {
      TORCH_NEURONX_ERROR("EnsureServiceRunning called with new path=", socket_path,
                          " but service already ready on path=", socket_path_,
                          " — ignoring new path");
    }
    return true;
  }
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

  TORCH_NEURONX_ERROR("EnsureServiceRunning: failed to start service socket=", socket_path,
                      " pid=", getpid());
  return false;
}

std::vector<uint8_t> CompilationServiceClient::CompileSync(
    const std::vector<uint8_t>& hlo_bytes, const std::string& ir_type, const std::string& op_name,
    const std::string& cache_key, const std::string& optimization_level,
    const std::string& additional_args, std::string* out_error) {
  auto set_err = [out_error](const std::string& msg) {
    TORCH_NEURONX_ERROR("Compilation service connection failed: ", msg);
    if (out_error) *out_error = msg;
  };

  if (!service_ready_.load()) {
    set_err("service not ready (EnsureServiceRunning was never called or failed) for op=" +
            op_name + " cache_key=" + cache_key);
    return {};
  }

  // Each call gets its own connection — no sharing between pool threads.
  // If ConnectToService fails, the service socket was torn down (the
  // subprocess that started the service exited). By design the service is
  // single-owner and peer ranks must not create a new one — surface the
  // error with full context.
  int fd = ConnectToService();
  if (fd < 0) {
    set_err("ConnectToService failed (socket=" + socket_path_ + " errno=" + std::to_string(errno) +
            ") for op=" + op_name + " cache_key=" + cache_key);
    return {};
  }

  uint64_t req_id = request_seq_.fetch_add(1);
  std::string hlo_path = WriteHloTempFile(hlo_bytes, req_id);
  if (hlo_path.empty()) {
    close(fd);
    set_err("WriteHloTempFile failed (pid=" + std::to_string(getpid()) +
            " req_id=" + std::to_string(req_id) + " hlo_bytes=" + std::to_string(hlo_bytes.size()) +
            ") for op=" + op_name + " cache_key=" + cache_key);
    return {};
  }

  CompileRequest req;
  req.req_id = req_id;
  req.hlo_file_path = hlo_path;
  req.ir_type = ir_type;
  req.op_name = op_name;
  req.cache_key = cache_key;
  req.optimization_level = optimization_level;
  req.additional_args = additional_args;
  auto payload = req.Serialize();

  std::vector<uint8_t> neff_bytes;

  if (!SendMessage(fd, ServiceMsgType::kCompileRequest, payload)) {
    close(fd);
    std::remove(hlo_path.c_str());
    set_err("SendMessage(kCompileRequest) failed (errno=" + std::to_string(errno) +
            " payload_size=" + std::to_string(payload.size()) + " req_id=" +
            std::to_string(req_id) + ") for op=" + op_name + " cache_key=" + cache_key);
    return {};
  }

  ServiceMsgType resp_type;
  std::vector<uint8_t> resp_payload;
  if (!RecvMessage(fd, &resp_type, &resp_payload)) {
    close(fd);
    std::remove(hlo_path.c_str());
    set_err("RecvMessage failed (errno=" + std::to_string(errno) + " req_id=" +
            std::to_string(req_id) + ") for op=" + op_name + " cache_key=" + cache_key);
    return {};
  }

  if (resp_type != ServiceMsgType::kCompileResponse) {
    close(fd);
    std::remove(hlo_path.c_str());
    set_err("unexpected response type=" + std::to_string(static_cast<int>(resp_type)) +
            " (expected kCompileResponse) req_id=" + std::to_string(req_id) + " for op=" + op_name +
            " cache_key=" + cache_key);
    return {};
  }

  auto resp = CompileResponse::Deserialize(resp_payload.data(), resp_payload.size());
  if (!resp.success) {
    close(fd);
    std::remove(hlo_path.c_str());
    if (out_error) *out_error = resp.error_message;
    return {};
  }

  if (resp.neff_cache_path.empty()) {
    close(fd);
    std::remove(hlo_path.c_str());
    set_err("response success=true but neff_cache_path is empty (req_id=" + std::to_string(req_id) +
            ") for op=" + op_name + " cache_key=" + cache_key);
    return {};
  }

  std::ifstream ifs(resp.neff_cache_path, std::ios::binary);
  if (!ifs) {
    close(fd);
    std::remove(hlo_path.c_str());
    set_err("failed to open NEFF cache path '" + resp.neff_cache_path +
            "' (errno=" + std::to_string(errno) + " req_id=" + std::to_string(req_id) +
            ") for op=" + op_name + " cache_key=" + cache_key);
    return {};
  }
  neff_bytes.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
  ifs.close();
  std::remove(resp.neff_cache_path.c_str());

  if (neff_bytes.empty()) {
    close(fd);
    std::remove(hlo_path.c_str());
    set_err("NEFF cache file '" + resp.neff_cache_path + "' read as 0 bytes (req_id=" +
            std::to_string(req_id) + ") for op=" + op_name + " cache_key=" + cache_key);
    return {};
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

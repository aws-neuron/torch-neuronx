#include "NeuronCompilationService.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/compilation/service/ServiceProtocol.h"

namespace at::neuron {

namespace {
constexpr size_t kMaxServiceWorkers = 128;
constexpr const char* kServiceWorkersEnv = "NEURON_COMPILE_SERVICE_WORKERS";

size_t ResolveWorkerCount(size_t requested) {
  if (const char* env = std::getenv(kServiceWorkersEnv)) {
    size_t n = std::strtoul(env, nullptr, 10);
    if (n > 0) requested = n;
  }
  if (requested == 0) {
    requested = std::max(4u, std::thread::hardware_concurrency() / 4);
  }
  return std::min(requested, kMaxServiceWorkers);
}
}  // namespace

NeuronCompilationService::NeuronCompilationService() = default;

NeuronCompilationService::~NeuronCompilationService() { Stop(); }

void NeuronCompilationService::Start(const std::string& socket_path, size_t num_workers) {
  if (running_.exchange(true)) return;

  socket_path_ = socket_path;
  num_workers = ResolveWorkerCount(num_workers);

  compilation_config_ = CompilationConfig::FromEnvironment();

  unlink(socket_path.c_str());

  listen_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    running_ = false;
    throw std::runtime_error("Failed to create service socket");
  }

  struct sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

  if (bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
    close(listen_fd_);
    listen_fd_ = -1;
    running_ = false;
    throw std::runtime_error("Failed to bind service socket: " + socket_path);
  }

  if (listen(listen_fd_, 256) < 0) {
    close(listen_fd_);
    listen_fd_ = -1;
    running_ = false;
    throw std::runtime_error("Failed to listen on service socket");
  }

  workers_.reserve(num_workers);
  for (size_t i = 0; i < num_workers; ++i) {
    workers_.emplace_back(&NeuronCompilationService::WorkerLoop, this);
  }

  acceptor_thread_ = std::thread(&NeuronCompilationService::AcceptorLoop, this);
  TORCH_NEURONX_DEBUG("Compilation service started on", socket_path, "workers=", num_workers,
                      "target=", compilation_config_.platform_target);
}

void NeuronCompilationService::Stop() {
  if (!running_.exchange(false)) return;

  if (listen_fd_ >= 0) {
    ::shutdown(listen_fd_, SHUT_RDWR);
    close(listen_fd_);
    listen_fd_ = -1;
  }
  if (acceptor_thread_.joinable()) acceptor_thread_.join();

  queue_cv_.notify_all();
  for (auto& w : workers_) {
    if (w.joinable()) w.join();
  }

  unlink(socket_path_.c_str());
  unlink((socket_path_ + ".startup_lock").c_str());
}

void NeuronCompilationService::AcceptorLoop() {
  while (running_.load()) {
    int client_fd = accept(listen_fd_, nullptr, nullptr);
    if (client_fd < 0) {
      if (!running_.load()) break;
      continue;
    }
    std::thread(&NeuronCompilationService::HandleClient, this, client_fd).detach();
  }
}

void NeuronCompilationService::HandleClient(int client_fd) {
  ServiceMsgType type;
  std::vector<uint8_t> payload;
  if (!RecvMessage(client_fd, &type, &payload) || type != ServiceMsgType::kCompileRequest) {
    close(client_fd);
    return;
  }

  auto req = CompileRequest::Deserialize(payload.data(), payload.size());

  auto job = std::make_shared<CompileJob>();
  job->hlo_file_path = req.hlo_file_path;
  job->ir_type = req.ir_type;
  job->op_name = req.op_name;
  job->cache_key = req.cache_key;
  job->optimization_level = req.optimization_level;
  job->additional_args = req.additional_args;

  std::future<CompileResult> result_future = job->result.get_future();

  size_t queue_depth_after = 0;
  {
    std::lock_guard lock(queue_mutex_);
    job_queue_.push(job);
    queue_depth_after = job_queue_.size();
  }
  queue_cv_.notify_one();
  TORCH_NEURONX_DEBUG("HandleClient: enqueued req_id=", req.req_id, " op=", req.op_name,
                      " cache_key=", req.cache_key, " queue_depth=", queue_depth_after);

  auto result = result_future.get();

  CompileResponse resp;
  resp.req_id = req.req_id;
  resp.success = !result.neff_path.empty();
  resp.neff_cache_path = result.neff_path;
  resp.error_message = result.error_message;
  auto resp_payload = resp.Serialize();
  if (!SendMessage(client_fd, ServiceMsgType::kCompileResponse, resp_payload)) {
    TORCH_NEURONX_ERROR(
        "HandleClient: SendMessage(kCompileResponse) failed for req_id=", req.req_id,
        " op=", req.op_name, " success=", resp.success, " errno=", errno);
  }

  close(client_fd);
}

void NeuronCompilationService::WorkerLoop() {
  while (true) {
    std::shared_ptr<CompileJob> job;
    {
      std::unique_lock lock(queue_mutex_);
      queue_cv_.wait(lock, [this] { return !running_.load() || !job_queue_.empty(); });
      if (!running_.load() && job_queue_.empty()) return;
      job = job_queue_.front();
      job_queue_.pop();
    }

    std::string neff_path;
    std::string error_msg;
    try {
      std::ifstream ifs(job->hlo_file_path, std::ios::binary);
      if (ifs) {
        std::vector<uint8_t> hlo_bytes((std::istreambuf_iterator<char>(ifs)),
                                       std::istreambuf_iterator<char>());
        ifs.close();

        // Build config from per-request values
        CompilationConfig request_config = compilation_config_;
        request_config.optimization_level = job->optimization_level;
        request_config.additional_args = job->additional_args;

        auto neff_bytes = NeuronCompiler::CompileHloToNeff(hlo_bytes, request_config, job->ir_type,
                                                           job->op_name, job->cache_key);

        if (!neff_bytes.empty()) {
          neff_path = job->hlo_file_path + ".neff";
          std::ofstream ofs(neff_path, std::ios::binary);
          if (ofs) {
            ofs.write(reinterpret_cast<const char*>(neff_bytes.data()),
                      static_cast<std::streamsize>(neff_bytes.size()));
          } else {
            neff_path.clear();
            error_msg = "Failed to write NEFF to temp file";
          }
        } else {
          error_msg =
              "NeuronCompiler::CompileHloToNeff returned empty NEFF for op=" + job->op_name +
              " cache_key=" + job->cache_key + " ir_type=" + job->ir_type;
          TORCH_NEURONX_ERROR(error_msg);
        }
      } else {
        error_msg = "Failed to read HLO file: " + job->hlo_file_path;
      }
    } catch (const std::exception& e) {
      error_msg = e.what();
    }

    job->result.set_value({neff_path, error_msg});
  }
}

}  // namespace at::neuron

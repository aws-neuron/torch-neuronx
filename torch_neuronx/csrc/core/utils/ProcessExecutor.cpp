#include "torch_neuronx/csrc/core/utils/ProcessExecutor.h"

#include <poll.h>
#include <unistd.h>

#include <cerrno>

#include "torch_neuronx/csrc/core/NeuronLogging.h"

// POSIX requires programs to declare `environ` themselves (it is not standardised
// in any header). glibc's <unistd.h> exposes it under default feature-test macros
// today, but a stricter sysroot would not. The declaration MUST be at file scope
// (or wrapped in `extern "C"`) — declaring it inside a C++ namespace produces a
// mangled namespace-scoped symbol (e.g. `at::neuron::environ`) that the linker
// cannot resolve against the C-linkage global.
extern "C" char** environ;

namespace at::neuron {

// PipeManager implementation

PipeManager::PipeManager() {
  if (pipe(fds_) == -1) {
    throw std::runtime_error("Failed to create pipe: " + std::string(strerror(errno)));
  }
}

PipeManager::~PipeManager() {
  CloseRead();
  CloseWrite();
}

PipeManager::PipeManager(PipeManager&& other) noexcept {
  fds_[0] = other.fds_[0];
  fds_[1] = other.fds_[1];
  other.fds_[0] = kInvalidFd;
  other.fds_[1] = kInvalidFd;
}

void PipeManager::CloseRead() {
  if (fds_[0] != kInvalidFd) {
    close(fds_[0]);
    fds_[0] = kInvalidFd;
  }
}

void PipeManager::CloseWrite() {
  if (fds_[1] != kInvalidFd) {
    close(fds_[1]);
    fds_[1] = kInvalidFd;
  }
}

// ProcessExecutor implementation

void ProcessExecutor::Execute(const std::vector<std::string>& args, const std::string& context,
                              const std::string& working_dir) {
  if (args.empty()) {
    throw std::runtime_error("No command arguments provided");
  }

  // Convert args to char* array for posix_spawn (same format as execvp)
  std::vector<char*> argv;
  for (const auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }
  argv.push_back(nullptr);

  // Create pipes using RAII
  PipeManager stdout_pipe, stderr_pipe;

  // RAII for posix_spawn_file_actions_t so destroy runs on every exit path.
  struct FileActionsGuard {
    posix_spawn_file_actions_t actions{};
    bool inited{false};
    int Init() {
      int rc = posix_spawn_file_actions_init(&actions);
      inited = (rc == 0);
      return rc;
    }
    ~FileActionsGuard() {
      if (inited) posix_spawn_file_actions_destroy(&actions);
    }
  };

  // Set up file actions for the spawned child. posix_spawn avoids fork()'s
  // copy-on-write of the parent's address space.
  FileActionsGuard guard;
  int rc = guard.Init();
  if (rc != 0) {
    throw std::runtime_error("posix_spawn_file_actions_init failed: " + std::string(strerror(rc)));
  }

  // Helper: throw on non-zero rc from posix_spawn_file_actions_*.
  auto check = [](int rc, const char* op) {
    if (rc != 0) {
      throw std::runtime_error(std::string("posix_spawn_file_actions_") + op +
                               " failed: " + strerror(rc));
    }
  };

  // Redirect stdout and stderr to pipes (dup2 actions execute in registration
  // order; close actions below run after dup2 so STDOUT/STDERR retain the pipe).
  check(posix_spawn_file_actions_adddup2(&guard.actions, stdout_pipe.write_fd(), STDOUT_FILENO),
        "adddup2(stdout)");
  check(posix_spawn_file_actions_adddup2(&guard.actions, stderr_pipe.write_fd(), STDERR_FILENO),
        "adddup2(stderr)");

  // Close pipe ends the child doesn't need (parent's read fds, plus the
  // original write fds whose duplicates already live on STDOUT/STDERR_FILENO).
  check(posix_spawn_file_actions_addclose(&guard.actions, stdout_pipe.read_fd()),
        "addclose(stdout_read)");
  check(posix_spawn_file_actions_addclose(&guard.actions, stderr_pipe.read_fd()),
        "addclose(stderr_read)");
  check(posix_spawn_file_actions_addclose(&guard.actions, stdout_pipe.write_fd()),
        "addclose(stdout_write)");
  check(posix_spawn_file_actions_addclose(&guard.actions, stderr_pipe.write_fd()),
        "addclose(stderr_write)");

  // Set working directory if specified.
  if (!working_dir.empty()) {
    check(posix_spawn_file_actions_addchdir_np(&guard.actions, working_dir.c_str()), "addchdir_np");
  }

  pid_t pid;
  int err = posix_spawnp(&pid, argv[0], &guard.actions, nullptr, argv.data(), environ);
  if (err != 0) {
    throw std::runtime_error("posix_spawn failed: " + std::string(strerror(err)));
  }

  WaitAndCheckResult(pid, stdout_pipe, stderr_pipe, context);
}

void ProcessExecutor::WaitAndCheckResult(pid_t pid, PipeManager& stdout_pipe,
                                         PipeManager& stderr_pipe, const std::string& context) {
  stdout_pipe.CloseWrite();
  stderr_pipe.CloseWrite();

  // Read both pipes concurrently using poll() to avoid deadlock when
  // the child fills one pipe buffer while we're blocked reading the other.
  std::string stdout_output, stderr_output;
  struct pollfd fds[2];
  fds[0] = {stdout_pipe.read_fd(), POLLIN, 0};
  fds[1] = {stderr_pipe.read_fd(), POLLIN, 0};
  int open_fds = 2;

  while (open_fds > 0) {
    int ret = poll(fds, 2, -1);
    if (ret < 0) {
      if (errno == EINTR) continue;
      TORCH_NEURONX_WARN("poll() failed during subprocess pipe read", "errno=", errno,
                         "context=", context);
      break;
    }
    for (int i = 0; i < 2; i++) {
      if (fds[i].fd < 0) continue;
      if (fds[i].revents & (POLLIN | POLLHUP)) {
        char buf[kPipeBufferSize] = {};
        ssize_t n = read(fds[i].fd, buf, sizeof(buf));
        if (n > 0) {
          (i == 0 ? stdout_output : stderr_output).append(buf, n);
        } else if (n == 0) {
          // EOF — child closed this pipe
          fds[i].fd = -1;
          open_fds--;
        } else {
          // read error
          if (errno == EINTR || errno == EAGAIN) continue;
          TORCH_NEURONX_WARN("read() failed during subprocess pipe read",
                             "fd=", (i == 0 ? "stdout" : "stderr"), "errno=", errno,
                             "context=", context);
          fds[i].fd = -1;
          open_fds--;
        }
      }
    }
  }

  int status;
  if (waitpid(pid, &status, 0) == -1) {
    throw std::runtime_error("waitpid failed: " + std::string(strerror(errno)));
  }

  int exit_code = WEXITSTATUS(status);

  if (exit_code != 0) {
    std::string error_msg = "Command failed";
    if (!context.empty()) {
      error_msg += " (" + context + ")";
    }
    error_msg += " with exit code " + std::to_string(exit_code);
    if (!stderr_output.empty()) {
      error_msg += ": " + stderr_output;
    }
    throw std::runtime_error(error_msg);
  }
}

}  // namespace at::neuron

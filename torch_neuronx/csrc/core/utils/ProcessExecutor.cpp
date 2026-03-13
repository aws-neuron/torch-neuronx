#include "torch_neuronx/csrc/core/utils/ProcessExecutor.h"

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

  // Convert args to char* array for execvp
  std::vector<char*> argv;
  for (const auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }
  argv.push_back(nullptr);

  // Create pipes using RAII
  PipeManager stdout_pipe, stderr_pipe;

  pid_t pid = fork();
  if (pid == 0) {
    SetupChildProcess(stdout_pipe, stderr_pipe, working_dir);
    ExecCommand(argv);
  } else if (pid > 0) {
    WaitAndCheckResult(pid, stdout_pipe, stderr_pipe, context);
  } else {
    throw std::runtime_error("Fork failed: " + std::string(strerror(errno)));
  }
}

void ProcessExecutor::SetupChildProcess(PipeManager& stdout_pipe, PipeManager& stderr_pipe,
                                        const std::string& working_dir) {
  stdout_pipe.CloseRead();
  stderr_pipe.CloseRead();

  if (!working_dir.empty()) {
    if (chdir(working_dir.c_str()) == -1) {
      exit(kExitDup2Failed);
    }
  }

  if (dup2(stdout_pipe.write_fd(), STDOUT_FILENO) == -1 ||
      dup2(stderr_pipe.write_fd(), STDERR_FILENO) == -1) {
    exit(kExitDup2Failed);
  }

  stdout_pipe.CloseWrite();
  stderr_pipe.CloseWrite();
}

void ProcessExecutor::ExecCommand(const std::vector<char*>& argv) {
  execvp(argv[0], argv.data());
  exit(kExitExecvpFailed);
}

void ProcessExecutor::WaitAndCheckResult(pid_t pid, PipeManager& stdout_pipe,
                                         PipeManager& stderr_pipe, const std::string& context) {
  stdout_pipe.CloseWrite();
  stderr_pipe.CloseWrite();

  std::string stdout_output = ReadPipeOutput(stdout_pipe.read_fd());
  std::string stderr_output = ReadPipeOutput(stderr_pipe.read_fd());

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

std::string ProcessExecutor::ReadPipeOutput(int fd) {
  std::string output;
  char buffer[kPipeBufferSize];
  ssize_t bytes_read;

  while ((bytes_read = read(fd, buffer, sizeof(buffer) - 1)) > 0) {
    buffer[bytes_read] = '\0';
    output += buffer;
  }

  return output;
}

}  // namespace at::neuron

#pragma once

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace at::neuron {

// RAII wrapper for pipe file descriptors.
// Automatically creates a pipe on construction and closes both ends on destruction.
class PipeManager {
 public:
  PipeManager();
  ~PipeManager();

  // Non-copyable, movable
  PipeManager(const PipeManager&) = delete;
  PipeManager& operator=(const PipeManager&) = delete;
  PipeManager(PipeManager&& other) noexcept;

  int read_fd() const { return fds_[0]; }
  int write_fd() const { return fds_[1]; }

  void CloseRead();
  void CloseWrite();

 private:
  static constexpr int kInvalidFd = -1;
  int fds_[2] = {kInvalidFd, kInvalidFd};
};

// Executes external commands and captures their output.
// Provides process management and error handling for subprocess execution.
class ProcessExecutor {
 public:
  // Executes a command and throws on failure.
  //
  // Args:
  //   args: Command arguments (first element is the executable)
  //   context: Optional context string to include in error message
  //   working_dir: Optional working directory for the command
  static void Execute(const std::vector<std::string>& args, const std::string& context = "",
                      const std::string& working_dir = "");

 private:
  static constexpr int kExitDup2Failed = 126;
  static constexpr int kExitExecvpFailed = 127;
  static constexpr size_t kPipeBufferSize = 1024;

  static void SetupChildProcess(PipeManager& stdout_pipe, PipeManager& stderr_pipe,
                                const std::string& working_dir);
  [[noreturn]] static void ExecCommand(const std::vector<char*>& argv);
  static void WaitAndCheckResult(pid_t pid, PipeManager& stdout_pipe, PipeManager& stderr_pipe,
                                 const std::string& context);
};

}  // namespace at::neuron

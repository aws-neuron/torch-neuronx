#include "NeuronCompiler.h"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace at::neuron {

namespace {

// File names used during compilation process.
constexpr const char* kHloFilename = "module.hlo.pb";
constexpr const char* kMlirFilename = "module.mlir";
constexpr const char* kNeffFilename = "module.neff";

// Compiler executable constant.
constexpr const char* kNeuronxCCBinary = "neuronx-cc";

// Command line arguments for neuronx-cc.
constexpr const char* kCompileCommand = "compile";
constexpr const char* kFrameworkFlag = "--framework";
constexpr const char* kTargetFlag = "--target";
constexpr const char* kLogicalNeuronCoresFlag = "--lnc";
constexpr const char* kOutputFlag = "--output";

// Process exit codes for child process failures.
constexpr int kExitDup2Failed = 126;
constexpr int kExitExecvpFailed = 127;

// Buffer and file descriptor constants.
constexpr size_t kPipeBufferSize = 1024;
constexpr int kInvalidFd = -1;

}  // namespace

// RAII wrapper for pipe file descriptors.
// Automatically creates a pipe on construction and closes both ends on destruction.
class PipeManager {
 public:
  PipeManager() {
    if (pipe(fds_) == -1) {
      throw std::runtime_error("Failed to create pipe: " + std::string(strerror(errno)));
    }
  }

  ~PipeManager() {
    CloseRead();
    CloseWrite();
  }

  // Non-copyable, movable
  PipeManager(const PipeManager&) = delete;
  PipeManager& operator=(const PipeManager&) = delete;

  PipeManager(PipeManager&& other) noexcept {
    fds_[0] = other.fds_[0];
    fds_[1] = other.fds_[1];
    other.fds_[0] = kInvalidFd;
    other.fds_[1] = kInvalidFd;
  }

  int read_fd() const { return fds_[0]; }
  int write_fd() const { return fds_[1]; }

  void CloseRead() {
    if (fds_[0] != kInvalidFd) {
      close(fds_[0]);
      fds_[0] = kInvalidFd;
    }
  }

  void CloseWrite() {
    if (fds_[1] != kInvalidFd) {
      close(fds_[1]);
      fds_[1] = kInvalidFd;
    }
  }

 private:
  int fds_[2] = {kInvalidFd, kInvalidFd};
};

// Executes external commands and captures their output.
// Provides process management and error handling for subprocess execution.
class ProcessExecutor {
 public:
  // Executes a command and throws on failure.
  // Args:
  //   args: Command arguments (first element is the executable)
  //   context: Optional context string to include in error message
  //   working_dir: Optional working directory for the command
  static void Execute(const std::vector<std::string>& args, const std::string& context = "",
                      const std::string& working_dir = "") {
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

 private:
  static void SetupChildProcess(PipeManager& stdout_pipe, PipeManager& stderr_pipe,
                                const std::string& working_dir) {
    // Close read ends in child
    stdout_pipe.CloseRead();
    stderr_pipe.CloseRead();

    // Change working directory if specified
    if (!working_dir.empty()) {
      if (chdir(working_dir.c_str()) == -1) {
        exit(kExitDup2Failed);
      }
    }

    // Redirect stdout and stderr to pipes
    if (dup2(stdout_pipe.write_fd(), STDOUT_FILENO) == -1 ||
        dup2(stderr_pipe.write_fd(), STDERR_FILENO) == -1) {
      exit(kExitDup2Failed);
    }

    // Close original write ends
    stdout_pipe.CloseWrite();
    stderr_pipe.CloseWrite();
  }

  [[noreturn]] static void ExecCommand(const std::vector<char*>& argv) {
    execvp(argv[0], argv.data());
    exit(kExitExecvpFailed);
  }

  static void WaitAndCheckResult(pid_t pid, PipeManager& stdout_pipe, PipeManager& stderr_pipe,
                                 const std::string& context) {
    // Close write ends in parent
    stdout_pipe.CloseWrite();
    stderr_pipe.CloseWrite();

    // Read output from both pipes
    std::string stdout_output = ReadPipeOutput(stdout_pipe.read_fd());
    std::string stderr_output = ReadPipeOutput(stderr_pipe.read_fd());

    // Wait for child process and get exit code
    int status;
    if (waitpid(pid, &status, 0) == -1) {
      throw std::runtime_error("waitpid failed: " + std::string(strerror(errno)));
    }

    int exit_code = WEXITSTATUS(status);

    // Throw on error with detailed context
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

  static std::string ReadPipeOutput(int fd) {
    std::string output;
    char buffer[kPipeBufferSize];
    ssize_t bytes_read;

    while ((bytes_read = read(fd, buffer, sizeof(buffer) - 1)) > 0) {
      buffer[bytes_read] = '\0';
      output += buffer;
    }

    return output;
  }
};

// RAII wrapper for temporary directory management.
// Automatically creates a temporary directory on construction and removes it on destruction.
class TempDirectory {
 public:
  TempDirectory() {
    char temp_template[] = "/tmp/neuron_compile_XXXXXX";
    char* temp_dir = mkdtemp(temp_template);
    if (!temp_dir) {
      throw std::runtime_error("Failed to create temporary directory: " +
                               std::string(strerror(errno)));
    }
    path_ = temp_dir;
  }

  ~TempDirectory() {
    if (!path_.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(path_, ec);
      // Don't throw in destructor - just log error if needed
      // In a production system, you might want to log this error
    }
  }

  // Non-copyable, movable
  TempDirectory(const TempDirectory&) = delete;
  TempDirectory& operator=(const TempDirectory&) = delete;

  TempDirectory(TempDirectory&& other) noexcept : path_(std::move(other.path_)) {
    other.path_.clear();
  }

  TempDirectory& operator=(TempDirectory&& other) noexcept {
    if (this != &other) {
      // Clean up current directory
      if (!path_.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
      }
      // Move from other
      path_ = std::move(other.path_);
      other.path_.clear();
    }
    return *this;
  }

  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

std::vector<uint8_t> NeuronCompiler::CompileHloToNeff(const std::vector<uint8_t>& hlo_bytes,
                                                      const CompilationConfig& config,
                                                      const std::string& ir_type) {
  // Create temporary directory using RAII
  TempDirectory temp_compilation_dir;

  // Determine file format based on ir_type parameter
  // "StableHLO" uses MLIR format, "XLA" (default) uses HLO protobuf
  std::string hlo_input_file =
      temp_compilation_dir.path() + "/" + (ir_type == "StableHLO" ? kMlirFilename : kHloFilename);
  std::string neff_output_file = temp_compilation_dir.path() + "/" + kNeffFilename;

  // Write HLO bytes to temporary file
  std::ofstream hlo_output_stream(hlo_input_file, std::ios::binary);
  if (!hlo_output_stream) {
    throw std::runtime_error("Failed to create HLO file: " + hlo_input_file);
  }
  hlo_output_stream.write(reinterpret_cast<const char*>(hlo_bytes.data()), hlo_bytes.size());
  hlo_output_stream.close();

  // Build neuronx-cc command arguments using config
  // execvp will search PATH automatically, so just use the binary name
  std::vector<std::string> compiler_args = {kNeuronxCCBinary,
                                            kCompileCommand,
                                            kFrameworkFlag,
                                            config.framework,
                                            kTargetFlag,
                                            config.platform_target,
                                            kLogicalNeuronCoresFlag,
                                            config.logical_neuron_cores,
                                            config.optimization_level};

  // Parse and add additional compiler arguments
  std::istringstream additional_args_stream(config.additional_args);
  std::string single_arg;
  while (additional_args_stream >> single_arg) {
    compiler_args.push_back(single_arg);
  }

  // Add input file and output specification
  compiler_args.push_back(hlo_input_file);
  compiler_args.push_back(kOutputFlag);
  compiler_args.push_back(neff_output_file);

  // Execute neuronx-cc compiler in the temporary directory
  ProcessExecutor::Execute(compiler_args, std::string(kNeuronxCCBinary) + " compilation",
                           temp_compilation_dir.path());

  // Read compiled NEFF file
  std::ifstream neff_input_stream(neff_output_file, std::ios::binary | std::ios::ate);
  if (!neff_input_stream) {
    throw std::runtime_error("Failed to read compiled NEFF file: " + neff_output_file);
  }

  std::streamsize neff_file_size = neff_input_stream.tellg();
  neff_input_stream.seekg(0, std::ios::beg);

  std::vector<uint8_t> compiled_neff_bytes(neff_file_size);
  if (!neff_input_stream.read(reinterpret_cast<char*>(compiled_neff_bytes.data()),
                              neff_file_size)) {
    throw std::runtime_error("Failed to read NEFF data from file: " + neff_output_file);
  }

  return compiled_neff_bytes;
}

bool NeuronCompiler::IsStableHLOEnabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("TORCH_NEURONX_ENABLE_STABLEHLO");
    // Default to enabled (true). Only disable if explicitly set to "0" or "false"
    if (!env) {
      return true;  // Default: enabled
    }
    std::string env_str(env);
    return !(env_str == "0" || env_str == "false");
  }();
  return enabled;
}

}  // namespace at::neuron

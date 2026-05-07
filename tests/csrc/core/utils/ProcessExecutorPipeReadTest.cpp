#include <gtest/gtest.h>

#include <chrono>
#include <future>

#include "torch_neuronx/csrc/core/utils/ProcessExecutor.h"

using namespace at::neuron;

class ProcessExecutorPipeReadTest : public ::testing::Test {};

// Helper: run ProcessExecutor::Execute in a thread with a timeout.
// Returns true if completed within timeout, false if it would have deadlocked.
bool ExecuteWithTimeout(const std::vector<std::string>& args, int timeout_seconds = 10) {
  auto future =
      std::async(std::launch::async, [&]() { ProcessExecutor::Execute(args, "pipe read test"); });
  return future.wait_for(std::chrono::seconds(timeout_seconds)) == std::future_status::ready;
}

// Core regression test: >64KB on stderr must not deadlock.
// Without the poll() fix, the parent blocks on read(stdout) while the child
// blocks on write(stderr) because the 64KB pipe buffer is full.
TEST_F(ProcessExecutorPipeReadTest, LargeStderrCompletesWithinTimeout) {
  std::vector<std::string> args = {
      "python3", "-c", "import sys; sys.stderr.write('x' * 100000); sys.stdout.write('ok\\n')"};

  EXPECT_TRUE(ExecuteWithTimeout(args)) << "ProcessExecutor deadlocked on large stderr output";
}

// Symmetric test: large stdout should also work.
TEST_F(ProcessExecutorPipeReadTest, LargeStdoutCompletesWithinTimeout) {
  std::vector<std::string> args = {"python3", "-c",
                                   "import sys; sys.stdout.write('x' * 100000 + '\\n')"};

  EXPECT_TRUE(ExecuteWithTimeout(args)) << "ProcessExecutor deadlocked on large stdout output";
}

// Both pipes produce >64KB simultaneously.
TEST_F(ProcessExecutorPipeReadTest, LargeBothPipesCompletesWithinTimeout) {
  std::vector<std::string> args = {
      "python3", "-c",
      "import sys; sys.stderr.write('e' * 100000); sys.stdout.write('o' * 100000 + '\\n')"};

  EXPECT_TRUE(ExecuteWithTimeout(args))
      << "ProcessExecutor deadlocked on large output on both pipes";
}

// Non-zero exit code should throw, not deadlock.
TEST_F(ProcessExecutorPipeReadTest, NonZeroExitWithLargeStderrThrows) {
  std::vector<std::string> args = {"python3", "-c",
                                   "import sys; sys.stderr.write('error ' * 20000); sys.exit(1)"};

  auto future = std::async(std::launch::async, [&]() {
    EXPECT_THROW(ProcessExecutor::Execute(args, "error test"), std::runtime_error);
  });
  EXPECT_NE(future.wait_for(std::chrono::seconds(10)), std::future_status::timeout)
      << "ProcessExecutor deadlocked on failed command with large stderr";
}

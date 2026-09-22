#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <string>

#include "torch_neuronx/csrc/core/utils/ProcessExecutor.h"

extern "C" char** environ;

namespace {
std::atomic<int> stable_spawn_calls{0};
}

extern "C" int torch_neuronx_posix_spawn_with_stable_environ(
    pid_t* pid, const char* file, const posix_spawn_file_actions_t* file_actions,
    const posix_spawnattr_t* attrp, char* const argv[], const char* working_dir) {
  (void)working_dir;
  stable_spawn_calls.fetch_add(1, std::memory_order_relaxed);
  return posix_spawnp(pid, file, file_actions, attrp, argv, environ);
}

using namespace at::neuron;

class ProcessExecutorPipeReadTest : public ::testing::Test {};

// Helper: run ProcessExecutor::Execute in a thread with a timeout.
// Returns true if completed within timeout, false if it would have deadlocked.
bool ExecuteWithTimeout(const std::vector<std::string>& args, int timeout_seconds = 10) {
  auto future =
      std::async(std::launch::async, [&]() { ProcessExecutor::Execute(args, "pipe read test"); });
  return future.wait_for(std::chrono::seconds(timeout_seconds)) == std::future_status::ready;
}

// Helper: create a unique temporary directory, returning its absolute path.
// Uses mkdtemp so parallel test runs don't collide.
std::string MakeTempDir() {
  const char* base = std::getenv("TEST_TMPDIR");
  std::string tmpl = std::string(base ? base : "/tmp") + "/procexec_XXXXXX";
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  char* dir = mkdtemp(buf.data());
  return dir ? std::string(dir) : std::string();
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

// posix_spawn: a nonexistent executable must surface as runtime_error, from
// posix_spawnp's ENOENT or the child's non-zero exit, never a hang.
TEST_F(ProcessExecutorPipeReadTest, NonexistentExecutableThrows) {
  std::vector<std::string> args = {"this_binary_does_not_exist_procexec_test", "--version"};

  auto future = std::async(std::launch::async, [&]() {
    EXPECT_THROW(ProcessExecutor::Execute(args, "missing binary test"), std::runtime_error);
  });
  EXPECT_EQ(future.wait_for(std::chrono::seconds(10)), std::future_status::ready)
      << "ProcessExecutor hung on a nonexistent executable";
}

// posix_spawn: with a non-empty working_dir, addchdir_np must run the
// child in that directory.
TEST_F(ProcessExecutorPipeReadTest, WorkingDirRunsChildInThatDirectory) {
  std::string dir = MakeTempDir();
  ASSERT_FALSE(dir.empty()) << "failed to create temp dir";

  // Write CWD to a relative path so it lands in the child's working_dir.
  std::vector<std::string> args = {"python3", "-c",
                                   "import os; open('cwd.txt', 'w').write(os.getcwd())"};
  ASSERT_NO_THROW(ProcessExecutor::Execute(args, "workdir test", dir));

  std::ifstream f(dir + "/cwd.txt");
  ASSERT_TRUE(f.good()) << "child did not create cwd.txt in working_dir " << dir;
  std::string child_cwd((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

  // os.getcwd() resolves symlinks so compare the
  // realpath of the requested dir rather than the raw string.
  char resolved[PATH_MAX];
  ASSERT_NE(realpath(dir.c_str(), resolved), nullptr);
  EXPECT_EQ(child_cwd, std::string(resolved));

  std::remove((dir + "/cwd.txt").c_str());
  rmdir(dir.c_str());
}

// posix_spawn: an empty working_dir must skip addchdir_np entirely and
// run the child in the parent's CWD.
TEST_F(ProcessExecutorPipeReadTest, EmptyWorkingDirUsesParentCwd) {
  char parent_cwd[PATH_MAX];
  ASSERT_NE(getcwd(parent_cwd, sizeof(parent_cwd)), nullptr);

  // Child exits 0 iff its CWD matches the parent's, otherwise Execute throws.
  std::string script =
      "import os,sys; sys.exit(0 if os.getcwd()==os.environ['EXPECTED_CWD'] "
      "else 1)";
  setenv("EXPECTED_CWD", parent_cwd, 1);
  std::vector<std::string> args = {"python3", "-c", script};
  EXPECT_NO_THROW(ProcessExecutor::Execute(args, "empty workdir test", ""));
  unsetenv("EXPECTED_CWD");
}

// When the env-lock preload exports its helper, ProcessExecutor must use it so
// compiler children receive a consistent environment snapshot.
TEST_F(ProcessExecutorPipeReadTest, UsesStableEnvironSpawnHelperWhenAvailable) {
  int calls_before = stable_spawn_calls.load(std::memory_order_relaxed);
  ASSERT_NO_THROW(ProcessExecutor::Execute({"true"}, "stable environ helper test"));
  EXPECT_EQ(stable_spawn_calls.load(std::memory_order_relaxed), calls_before + 1);
}

#include <gtest/gtest.h>
#include <sys/stat.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>

#include "tests/csrc/mocks/MockNRT.h"
#include "torch_neuronx/csrc/core/compilation/service/CompilationServiceClient.h"
#include "torch_neuronx/csrc/core/compilation/service/NeuronCompilationService.h"

using namespace at::neuron;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

class NeuronCompilationServiceTest : public ::testing::Test {
 protected:
  torch_neuronx::testing::MockNRTSession mock_session_;
  torch_neuronx::testing::MockNRT* mock_nrt_ = torch_neuronx::testing::MockNRT::GetInstance();

  void SetUp() override {
    nrt_instance_info_t info{};
    info.family = NRT_INSTANCE_TRN3;
    EXPECT_CALL(*mock_nrt_, nrt_get_instance_info(_, _))
        .Times(::testing::AnyNumber())
        .WillRepeatedly(DoAll(SetArgPointee<0>(info), Return(NRT_SUCCESS)));

    socket_path_ = "/tmp/neuron_test_" + std::to_string(getpid()) + "_" +
                   std::to_string(test_counter_++) + ".sock";
    setenv("TORCH_NEURONX_NEFF_DISABLE_CACHE", "true", 1);
  }

  void TearDown() override {
    unsetenv("TORCH_NEURONX_NEFF_DISABLE_CACHE");
    unlink(socket_path_.c_str());
  }

  std::string socket_path_;
  static inline int test_counter_ = 0;
};

TEST_F(NeuronCompilationServiceTest, StartAndStop) {
  NeuronCompilationService service;
  ASSERT_NO_THROW(service.Start(socket_path_, 2));
  EXPECT_TRUE(service.IsRunning());
  service.Stop();
  EXPECT_FALSE(service.IsRunning());
}

TEST_F(NeuronCompilationServiceTest, ClientEnsureServiceRunning) {
  CompilationServiceClient client;
  EXPECT_TRUE(client.EnsureServiceRunning(socket_path_));

  // Second call is a no-op
  EXPECT_TRUE(client.EnsureServiceRunning(socket_path_));
}

TEST_F(NeuronCompilationServiceTest, PropagatesCompilerError) {
  NeuronCompilationService service;
  service.Start(socket_path_, 2);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  CompilationServiceClient client;
  ASSERT_TRUE(client.EnsureServiceRunning(socket_path_));

  // Send garbage HLO — neuronx-cc will fail with a real error message
  std::string error;
  auto result =
      client.CompileSync({0x01, 0x02, 0x03}, "XLA", "op_name", "cache_key", "-O1", "", &error);
  EXPECT_TRUE(result.empty());
  EXPECT_FALSE(error.empty()) << "Service should propagate the neuronx-cc error message";
  EXPECT_TRUE(error.find("Command failed") != std::string::npos ||
              error.find("Failed to read file") != std::string::npos)
      << "Got: " << error;
}

TEST_F(NeuronCompilationServiceTest, CompileSyncBeforeServiceReturnsEmpty) {
  CompilationServiceClient client;
  // service_ready_ is false
  auto result = client.CompileSync({0x01}, "XLA", "op_name", "cache_key", "-O1", "");
  EXPECT_TRUE(result.empty());
}

TEST_F(NeuronCompilationServiceTest, ConcurrentCompileSync) {
  NeuronCompilationService service;
  service.Start(socket_path_, 4);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  CompilationServiceClient client;
  ASSERT_TRUE(client.EnsureServiceRunning(socket_path_));

  // Multiple threads calling CompileSync concurrently — each gets own connection
  constexpr int kNumThreads = 8;
  std::atomic<int> completed{0};

  std::vector<std::thread> threads;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&]() {
      auto result = client.CompileSync({0xDE, 0xAD}, "XLA", "op_name", "cache_key", "-O1", "");
      // Compilation fails (garbage HLO) but no crash or hang
      completed++;
    });
  }

  for (auto& t : threads) t.join();
  EXPECT_EQ(completed.load(), kNumThreads);

  service.Stop();
}

TEST_F(NeuronCompilationServiceTest, LazyStartupFromMultipleClients) {
  constexpr int kNumClients = 4;
  std::atomic<int> ready{0};

  std::vector<std::thread> threads;
  for (int i = 0; i < kNumClients; ++i) {
    threads.emplace_back([&]() {
      CompilationServiceClient client;
      if (client.EnsureServiceRunning(socket_path_)) ready++;
    });
  }

  for (auto& t : threads) t.join();
  EXPECT_EQ(ready.load(), kNumClients);
}

class FailedCompilationTest : public NeuronCompilationServiceTest {
 protected:
  void SetUp() override {
    NeuronCompilationServiceTest::SetUp();
    char temp_template[] = "/tmp/neuron_failed_concat_XXXXXX";
    char* dir = mkdtemp(temp_template);
    ASSERT_NE(dir, nullptr);
    temp_dir_ = dir;
    original_path_ = getenv("PATH") ? getenv("PATH") : "";
    setenv("PATH", (temp_dir_ + ":" + original_path_).c_str(), 1);
    WriteCompiler(false);
  }

  void TearDown() override {
    setenv("PATH", original_path_.c_str(), 1);
    if (!temp_dir_.empty()) std::filesystem::remove_all(temp_dir_);
    NeuronCompilationServiceTest::TearDown();
  }

  void WriteCompiler(bool empty_neff) {
    std::string path = temp_dir_ + "/neuronx-cc";
    std::ofstream script(path);
    script << "#!/bin/bash\n"
           << "if [[ \"$1\" == '--version' ]]; then echo 'test-compiler'; exit 0; fi\n"
           << "echo compile >> '" << temp_dir_ << "/invocations'\n";
    if (empty_neff) {
      script << "while [[ $# -gt 0 ]]; do\n"
             << "  if [[ \"$1\" == '--output' ]]; then : > \"$2\"; exit 0; fi\n"
             << "  shift\n"
             << "done\n";
    }
    script << "echo 'Test compiler failure' >&2\nexit 1\n";
    script.close();
    ASSERT_EQ(chmod(path.c_str(), 0755), 0);
  }

  int CompileCount() {
    std::ifstream input(temp_dir_ + "/invocations");
    int count = 0;
    std::string line;
    while (std::getline(input, line)) ++count;
    return count;
  }

  std::string temp_dir_;
  std::string original_path_;
};

TEST_F(FailedCompilationTest, ReusesAnyCompilerErrorAcrossClients) {
  NeuronCompilationService service;
  service.Start(socket_path_, 1);
  CompilationServiceClient first;
  CompilationServiceClient second;
  ASSERT_TRUE(first.EnsureServiceRunning(socket_path_));
  ASSERT_TRUE(second.EnsureServiceRunning(socket_path_));

  std::string first_error;
  EXPECT_TRUE(first.CompileSync({1, 2}, "StableHLO", "aten::add", "first", "-O1", "", &first_error)
                  .empty());
  EXPECT_NE(first_error.find("Test compiler failure"), std::string::npos);
  EXPECT_EQ(first_error.find("[NCC_IBIR243]"), std::string::npos);
  ASSERT_EQ(CompileCount(), 1);

  std::string second_error;
  EXPECT_TRUE(
      second.CompileSync({1, 2}, "StableHLO", "aten::add", "first", "-O1", "", &second_error)
          .empty());
  EXPECT_EQ(second_error, first_error);
  EXPECT_EQ(CompileCount(), 1);
}

TEST_F(FailedCompilationTest, ReusesFailedKeyRegardlessOfOperationName) {
  NeuronCompilationService service;
  service.Start(socket_path_, 1);
  CompilationServiceClient client;
  ASSERT_TRUE(client.EnsureServiceRunning(socket_path_));

  EXPECT_TRUE(client.CompileSync({1, 2}, "StableHLO", "merged_op", "concat", "-O1", "").empty());
  ASSERT_EQ(CompileCount(), 1);
  for (int i = 0; i < 2; ++i) {
    EXPECT_TRUE(
        client.CompileSync({1, 2}, "StableHLO", "individual|op", "concat", "-O1", "").empty());
    EXPECT_EQ(CompileCount(), 1);
  }
}

TEST_F(FailedCompilationTest, SeparatesGeneratedCacheKeys) {
  NeuronCompilationService service;
  service.Start(socket_path_, 1);
  CompilationServiceClient client;
  ASSERT_TRUE(client.EnsureServiceRunning(socket_path_));

  const std::string alias_off_key = "add_key|mul_key|alias_enabled:0";
  const std::string alias_on_key = "add_key|mul_key|alias_enabled:1";
  EXPECT_TRUE(
      client.CompileSync({1, 2}, "StableHLO", "merged_op", alias_off_key, "-O1", "").empty());
  EXPECT_EQ(CompileCount(), 1);
  EXPECT_TRUE(
      client.CompileSync({1, 2}, "StableHLO", "merged_op", alias_on_key, "-O1", "").empty());
  EXPECT_EQ(CompileCount(), 2);
  EXPECT_TRUE(
      client.CompileSync({1, 2}, "StableHLO", "merged_op", alias_off_key, "-O1", "").empty());
  EXPECT_EQ(CompileCount(), 2);
}

TEST_F(FailedCompilationTest, ReusesEmptyNeffFailure) {
  WriteCompiler(true);
  NeuronCompilationService service;
  service.Start(socket_path_, 1);
  CompilationServiceClient client;
  ASSERT_TRUE(client.EnsureServiceRunning(socket_path_));

  std::string first_error;
  EXPECT_TRUE(
      client.CompileSync({1, 2}, "StableHLO", "merged_op", "key", "-O1", "", &first_error).empty());
  EXPECT_NE(first_error.find("returned empty NEFF"), std::string::npos);
  ASSERT_EQ(CompileCount(), 1);

  std::string second_error;
  EXPECT_TRUE(client.CompileSync({1, 2}, "StableHLO", "merged_op", "key", "-O1", "", &second_error)
                  .empty());
  EXPECT_EQ(second_error, first_error);
  EXPECT_EQ(CompileCount(), 1);
}

TEST_F(FailedCompilationTest, CacheHitAtCapacityPreservesOtherFailures) {
  NeuronCompilationService service;
  service.Start(socket_path_, 1);
  CompilationServiceClient client;
  ASSERT_TRUE(client.EnsureServiceRunning(socket_path_));

  constexpr int kCapacity = 1024;
  for (int i = 0; i < kCapacity; ++i) {
    ASSERT_TRUE(
        client.CompileSync({1, 2}, "StableHLO", "merged_op", "key" + std::to_string(i), "-O1", "")
            .empty());
  }
  ASSERT_EQ(CompileCount(), kCapacity);
  for (int i = 0; i < 2; ++i) {
    EXPECT_TRUE(
        client.CompileSync({1, 2}, "StableHLO", "merged_op", "key" + std::to_string(i), "-O1", "")
            .empty());
  }
  EXPECT_EQ(CompileCount(), kCapacity);
}

TEST_F(FailedCompilationTest, NewFailureAtCapacityClearsPreviousFailures) {
  NeuronCompilationService service;
  service.Start(socket_path_, 1);
  CompilationServiceClient client;
  ASSERT_TRUE(client.EnsureServiceRunning(socket_path_));

  constexpr int kCapacity = 1024;
  for (int i = 0; i <= kCapacity; ++i) {
    ASSERT_TRUE(
        client.CompileSync({1, 2}, "StableHLO", "merged_op", "key" + std::to_string(i), "-O1", "")
            .empty());
  }
  ASSERT_EQ(CompileCount(), kCapacity + 1);
  EXPECT_TRUE(client
                  .CompileSync({1, 2}, "StableHLO", "merged_op", "key" + std::to_string(kCapacity),
                               "-O1", "")
                  .empty());
  EXPECT_EQ(CompileCount(), kCapacity + 1);
  EXPECT_TRUE(client.CompileSync({1, 2}, "StableHLO", "merged_op", "key0", "-O1", "").empty());
  EXPECT_EQ(CompileCount(), kCapacity + 2);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

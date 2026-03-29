#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "torch_neuronx/csrc/core/compilation/service/CompilationServiceClient.h"
#include "torch_neuronx/csrc/core/compilation/service/NeuronCompilationService.h"

using namespace at::neuron;

class NeuronCompilationServiceTest : public ::testing::Test {
 protected:
  void SetUp() override {
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
  auto result = client.CompileSync({0x01, 0x02, 0x03}, "XLA", "-O1", "", &error);
  EXPECT_TRUE(result.empty());
  EXPECT_FALSE(error.empty()) << "Service should propagate the neuronx-cc error message";
  EXPECT_NE(error.find("Command failed"), std::string::npos) << "Got: " << error;
}

TEST_F(NeuronCompilationServiceTest, CompileSyncBeforeServiceReturnsEmpty) {
  CompilationServiceClient client;
  // service_ready_ is false
  auto result = client.CompileSync({0x01}, "XLA", "-O1", "");
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
      auto result = client.CompileSync({0xDE, 0xAD}, "XLA", "-O1", "");
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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

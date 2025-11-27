#include <gtest/gtest.h>

#include <vector>

#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"

using namespace at::neuron;
using namespace at::neuron::testing;

class KernelExecutionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create test tensors
    input_tensor_ = create_test_tensor({2, 3});
    output_tensor_ = create_test_tensor({2, 3});
    inputs_ = {input_tensor_};
    outputs_ = {output_tensor_};
  }

  torch::Tensor input_tensor_;
  torch::Tensor output_tensor_;
  std::vector<torch::Tensor> inputs_;
  std::vector<torch::Tensor> outputs_;
};

TEST_F(KernelExecutionTest, DtypeConversionNeeded) {
  // Test types that need conversion
  EXPECT_TRUE(NeuronKernelExecution::NeedsDtypeConversion(torch::kInt64));
  EXPECT_TRUE(NeuronKernelExecution::NeedsDtypeConversion(torch::kDouble));

  // Test types that don't need conversion
  EXPECT_FALSE(NeuronKernelExecution::NeedsDtypeConversion(torch::kFloat32));
  EXPECT_FALSE(NeuronKernelExecution::NeedsDtypeConversion(torch::kFloat16));
  EXPECT_FALSE(NeuronKernelExecution::NeedsDtypeConversion(torch::kBFloat16));
  EXPECT_FALSE(NeuronKernelExecution::NeedsDtypeConversion(torch::kInt32));
}

TEST_F(KernelExecutionTest, CompilableKernelConstruction) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("compilable_op", inputs_, outputs_, "cache_key_123",
                                      hlo_bytes, false);

  EXPECT_EQ(kernel.GetOpName(), "compilable_op");
  EXPECT_TRUE(kernel.RequiresCompilation());
  EXPECT_TRUE(kernel.GetCacheKey().find("cache_key_123") == 0);
  EXPECT_FALSE(kernel.HasCachedNeff());
  EXPECT_FALSE(kernel.HasCollectives());
  EXPECT_TRUE(kernel.IsValid());
}

TEST_F(KernelExecutionTest, CompilableKernelWithCollectives) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("collective_op", inputs_, outputs_, "cache_key", hlo_bytes,
                                      true);

  EXPECT_TRUE(kernel.HasCollectives());
  EXPECT_TRUE(kernel.IsValid());
}

TEST_F(KernelExecutionTest, CompilableKernelCompilation) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("compilable_op", inputs_, outputs_, "cache_key", hlo_bytes,
                                      false);

  // Test compilation - will throw since we don't have real compiler
  EXPECT_THROW(kernel.CompileToNeff(), std::exception);
}

TEST_F(KernelExecutionTest, CompilableKernelValidation) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};

  // Test with empty cache key (should be invalid)
  XLACompilableKernelExecution kernel1("op", inputs_, outputs_, "", hlo_bytes, false);
  EXPECT_FALSE(kernel1.IsValid());

  // Test with empty op name (should be invalid)
  XLACompilableKernelExecution kernel2("", inputs_, outputs_, "cache_key", hlo_bytes, false);
  EXPECT_FALSE(kernel2.IsValid());
}

// DirectKernelExecution is abstract - tested through CollectiveDirectKernelExecution below

TEST_F(KernelExecutionTest, XLAKernelConstruction) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};  // "HLO"

  XLACompilableKernelExecution kernel("xla_op", inputs_, outputs_, "xla_cache_key", hlo_bytes,
                                      false);

  EXPECT_EQ(kernel.GetOpName(), "xla_op");
  EXPECT_TRUE(kernel.RequiresCompilation());
  EXPECT_TRUE(kernel.GetCacheKey().find("xla_cache_key") == 0);
  EXPECT_EQ(kernel.GetHloBytes(), hlo_bytes);
  EXPECT_FALSE(kernel.HasCollectives());
  EXPECT_TRUE(kernel.IsValid());
}

TEST_F(KernelExecutionTest, XLAKernelWithEmptyHLO) {
  std::vector<uint8_t> empty_hlo;

  XLACompilableKernelExecution kernel("xla_op", inputs_, outputs_, "cache_key", empty_hlo, false);

  EXPECT_FALSE(kernel.IsValid());  // Should be invalid with empty HLO
}

TEST_F(KernelExecutionTest, XLAKernelCompilation) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F, 0x01, 0x02};

  XLACompilableKernelExecution kernel("xla_op", inputs_, outputs_, "cache_key", hlo_bytes, false);

  // TODO(rpsilva): Integrate or mock the compiler.
  EXPECT_THROW(kernel.CompileToNeff(), std::exception);
}

TEST_F(KernelExecutionTest, NKIKernelConstruction) {
  NKICompilableKernelExecution kernel("nki_op", "test_kernel", "/path/to/module.py", inputs_,
                                      outputs_, "nki_cache_key", false);

  EXPECT_EQ(kernel.GetOpName(), "nki_op");
  EXPECT_TRUE(kernel.RequiresCompilation());
  EXPECT_TRUE(kernel.GetCacheKey().find("nki_cache_key") == 0);
  EXPECT_EQ(kernel.GetKernelName(), "test_kernel");
  EXPECT_EQ(kernel.GetModulePath(), "/path/to/module.py");
  EXPECT_FALSE(kernel.HasCollectives());
  EXPECT_TRUE(kernel.IsValid());
}

TEST_F(KernelExecutionTest, NKIKernelWithEmptyKernelName) {
  NKICompilableKernelExecution kernel("nki_op", "", "/path/to/module.py", inputs_, outputs_,
                                      "cache_key", false);

  EXPECT_FALSE(kernel.IsValid());  // Should be invalid with empty kernel name
}

TEST_F(KernelExecutionTest, NKIKernelWithEmptyModulePath) {
  NKICompilableKernelExecution kernel("nki_op", "test_kernel", "", inputs_, outputs_, "cache_key",
                                      false);

  EXPECT_FALSE(kernel.IsValid());  // Should be invalid with empty module path
}

TEST_F(KernelExecutionTest, NKIKernelCompilation) {
  NKICompilableKernelExecution kernel("nki_op", "test_kernel", "/path/to/module.py", inputs_,
                                      outputs_, "cache_key", false);

  // Test compilation - this will throw since NKI compilation is not implemented yet
  EXPECT_THROW(kernel.CompileToNeff(), std::logic_error);
}

TEST_F(KernelExecutionTest, CollectiveKernelConstruction) {
  CollectiveDirectKernelExecution kernel(
      "allreduce_op", CollectiveDirectKernelExecution::CollectiveType::kAllReduce, inputs_,
      outputs_, c10d::ReduceOp::SUM);

  EXPECT_EQ(kernel.GetOpName(), "allreduce_op");
  EXPECT_FALSE(kernel.RequiresCompilation());
  EXPECT_EQ(kernel.GetKernelType(), DirectKernelExecution::KernelType::kCollective);
  EXPECT_EQ(kernel.GetCollectiveType(),
            CollectiveDirectKernelExecution::CollectiveType::kAllReduce);
  EXPECT_EQ(kernel.GetReduceOp(), c10d::ReduceOp::SUM);
  EXPECT_TRUE(kernel.IsValid());
}

TEST_F(KernelExecutionTest, CollectiveKernelAllTypes) {
  // Test all collective types
  std::vector<CollectiveDirectKernelExecution::CollectiveType> types = {
      CollectiveDirectKernelExecution::CollectiveType::kAllGather,
      CollectiveDirectKernelExecution::CollectiveType::kAllReduce,
      CollectiveDirectKernelExecution::CollectiveType::kReduceScatter,
      CollectiveDirectKernelExecution::CollectiveType::kAllToAll,
      CollectiveDirectKernelExecution::CollectiveType::kBroadcast,
      CollectiveDirectKernelExecution::CollectiveType::kReduce};

  for (auto type : types) {
    CollectiveDirectKernelExecution kernel("collective", type, inputs_, outputs_,
                                           c10d::ReduceOp::SUM);
    EXPECT_EQ(kernel.GetCollectiveType(), type);
    EXPECT_TRUE(kernel.IsValid());
  }
}

TEST_F(KernelExecutionTest, CollectiveKernelReduceOps) {
  // Test different reduce operations
  std::vector<c10d::ReduceOp> reduce_ops = {
      c10d::ReduceOp::SUM,  c10d::ReduceOp::PRODUCT, c10d::ReduceOp::MIN,  c10d::ReduceOp::MAX,
      c10d::ReduceOp::BAND, c10d::ReduceOp::BOR,     c10d::ReduceOp::BXOR, c10d::ReduceOp::AVG};

  for (auto reduce_op : reduce_ops) {
    CollectiveDirectKernelExecution kernel(
        "allreduce", CollectiveDirectKernelExecution::CollectiveType::kAllReduce, inputs_, outputs_,
        reduce_op);
    EXPECT_EQ(kernel.GetReduceOp(), reduce_op);
    EXPECT_TRUE(kernel.IsValid());
  }
}

TEST_F(KernelExecutionTest, PolymorphicBehavior) {
  // Create different kernel types and verify polymorphic behavior through base pointer
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};

  std::vector<std::unique_ptr<NeuronKernelExecution>> kernels;
  kernels.push_back(std::make_unique<XLACompilableKernelExecution>("xla", inputs_, outputs_, "key",
                                                                   hlo_bytes, false));
  kernels.push_back(std::make_unique<NKICompilableKernelExecution>(
      "nki", "kernel", "/path", inputs_, outputs_, "key", false));
  kernels.push_back(std::make_unique<CollectiveDirectKernelExecution>(
      "collective", CollectiveDirectKernelExecution::CollectiveType::kAllReduce, inputs_, outputs_,
      c10d::ReduceOp::SUM));

  // Verify polymorphic behavior
  EXPECT_TRUE(kernels[0]->RequiresCompilation());   // XLA kernel
  EXPECT_TRUE(kernels[1]->RequiresCompilation());   // NKI kernel
  EXPECT_FALSE(kernels[2]->RequiresCompilation());  // Collective kernel

  for (const auto& kernel : kernels) {
    EXPECT_TRUE(kernel->IsValid());
    EXPECT_FALSE(kernel->GetOpName().empty());
  }
}

TEST_F(KernelExecutionTest, CacheKeyIncludesCompilerExtension) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};

  XLACompilableKernelExecution kernel("xla_op", inputs_, outputs_, "base_key", hlo_bytes, false);

  // Cache key should start with base key
  std::string cache_key = kernel.GetCacheKey();
  EXPECT_TRUE(cache_key.find("base_key") == 0);

  // Cache key should be longer than base key (includes extension)
  EXPECT_GT(cache_key.length(), std::string("base_key").length());

  // Cache key should contain platform-specific information
  // Format: base_key_<platform>_<cores>_<opt_level>_<args>
  EXPECT_TRUE(cache_key.find("_") != std::string::npos);
}

TEST_F(KernelExecutionTest, CompilableKernelAccessors) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};

  XLACompilableKernelExecution kernel("xla_op", inputs_, outputs_, "key", hlo_bytes, false);

  // Test that we can access compilation configuration
  EXPECT_FALSE(kernel.GetAdditionalArgs().empty());
  EXPECT_FALSE(kernel.GetOptimizationLevel().empty());

  // XLA should have default args
  EXPECT_TRUE(kernel.GetAdditionalArgs().find("--model-type") != std::string::npos);
  EXPECT_TRUE(kernel.GetAdditionalArgs().find("--auto-cast=none") != std::string::npos);

  // Default optimization level should be -O1
  EXPECT_EQ(kernel.GetOptimizationLevel(), "-O1");
}

TEST_F(KernelExecutionTest, XLAEnvironmentVariableSupport) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};

  // Set environment variables
  setenv("NEURON_CC_FLAGS", "--custom-flag=value", 1);
  setenv("NEURON_COMPILER_OPT_LEVEL", "-O2", 1);

  XLACompilableKernelExecution kernel("xla_op", inputs_, outputs_, "key", hlo_bytes, false);

  // Check that environment variables are reflected
  EXPECT_TRUE(kernel.GetAdditionalArgs().find("--custom-flag=value") != std::string::npos);
  EXPECT_EQ(kernel.GetOptimizationLevel(), "-O2");

  // Cache key should reflect the custom settings
  std::string cache_key = kernel.GetCacheKey();
  EXPECT_TRUE(cache_key.find("O2") != std::string::npos);

  // Clean up
  unsetenv("NEURON_CC_FLAGS");
  unsetenv("NEURON_COMPILER_OPT_LEVEL");
}

TEST_F(KernelExecutionTest, NKIEnvironmentVariableSupport) {
  // Set environment variables
  setenv("NEURON_NKI_COMPILE_FLAGS", "--nki-custom-flag", 1);
  setenv("NEURON_COMPILER_OPT_LEVEL", "-O3", 1);

  NKICompilableKernelExecution kernel("nki_op", "kernel", "/path", inputs_, outputs_, "key", false);

  // Check that environment variables are reflected
  EXPECT_TRUE(kernel.GetAdditionalArgs().find("--nki-custom-flag") != std::string::npos);
  EXPECT_EQ(kernel.GetOptimizationLevel(), "-O3");

  // Clean up
  unsetenv("NEURON_NKI_COMPILE_FLAGS");
  unsetenv("NEURON_COMPILER_OPT_LEVEL");
}

TEST_F(KernelExecutionTest, CacheKeyUniquenessWithDifferentArgs) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};

  // Create two kernels with same base key but different environment settings
  setenv("NEURON_COMPILER_OPT_LEVEL", "-O1", 1);
  XLACompilableKernelExecution kernel1("op", inputs_, outputs_, "same_base", hlo_bytes, false);
  std::string key1 = kernel1.GetCacheKey();

  setenv("NEURON_COMPILER_OPT_LEVEL", "-O2", 1);
  XLACompilableKernelExecution kernel2("op", inputs_, outputs_, "same_base", hlo_bytes, false);
  std::string key2 = kernel2.GetCacheKey();

  // Keys should be different due to different optimization levels
  EXPECT_NE(key1, key2);

  // Both should start with the same base
  EXPECT_TRUE(key1.find("same_base") == 0);
  EXPECT_TRUE(key2.find("same_base") == 0);

  unsetenv("NEURON_COMPILER_OPT_LEVEL");
}

TEST_F(KernelExecutionTest, CachedNeffManagement) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("op", inputs_, outputs_, "key", hlo_bytes, false);

  // Initially no cached NEFF
  EXPECT_FALSE(kernel.HasCachedNeff());

  // Set a cached NEFF - create a NeffBytesPtr (unique_ptr with CacheEntryGuard deleter)
  auto neff_data = new std::vector<uint8_t>{0x4E, 0x45, 0x46, 0x46};  // "NEFF"
  NeffBytesPtr neff_ptr(neff_data, CacheEntryGuard{});

  // Store expected data before moving the pointer
  std::vector<uint8_t> expected_data = *neff_data;

  kernel.SetCachedNeff(std::move(neff_ptr));

  // Now should have cached NEFF
  EXPECT_TRUE(kernel.HasCachedNeff());
  EXPECT_EQ(kernel.GetCachedNeff(), expected_data);
}

TEST_F(KernelExecutionTest, XLACompilationConfigPropagation) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};

  setenv("NEURON_COMPILER_OPT_LEVEL", "-O2", 1);
  setenv("NEURON_CC_FLAGS", "--test-flag", 1);

  XLACompilableKernelExecution kernel("op", inputs_, outputs_, "key", hlo_bytes, false);

  // Verify that compilation config would include these settings
  EXPECT_EQ(kernel.GetOptimizationLevel(), "-O2");
  EXPECT_TRUE(kernel.GetAdditionalArgs().find("--test-flag") != std::string::npos);

  unsetenv("NEURON_COMPILER_OPT_LEVEL");
  unsetenv("NEURON_CC_FLAGS");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

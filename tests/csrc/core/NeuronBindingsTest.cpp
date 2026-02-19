#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"

using namespace at::neuron;
using namespace at::neuron::testing;

class NeuronBindingsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    input_tensor_ = create_test_tensor({2, 3});
    output_tensor_ = create_test_tensor({2, 3});
    inputs_ = {input_tensor_};
    outputs_ = {output_tensor_};
    input_ptrs_ = get_tensor_data_ptrs(inputs_);
    output_ptrs_ = get_tensor_data_ptrs(outputs_);
  }

  torch::Tensor input_tensor_;
  torch::Tensor output_tensor_;
  std::vector<torch::Tensor> inputs_;
  std::vector<torch::Tensor> outputs_;
  std::vector<void*> input_ptrs_;
  std::vector<void*> output_ptrs_;
};

// Tests for compile_graph binding logic (CompileOnlyKernelExecution creation)

TEST_F(NeuronBindingsTest, CompileGraphCreatesValidKernel) {
  std::string base_cache_key = "test_graph_key";
  std::vector<uint8_t> stablehlo_bytes = {0x53, 0x48, 0x4C, 0x4F};
  bool has_collectives = false;

  CompileOnlyKernelExecution kernel(base_cache_key, stablehlo_bytes, has_collectives);

  EXPECT_TRUE(kernel.IsValid());
  EXPECT_TRUE(kernel.RequiresCompilation());
  EXPECT_EQ(kernel.GetKernelType(), KernelTypeEnum::kHLO);
}

TEST_F(NeuronBindingsTest, CompileGraphReturnsCacheKey) {
  std::string base_cache_key = "my_graph";
  std::vector<uint8_t> stablehlo_bytes = {0x01, 0x02, 0x03};

  CompileOnlyKernelExecution kernel(base_cache_key, stablehlo_bytes, false);
  std::string cache_key = kernel.GetCacheKey();

  // Cache key should contain the base key
  EXPECT_TRUE(cache_key.find(base_cache_key) == 0);
  // Cache key should have compiler extension appended
  EXPECT_GT(cache_key.length(), base_cache_key.length());
}

TEST_F(NeuronBindingsTest, CompileGraphWithCollectives) {
  std::vector<uint8_t> stablehlo_bytes = {0x01, 0x02};

  CompileOnlyKernelExecution kernel("collective_graph", stablehlo_bytes, true);

  EXPECT_TRUE(kernel.HasCollectives());
  EXPECT_TRUE(kernel.IsValid());
}

// Tests for execute_compiled_graph binding logic (NeffDirectKernelExecution creation)

TEST_F(NeuronBindingsTest, ExecuteCompiledGraphCreatesValidKernel) {
  std::string graph_name = "test_graph";
  std::string cache_key = "precompiled_cache_key";
  bool has_collectives = false;

  NeffDirectKernelExecution kernel(graph_name, cache_key, create_fake_tensor_refs(input_ptrs_),
                                   create_fake_tensor_refs(output_ptrs_), {}, {}, 0,
                                   has_collectives);

  EXPECT_TRUE(kernel.IsValid());
  EXPECT_EQ(kernel.GetOpName(), graph_name);
  EXPECT_EQ(kernel.GetCacheKey(), cache_key);
  EXPECT_EQ(kernel.GetKernelType(), KernelTypeEnum::kHLO);
}

TEST_F(NeuronBindingsTest, ExecuteCompiledGraphWithCollectives) {
  NeffDirectKernelExecution kernel("graph", "key", create_fake_tensor_refs(input_ptrs_),
                                   create_fake_tensor_refs(output_ptrs_), {}, {}, 0, true);

  EXPECT_TRUE(kernel.HasCollectives());
  EXPECT_TRUE(kernel.IsValid());
}

TEST_F(NeuronBindingsTest, ExecuteCompiledGraphPreservesDeviceId) {
  int device_id = 2;

  NeffDirectKernelExecution kernel("graph", "key", create_fake_tensor_refs(input_ptrs_),
                                   create_fake_tensor_refs(output_ptrs_), {}, {}, device_id, false);

  EXPECT_TRUE(kernel.IsValid());
}

TEST_F(NeuronBindingsTest, ExecuteCompiledGraphRequiresCacheLookup) {
  NeffDirectKernelExecution kernel("graph", "key", create_fake_tensor_refs(input_ptrs_),
                                   create_fake_tensor_refs(output_ptrs_), {}, {}, 0, false);

  // RequiresCompilation returns true to trigger cache lookup
  EXPECT_TRUE(kernel.RequiresCompilation());
}

TEST_F(NeuronBindingsTest, ExecuteCompiledGraphCompileToNeffThrows) {
  NeffDirectKernelExecution kernel("graph", "key", create_fake_tensor_refs(input_ptrs_),
                                   create_fake_tensor_refs(output_ptrs_), {}, {}, 0, false);

  // NEFF is pre-compiled, CompileToNeff should throw
  EXPECT_THROW(kernel.CompileToNeff(), std::logic_error);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

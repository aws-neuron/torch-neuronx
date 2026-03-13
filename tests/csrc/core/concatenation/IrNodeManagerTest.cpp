#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>

#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/concatenation/IrNode.h"
#include "torch_neuronx/csrc/core/concatenation/IrNodeManager.h"

using namespace at::neuron;
using namespace torch_neuronx;

class IrNodeManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    manager = std::make_unique<IrNodeManager>();

    // Clear environment variable for consistent testing
    unsetenv("TORCH_NEURONX_SKIP_INTERMEDIATES");
  }

  void TearDown() override {
    // Clean up environment
    unsetenv("TORCH_NEURONX_SKIP_INTERMEDIATES");
  }

  std::unique_ptr<IrNodeManager> manager;

  std::vector<uint8_t> createDummyIr() {
    std::string mlir = R"(
module @jit__lambda2 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.multiply %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";
    return std::vector<uint8_t>(mlir.begin(), mlir.end());
  }

  void* createPtr() {
    static int counter = 0;
    return reinterpret_cast<void*>(++counter);
  }

  std::unique_ptr<StableHloNode> createStableHloNode(const std::string& op_name = "test_op",
                                                     const std::string& cache_key = "") {
    auto ir = createDummyIr();
    std::string key = cache_key.empty() ? "cache_key_" + std::to_string(rand()) : cache_key;
    return std::make_unique<StableHloNode>(
        op_name, key, std::move(ir), std::vector<void*>{createPtr()},
        std::vector<void*>{createPtr()}, false, manager->GetContext());
  }
};

TEST_F(IrNodeManagerTest, CreateConcatNodeNullInputs) {
  auto node1 = createStableHloNode();
  EXPECT_THROW(manager->CreateConcatNode(node1.get(), nullptr), std::invalid_argument);

  EXPECT_THROW(manager->CreateConcatNode(nullptr, node1.get()), std::invalid_argument);
}

TEST_F(IrNodeManagerTest, CreateConcatNodeNonStableHLO) {
  auto ir = createDummyIr();
  std::vector<void*> inputs = {createPtr()};
  std::vector<void*> outputs = {createPtr()};

  // Create non-StableHLO nodes
  IrNode hlo_node1("op1", "key1", ir, IrNodeType::HLO, std::vector<void*>(inputs),
                   std::vector<void*>(outputs), false);
  IrNode hlo_node2("op2", "key2", ir, IrNodeType::HLO, std::vector<void*>(inputs),
                   std::vector<void*>(outputs), false);

  EXPECT_THROW(manager->CreateConcatNode(&hlo_node1, &hlo_node2), std::runtime_error);
}

TEST_F(IrNodeManagerTest, CreateConcatNodeCacheHit) {
  auto node1 = createStableHloNode("op1", "key1");
  auto node2 = createStableHloNode("op2", "key2");

  // First call should create and cache
  auto result1 = manager->CreateConcatNode(node1.get(), node2.get());

  if (result1) {
    // Verify optimized module has only 1 main function (submains are inlined)
    std::string ir_string(result1->ir_serialized.begin(), result1->ir_serialized.end());
    EXPECT_TRUE(ir_string.find("main") != std::string::npos);
    // After optimization/inlining, submain functions should be removed
    EXPECT_TRUE(ir_string.find("submain1") == std::string::npos);
    EXPECT_TRUE(ir_string.find("submain2") == std::string::npos);

    // Verify combined input/output counts
    EXPECT_EQ(result1->inputs.size(), node1->inputs.size() + node2->inputs.size());
    EXPECT_EQ(result1->outputs.size(), node1->outputs.size() + node2->outputs.size());
  }
}

TEST_F(IrNodeManagerTest, CreateConcatNodeSkipIntermediatesDisabled) {
  // Ensure environment variable is not set
  unsetenv("TORCH_NEURONX_SKIP_INTERMEDIATES");

  auto node1 = createStableHloNode("op1", "key1");
  auto node2 = createStableHloNode("op2", "key2");

  // This should use the conservative merge function
  auto result = manager->CreateConcatNode(node1.get(), node2.get());

  if (result) {
    // Create union sets for inputs and outputs
    std::set<void*> expected_inputs(node1->inputs.begin(), node1->inputs.end());
    expected_inputs.insert(node2->inputs.begin(), node2->inputs.end());

    std::set<void*> expected_outputs(node1->outputs.begin(), node1->outputs.end());
    expected_outputs.insert(node2->outputs.begin(), node2->outputs.end());

    std::set<void*> actual_inputs(result->inputs.begin(), result->inputs.end());
    std::set<void*> actual_outputs(result->outputs.begin(), result->outputs.end());

    EXPECT_EQ(actual_inputs, expected_inputs);
    EXPECT_EQ(actual_outputs, expected_outputs);
  }
}

TEST_F(IrNodeManagerTest, CreateConcatNodeSkipIntermediatesEnabled) {
  // Set environment variable to enable skip intermediates
  setenv("TORCH_NEURONX_SKIP_INTERMEDIATES", "1", 1);

  auto node1 = createStableHloNode("op1", "key1");
  auto node2 = createStableHloNode("op2", "key2");

  // This should use the unsafe merge function
  auto result = manager->CreateConcatNode(node1.get(), node2.get());

  if (result) {
    // Verify optimized module has only 1 main function (submains are inlined)
    std::string ir_string(result->ir_serialized.begin(), result->ir_serialized.end());
    EXPECT_TRUE(ir_string.find("main") != std::string::npos);
    // After optimization/inlining, submain functions should be removed
    EXPECT_TRUE(ir_string.find("submain1") == std::string::npos);
    EXPECT_TRUE(ir_string.find("submain2") == std::string::npos);

    // Verify combined input/output counts
    EXPECT_EQ(result->inputs.size(), node1->inputs.size() + node2->inputs.size());
    EXPECT_EQ(result->outputs.size(), node1->outputs.size() + node2->outputs.size());
  }
}

TEST_F(IrNodeManagerTest, GenerateConcatenationCacheKey) {
  auto node1 = createStableHloNode("op1", "key1");
  auto node2 = createStableHloNode("op2", "key2");

  std::string cache_key = manager->GenerateConcatenationCacheKey(node1.get(), node2.get());

  EXPECT_EQ(cache_key, "key1|key2|alias:");
}

TEST_F(IrNodeManagerTest, GenerateConcatenationCacheKeyNullNodes) {
  auto node1 = createStableHloNode("op1", "key1");

  EXPECT_THROW(manager->GenerateConcatenationCacheKey(nullptr, node1.get()), std::invalid_argument);

  EXPECT_THROW(manager->GenerateConcatenationCacheKey(node1.get(), nullptr), std::invalid_argument);

  EXPECT_THROW(manager->GenerateConcatenationCacheKey(nullptr, nullptr), std::invalid_argument);
}

TEST_F(IrNodeManagerTest, GenerateConcatenationOpName) {
  auto node1 = createStableHloNode("op1", "key1");
  auto node2 = createStableHloNode("op2", "key2");

  std::string op_name = manager->GenerateConcatenationOpName(node1.get(), node2.get());

  EXPECT_EQ(op_name, "op1|op2");
}

TEST_F(IrNodeManagerTest, GenerateConcatenationOpNameNullNodes) {
  auto node1 = createStableHloNode("op1", "key1");

  EXPECT_THROW(manager->GenerateConcatenationOpName(nullptr, node1.get()), std::invalid_argument);

  EXPECT_THROW(manager->GenerateConcatenationOpName(node1.get(), nullptr), std::invalid_argument);
}

TEST_F(IrNodeManagerTest, CreateNodeFromOperationContextNull) {
  EXPECT_THROW(manager->CreateNodeFromOperationContext(nullptr), std::invalid_argument);
}

TEST_F(IrNodeManagerTest, GetByKeyEmptyKey) {
  auto result = manager->GetByKey("");
  EXPECT_EQ(result, nullptr);
}

TEST_F(IrNodeManagerTest, GetByKeyNonExistentKey) {
  auto result = manager->GetByKey("non_existent_key");
  EXPECT_EQ(result, nullptr);
}

TEST_F(IrNodeManagerTest, GetByKeyExistingKey) {
  auto node1 = createStableHloNode("op1", "existing_key");
  auto node2 = createStableHloNode("op2", "other_key");

  // Create a concat node to populate cache
  auto concat_result = manager->CreateConcatNode(node1.get(), node2.get());

  if (concat_result) {
    // Get the cached node by its key
    auto retrieved = manager->GetByKey(concat_result->cache_key);
    EXPECT_EQ(retrieved->module.get(), concat_result->module.get());
  }
}

TEST_F(IrNodeManagerTest, EvictNonExistentKey) {
  size_t evicted = manager->Evict("non_existent_key");
  EXPECT_EQ(evicted, 0);
}

TEST_F(IrNodeManagerTest, EvictExistingKey) {
  auto node1 = createStableHloNode("op1", "key1");
  auto node2 = createStableHloNode("op2", "key2");

  // Create a concat node to populate cache
  auto concat_result = manager->CreateConcatNode(node1.get(), node2.get());

  if (concat_result) {
    // Verify node exists
    auto retrieved = manager->GetByKey(concat_result->cache_key);
    EXPECT_EQ(retrieved->module.get(), concat_result->module.get());

    // Evict the node
    size_t evicted = manager->Evict(concat_result->cache_key);
    EXPECT_EQ(evicted, 1);

    // Verify node no longer exists
    auto retrieved_after = manager->GetByKey(concat_result->cache_key);
    EXPECT_EQ(retrieved_after, nullptr);
  }
}

TEST_F(IrNodeManagerTest, MultipleConcatOperations) {
  auto node1 = createStableHloNode("MultipleConcatOperations-op1", "key1");
  auto node2 = createStableHloNode("MultipleConcatOperations-op2", "key2");
  auto node3 = createStableHloNode("MultipleConcatOperations-op3", "key3");
  auto node4 = createStableHloNode("MultipleConcatOperations-op4", "key4");

  // Create multiple concat operations
  auto concat1 = manager->CreateConcatNode(node1.get(), node2.get());
  auto concat2 = manager->CreateConcatNode(node3.get(), node4.get());

  if (!concat1 || !concat2) {
    FAIL() << "Concatenation failed";
  }

  // Results should not be different (if both succeed)
  EXPECT_EQ(concat1->ir_serialized, concat2->ir_serialized);
}

TEST_F(IrNodeManagerTest, CacheAccess) {
  // Test that cache operations are consistent
  auto node1 = createStableHloNode("op1", "key1");
  auto node2 = createStableHloNode("op2", "key2");

  // Multiple calls should return same cached result
  auto result1 = manager->CreateConcatNode(node1.get(), node2.get());
  auto result2 = manager->CreateConcatNode(node1.get(), node2.get());

  if (result1) {
    EXPECT_EQ(result1->module.get(), result2->module.get());
  }
}

TEST_F(IrNodeManagerTest, ConcatAllEmptyList) {
  std::list<IrNode*> empty_list;
  auto result = manager->ConcatAll(empty_list);

  EXPECT_TRUE(result.compilable_irs.empty());
  EXPECT_TRUE(result.remain_irs.empty());
  EXPECT_TRUE(result.concat_ir_list.empty());
}

TEST_F(IrNodeManagerTest, ConcatAllSingleNode) {
  auto node = createStableHloNode("single_op", "single_key");
  std::list<IrNode*> single_list = {node.get()};

  auto result = manager->ConcatAll(single_list);

  EXPECT_EQ(result.compilable_irs.size(), 1);
  EXPECT_EQ(result.compilable_irs.front(), node.get());
  EXPECT_TRUE(result.remain_irs.empty());
  EXPECT_TRUE(result.concat_ir_list.empty());
}

TEST_F(IrNodeManagerTest, ConcatAllTwoNodes) {
  auto node1 = createStableHloNode("op1", "key1");
  auto node2 = createStableHloNode("op2", "key2");
  std::list<IrNode*> two_nodes = {node1.get(), node2.get()};

  auto result = manager->ConcatAll(two_nodes);

  EXPECT_EQ(result.compilable_irs.size(), 1);
  EXPECT_TRUE(result.remain_irs.empty());
  EXPECT_EQ(result.concat_ir_list.size(), 1);
}

TEST_F(IrNodeManagerTest, ConcatAllMultipleNodes) {
  auto node1 = createStableHloNode("op1", "key1");
  auto node2 = createStableHloNode("op2", "key2");
  auto node3 = createStableHloNode("op3", "key3");
  auto node4 = createStableHloNode("op4", "key4");
  std::list<IrNode*> four_nodes = {node1.get(), node2.get(), node3.get(), node4.get()};

  auto result = manager->ConcatAll(four_nodes);

  EXPECT_EQ(result.compilable_irs.size(), 1);
  EXPECT_TRUE(result.remain_irs.empty());
  EXPECT_EQ(result.concat_ir_list.size(), 1);
}

TEST_F(IrNodeManagerTest, ConcatAllWithInvalidNode) {
  // Create valid StableHLO nodes
  auto node1 = createStableHloNode("op1", "key1");
  auto node2 = createStableHloNode("op2", "key2");

  // Create an invalid non-StableHLO node
  auto ir = createDummyIr();
  std::vector<void*> inputs = {createPtr()};
  std::vector<void*> outputs = {createPtr()};
  IrNode invalid_node("invalid_op", "invalid_key", ir, IrNodeType::HLO, std::vector<void*>(inputs),
                      std::vector<void*>(outputs), false);

  // Create list with mix of valid and invalid nodes
  std::list<IrNode*> mixed_nodes = {node1.get(), &invalid_node, node2.get()};

  // ConcatAll should throw when encountering non-StableHLO nodes
  EXPECT_THROW(manager->ConcatAll(mixed_nodes), std::runtime_error);
}

// ============================================================================
// Aliasing Tests
// ============================================================================

TEST_F(IrNodeManagerTest, CacheKeyNoAliasing) {
  auto ir = createDummyIr();
  void* in1 = createPtr();
  void* in2 = createPtr();
  void* out1 = createPtr();
  void* out2 = createPtr();

  auto node1 = std::make_unique<StableHloNode>("op1", "key1", std::vector<uint8_t>(ir),
                                               std::vector<void*>{in1}, std::vector<void*>{out1},
                                               false, manager->GetContext());
  auto node2 = std::make_unique<StableHloNode>("op2", "key2", std::vector<uint8_t>(ir),
                                               std::vector<void*>{in2}, std::vector<void*>{out2},
                                               false, manager->GetContext());

  std::string cache_key = manager->GenerateConcatenationCacheKey(node1.get(), node2.get());
  EXPECT_EQ(cache_key, "key1|key2|alias:");
}

TEST_F(IrNodeManagerTest, CacheKeyNode1SelfAliasing) {
  auto ir = createDummyIr();
  void* shared = createPtr();
  void* in2 = createPtr();
  void* out2 = createPtr();

  auto node1 = std::make_unique<StableHloNode>(
      "op1", "key1", std::vector<uint8_t>(ir), std::vector<void*>{shared},
      std::vector<void*>{shared}, false, manager->GetContext());
  auto node2 = std::make_unique<StableHloNode>("op2", "key2", std::vector<uint8_t>(ir),
                                               std::vector<void*>{in2}, std::vector<void*>{out2},
                                               false, manager->GetContext());

  std::string cache_key = manager->GenerateConcatenationCacheKey(node1.get(), node2.get());
  EXPECT_EQ(cache_key, "key1|key2|alias:1i0->1o0,");
}

TEST_F(IrNodeManagerTest, CacheKeyNode2SelfAliasing) {
  auto ir = createDummyIr();
  void* in1 = createPtr();
  void* out1 = createPtr();
  void* shared = createPtr();

  auto node1 = std::make_unique<StableHloNode>("op1", "key1", std::vector<uint8_t>(ir),
                                               std::vector<void*>{in1}, std::vector<void*>{out1},
                                               false, manager->GetContext());
  auto node2 = std::make_unique<StableHloNode>(
      "op2", "key2", std::vector<uint8_t>(ir), std::vector<void*>{shared},
      std::vector<void*>{shared}, false, manager->GetContext());

  std::string cache_key = manager->GenerateConcatenationCacheKey(node1.get(), node2.get());
  EXPECT_EQ(cache_key, "key1|key2|alias:2i0->2o0,");
}

TEST_F(IrNodeManagerTest, CacheKeyCrossNodeAliasing) {
  auto ir = createDummyIr();
  void* shared = createPtr();
  void* out1 = createPtr();
  void* in2 = createPtr();

  auto node1 = std::make_unique<StableHloNode>("op1", "key1", std::vector<uint8_t>(ir),
                                               std::vector<void*>{shared}, std::vector<void*>{out1},
                                               false, manager->GetContext());
  auto node2 = std::make_unique<StableHloNode>("op2", "key2", std::vector<uint8_t>(ir),
                                               std::vector<void*>{in2}, std::vector<void*>{shared},
                                               false, manager->GetContext());

  std::string cache_key = manager->GenerateConcatenationCacheKey(node1.get(), node2.get());
  EXPECT_EQ(cache_key, "key1|key2|alias:1i0->2o0,");
}

TEST_F(IrNodeManagerTest, CacheKeyMultipleAliasing) {
  auto ir = createDummyIr();
  void* shared1 = createPtr();
  void* shared2 = createPtr();
  void* shared3 = createPtr();

  auto node1 = std::make_unique<StableHloNode>(
      "op1", "key1", std::vector<uint8_t>(ir), std::vector<void*>{shared1, shared2},
      std::vector<void*>{shared1, shared3}, false, manager->GetContext());
  auto node2 = std::make_unique<StableHloNode>(
      "op2", "key2", std::vector<uint8_t>(ir), std::vector<void*>{shared2, shared3},
      std::vector<void*>{shared2, shared3}, false, manager->GetContext());

  std::string cache_key = manager->GenerateConcatenationCacheKey(node1.get(), node2.get());
  EXPECT_EQ(cache_key, "key1|key2|alias:1i0->1o0,2i0->2o0,2i1->2o1,1i1->2o0,");
}

TEST_F(IrNodeManagerTest, CacheKeyNullPointers) {
  auto ir = createDummyIr();
  void* shared = createPtr();

  auto node1 = std::make_unique<StableHloNode>(
      "op1", "key1", std::vector<uint8_t>(ir), std::vector<void*>{shared, nullptr},
      std::vector<void*>{shared, nullptr}, false, manager->GetContext());
  auto node2 = std::make_unique<StableHloNode>(
      "op2", "key2", std::vector<uint8_t>(ir), std::vector<void*>{nullptr},
      std::vector<void*>{nullptr}, false, manager->GetContext());

  std::string cache_key = manager->GenerateConcatenationCacheKey(node1.get(), node2.get());
  EXPECT_EQ(cache_key, "key1|key2|alias:1i0->1o0,");
}

TEST_F(IrNodeManagerTest, CacheKeyDifferentAliasingPatterns) {
  auto ir = createDummyIr();
  void* shared = createPtr();
  void* in1 = createPtr();
  void* out1 = createPtr();
  void* in2 = createPtr();
  void* out2 = createPtr();

  // Pattern 1: node1 self-aliasing
  auto node1a = std::make_unique<StableHloNode>(
      "op1", "key1", std::vector<uint8_t>(ir), std::vector<void*>{shared},
      std::vector<void*>{shared}, false, manager->GetContext());
  auto node2a = std::make_unique<StableHloNode>("op2", "key2", std::vector<uint8_t>(ir),
                                                std::vector<void*>{in2}, std::vector<void*>{out2},
                                                false, manager->GetContext());

  // Pattern 2: node2 self-aliasing
  auto node1b = std::make_unique<StableHloNode>("op1", "key1", std::vector<uint8_t>(ir),
                                                std::vector<void*>{in1}, std::vector<void*>{out1},
                                                false, manager->GetContext());
  auto node2b = std::make_unique<StableHloNode>(
      "op2", "key2", std::vector<uint8_t>(ir), std::vector<void*>{shared},
      std::vector<void*>{shared}, false, manager->GetContext());

  std::string key1 = manager->GenerateConcatenationCacheKey(node1a.get(), node2a.get());
  std::string key2 = manager->GenerateConcatenationCacheKey(node1b.get(), node2b.get());

  EXPECT_NE(key1, key2);
  EXPECT_EQ(key1, "key1|key2|alias:1i0->1o0,");
  EXPECT_EQ(key2, "key1|key2|alias:2i0->2o0,");
}

TEST_F(IrNodeManagerTest, CacheMissDifferentAliasing) {
  auto ir = createDummyIr();
  void* shared = createPtr();
  void* in = createPtr();
  void* out = createPtr();

  // First pair: node1 self-aliasing
  auto node1a = std::make_unique<StableHloNode>(
      "op1", "key1", std::vector<uint8_t>(ir), std::vector<void*>{shared},
      std::vector<void*>{shared}, false, manager->GetContext());
  auto node2a = std::make_unique<StableHloNode>("op2", "key2", std::vector<uint8_t>(ir),
                                                std::vector<void*>{in}, std::vector<void*>{out},
                                                false, manager->GetContext());

  // Second pair: no aliasing
  auto node1b = std::make_unique<StableHloNode>("op1", "key1", std::vector<uint8_t>(ir),
                                                std::vector<void*>{in}, std::vector<void*>{out},
                                                false, manager->GetContext());
  auto node2b = std::make_unique<StableHloNode>("op2", "key2", std::vector<uint8_t>(ir),
                                                std::vector<void*>{in}, std::vector<void*>{out},
                                                false, manager->GetContext());

  auto result1 = manager->CreateConcatNode(node1a.get(), node2a.get());
  auto result2 = manager->CreateConcatNode(node1b.get(), node2b.get());

  ASSERT_NE(result1, nullptr);
  ASSERT_NE(result2, nullptr);
  EXPECT_NE(result1->cache_key, result2->cache_key);
  EXPECT_NE(result1->module.get(), result2->module.get());
}

TEST_F(IrNodeManagerTest, CacheHitSameAliasing) {
  auto ir = createDummyIr();
  void* shared1 = createPtr();
  void* shared2 = createPtr();

  // First pair
  auto node1a = std::make_unique<StableHloNode>(
      "op1", "key1", std::vector<uint8_t>(ir), std::vector<void*>{shared1},
      std::vector<void*>{shared1}, false, manager->GetContext());
  auto node2a = std::make_unique<StableHloNode>(
      "op2", "key2", std::vector<uint8_t>(ir), std::vector<void*>{shared2},
      std::vector<void*>{shared2}, false, manager->GetContext());

  // Second pair with same aliasing pattern
  auto node1b = std::make_unique<StableHloNode>(
      "op1", "key1", std::vector<uint8_t>(ir), std::vector<void*>{shared1},
      std::vector<void*>{shared1}, false, manager->GetContext());
  auto node2b = std::make_unique<StableHloNode>(
      "op2", "key2", std::vector<uint8_t>(ir), std::vector<void*>{shared2},
      std::vector<void*>{shared2}, false, manager->GetContext());

  auto result1 = manager->CreateConcatNode(node1a.get(), node2a.get());
  auto result2 = manager->CreateConcatNode(node1b.get(), node2b.get());

  if (result1 && result2) {
    EXPECT_EQ(result1->cache_key, result2->cache_key);
    EXPECT_EQ(result1->module.get(), result2->module.get());
  }
}

TEST_F(IrNodeManagerTest, CacheKeyMultiInputOutput) {
  auto ir = createDummyIr();
  void* p1 = createPtr();
  void* p2 = createPtr();
  void* p3 = createPtr();
  void* p4 = createPtr();

  auto node1 = std::make_unique<StableHloNode>(
      "op1", "key1", std::vector<uint8_t>(ir), std::vector<void*>{p1, p2},
      std::vector<void*>{p2, p3}, false, manager->GetContext());
  auto node2 = std::make_unique<StableHloNode>(
      "op2", "key2", std::vector<uint8_t>(ir), std::vector<void*>{p3, p4},
      std::vector<void*>{p3, p4}, false, manager->GetContext());

  std::string cache_key = manager->GenerateConcatenationCacheKey(node1.get(), node2.get());
  EXPECT_EQ(cache_key, "key1|key2|alias:1i1->1o0,2i0->2o0,2i1->2o1,");
}

// ============================================================================
// Overlap Discard Tests (TORCH_NEURONX_DISABLE_CONCAT_FOR_OVERLAP_IO_TENSOR flag)
// ============================================================================

TEST_F(IrNodeManagerTest, DiscardConcatEnabledByDefault) {
  auto ir = createDummyIr();
  void* shared = createPtr();

  auto node1 = std::make_unique<StableHloNode>(
      "op1", "key1", std::vector<uint8_t>(ir), std::vector<void*>{shared},
      std::vector<void*>{shared}, false, manager->GetContext());
  auto node2 = std::make_unique<StableHloNode>(
      "op2", "key2", std::vector<uint8_t>(ir), std::vector<void*>{createPtr()},
      std::vector<void*>{createPtr()}, false, manager->GetContext());

  std::list<IrNode*> nodes = {node1.get(), node2.get()};
  auto result = manager->ConcatAll(nodes);

  // By default, concatenation should be discarded due to aliasing
  EXPECT_EQ(result.compilable_irs.size(), 2);
  EXPECT_EQ(result.concat_ir_list.size(), 0);
}

TEST_F(IrNodeManagerTest, DiscardConcatDisabled) {
  setenv("TORCH_NEURONX_DISABLE_CONCAT_FOR_OVERLAP_IO_TENSOR", "0", 1);

  // Need to create new manager to pick up env var
  auto test_manager = std::make_unique<IrNodeManager>();

  auto ir = createDummyIr();
  void* shared = createPtr();

  auto node1 = std::make_unique<StableHloNode>(
      "op1", "key1", std::vector<uint8_t>(ir), std::vector<void*>{shared},
      std::vector<void*>{shared}, false, test_manager->GetContext());
  auto node2 = std::make_unique<StableHloNode>(
      "op2", "key2", std::vector<uint8_t>(ir), std::vector<void*>{createPtr()},
      std::vector<void*>{createPtr()}, false, test_manager->GetContext());

  std::list<IrNode*> nodes = {node1.get(), node2.get()};
  auto result = test_manager->ConcatAll(nodes);

  // With flag=0, concatenation should succeed despite aliasing
  EXPECT_EQ(result.compilable_irs.size(), 1);
  EXPECT_EQ(result.concat_ir_list.size(), 1);

  unsetenv("TORCH_NEURONX_DISABLE_CONCAT_FOR_OVERLAP_IO_TENSOR");
}

TEST_F(IrNodeManagerTest, DiscardConcatNoOverlap) {
  // Default behavior (enabled), but no overlap so concatenation succeeds
  auto ir = createDummyIr();

  auto node1 = std::make_unique<StableHloNode>(
      "op1", "key1", std::vector<uint8_t>(ir), std::vector<void*>{createPtr()},
      std::vector<void*>{createPtr()}, false, manager->GetContext());
  auto node2 = std::make_unique<StableHloNode>(
      "op2", "key2", std::vector<uint8_t>(ir), std::vector<void*>{createPtr()},
      std::vector<void*>{createPtr()}, false, manager->GetContext());

  std::list<IrNode*> nodes = {node1.get(), node2.get()};
  auto result = manager->ConcatAll(nodes);

  // No overlap, concatenation should succeed
  EXPECT_EQ(result.compilable_irs.size(), 1);
  EXPECT_EQ(result.concat_ir_list.size(), 1);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

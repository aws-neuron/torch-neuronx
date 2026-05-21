#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "torch_neuronx/csrc/core/concatenation/IrNode.h"
#include "torch_neuronx/csrc/core/concatenation/IrNodeManager.h"
#include "torch_neuronx/csrc/core/concatenation/OpConcatUtils.h"

using namespace torch_neuronx;

/**
 * Conditions tested (from design.md isBugCondition):
 *   Condition 1: mlir_result_count != node_output_count after multi-prune
 *   Condition 2: cached merge_mapping.total_outputs != node_output_count
 *   Condition 3: PruneMlirFunction failure leaves corrupted MLIR state
 *   Condition 4: Pruned cache entry has null MergeMapping
 */
class ConcatPruneExplorationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    manager = std::make_unique<IrNodeManager>();
    unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION_ALIASING");
  }

  void TearDown() override { unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION_ALIASING"); }

  std::unique_ptr<IrNodeManager> manager;

  // Counter for unique pointer addresses
  static int ptr_counter_;

  void* createPtr() { return reinterpret_cast<void*>(static_cast<uintptr_t>(++ptr_counter_)); }

  /**
   * Create a single-input, single-output StableHLO MLIR module.
   * Each node has a unique MLIR module: func @main(%arg0: tensor<f32>) -> tensor<f32>
   */
  std::vector<uint8_t> createSingleOutputIr() {
    std::string mlir = R"(
module @jit_op {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.multiply %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";
    return std::vector<uint8_t>(mlir.begin(), mlir.end());
  }

  /**
   * Create a StableHloNode with 1 input and 1 output.
   */
  std::unique_ptr<StableHloNode> createNode(const std::string& op_name,
                                            const std::string& cache_key, void* input_addr,
                                            void* output_addr) {
    auto ir = createSingleOutputIr();
    auto node = std::make_unique<StableHloNode>(
        op_name, cache_key, std::move(ir), std::vector<void*>{input_addr},
        std::vector<void*>{output_addr}, false, manager->GetContext());
    return node;
  }

  /**
   * Merge 4 individual nodes into a single merged node with 4 outputs.
   * Returns the merged node (owned by concat_ir_list in result).
   * Also returns the IrConcatResult to keep ownership alive.
   */
  struct MergedNodeResult {
    StableHloNode* node;  // raw pointer into concat_ir_list
    std::unique_ptr<StableHloNode> node1, node2, node3, node4;
    std::vector<void*> output_addrs;
    IrConcatResult concat_result;
  };

  MergedNodeResult createMergedNode4Outputs() {
    MergedNodeResult result;
    result.output_addrs = {createPtr(), createPtr(), createPtr(), createPtr()};

    result.node1 = createNode("op1", "key1", createPtr(), result.output_addrs[0]);
    result.node2 = createNode("op2", "key2", createPtr(), result.output_addrs[1]);
    result.node3 = createNode("op3", "key3", createPtr(), result.output_addrs[2]);
    result.node4 = createNode("op4", "key4", createPtr(), result.output_addrs[3]);

    // Merge: op1 + op2
    auto merged12 = manager->CreateConcatNode(result.node1.get(), result.node2.get());
    EXPECT_NE(merged12, nullptr) << "Failed to merge op1 + op2";

    // Merge: (op1|op2) + op3
    auto merged123 = manager->CreateConcatNode(merged12.get(), result.node3.get());
    EXPECT_NE(merged123, nullptr) << "Failed to merge (op1|op2) + op3";

    // Merge: (op1|op2|op3) + op4
    auto merged1234 = manager->CreateConcatNode(merged123.get(), result.node4.get());
    EXPECT_NE(merged1234, nullptr) << "Failed to merge (op1|op2|op3) + op4";

    result.node = merged1234.get();

    // Store in concat_result to keep ownership
    result.concat_result.concat_ir_list.push_back(std::move(merged12));
    result.concat_result.concat_ir_list.push_back(std::move(merged123));
    result.concat_result.concat_ir_list.push_back(std::move(merged1234));

    return result;
  }

  /**
   * Helper: get MLIR funcOp result count from a StableHloNode
   */
  size_t getFuncOpResultCount(StableHloNode* node) {
    auto funcOp = (*node->module)->lookupSymbol<mlir::func::FuncOp>("main");
    EXPECT_TRUE(funcOp) << "Could not find 'main' function in module";
    return funcOp.getNumResults();
  }
};

int ConcatPruneExplorationTest::ptr_counter_ = 0x1000;

/**
 * Test Case 1 — Multi-Prune Index Corruption
 * Create a merged StableHloNode with 4 outputs via CreateConcatNode.
 * Call PruneSingleOutput to remove output 0.
 * Then call PruneSingleOutput again to remove what was originally output 2
 * (now at index 1 after first prune).
 *
 * Assert after each prune:
 *   node->outputs.size() == funcOp.getNumResults()
 *   node->output_data_refs.size() == node->outputs.size()
 */
TEST_F(ConcatPruneExplorationTest, MultiPruneIndexCorruption) {
  auto merged = createMergedNode4Outputs();
  auto* node = merged.node;

  ASSERT_NE(node, nullptr);
  ASSERT_EQ(node->outputs.size(), 4u) << "Merged node should have 4 outputs";
  ASSERT_EQ(getFuncOpResultCount(node), 4u) << "MLIR function should have 4 results";

  // --- First prune: remove output at index 0 ---
  void* addr_to_prune_first = merged.output_addrs[0];
  bool prune1_ok = manager->PruneSingleOutput(node, addr_to_prune_first);
  ASSERT_TRUE(prune1_ok) << "First PruneSingleOutput should succeed";

  // After first prune: 3 outputs remain
  EXPECT_EQ(node->outputs.size(), 3u) << "After first prune, node should have 3 outputs";
  EXPECT_EQ(getFuncOpResultCount(node), node->outputs.size())
      << "Condition 1: MLIR result count must match node->outputs.size() after first prune";
  // NOTE: output_data_refs is not populated in unit tests (no real NRT tensors).
  // The bug condition we're testing is MLIR result count vs outputs size.

  // --- Second prune: remove what was originally output 2 (now at index 1) ---
  // After first prune removed index 0, the original output_addrs[2] is now at index 1
  void* addr_to_prune_second = merged.output_addrs[2];
  bool prune2_ok = manager->PruneSingleOutput(node, addr_to_prune_second);
  ASSERT_TRUE(prune2_ok) << "Second PruneSingleOutput should succeed";

  // After second prune: 2 outputs remain
  EXPECT_EQ(node->outputs.size(), 2u) << "After second prune, node should have 2 outputs";
  EXPECT_EQ(getFuncOpResultCount(node), node->outputs.size())
      << "Condition 1: MLIR result count must match node->outputs.size() after second prune";
}

/**
 * Test Case 2 — Cache Stale MergeMapping
 * Merge two nodes via CreateConcatNode, cache via CacheIrResource (automatic).
 * Prune one output on the merged node.
 * Retrieve the original cache entry via GetByKey.
 * Assert: cached_resource->merge_mapping->total_outputs == module.main_function.num_results
 */
TEST_F(ConcatPruneExplorationTest, CacheStaleMergeMapping) {
  void* out1 = createPtr();
  void* out2 = createPtr();

  auto node1 = createNode("op1", "stale_key1", createPtr(), out1);
  auto node2 = createNode("op2", "stale_key2", createPtr(), out2);

  // Merge: creates cached entry with key "stale_key1|stale_key2|alias_enabled:0"
  auto merged = manager->CreateConcatNode(node1.get(), node2.get());
  ASSERT_NE(merged, nullptr) << "Merge should succeed";
  ASSERT_EQ(merged->outputs.size(), 2u);

  std::string original_cache_key = merged->cache_key;

  // Retrieve the cached resource BEFORE pruning
  auto* cached_before = manager->GetByKey(original_cache_key);
  ASSERT_NE(cached_before, nullptr) << "Cache entry should exist for merged node";
  ASSERT_NE(cached_before->merge_mapping, nullptr) << "Cached merge_mapping should be non-null";

  // Record the original module's result count via the cached resource
  auto funcOp_before = (*cached_before->module)->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(funcOp_before);
  size_t results_before = funcOp_before.getNumResults();
  EXPECT_EQ(results_before, 2u);

  // --- Prune one output on the merged node ---
  bool prune_ok = manager->PruneSingleOutput(merged.get(), out1);
  ASSERT_TRUE(prune_ok) << "Prune should succeed";
  EXPECT_EQ(merged->outputs.size(), 1u);

  // --- Retrieve the ORIGINAL cache entry again ---
  auto* cached_after = manager->GetByKey(original_cache_key);
  ASSERT_NE(cached_after, nullptr) << "Original cache entry should still exist";
  ASSERT_NE(cached_after->merge_mapping, nullptr)
      << "Cached merge_mapping should still be non-null";

  // The cached module's result count should still match the cached mapping
  auto funcOp_after = (*cached_after->module)->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(funcOp_after);
  size_t module_results = funcOp_after.getNumResults();

  EXPECT_EQ(cached_after->merge_mapping->total_outputs, module_results)
      << "Condition 2: cached merge_mapping->total_outputs ("
      << cached_after->merge_mapping->total_outputs << ") must match module's actual result count ("
      << module_results << "). If they differ, the shared module was mutated in-place by pruning.";
}

/**
 * Test Case 3 — PruneMlirFunction Failure Corruption
 * Create a merged node, force PruneMlirFunction to fail by pruning all outputs
 * (PruneSingleOutput returns false when all outputs would be removed).
 * Verify the node's MLIR module is not left partially modified:
 *   function type result count must match return op operand count.
 */
TEST_F(ConcatPruneExplorationTest, PruneMlirFunctionFailureCorruption) {
  void* out1 = createPtr();
  void* out2 = createPtr();

  auto node1 = createNode("op1", "fail_key1", createPtr(), out1);
  auto node2 = createNode("op2", "fail_key2", createPtr(), out2);

  auto merged = manager->CreateConcatNode(node1.get(), node2.get());
  ASSERT_NE(merged, nullptr);
  ASSERT_EQ(merged->outputs.size(), 2u);

  // Record pre-prune state
  auto funcOp_pre = (*merged->module)->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(funcOp_pre);
  size_t results_pre = funcOp_pre.getNumResults();
  EXPECT_EQ(results_pre, 2u);

  // First, prune one output successfully to reduce to 1 output
  bool prune1_ok = manager->PruneSingleOutput(merged.get(), out1);
  ASSERT_TRUE(prune1_ok) << "First prune should succeed (2 -> 1 output)";
  EXPECT_EQ(merged->outputs.size(), 1u);

  // Now attempt to prune the last remaining output — this should fail
  // (PruneSingleOutput returns false when pruning would remove all outputs)
  bool prune2_ok = manager->PruneSingleOutput(merged.get(), out2);
  EXPECT_FALSE(prune2_ok) << "Pruning all outputs should fail (return false)";

  // After the failed prune, verify MLIR module consistency:
  // The function type result count must match the return op operand count
  auto funcOp_post = (*merged->module)->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(funcOp_post) << "Module should still have a 'main' function";

  size_t func_type_results = funcOp_post.getNumResults();

  // Find the return op and check its operand count
  mlir::func::ReturnOp returnOp;
  funcOp_post.walk([&](mlir::func::ReturnOp op) { returnOp = op; });
  ASSERT_TRUE(returnOp) << "Function should have a return op";

  size_t return_operands = returnOp.getNumOperands();

  EXPECT_EQ(func_type_results, return_operands)
      << "Condition 3: After failed prune, function type result count (" << func_type_results
      << ") must match return op operand count (" << return_operands
      << "). Mismatch indicates partially corrupted MLIR state.";

  // Also verify the node's output vector is still consistent with MLIR
  EXPECT_EQ(merged->outputs.size(), func_type_results)
      << "Node outputs size should match MLIR function result count after failed prune";
}

/**
 * Test Case 4 — Prune-Then-Cache-Hit Null MergeMapping
 * Merge two nodes, prune one output, cache the pruned result via CacheIrResource
 * (automatic in PruneSingleOutput). Look up the pruned cache key via GetByKey.
 * Assert:
 *   ir_resource->merge_mapping != nullptr
 *   merge_mapping->total_outputs == node->outputs.size()
 */
TEST_F(ConcatPruneExplorationTest, PruneThenCacheHitNullMergeMapping) {
  void* out1 = createPtr();
  void* out2 = createPtr();

  auto node1 = createNode("op1", "null_mm_key1", createPtr(), out1);
  auto node2 = createNode("op2", "null_mm_key2", createPtr(), out2);

  auto merged = manager->CreateConcatNode(node1.get(), node2.get());
  ASSERT_NE(merged, nullptr);
  ASSERT_EQ(merged->outputs.size(), 2u);

  // Prune one output — this caches the pruned result
  bool prune_ok = manager->PruneSingleOutput(merged.get(), out1);
  ASSERT_TRUE(prune_ok) << "Prune should succeed";
  EXPECT_EQ(merged->outputs.size(), 1u);

  // The pruned node should have a new cache key
  std::string pruned_cache_key = merged->cache_key;
  EXPECT_NE(pruned_cache_key, "null_mm_key1|null_mm_key2|alias_enabled:0")
      << "Cache key should have changed after pruning";

  // Look up the pruned cache entry
  auto* ir_resource = manager->GetByKey(pruned_cache_key);
  ASSERT_NE(ir_resource, nullptr) << "Pruned cache entry should exist at key: " << pruned_cache_key;

  // Assert MergeMapping is non-null
  EXPECT_NE(ir_resource->merge_mapping, nullptr)
      << "Pruned cache entry must have a non-null MergeMapping. "
      << "On unfixed code, CacheIrResource is called with nullptr merge_mapping for pruned nodes.";

  // If MergeMapping exists, verify it's consistent with the node's output count
  if (ir_resource->merge_mapping) {
    EXPECT_EQ(ir_resource->merge_mapping->total_outputs, merged->outputs.size())
        << "MergeMapping total_outputs (" << ir_resource->merge_mapping->total_outputs
        << ") must match node->outputs.size() (" << merged->outputs.size() << ")";

    // Verify all output_mapping keys are in valid range [0, outputs.size())
    for (const auto& [idx, mapping] : ir_resource->merge_mapping->output_mapping) {
      EXPECT_LT(idx, merged->outputs.size())
          << "MergeMapping output_mapping key " << idx << " is out of range [0, "
          << merged->outputs.size() << ")";
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

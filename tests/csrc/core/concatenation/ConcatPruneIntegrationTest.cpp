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
 * Integration tests for the concat-prune bugfix.
 *
 * These tests exercise scenarios that the exploration/preservation tests
 * do NOT cover:
 *   1. Cache-hit path after pruning with NEW output addresses
 *   2. Sequential prune cache reuse across iterations
 *   3. ConcatAll with interleaved DeallocHintNodes
 *   4. Clone-before-prune: original cache entry module independence
 *   5. MergeMapping rebuild correctness after pruning
 */
class ConcatPruneIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    manager = std::make_unique<IrNodeManager>();
    unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION_ALIASING");
  }

  void TearDown() override { unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION_ALIASING"); }

  std::unique_ptr<IrNodeManager> manager;

  static int ptr_counter_;

  void* createPtr() { return reinterpret_cast<void*>(static_cast<uintptr_t>(++ptr_counter_)); }

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

  std::unique_ptr<StableHloNode> createNode(const std::string& op_name,
                                            const std::string& cache_key, void* input_addr,
                                            void* output_addr) {
    auto ir = createSingleOutputIr();
    auto node = std::make_unique<StableHloNode>(
        op_name, cache_key, std::move(ir), std::vector<void*>{input_addr},
        std::vector<void*>{output_addr}, false, manager->GetContext());
    return node;
  }

  size_t getFuncOpResultCount(StableHloNode* node) {
    auto funcOp = (*node->module)->lookupSymbol<mlir::func::FuncOp>("main");
    EXPECT_TRUE(funcOp) << "Could not find 'main' function in module";
    return funcOp.getNumResults();
  }
};

int ConcatPruneIntegrationTest::ptr_counter_ = 0x9000;

// =============================================================================
// Test 1: CacheHitPruneReuse
//
// Merge 3 nodes, prune output 0 (cache miss). Then create a NEW set of 3 nodes
// with the SAME cache keys but DIFFERENT output addresses. Merge them, prune
// output 0 again. This should be a cache HIT. Verify:
//   - The cache-hit path returns true
//   - node->outputs.size() matches funcOp.getNumResults() after cache-hit restore
//   - The output addresses are the NEW addresses (not the old ones)
// =============================================================================

TEST_F(ConcatPruneIntegrationTest, CacheHitPruneReuse) {
  // --- Iteration 1: cache miss path ---
  void* old_out0 = createPtr();
  void* old_out1 = createPtr();
  void* old_out2 = createPtr();

  auto n1 = createNode("op1", "reuse_k1", createPtr(), old_out0);
  auto n2 = createNode("op2", "reuse_k2", createPtr(), old_out1);
  auto n3 = createNode("op3", "reuse_k3", createPtr(), old_out2);

  // Merge: n1 + n2
  auto merged12 = manager->CreateConcatNode(n1.get(), n2.get());
  ASSERT_NE(merged12, nullptr);
  // Merge: (n1|n2) + n3
  auto merged123 = manager->CreateConcatNode(merged12.get(), n3.get());
  ASSERT_NE(merged123, nullptr);
  ASSERT_EQ(merged123->outputs.size(), 3u);

  // Prune output 0 (cache miss — first time seeing this pruned key)
  bool prune1 = manager->PruneSingleOutput(merged123.get(), old_out0);
  ASSERT_TRUE(prune1) << "First prune (cache miss) should succeed";
  EXPECT_EQ(merged123->outputs.size(), 2u);
  EXPECT_EQ(getFuncOpResultCount(merged123.get()), 2u);

  std::string pruned_cache_key = merged123->cache_key;

  // --- Iteration 2: same cache keys, DIFFERENT addresses ---
  void* new_out0 = createPtr();
  void* new_out1 = createPtr();
  void* new_out2 = createPtr();

  auto n1b = createNode("op1", "reuse_k1", createPtr(), new_out0);
  auto n2b = createNode("op2", "reuse_k2", createPtr(), new_out1);
  auto n3b = createNode("op3", "reuse_k3", createPtr(), new_out2);

  // Merge again — cache HIT on the merge key
  auto merged12b = manager->CreateConcatNode(n1b.get(), n2b.get());
  ASSERT_NE(merged12b, nullptr);
  auto merged123b = manager->CreateConcatNode(merged12b.get(), n3b.get());
  ASSERT_NE(merged123b, nullptr);
  ASSERT_EQ(merged123b->outputs.size(), 3u);

  // Prune output 0 again — should be a cache HIT on the pruned key
  bool prune2 = manager->PruneSingleOutput(merged123b.get(), new_out0);
  ASSERT_TRUE(prune2) << "Second prune (cache hit) should succeed";

  // Verify: outputs.size() matches funcOp.getNumResults()
  EXPECT_EQ(merged123b->outputs.size(), getFuncOpResultCount(merged123b.get()))
      << "After cache-hit prune, MLIR result count must match outputs size";
  EXPECT_EQ(merged123b->outputs.size(), 2u);

  // Verify: output addresses are the NEW addresses, not the old ones
  std::unordered_set<void*> new_remaining = {new_out1, new_out2};
  std::unordered_set<void*> actual_outputs(merged123b->outputs.begin(), merged123b->outputs.end());
  EXPECT_EQ(actual_outputs, new_remaining)
      << "Cache-hit prune must use the NEW output addresses, not stale old ones";

  // Verify: old addresses are NOT present
  for (void* addr : merged123b->outputs) {
    EXPECT_NE(addr, old_out0);
    EXPECT_NE(addr, old_out1);
    EXPECT_NE(addr, old_out2);
  }
}

// =============================================================================
// Test 2: SequentialPruneCacheReuse
//
// Merge 4 nodes, prune output 0 (cache miss), then prune output 1 from the
// result (cache miss again — different key). Then create a NEW set of 4 nodes
// with same cache keys, merge, prune output 0 (cache HIT), prune output 1
// (cache HIT). Verify consistency at each step.
// =============================================================================

TEST_F(ConcatPruneIntegrationTest, SequentialPruneCacheReuse) {
  // --- Iteration 1: two sequential prunes (both cache misses) ---
  void* out0 = createPtr();
  void* out1 = createPtr();
  void* out2 = createPtr();
  void* out3 = createPtr();

  auto a1 = createNode("op1", "seq_k1", createPtr(), out0);
  auto a2 = createNode("op2", "seq_k2", createPtr(), out1);
  auto a3 = createNode("op3", "seq_k3", createPtr(), out2);
  auto a4 = createNode("op4", "seq_k4", createPtr(), out3);

  auto m12 = manager->CreateConcatNode(a1.get(), a2.get());
  ASSERT_NE(m12, nullptr);
  auto m123 = manager->CreateConcatNode(m12.get(), a3.get());
  ASSERT_NE(m123, nullptr);
  auto m1234 = manager->CreateConcatNode(m123.get(), a4.get());
  ASSERT_NE(m1234, nullptr);
  ASSERT_EQ(m1234->outputs.size(), 4u);

  // Prune output 0 (cache miss)
  bool p1 = manager->PruneSingleOutput(m1234.get(), out0);
  ASSERT_TRUE(p1);
  EXPECT_EQ(m1234->outputs.size(), 3u);
  EXPECT_EQ(getFuncOpResultCount(m1234.get()), 3u);

  // Prune output 1 (originally at index 1, now at index 0 after first prune)
  // But PruneSingleOutput matches by ADDRESS, not index — so prune out1
  bool p2 = manager->PruneSingleOutput(m1234.get(), out1);
  ASSERT_TRUE(p2);
  EXPECT_EQ(m1234->outputs.size(), 2u);
  EXPECT_EQ(getFuncOpResultCount(m1234.get()), 2u);

  // --- Iteration 2: same cache keys, new addresses ---
  void* new_out0 = createPtr();
  void* new_out1 = createPtr();
  void* new_out2 = createPtr();
  void* new_out3 = createPtr();

  auto b1 = createNode("op1", "seq_k1", createPtr(), new_out0);
  auto b2 = createNode("op2", "seq_k2", createPtr(), new_out1);
  auto b3 = createNode("op3", "seq_k3", createPtr(), new_out2);
  auto b4 = createNode("op4", "seq_k4", createPtr(), new_out3);

  auto m12b = manager->CreateConcatNode(b1.get(), b2.get());
  ASSERT_NE(m12b, nullptr);
  auto m123b = manager->CreateConcatNode(m12b.get(), b3.get());
  ASSERT_NE(m123b, nullptr);
  auto m1234b = manager->CreateConcatNode(m123b.get(), b4.get());
  ASSERT_NE(m1234b, nullptr);
  ASSERT_EQ(m1234b->outputs.size(), 4u);

  // Prune output 0 (cache HIT)
  bool p3 = manager->PruneSingleOutput(m1234b.get(), new_out0);
  ASSERT_TRUE(p3);
  EXPECT_EQ(m1234b->outputs.size(), 3u);
  EXPECT_EQ(getFuncOpResultCount(m1234b.get()), 3u);

  // Prune output 1 (cache HIT)
  bool p4 = manager->PruneSingleOutput(m1234b.get(), new_out1);
  ASSERT_TRUE(p4);
  EXPECT_EQ(m1234b->outputs.size(), 2u);
  EXPECT_EQ(getFuncOpResultCount(m1234b.get()), 2u);

  // Verify remaining outputs are the new addresses for out2 and out3
  std::unordered_set<void*> expected = {new_out2, new_out3};
  std::unordered_set<void*> actual(m1234b->outputs.begin(), m1234b->outputs.end());
  EXPECT_EQ(actual, expected)
      << "After sequential cache-hit prunes, remaining outputs must be new addresses";
}

// =============================================================================
// Test 3: ConcatAllWithDeallocHints
//
// Use IrNodeManager::ConcatAll with a mixed list of StableHloNodes and
// DeallocHintNodes. Create 3 HLO nodes, interleave a DeallocHintNode after
// the 2nd node (dealloc address = output of node 1). Verify:
//   - ConcatAll produces a merged node with outputs.size() == 2 (one was pruned)
//   - The pruned output address is NOT in the result
//   - The MLIR function result count matches
// =============================================================================

TEST_F(ConcatPruneIntegrationTest, ConcatAllWithDeallocHints) {
  void* out0 = createPtr();
  void* out1 = createPtr();
  void* out2 = createPtr();

  auto node0 = createNode("op0", "dall_k0", createPtr(), out0);
  auto node1 = createNode("op1", "dall_k1", createPtr(), out1);
  auto node2 = createNode("op2", "dall_k2", createPtr(), out2);

  // DeallocHintNode: dealloc address = output of node 0
  DeallocHintNode dealloc_hint(out0);

  // Build input list: [node0, node1, dealloc(out0), node2]
  // Processing order:
  //   node0 → starter
  //   node1 → merge with node0 → merged has outputs {out0, out1}
  //   dealloc(out0) → prune out0 from merged → merged has outputs {out1}
  //   node2 → merge with pruned → merged has outputs {out1, out2}
  std::list<IrNode*> ir_list;
  ir_list.push_back(node0.get());
  ir_list.push_back(node1.get());
  ir_list.push_back(&dealloc_hint);
  ir_list.push_back(node2.get());

  auto result = manager->ConcatAll(ir_list);

  ASSERT_EQ(result.compilable_irs.size(), 1u) << "ConcatAll should produce 1 merged compilable IR";

  auto* merged = dynamic_cast<StableHloNode*>(result.compilable_irs.front());
  ASSERT_NE(merged, nullptr) << "Merged node should be a StableHloNode";

  // After pruning out0, the merged node should have 2 outputs: out1 and out2
  EXPECT_EQ(merged->outputs.size(), 2u)
      << "Merged node should have 2 outputs (one was pruned by dealloc hint)";

  // MLIR function result count must match
  EXPECT_EQ(getFuncOpResultCount(merged), merged->outputs.size())
      << "MLIR function result count must match outputs size after ConcatAll with dealloc";

  // The pruned address (out0) must NOT be in the result
  for (void* addr : merged->outputs) {
    EXPECT_NE(addr, out0) << "Pruned address (out0) should not appear in merged outputs";
  }

  // The remaining addresses should be out1 and out2
  std::unordered_set<void*> expected = {out1, out2};
  std::unordered_set<void*> actual(merged->outputs.begin(), merged->outputs.end());
  EXPECT_EQ(actual, expected) << "Remaining outputs should be out1 and out2 after pruning out0";
}

// =============================================================================
// Test 4: ClonedModuleIndependence
//
// Merge 2 nodes, get the cache entry. Prune one output (which clones the
// module). Verify the ORIGINAL cache entry's module is UNCHANGED (still has
// the original result count). This directly tests our clone-before-prune fix.
// =============================================================================

TEST_F(ConcatPruneIntegrationTest, ClonedModuleIndependence) {
  void* out0 = createPtr();
  void* out1 = createPtr();

  auto n1 = createNode("op1", "clone_k1", createPtr(), out0);
  auto n2 = createNode("op2", "clone_k2", createPtr(), out1);

  auto merged = manager->CreateConcatNode(n1.get(), n2.get());
  ASSERT_NE(merged, nullptr);
  ASSERT_EQ(merged->outputs.size(), 2u);

  std::string original_cache_key = merged->cache_key;

  // Retrieve the original cache entry BEFORE pruning
  auto* cached_before = manager->GetByKey(original_cache_key);
  ASSERT_NE(cached_before, nullptr);

  auto funcOp_before = (*cached_before->module)->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(funcOp_before);
  size_t original_result_count = funcOp_before.getNumResults();
  EXPECT_EQ(original_result_count, 2u) << "Original module should have 2 results";

  // --- Prune one output (this should clone the module) ---
  bool prune_ok = manager->PruneSingleOutput(merged.get(), out0);
  ASSERT_TRUE(prune_ok);
  EXPECT_EQ(merged->outputs.size(), 1u);
  EXPECT_EQ(getFuncOpResultCount(merged.get()), 1u);

  // --- Verify the ORIGINAL cache entry's module is UNCHANGED ---
  auto* cached_after = manager->GetByKey(original_cache_key);
  ASSERT_NE(cached_after, nullptr) << "Original cache entry should still exist";

  auto funcOp_after = (*cached_after->module)->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(funcOp_after);
  size_t after_result_count = funcOp_after.getNumResults();

  EXPECT_EQ(after_result_count, original_result_count)
      << "Clone-before-prune fix: original cache entry's module must be UNCHANGED. "
      << "Expected " << original_result_count << " results but got " << after_result_count
      << ". This means PruneSingleOutput mutated the shared module in-place.";

  // Also verify the original cache entry's MergeMapping is still consistent
  ASSERT_NE(cached_after->merge_mapping, nullptr);
  EXPECT_EQ(cached_after->merge_mapping->total_outputs, original_result_count)
      << "Original cache entry's MergeMapping should still reflect the original output count";
}

// =============================================================================
// Test 5: MergeMappingRebuildCorrectness
//
// Merge 3 nodes (outputs at indices 0,1,2 mapping to modules 1,2,3).
// Prune index 1. Verify the rebuilt MergeMapping has:
//   - total_outputs == 2
//   - output_mapping[0] maps to module 1's output
//   - output_mapping[1] maps to module 3's output (was index 2, now re-indexed)
// =============================================================================

TEST_F(ConcatPruneIntegrationTest, MergeMappingRebuildCorrectness) {
  void* out0 = createPtr();
  void* out1 = createPtr();
  void* out2 = createPtr();

  auto n1 = createNode("op1", "mm_k1", createPtr(), out0);
  auto n2 = createNode("op2", "mm_k2", createPtr(), out1);
  auto n3 = createNode("op3", "mm_k3", createPtr(), out2);

  // Merge: n1 + n2
  auto merged12 = manager->CreateConcatNode(n1.get(), n2.get());
  ASSERT_NE(merged12, nullptr);
  // Merge: (n1|n2) + n3
  auto merged123 = manager->CreateConcatNode(merged12.get(), n3.get());
  ASSERT_NE(merged123, nullptr);
  ASSERT_EQ(merged123->outputs.size(), 3u);

  // Verify pre-prune MergeMapping from cache
  std::string pre_prune_key = merged123->cache_key;
  auto* pre_cached = manager->GetByKey(pre_prune_key);
  ASSERT_NE(pre_cached, nullptr);
  ASSERT_NE(pre_cached->merge_mapping, nullptr);
  EXPECT_EQ(pre_cached->merge_mapping->total_outputs, 3u);

  // Record the original output_mapping for indices 0 and 2
  // These should survive the prune (index 1 is being pruned)
  auto orig_mapping_0 = pre_cached->merge_mapping->output_mapping[0];
  auto orig_mapping_2 = pre_cached->merge_mapping->output_mapping[2];

  // --- Prune output at index 1 (address = out1) ---
  bool prune_ok = manager->PruneSingleOutput(merged123.get(), out1);
  ASSERT_TRUE(prune_ok);
  EXPECT_EQ(merged123->outputs.size(), 2u);
  EXPECT_EQ(getFuncOpResultCount(merged123.get()), 2u);

  // Retrieve the pruned cache entry
  std::string pruned_key = merged123->cache_key;
  EXPECT_NE(pruned_key, pre_prune_key) << "Pruned cache key should differ from original";

  auto* pruned_cached = manager->GetByKey(pruned_key);
  ASSERT_NE(pruned_cached, nullptr) << "Pruned cache entry should exist";
  ASSERT_NE(pruned_cached->merge_mapping, nullptr)
      << "Pruned cache entry must have a non-null MergeMapping";

  auto* pm = pruned_cached->merge_mapping.get();

  // Verify: total_outputs == 2
  EXPECT_EQ(pm->total_outputs, 2u) << "After pruning 1 of 3 outputs, total_outputs should be 2";

  // Verify: output_mapping[0] maps to the same module as original index 0
  // (module 1's output — the first node's contribution)
  ASSERT_NE(pm->output_mapping.find(0), pm->output_mapping.end())
      << "output_mapping should have entry for index 0";
  EXPECT_EQ(pm->output_mapping[0], orig_mapping_0)
      << "output_mapping[0] should map to module 1's output (unchanged)";

  // Verify: output_mapping[1] maps to what was originally at index 2
  // (module 3's output — the third node's contribution, re-indexed from 2 to 1)
  ASSERT_NE(pm->output_mapping.find(1), pm->output_mapping.end())
      << "output_mapping should have entry for index 1";
  EXPECT_EQ(pm->output_mapping[1], orig_mapping_2)
      << "output_mapping[1] should map to module 3's output (was index 2, re-indexed to 1)";

  // Verify: all keys in output_mapping are in valid range [0, total_outputs)
  for (const auto& [idx, info] : pm->output_mapping) {
    EXPECT_LT(idx, pm->total_outputs)
        << "output_mapping key " << idx << " out of range [0, " << pm->total_outputs << ")";
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

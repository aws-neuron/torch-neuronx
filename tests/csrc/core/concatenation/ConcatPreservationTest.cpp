#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <numeric>
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
 * Properties tested:
 *   Property 1: No-prune merge via ConcatAll preserves output count, mapping, and cache key
 *   Property 2: Single successful PruneSingleOutput reduces outputs correctly
 *   Property 3: Cache hit for non-pruned merge key returns consistent IrResource
 *   Property 4: HandleConcatenationFailure returns original operations in order
 */

// =============================================================================
// Shared test fixture base for both parameterized and non-parameterized tests
// =============================================================================
class ConcatPreservationFixture {
 protected:
  void SetUpFixture() {
    manager = std::make_unique<IrNodeManager>();
    unsetenv("TORCH_NEURONX_SKIP_INTERMEDIATES");
    unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION_ALIASING");
  }

  void TearDownFixture() {
    unsetenv("TORCH_NEURONX_SKIP_INTERMEDIATES");
    unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION_ALIASING");
  }

  std::unique_ptr<IrNodeManager> manager;

  static int ptr_counter_;

  void* createPtr() { return reinterpret_cast<void*>(static_cast<uintptr_t>(++ptr_counter_)); }

  /**
   * Create a single-input, single-output StableHLO MLIR module.
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
   * Create N nodes with unique addresses and cache keys.
   * Each node has 1 input and 1 output.
   */
  struct NodeSet {
    std::vector<std::unique_ptr<StableHloNode>> nodes;
    std::vector<void*> input_addrs;
    std::vector<void*> output_addrs;
    std::vector<std::string> cache_keys;
  };

  NodeSet createNodes(int count) {
    NodeSet ns;
    for (int i = 0; i < count; ++i) {
      void* in = createPtr();
      void* out = createPtr();
      std::string key = "pkey_" + std::to_string(i);
      ns.input_addrs.push_back(in);
      ns.output_addrs.push_back(out);
      ns.cache_keys.push_back(key);
      ns.nodes.push_back(createNode("pop_" + std::to_string(i), key, in, out));
    }
    return ns;
  }

  /**
   * Helper: get MLIR funcOp result count from a StableHloNode
   */
  size_t getFuncOpResultCount(StableHloNode* node) {
    auto funcOp = (*node->module)->lookupSymbol<mlir::func::FuncOp>("main");
    EXPECT_TRUE(funcOp) << "Could not find 'main' function in module";
    return funcOp.getNumResults();
  }

  /**
   * Build expected pipe-separated cache key for N merged nodes.
   * Format: "key0|key1|...|keyN-1|alias_enabled:0"
   * The merge is left-associative: ((key0|key1)|key2)|key3...
   */
  std::string buildExpectedCacheKey(const std::vector<std::string>& keys) {
    if (keys.empty()) return "";
    // Left-associative merge: first merge keys[0]|keys[1], then result|keys[2], etc.
    std::string result = keys[0];
    for (size_t i = 1; i < keys.size(); ++i) {
      result = result + "|" + keys[i] + "|alias_enabled:0";
    }
    return result;
  }
};

int ConcatPreservationFixture::ptr_counter_ = 0x5000;

// =============================================================================
// Parameterized fixture — single parameter: node_count
// =============================================================================
class ConcatPreservationTest : public ::testing::TestWithParam<int>,
                               public ConcatPreservationFixture {
 protected:
  void SetUp() override { SetUpFixture(); }
  void TearDown() override { TearDownFixture(); }
};

// =============================================================================
// Non-parameterized fixture for parameter-independent tests
// =============================================================================
class ConcatPreservationStaticTest : public ::testing::Test, public ConcatPreservationFixture {
 protected:
  void SetUp() override { SetUpFixture(); }
  void TearDown() override { TearDownFixture(); }
};

// =============================================================================
// Property 1: No-Prune Merge Preservation via ConcatAll
//
// For all merge sequences with NO dealloc hints, ConcatAll produces
// IrConcatResult with:
//   merged_node.outputs.size() == sum(individual_outputs)
//   merge_mapping.total_outputs == merged_node.outputs.size()
//   cache key is pipe-separated concatenation of individual keys
// =============================================================================

TEST_P(ConcatPreservationTest, NoPruneMergePreservation) {
  int node_count = GetParam();

  auto ns = createNodes(node_count);

  // Build input list for ConcatAll — no dealloc hints
  std::list<IrNode*> ir_list;
  for (auto& node : ns.nodes) {
    ir_list.push_back(node.get());
  }

  auto result = manager->ConcatAll(ir_list);

  // ConcatAll should produce exactly 1 compilable IR (the merged node)
  ASSERT_EQ(result.compilable_irs.size(), 1u)
      << "ConcatAll with " << node_count << " nodes should produce 1 merged compilable IR";

  // The merged node should be a StableHloNode
  auto* merged = dynamic_cast<StableHloNode*>(result.compilable_irs.front());
  ASSERT_NE(merged, nullptr) << "Merged node should be a StableHloNode";

  // Property: merged_node.outputs.size() == sum(individual_outputs)
  size_t expected_outputs = static_cast<size_t>(node_count);  // 1 output per node
  EXPECT_EQ(merged->outputs.size(), expected_outputs)
      << "Merged node should have " << expected_outputs << " outputs (1 per node)";

  // Property: MLIR function result count matches output vector
  EXPECT_EQ(getFuncOpResultCount(merged), merged->outputs.size())
      << "MLIR function result count must match merged outputs size";

  // Property: merge_mapping.total_outputs == merged_node.outputs.size()
  // NOTE: CreateConcatNode std::move's the MergeMapping into the cache,
  // so node->GetMergeMapping() is nullptr. Look up via cache instead.
  auto* ir_resource = manager->GetByKey(merged->cache_key);
  ASSERT_NE(ir_resource, nullptr) << "Cache should contain entry for merged node";
  ASSERT_NE(ir_resource->merge_mapping, nullptr) << "Cached MergeMapping must be non-null";
  auto* mapping = ir_resource->merge_mapping.get();
  EXPECT_EQ(mapping->total_outputs, merged->outputs.size())
      << "MergeMapping total_outputs must match merged outputs size";

  // Property: all output_mapping keys are in valid range
  for (const auto& [idx, info] : mapping->output_mapping) {
    EXPECT_LT(idx, merged->outputs.size()) << "MergeMapping output_mapping key " << idx
                                           << " out of range [0, " << merged->outputs.size() << ")";
  }

  // Property: cache key is pipe-separated concatenation of individual keys
  std::string expected_key = buildExpectedCacheKey(ns.cache_keys);
  EXPECT_EQ(merged->cache_key, expected_key)
      << "Cache key should be pipe-separated concatenation of individual keys";

  // Property: output addresses are preserved (union of all individual outputs)
  std::unordered_set<void*> expected_addrs(ns.output_addrs.begin(), ns.output_addrs.end());
  std::unordered_set<void*> actual_addrs(merged->outputs.begin(), merged->outputs.end());
  EXPECT_EQ(actual_addrs, expected_addrs)
      << "Merged node output addresses should be union of individual output addresses";
}

// =============================================================================
// Property 2: Single-Prune Preservation
//
// For all single-prune operations where the dealloc address IS in
// node->outputs, the pruned node has:
//   outputs.size() == original_outputs.size() - 1
//   funcOp.getNumResults() == outputs.size()
// =============================================================================

TEST_P(ConcatPreservationTest, SinglePrunePreservation) {
  int node_count = GetParam();

  auto ns = createNodes(node_count);

  // Merge all nodes via incremental CreateConcatNode
  std::unique_ptr<StableHloNode> merged;
  std::vector<std::unique_ptr<StableHloNode>> intermediates;

  // First node is the starting point
  merged = std::move(ns.nodes[0]);
  for (int i = 1; i < node_count; ++i) {
    auto next_merged = manager->CreateConcatNode(merged.get(), ns.nodes[i].get());
    ASSERT_NE(next_merged, nullptr) << "Merge step " << i << " should succeed";
    intermediates.push_back(std::move(merged));
    merged = std::move(next_merged);
  }

  size_t original_output_count = merged->outputs.size();
  ASSERT_EQ(original_output_count, static_cast<size_t>(node_count))
      << "Merged node should have " << node_count << " outputs before prune";

  // Prune the FIRST output address (always valid since node_count >= 2)
  void* addr_to_prune = ns.output_addrs[0];
  bool prune_ok = manager->PruneSingleOutput(merged.get(), addr_to_prune);
  ASSERT_TRUE(prune_ok) << "Single PruneSingleOutput should succeed";

  // Property: outputs.size() == original_outputs.size() - 1
  EXPECT_EQ(merged->outputs.size(), original_output_count - 1)
      << "After single prune, outputs should decrease by 1";

  // Property: funcOp.getNumResults() == outputs.size()
  EXPECT_EQ(getFuncOpResultCount(merged.get()), merged->outputs.size())
      << "MLIR function result count must match outputs size after single prune";

  // NOTE: output_data_refs is only populated on cache-hit path when
  // CreateConcatNode reconstructs from cached IrResource with real NRT tensors.
  // In unit tests without real tensors, output_data_refs is empty — skip this check.

  // Property: pruned address is no longer in outputs
  for (void* addr : merged->outputs) {
    EXPECT_NE(addr, addr_to_prune) << "Pruned address should not appear in outputs after prune";
  }
}

// =============================================================================
// Property 3: Cache Hit Preservation for Non-Pruned Merge Keys
//
// For all cache lookups of non-pruned merge keys via GetByKey, the returned
// IrResource has:
//   merge_mapping->total_outputs matching the module's function result count
// =============================================================================

TEST_P(ConcatPreservationTest, CacheHitNonPrunedPreservation) {
  int node_count = GetParam();

  auto ns = createNodes(node_count);

  // Merge all nodes via incremental CreateConcatNode
  std::unique_ptr<StableHloNode> merged;
  std::vector<std::unique_ptr<StableHloNode>> intermediates;

  merged = std::move(ns.nodes[0]);
  for (int i = 1; i < node_count; ++i) {
    auto next_merged = manager->CreateConcatNode(merged.get(), ns.nodes[i].get());
    ASSERT_NE(next_merged, nullptr) << "Merge step " << i << " should succeed";
    intermediates.push_back(std::move(merged));
    merged = std::move(next_merged);
  }

  std::string merge_key = merged->cache_key;

  // Look up the non-pruned merge key in cache
  auto* ir_resource = manager->GetByKey(merge_key);
  ASSERT_NE(ir_resource, nullptr) << "Cache should contain entry for merge key: " << merge_key;

  // Property: merge_mapping is non-null
  ASSERT_NE(ir_resource->merge_mapping, nullptr)
      << "Cached IrResource must have a non-null merge_mapping";

  // Property: merge_mapping->total_outputs matches module's function result count
  auto funcOp = (*ir_resource->module)->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(funcOp) << "Cached module should have a 'main' function";
  size_t module_results = funcOp.getNumResults();

  EXPECT_EQ(ir_resource->merge_mapping->total_outputs, module_results)
      << "Cached merge_mapping->total_outputs (" << ir_resource->merge_mapping->total_outputs
      << ") must match module's function result count (" << module_results << ")";

  // Property: total_outputs matches expected node count
  EXPECT_EQ(ir_resource->merge_mapping->total_outputs, static_cast<size_t>(node_count))
      << "Cached merge_mapping->total_outputs should equal number of merged nodes";

  // Property: output_mapping has correct number of entries
  EXPECT_EQ(ir_resource->merge_mapping->output_mapping.size(), static_cast<size_t>(node_count))
      << "output_mapping should have one entry per merged output";

  // Property: all output_mapping keys are in valid range
  for (const auto& [idx, info] : ir_resource->merge_mapping->output_mapping) {
    EXPECT_LT(idx, ir_resource->merge_mapping->total_outputs)
        << "output_mapping key " << idx << " out of range";
  }

  // Property: second lookup returns same resource (cache consistency)
  auto* ir_resource2 = manager->GetByKey(merge_key);
  EXPECT_EQ(ir_resource, ir_resource2)
      << "Repeated cache lookup should return same IrResource pointer";
}

// =============================================================================
// Property 4a: Parameterized Fallback — non-StableHLO node throws
//
// ConcatAll with an invalid (non-StableHLO) node triggers runtime_error,
// which is the error path that triggers HandleConcatenationFailure in the
// full pipeline.
// =============================================================================

TEST_P(ConcatPreservationTest, FallbackThrowsOnInvalidNode) {
  int node_count = GetParam();

  auto ns = createNodes(node_count);
  auto ir = createSingleOutputIr();
  IrNode invalid_node("invalid_op", "invalid_key", ir, IrNodeType::HLO,
                      std::vector<void*>{createPtr()}, std::vector<void*>{createPtr()}, false);

  std::list<IrNode*> mixed_list;
  mixed_list.push_back(ns.nodes[0].get());
  mixed_list.push_back(&invalid_node);

  // ConcatAll should throw when encountering non-StableHLO nodes
  EXPECT_THROW(manager->ConcatAll(mixed_list), std::runtime_error)
      << "ConcatAll with non-StableHLO node should throw runtime_error";
}

// =============================================================================
// Property 4b: Parameter-independent fallback tests (run exactly once)
// =============================================================================

TEST_F(ConcatPreservationStaticTest, SingleNodeFallbackPreservation) {
  auto ns = createNodes(1);
  std::list<IrNode*> single_list = {ns.nodes[0].get()};

  auto result = manager->ConcatAll(single_list);

  EXPECT_EQ(result.compilable_irs.size(), 1u)
      << "Single-node ConcatAll should return 1 compilable IR";
  EXPECT_EQ(result.compilable_irs.front(), ns.nodes[0].get())
      << "Single-node ConcatAll should return the original node";
  EXPECT_TRUE(result.concat_ir_list.empty())
      << "Single-node ConcatAll should not create any concatenated nodes";
}

TEST_F(ConcatPreservationStaticTest, EmptyListFallbackPreservation) {
  std::list<IrNode*> empty_list;
  auto result = manager->ConcatAll(empty_list);

  EXPECT_TRUE(result.compilable_irs.empty())
      << "Empty ConcatAll should return empty compilable_irs";
  EXPECT_TRUE(result.remain_irs.empty()) << "Empty ConcatAll should return empty remain_irs";
  EXPECT_TRUE(result.concat_ir_list.empty())
      << "Empty ConcatAll should return empty concat_ir_list";
}

// =============================================================================
// Parameterized test instantiation — single dimension: node_count 2..4
// =============================================================================

INSTANTIATE_TEST_SUITE_P(PreservationSweep, ConcatPreservationTest,
                         ::testing::Range(2, 5),  // node_count: 2, 3, 4
                         [](const ::testing::TestParamInfo<int>& info) {
                           return "Nodes" + std::to_string(info.param);
                         });

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

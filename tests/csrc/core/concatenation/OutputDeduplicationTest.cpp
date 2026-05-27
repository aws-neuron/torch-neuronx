#include <gtest/gtest.h>

#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "torch_neuronx/csrc/core/KernelExecution.h"

/**
 * Unit tests for the output deduplication logic used in ConcatenationCore
 * when TORCH_NEURONX_ENABLE_CONCATENATION_ALIASING=1 is set.
 *
 * The deduplication removes duplicate output addresses keeping only the
 * last occurrence of each address ("last writer wins"). This ensures the
 * NRT receives unique output bindings while preserving the final value
 * semantics of in-place operations.
 *
 * These tests exercise the deduplication algorithm in isolation without
 * requiring a full concatenation pipeline setup.
 */

namespace {

struct DeduplicationResult {
  std::vector<void*> outputs;
  std::vector<at::neuron::TensorDataRef> output_data_refs;
  std::vector<at::neuron::TensorContext> output_contexts;
};

// Extracted deduplication algorithm matching ConcatenationCore::CreateOpContext
DeduplicationResult DeduplicateOutputs(std::vector<void*> outputs,
                                       std::vector<at::neuron::TensorDataRef> output_data_refs,
                                       std::vector<at::neuron::TensorContext> output_contexts) {
  std::unordered_set<void*> unique_outputs(outputs.begin(), outputs.end());
  if (unique_outputs.size() == outputs.size()) {
    // No duplicates — return as-is
    return {outputs, output_data_refs, output_contexts};
  }

  // Last-occurrence deduplication
  std::unordered_map<void*, size_t> last_idx;
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i]) last_idx[outputs[i]] = i;
  }

  std::vector<void*> deduped_outputs;
  std::vector<at::neuron::TensorDataRef> deduped_refs;
  std::vector<at::neuron::TensorContext> deduped_contexts;
  std::unordered_set<void*> seen;

  for (size_t i = 0; i < outputs.size(); i++) {
    void* addr = outputs[i];
    if (!addr || (last_idx[addr] == i && seen.insert(addr).second)) {
      deduped_outputs.push_back(addr);
      if (i < output_data_refs.size()) deduped_refs.push_back(output_data_refs[i]);
      if (i < output_contexts.size()) deduped_contexts.push_back(output_contexts[i]);
    }
  }

  return {deduped_outputs, deduped_refs, deduped_contexts};
}

}  // namespace

class OutputDeduplicationTest : public ::testing::Test {
 protected:
  // Use fixed addresses for deterministic testing
  void* addr_A = reinterpret_cast<void*>(0x1000);
  void* addr_B = reinterpret_cast<void*>(0x2000);
  void* addr_C = reinterpret_cast<void*>(0x3000);
};

TEST_F(OutputDeduplicationTest, NoDuplicatesUnchanged) {
  // outputs = [A, B, C] — all unique, no deduplication needed
  std::vector<void*> outputs = {addr_A, addr_B, addr_C};
  std::vector<at::neuron::TensorDataRef> refs = {at::neuron::TensorDataRef(addr_A),
                                                 at::neuron::TensorDataRef(addr_B),
                                                 at::neuron::TensorDataRef(addr_C)};
  std::vector<at::neuron::TensorContext> contexts(3);

  auto result = DeduplicateOutputs(outputs, refs, contexts);

  ASSERT_EQ(result.outputs.size(), 3u);
  EXPECT_EQ(result.outputs[0], addr_A);
  EXPECT_EQ(result.outputs[1], addr_B);
  EXPECT_EQ(result.outputs[2], addr_C);
}

TEST_F(OutputDeduplicationTest, DuplicateKeepsLastOccurrence) {
  // outputs = [A, B, A] — A appears at idx 0 and 2; keep idx 2 (last writer)
  std::vector<void*> outputs = {addr_A, addr_B, addr_A};
  std::vector<at::neuron::TensorDataRef> refs = {at::neuron::TensorDataRef(addr_A),
                                                 at::neuron::TensorDataRef(addr_B),
                                                 at::neuron::TensorDataRef(addr_A)};
  std::vector<at::neuron::TensorContext> contexts(3);

  auto result = DeduplicateOutputs(outputs, refs, contexts);

  // Should be [B, A] — B at idx 1 (unique), A at idx 2 (last occurrence)
  ASSERT_EQ(result.outputs.size(), 2u);
  EXPECT_EQ(result.outputs[0], addr_B);
  EXPECT_EQ(result.outputs[1], addr_A);
  // The ref for A should be from index 2 (the last writer)
  EXPECT_EQ(result.output_data_refs[1].ptr, addr_A);
}

TEST_F(OutputDeduplicationTest, TripleDuplicateKeepsLast) {
  // outputs = [A, A, A] — all same address, keep only the last
  std::vector<void*> outputs = {addr_A, addr_A, addr_A};
  std::vector<at::neuron::TensorDataRef> refs = {at::neuron::TensorDataRef(addr_A),
                                                 at::neuron::TensorDataRef(addr_A),
                                                 at::neuron::TensorDataRef(addr_A)};
  std::vector<at::neuron::TensorContext> contexts(3);

  auto result = DeduplicateOutputs(outputs, refs, contexts);

  ASSERT_EQ(result.outputs.size(), 1u);
  EXPECT_EQ(result.outputs[0], addr_A);
}

TEST_F(OutputDeduplicationTest, MultipleDuplicatesEachKeepsLast) {
  // outputs = [A, B, A, B] — A last at idx 2, B last at idx 3
  std::vector<void*> outputs = {addr_A, addr_B, addr_A, addr_B};
  std::vector<at::neuron::TensorDataRef> refs = {
      at::neuron::TensorDataRef(addr_A), at::neuron::TensorDataRef(addr_B),
      at::neuron::TensorDataRef(addr_A), at::neuron::TensorDataRef(addr_B)};
  std::vector<at::neuron::TensorContext> contexts(4);

  auto result = DeduplicateOutputs(outputs, refs, contexts);

  // Should be [A, B] — both from their last occurrence (idx 2 and 3)
  ASSERT_EQ(result.outputs.size(), 2u);
  EXPECT_EQ(result.outputs[0], addr_A);
  EXPECT_EQ(result.outputs[1], addr_B);
}

TEST_F(OutputDeduplicationTest, NullPtrsPreserved) {
  // outputs = [nullptr, A, nullptr, A] — nulls always kept, A deduped to last
  std::vector<void*> outputs = {nullptr, addr_A, nullptr, addr_A};
  std::vector<at::neuron::TensorDataRef> refs = {
      at::neuron::TensorDataRef(nullptr), at::neuron::TensorDataRef(addr_A),
      at::neuron::TensorDataRef(nullptr), at::neuron::TensorDataRef(addr_A)};
  std::vector<at::neuron::TensorContext> contexts(4);

  auto result = DeduplicateOutputs(outputs, refs, contexts);

  // nulls at idx 0 and 2 preserved, A from idx 3 (last)
  ASSERT_EQ(result.outputs.size(), 3u);
  EXPECT_EQ(result.outputs[0], nullptr);
  EXPECT_EQ(result.outputs[1], nullptr);
  EXPECT_EQ(result.outputs[2], addr_A);
}

TEST_F(OutputDeduplicationTest, InPlacePattern_MmAddMm) {
  // Simulates: mm(outputs=[A]) merged with add_(outputs=[A]) merged with mm2(outputs=[B])
  // If module1_skip fails, we get: outputs = [A, A, B]
  // Dedup should produce: [A, B] (A from idx 1, B from idx 2)
  std::vector<void*> outputs = {addr_A, addr_A, addr_B};
  std::vector<at::neuron::TensorDataRef> refs = {at::neuron::TensorDataRef(addr_A),
                                                 at::neuron::TensorDataRef(addr_A),
                                                 at::neuron::TensorDataRef(addr_B)};
  std::vector<at::neuron::TensorContext> contexts(3);

  auto result = DeduplicateOutputs(outputs, refs, contexts);

  ASSERT_EQ(result.outputs.size(), 2u);
  EXPECT_EQ(result.outputs[0], addr_A);  // Last A is at idx 1
  EXPECT_EQ(result.outputs[1], addr_B);
}

TEST_F(OutputDeduplicationTest, OrderPreservation) {
  // outputs = [C, A, B, A] — should become [C, B, A] preserving relative order of uniques
  // C at idx 0 (unique), B at idx 2 (unique), A at idx 3 (last occurrence)
  std::vector<void*> outputs = {addr_C, addr_A, addr_B, addr_A};
  std::vector<at::neuron::TensorDataRef> refs = {
      at::neuron::TensorDataRef(addr_C), at::neuron::TensorDataRef(addr_A),
      at::neuron::TensorDataRef(addr_B), at::neuron::TensorDataRef(addr_A)};
  std::vector<at::neuron::TensorContext> contexts(4);

  auto result = DeduplicateOutputs(outputs, refs, contexts);

  // C (idx 0, unique), B (idx 2, unique), A (idx 3, last occurrence)
  ASSERT_EQ(result.outputs.size(), 3u);
  EXPECT_EQ(result.outputs[0], addr_C);
  EXPECT_EQ(result.outputs[1], addr_B);
  EXPECT_EQ(result.outputs[2], addr_A);
}

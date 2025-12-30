#pragma once

#include <string>
#include <vector>

#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/concatenation/ConcatenationCore.h"

namespace at::neuron::testing {

/**
 * @brief Mock ConcatenationCore for testing ConcatenationEngine
 *
 * This mock allows tests to configure various concatenation behaviors:
 * - Success with concatenation of N operations
 * - Failure scenarios
 * - Partial processing
 * - No concatenation (pass-through)
 */
class MockConcatenationCore {
 public:
  enum class Behavior {
    SUCCESS_NO_CONCAT,   // Return all operations unchanged
    SUCCESS_CONCAT_2,    // Concatenate first 2 operations
    SUCCESS_CONCAT_ALL,  // Concatenate all operations into 1
    FAILURE,             // Return failure
    PARTIAL_CONCAT       // Concatenate some, leave others
  };

  MockConcatenationCore() : behavior_(Behavior::SUCCESS_NO_CONCAT), call_count_(0) {}

  // Configure behavior for the next call
  void SetBehavior(Behavior behavior) { behavior_ = behavior; }

  // Set custom behavior for partial concatenation
  void SetPartialConcatBehavior(size_t ops_to_concat, size_t ops_consumed) {
    behavior_ = Behavior::PARTIAL_CONCAT;
    partial_ops_to_concat_ = ops_to_concat;
    partial_ops_consumed_ = ops_consumed;
  }

  // Get call statistics
  size_t GetProcessCallCount() const { return call_count_; }
  void ResetCallCount() { call_count_ = 0; }

  // Mock implementation of processBufferedOperations
  ConcatenationCore::ConcatenationResult processBufferedOperations(
      const std::vector<OperationContext*>& buffered_ops) {
    call_count_++;

    ConcatenationCore::ConcatenationResult result;

    if (buffered_ops.empty()) {
      result.success = true;
      result.original_operations_consumed = 0;
      return result;
    }

    switch (behavior_) {
      case Behavior::SUCCESS_NO_CONCAT:
        // Return all operations unchanged (not concatenated)
        for (auto* op : buffered_ops) {
          ConcatenationCore::ConcatenationResult::ProcessedOperation proc_op;
          proc_op.operation = op;
          proc_op.cascading_ops = {};  // Empty = not concatenated
          result.processed_operations.push_back(proc_op);
        }
        result.original_operations_consumed = buffered_ops.size();
        result.success = true;
        break;

      case Behavior::SUCCESS_CONCAT_2:
        if (buffered_ops.size() >= 2) {
          // Simulate concatenating first 2 operations into 1
          ConcatenationCore::ConcatenationResult::ProcessedOperation concat_op;
          concat_op.operation = buffered_ops[0];  // Use first op as the concatenated operation
          concat_op.cascading_ops = {buffered_ops[0], buffered_ops[1]};  // Mark as concatenated
          result.processed_operations.push_back(concat_op);
          result.original_operations_consumed = 2;

          // Add remaining operations as non-concatenated
          for (size_t i = 2; i < buffered_ops.size(); ++i) {
            ConcatenationCore::ConcatenationResult::ProcessedOperation proc_op;
            proc_op.operation = buffered_ops[i];
            proc_op.cascading_ops = {};
            result.processed_operations.push_back(proc_op);
          }
        } else {
          // Not enough operations to concatenate
          for (auto* op : buffered_ops) {
            ConcatenationCore::ConcatenationResult::ProcessedOperation proc_op;
            proc_op.operation = op;
            proc_op.cascading_ops = {};
            result.processed_operations.push_back(proc_op);
          }
          result.original_operations_consumed = buffered_ops.size();
        }
        result.success = true;
        break;

      case Behavior::SUCCESS_CONCAT_ALL:
        if (!buffered_ops.empty()) {
          // Simulate concatenating all operations into 1
          ConcatenationCore::ConcatenationResult::ProcessedOperation concat_op;
          concat_op.operation = buffered_ops[0];   // Use first op as the concatenated operation
          concat_op.cascading_ops = buffered_ops;  // All ops are cascading
          result.processed_operations.push_back(concat_op);
          result.original_operations_consumed = buffered_ops.size();
        }
        result.success = true;
        break;

      case Behavior::FAILURE:
        result.success = false;
        result.error_message = "Mock concatenation failure";
        result.original_operations_consumed = 0;
        break;

      case Behavior::PARTIAL_CONCAT:
        // Custom partial concatenation
        if (buffered_ops.size() >= partial_ops_to_concat_) {
          // Create concatenated operation
          ConcatenationCore::ConcatenationResult::ProcessedOperation concat_op;
          concat_op.operation = buffered_ops[0];
          std::vector<OperationContext*> concat_ops;
          for (size_t i = 0; i < partial_ops_to_concat_; ++i) {
            concat_ops.push_back(buffered_ops[i]);
          }
          concat_op.cascading_ops = concat_ops;
          result.processed_operations.push_back(concat_op);
          result.original_operations_consumed = partial_ops_consumed_;
        } else {
          for (auto* op : buffered_ops) {
            ConcatenationCore::ConcatenationResult::ProcessedOperation proc_op;
            proc_op.operation = op;
            proc_op.cascading_ops = {};
            result.processed_operations.push_back(proc_op);
          }
          result.original_operations_consumed = buffered_ops.size();
        }
        result.success = true;
        break;
    }

    return result;
  }

  // Mock cache methods
  void clearCache() { cache_cleared_ = true; }
  size_t getCacheSize() const { return mock_cache_size_; }
  void invalidateCacheEntry(const std::string& key) { invalidated_keys_.push_back(key); }

  // Test helpers
  bool WasCacheCleared() const { return cache_cleared_; }
  const std::vector<std::string>& GetInvalidatedKeys() const { return invalidated_keys_; }
  void SetMockCacheSize(size_t size) { mock_cache_size_ = size; }

 private:
  Behavior behavior_;
  size_t call_count_;
  size_t partial_ops_to_concat_ = 0;
  size_t partial_ops_consumed_ = 0;

  // Test state tracking
  bool cache_cleared_ = false;
  size_t mock_cache_size_ = 0;
  std::vector<std::string> invalidated_keys_;
};

}  // namespace at::neuron::testing

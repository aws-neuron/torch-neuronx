#include <gtest/gtest.h>

#include <list>
#include <memory>
#include <string>
#include <vector>

#include "tests/csrc/mocks/MockStreamImpl.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"
#include "torch_neuronx/csrc/core/concatenation/ConcatenationCore.h"

using namespace torch_neuronx;

class ConcatenationCoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    core = ConcatenationCoreFactory::CreateInstance();
    stream_mock = std::make_unique<at::neuron::testing::MockStreamImpl>();
  }

  std::unique_ptr<ConcatenationCore> core;
  std::unique_ptr<at::neuron::testing::MockStreamImpl> stream_mock;

  std::shared_ptr<at::neuron::OperationContext> createDotOperation(
      std::string name = "dot_op", std::string cache = "dot_cache") {
    std::string dot_mlir = R"(
module @dot_example {
  func.func public @main(%arg0: tensor<4x8xf32>, %arg1: tensor<8xf32>) -> tensor<4xf32> {
    %0 = stablehlo.dot %arg0, %arg1 : (tensor<4x8xf32>, tensor<8xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)";
    std::vector<torch::Tensor> inputs = {torch::randn({4, 8}), torch::randn({8})};
    std::vector<torch::Tensor> outputs = {torch::randn({4})};
    std::vector<uint8_t> ir_data(dot_mlir.begin(), dot_mlir.end());
    std::vector<at::neuron::TensorDataRef> input_refs;
    std::vector<at::neuron::TensorDataRef> output_refs;
    for (auto& tensor : inputs) {
      input_refs.emplace_back(tensor.data_ptr());
    }
    for (auto& tensor : outputs) {
      output_refs.emplace_back(tensor.data_ptr());
    }

    auto kernel = std::make_unique<at::neuron::XLACompilableKernelExecution>(
        name, std::move(input_refs), std::move(output_refs),
        std::vector<at::neuron::TensorContext>{}, std::vector<at::neuron::TensorContext>{}, cache,
        ir_data, false, 0);
    auto op = std::make_shared<at::neuron::OperationContext>(std::move(kernel));
    // Set mock stream to avoid null stream validation failures
    op->stream = reinterpret_cast<at::neuron::StreamImpl*>(stream_mock.get());
    return op;
  }

  // Store tensors to keep them alive and ensure unique addresses
  std::vector<std::vector<torch::Tensor>> tensor_storage;

  std::shared_ptr<at::neuron::OperationContext> createAddOperation(
      std::string name = "add_op", std::string cache = "add_cache") {
    std::string add_mlir = R"(
module @add_example {
  func.func public @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)";
    std::vector<torch::Tensor> inputs = {torch::randn({4})};
    std::vector<torch::Tensor> outputs = {torch::randn({4})};

    // Store tensors to prevent address reuse
    tensor_storage.push_back(inputs);
    tensor_storage.push_back(outputs);

    std::vector<uint8_t> ir_data(add_mlir.begin(), add_mlir.end());
    std::vector<at::neuron::TensorDataRef> input_refs;
    std::vector<at::neuron::TensorDataRef> output_refs;
    for (auto& tensor : inputs) {
      input_refs.emplace_back(tensor.data_ptr());
    }
    for (auto& tensor : outputs) {
      output_refs.emplace_back(tensor.data_ptr());
    }

    auto kernel = std::make_unique<at::neuron::XLACompilableKernelExecution>(
        name, std::move(input_refs), std::move(output_refs),
        std::vector<at::neuron::TensorContext>{}, std::vector<at::neuron::TensorContext>{}, cache,
        ir_data, false, 0);
    auto op = std::make_shared<at::neuron::OperationContext>(std::move(kernel));
    // Set mock stream to avoid null stream validation failures
    op->stream = reinterpret_cast<at::neuron::StreamImpl*>(stream_mock.get());
    return op;
  }

  std::shared_ptr<at::neuron::OperationContext> createTransposeOperation(
      std::string name = "transpose_op", std::string cache = "transpose_cache") {
    std::string transpose_mlir = R"(
module @transpose_example {
  func.func public @main(%arg0: tensor<4x8xf32>) -> tensor<8x4xf32> {
    %0 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<4x8xf32>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x8xf32>) -> tensor<8x4xf32>
    return %2 : tensor<8x4xf32>
  }
}
)";
    std::vector<torch::Tensor> inputs = {torch::randn({4})};
    std::vector<torch::Tensor> outputs = {torch::randn({4})};
    std::vector<uint8_t> ir_data(transpose_mlir.begin(), transpose_mlir.end());
    std::vector<at::neuron::TensorDataRef> input_refs;
    std::vector<at::neuron::TensorDataRef> output_refs;
    for (auto& tensor : inputs) {
      input_refs.emplace_back(tensor.data_ptr());
    }
    for (auto& tensor : outputs) {
      output_refs.emplace_back(tensor.data_ptr());
    }
    auto kernel = std::make_unique<at::neuron::XLACompilableKernelExecution>(
        name, std::move(input_refs), std::move(output_refs),
        std::vector<at::neuron::TensorContext>{}, std::vector<at::neuron::TensorContext>{}, cache,
        ir_data, false, 0);
    auto op = std::make_shared<at::neuron::OperationContext>(std::move(kernel));
    // Set mock stream to avoid null stream validation failures
    op->stream = reinterpret_cast<at::neuron::StreamImpl*>(stream_mock.get());
    return op;
  }
};

TEST_F(ConcatenationCoreTest, FactoryCreateInstance) {
  auto core = ConcatenationCoreFactory::CreateInstance();
  ASSERT_NE(core, nullptr);

  auto strategies = core->GetConcatStrategies();
  EXPECT_FALSE(strategies.empty());
}

TEST_F(ConcatenationCoreTest, FactoryCreateInstanceWithCustomStrategies) {
  std::vector<std::unique_ptr<AbstractIrConcatStrategy>> custom_strategies;
  // Add mock strategies if available

  auto core = ConcatenationCoreFactory::CreateInstance(std::move(custom_strategies));
  ASSERT_NE(core, nullptr);
}

TEST_F(ConcatenationCoreTest, ProcessEmptyOperations) {
  std::list<at::neuron::OperationContext*> empty_ops;

  auto result = core->ProcessBufferedOperations(empty_ops);

  // Empty input should result in empty output
  EXPECT_TRUE(result.processed_operations.empty());
  EXPECT_EQ(result.original_operations_consumed, 0);
}

TEST_F(ConcatenationCoreTest, ProcessSingleOperation) {
  auto op = createAddOperation("single_op", "single_cache");
  std::list<at::neuron::OperationContext*> ops = {op.get()};

  auto result = core->ProcessBufferedOperations(ops);

  // Single operation can't be concatenated - should return the original operation
  EXPECT_EQ(result.processed_operations.size(), 1);
  EXPECT_EQ(result.original_operations_consumed, 1);
}

TEST_F(ConcatenationCoreTest, ProcessMultipleOperations) {
  std::list<std::shared_ptr<at::neuron::OperationContext>> ops;
  std::list<at::neuron::OperationContext*> op_contexts;

  // Create 5 INDEPENDENT operations (each with unique input/output buffers)
  // This avoids circular dependencies from buffer reuse
  for (int i = 0; i < 5; ++i) {
    auto op = createAddOperation("op_" + std::to_string(i), "cache_" + std::to_string(i));
    ops.push_back(op);
    op_contexts.push_back(op.get());
  }

  TORCH_NEURONX_DEBUG("Processing", ops.size(), "operations");
  for (const auto* op : op_contexts) {
    auto* xla_kernel =
        dynamic_cast<at::neuron::XLACompilableKernelExecution*>(op->kernel_execution.get());
    if (xla_kernel) {
      auto hlo = xla_kernel->GetHloBytes();
      std::string hlo_str(hlo.begin(), hlo.end());
      TORCH_NEURONX_DEBUG("Operation hlo:", hlo_str);
    }
  }
  auto result = core->ProcessBufferedOperations(op_contexts);

  TORCH_NEURONX_DEBUG("Result - processed:", result.processed_operations.size(),
                      ", consumed:", result.original_operations_consumed);

  // With ConcatenateAllStrategy and independent operations, expect all consumed
  EXPECT_EQ(result.original_operations_consumed, 5);
  // Result should contain operations (original or concatenated)
  EXPECT_FALSE(result.processed_operations.empty());
}

TEST_F(ConcatenationCoreTest, GetConcatStrategies) {
  auto strategies = core->GetConcatStrategies();

  EXPECT_FALSE(strategies.empty());
  for (auto* strategy : strategies) {
    EXPECT_NE(strategy, nullptr);
  }
}

TEST_F(ConcatenationCoreTest, ClearCache) {
  // Test cache clearing functionality
  core->ClearCache();

  size_t cache_size = core->GetCacheSize();
  EXPECT_EQ(cache_size, 0);
}

TEST_F(ConcatenationCoreTest, GetCacheSize) {
  size_t initial_size = core->GetCacheSize();
  EXPECT_EQ(initial_size, 0);

  // Process some operations to potentially populate cache
  auto op = createAddOperation("cache_test", "cache_test_key");
  std::list<at::neuron::OperationContext*> ops = {op.get()};

  core->ProcessBufferedOperations(ops);

  // Cache size should remain 0 based on current implementation
  EXPECT_EQ(core->GetCacheSize(), 0);
}

TEST_F(ConcatenationCoreTest, ProcessOperationsWithDifferentTypes) {
  std::list<at::neuron::OperationContext*> ops;

  // Create operations with different characteristics
  auto matmul_op = createDotOperation("matmul", "matmul_cache");
  auto transpose_op = createTransposeOperation("transpose", "transpose_cache");
  auto add_op = createAddOperation("add", "add_cache");

  ops.push_back(matmul_op.get());
  ops.push_back(transpose_op.get());
  ops.push_back(add_op.get());

  auto result = core->ProcessBufferedOperations(ops);

  EXPECT_FALSE(result.processed_operations.empty());
}

TEST_F(ConcatenationCoreTest, ProcessLargeNumberOfOperations) {
  std::list<at::neuron::OperationContext*> ops;
  std::list<std::shared_ptr<at::neuron::OperationContext>> op_contexts;

  // Create a large batch of INDEPENDENT operations (each with unique buffers)
  for (int i = 0; i < 100; ++i) {
    auto op =
        createAddOperation("batch_op_" + std::to_string(i), "batch_cache_" + std::to_string(i));
    op_contexts.push_back(op);
    ops.push_back(op.get());
  }

  auto result = core->ProcessBufferedOperations(ops);

  // All operations consumed
  EXPECT_EQ(result.original_operations_consumed, 100);
  // Result should have operations (original or concatenated)
  EXPECT_FALSE(result.processed_operations.empty());
}

TEST_F(ConcatenationCoreTest, ProcessOperationsResultStructure) {
  auto op = createAddOperation("result_test", "result_cache");
  std::list<at::neuron::OperationContext*> ops = {op.get()};

  auto result = core->ProcessBufferedOperations(ops);

  // Verify result structure
  EXPECT_EQ(result.original_operations_consumed, 1);
  EXPECT_EQ(result.processed_operations.size(), 1);

  // Verify processed operations are valid
  for (const auto* op : result.processed_operations) {
    EXPECT_NE(op, nullptr);
  }
}

TEST_F(ConcatenationCoreTest, SequentialProcessing) {
  // Test processing operations in sequence
  std::list<std::shared_ptr<at::neuron::OperationContext>> op_contexts;
  for (int batch = 0; batch < 3; ++batch) {
    auto op =
        createAddOperation("seq_op_" + std::to_string(batch), "seq_cache_" + std::to_string(batch));
    op_contexts.push_back(op);
    std::list<at::neuron::OperationContext*> ops = {op.get()};

    auto result = core->ProcessBufferedOperations(ops);
    EXPECT_EQ(result.processed_operations.size(), 1);
    EXPECT_EQ(result.original_operations_consumed, 1);
  }
}

TEST_F(ConcatenationCoreTest, ProcessDotAndAddOperationsConcatenation) {
  TORCH_NEURONX_DEBUG("Starting ProcessDotAndAddOperationsConcatenation test");

  std::list<at::neuron::OperationContext*> ops;

  auto add_op = createAddOperation();
  auto dot_op = createDotOperation();
  TORCH_NEURONX_DEBUG("Created dot and add operations");

  ops.push_back(add_op.get());
  ops.push_back(dot_op.get());
  TORCH_NEURONX_DEBUG("Added", ops.size(), "operations to list");

  auto result = core->ProcessBufferedOperations(ops);

  TORCH_NEURONX_DEBUG("Processed operations count:", result.processed_operations.size());
  TORCH_NEURONX_DEBUG("Original operations consumed:", result.original_operations_consumed);

  EXPECT_FALSE(result.processed_operations.empty());

  // Check if any operation has concatenated_operation containing both dot and add
  bool found_concatenated = false;
  int op_index = 0;
  for (const auto* op : result.processed_operations) {
    // Check the concatenated_operation field for concatenated MLIR
    if (op->HasConcatenatedOperation()) {
      auto* concat_op = op->GetConcatenatedOperation();
      auto* kernel = concat_op->kernel_execution.get();
      auto* xla_kernel = dynamic_cast<at::neuron::XLACompilableKernelExecution*>(kernel);
      if (xla_kernel) {
        auto hlo = xla_kernel->GetHloBytes();
        std::string ir_str(hlo.begin(), hlo.end());
        TORCH_NEURONX_DEBUG("Operation", op_index, "concatenated IR content:", ir_str);

        bool has_dot = ir_str.find("stablehlo.dot") != std::string::npos;
        bool has_add = ir_str.find("stablehlo.add") != std::string::npos;

        TORCH_NEURONX_DEBUG("Operation", op_index, "- has dot:", has_dot, ", has add:", has_add);

        if (has_dot && has_add) {
          found_concatenated = true;
          TORCH_NEURONX_DEBUG("Found concatenated operation at index", op_index);
          break;
        }
      } else {
        TORCH_NEURONX_DEBUG("Operation", op_index, "concatenated_operation has null kernel");
        FAIL() << "Operation " << op_index << " has concatenated_operation with null kernel";
      }
    } else {
      TORCH_NEURONX_DEBUG("Operation", op_index, "has no concatenated_operation");
      FAIL() << "Operation " << op_index << " has no concatenated_operation - concatenation failed";
    }
    op_index++;
  }

  TORCH_NEURONX_DEBUG("Found concatenated operation:", found_concatenated);
  EXPECT_TRUE(found_concatenated) << "Expected concatenated operation to contain both dot and add";
}

// Test that verifies the ownership model between ConcatenationState and operations:
// - Cascading ops use shared_ptr (concatenation_state_) to keep state alive
// - Concatenated op uses raw pointer (concatenation_state_raw_) to avoid circular reference
TEST_F(ConcatenationCoreTest, ConcatenationStateOwnershipModel) {
  std::list<at::neuron::OperationContext*> ops;
  std::list<std::shared_ptr<at::neuron::OperationContext>> op_contexts;

  // Create 2 operations that will be concatenated
  auto add_op1 = createAddOperation("add_1", "add_cache_1");
  auto add_op2 = createAddOperation("add_2", "add_cache_2");
  op_contexts.push_back(add_op1);
  op_contexts.push_back(add_op2);
  ops.push_back(add_op1.get());
  ops.push_back(add_op2.get());

  auto result = core->ProcessBufferedOperations(ops);

  EXPECT_EQ(result.processed_operations.size(), 2);

  // Verify both cascading operations have concatenation state
  auto* cascading_op1 = result.processed_operations.front();
  auto* cascading_op2 = result.processed_operations.back();

  ASSERT_TRUE(cascading_op1->HasConcatenatedOperation());
  ASSERT_TRUE(cascading_op2->HasConcatenatedOperation());

  // Verify cascading ops use shared_ptr (concatenation_state_), not raw pointer
  EXPECT_NE(cascading_op1->concatenation_state_, nullptr)
      << "Cascading op should use shared_ptr concatenation_state_";
  EXPECT_NE(cascading_op2->concatenation_state_, nullptr)
      << "Cascading op should use shared_ptr concatenation_state_";
  EXPECT_EQ(cascading_op1->concatenation_state_raw_, nullptr)
      << "Cascading op should NOT use raw pointer concatenation_state_raw_";
  EXPECT_EQ(cascading_op2->concatenation_state_raw_, nullptr)
      << "Cascading op should NOT use raw pointer concatenation_state_raw_";

  // Both cascading ops should share the same ConcatenationState
  EXPECT_EQ(cascading_op1->concatenation_state_.get(), cascading_op2->concatenation_state_.get())
      << "Both cascading ops should share the same ConcatenationState";

  // Get the concatenated operation
  auto* concat_op = cascading_op1->GetConcatenatedOperation();
  ASSERT_NE(concat_op, nullptr);

  // Verify concatenated op uses raw pointer (concatenation_state_raw_), not shared_ptr
  EXPECT_EQ(concat_op->concatenation_state_, nullptr)
      << "Concatenated op should NOT use shared_ptr to avoid circular reference";
  EXPECT_NE(concat_op->concatenation_state_raw_, nullptr)
      << "Concatenated op should use raw pointer concatenation_state_raw_";

  // Verify concatenated op's raw pointer points to the same state
  EXPECT_EQ(concat_op->concatenation_state_raw_, cascading_op1->concatenation_state_.get())
      << "Concatenated op's raw pointer should point to the same ConcatenationState";

  // Verify IsConcatenatedOperation() works correctly
  EXPECT_TRUE(concat_op->IsConcatenatedOperation())
      << "Concatenated op should return true for IsConcatenatedOperation()";
  EXPECT_FALSE(cascading_op1->IsConcatenatedOperation())
      << "Cascading op should return false for IsConcatenatedOperation()";
  EXPECT_FALSE(cascading_op2->IsConcatenatedOperation())
      << "Cascading op should return false for IsConcatenatedOperation()";

  // Verify GetConcatenationState() returns the correct state for all ops
  EXPECT_EQ(cascading_op1->GetConcatenationState(), cascading_op1->concatenation_state_.get());
  EXPECT_EQ(cascading_op2->GetConcatenationState(), cascading_op2->concatenation_state_.get());
  EXPECT_EQ(concat_op->GetConcatenationState(), concat_op->concatenation_state_raw_);
  EXPECT_EQ(cascading_op1->GetConcatenationState(), concat_op->GetConcatenationState())
      << "All ops in the group should return the same ConcatenationState";
}

// =============================================================================
// TensorDataRef Validation Tests
// =============================================================================
// These tests verify the validation logic in CreateOpContextFromIrNode that
// ensures TensorDataRef is properly populated for concatenated operations.
// =============================================================================

// Test that verifies TensorDataRef validation: throws when outputs exist but output_data_refs is
// empty
TEST_F(ConcatenationCoreTest, CreateOpContextFromIrNode_ThrowsWhenOutputDataRefsMissing) {
  // Create an IrNode with outputs but no output_data_refs
  std::vector<void*> inputs = {reinterpret_cast<void*>(0x1000)};
  std::vector<void*> outputs = {reinterpret_cast<void*>(0x2000)};
  std::vector<uint8_t> ir_data = {};

  auto ir_node = std::make_unique<IrNode>("test_op", "test_cache", ir_data, IrNodeType::STABLEHLO,
                                          std::move(inputs), std::move(outputs), false);

  // Add input_data_refs but NOT output_data_refs
  ir_node->input_data_refs.emplace_back(reinterpret_cast<void*>(0x1000));
  // output_data_refs intentionally left empty

  // Should throw because outputs exist but output_data_refs is empty
  EXPECT_THROW(
      {
        try {
          core->CreateOpContextFromIrNode(ir_node.get(), 0);
        } catch (const std::runtime_error& e) {
          EXPECT_TRUE(std::string(e.what()).find("output_data_refs") != std::string::npos)
              << "Error message should mention output_data_refs";
          throw;
        }
      },
      std::runtime_error);
}

// Test that verifies TensorDataRef validation: throws when inputs exist but input_data_refs is
// empty
TEST_F(ConcatenationCoreTest, CreateOpContextFromIrNode_ThrowsWhenInputDataRefsMissing) {
  // Create an IrNode with inputs but no input_data_refs
  std::vector<void*> inputs = {reinterpret_cast<void*>(0x1000)};
  std::vector<void*> outputs = {reinterpret_cast<void*>(0x2000)};
  std::vector<uint8_t> ir_data = {};

  auto ir_node = std::make_unique<IrNode>("test_op", "test_cache", ir_data, IrNodeType::STABLEHLO,
                                          std::move(inputs), std::move(outputs), false);

  // Add output_data_refs but NOT input_data_refs
  ir_node->output_data_refs.emplace_back(reinterpret_cast<void*>(0x2000));
  // input_data_refs intentionally left empty

  // Should throw because inputs exist but input_data_refs is empty
  EXPECT_THROW(
      {
        try {
          core->CreateOpContextFromIrNode(ir_node.get(), 0);
        } catch (const std::runtime_error& e) {
          EXPECT_TRUE(std::string(e.what()).find("input_data_refs") != std::string::npos)
              << "Error message should mention input_data_refs";
          throw;
        }
      },
      std::runtime_error);
}

// Test that verifies TensorDataRef validation: throws when sizes mismatch
TEST_F(ConcatenationCoreTest, CreateOpContextFromIrNode_ThrowsOnSizeMismatch) {
  // Create an IrNode with mismatched output_data_refs size
  std::vector<void*> inputs = {reinterpret_cast<void*>(0x1000)};
  std::vector<void*> outputs = {reinterpret_cast<void*>(0x2000), reinterpret_cast<void*>(0x3000)};
  std::vector<uint8_t> ir_data = {};

  auto ir_node = std::make_unique<IrNode>("test_op", "test_cache", ir_data, IrNodeType::STABLEHLO,
                                          std::move(inputs), std::move(outputs), false);

  // Add data refs but with wrong size
  ir_node->input_data_refs.emplace_back(reinterpret_cast<void*>(0x1000));
  ir_node->output_data_refs.emplace_back(reinterpret_cast<void*>(0x2000));  // Only 1, but 2 outputs

  // Should throw because output_data_refs size (1) != outputs size (2)
  EXPECT_THROW(
      {
        try {
          core->CreateOpContextFromIrNode(ir_node.get(), 0);
        } catch (const std::runtime_error& e) {
          std::string err_msg = e.what();
          EXPECT_TRUE(err_msg.find("size") != std::string::npos ||
                      err_msg.find("mismatch") != std::string::npos)
              << "Error message should mention size mismatch: " << err_msg;
          throw;
        }
      },
      std::runtime_error);
}

// Test that verifies TensorDataRef validation passes when data refs are properly populated
TEST_F(ConcatenationCoreTest, CreateOpContextFromIrNode_SucceedsWithProperDataRefs) {
  // Create an IrNode with properly matched data refs
  std::vector<void*> inputs = {reinterpret_cast<void*>(0x1000)};
  std::vector<void*> outputs = {reinterpret_cast<void*>(0x2000)};

  // Create a minimal valid MLIR for the operation
  std::string mlir = R"(
module @test {
  func.func public @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
}
)";
  std::vector<uint8_t> ir_data(mlir.begin(), mlir.end());

  auto ir_node = std::make_unique<IrNode>("test_op", "test_cache", ir_data, IrNodeType::STABLEHLO,
                                          std::move(inputs), std::move(outputs), false);

  // Add properly matched data refs
  ir_node->input_data_refs.emplace_back(reinterpret_cast<void*>(0x1000));
  ir_node->output_data_refs.emplace_back(reinterpret_cast<void*>(0x2000));

  // Should NOT throw when data refs are properly populated
  EXPECT_NO_THROW(core->CreateOpContextFromIrNode(ir_node.get(), 0));
}

// Test that verifies TensorDataRef validation with empty inputs/outputs succeeds
TEST_F(ConcatenationCoreTest, CreateOpContextFromIrNode_SucceedsWithEmptyInputsOutputs) {
  // Create an IrNode with no inputs or outputs (edge case)
  std::vector<void*> inputs = {};
  std::vector<void*> outputs = {};

  std::string mlir = R"(
module @test {
  func.func public @main() -> () {
    return
  }
}
)";
  std::vector<uint8_t> ir_data(mlir.begin(), mlir.end());

  auto ir_node = std::make_unique<IrNode>("test_op", "test_cache", ir_data, IrNodeType::STABLEHLO,
                                          std::move(inputs), std::move(outputs), false);

  // No data refs needed for empty inputs/outputs

  // Should NOT throw with empty inputs/outputs
  EXPECT_NO_THROW(core->CreateOpContextFromIrNode(ir_node.get(), 0));
}

TEST_F(ConcatenationCoreTest, ProcessTenAddsFollowedByOneMatmul) {
  std::list<at::neuron::OperationContext*> ops;
  std::list<std::shared_ptr<at::neuron::OperationContext>> op_contexts;

  // Create 10 add operations
  for (int i = 0; i < 10; ++i) {
    auto op = createAddOperation("add_" + std::to_string(i), "add_cache_" + std::to_string(i));
    op_contexts.push_back(op);
    ops.push_back(op.get());
  }

  // Add 1 matmul operation
  auto matmul_op = createDotOperation("matmul", "matmul_cache");
  op_contexts.push_back(matmul_op);
  ops.push_back(matmul_op.get());

  auto result = core->ProcessBufferedOperations(ops);

  // Check that the result contains concatenated IR with 10 adds and 1 matmul
  EXPECT_EQ(result.original_operations_consumed, 11);
  EXPECT_EQ(result.processed_operations.size(), 11);

  // Get the concatenated operation from the first processed operation
  auto* first_op = result.processed_operations.front();
  ASSERT_TRUE(first_op->HasConcatenatedOperation());
  auto* concat_op = first_op->GetConcatenatedOperation();
  auto* xla_kernel =
      dynamic_cast<at::neuron::XLACompilableKernelExecution*>(concat_op->kernel_execution.get());
  auto hlo = xla_kernel->GetHloBytes();
  std::string hlo_str(hlo.begin(), hlo.end());

  // Count occurrences of add and dot operations in the IR
  size_t add_count = 0;
  size_t dot_count = 0;
  size_t pos = 0;

  while ((pos = hlo_str.find("stablehlo.add", pos)) != std::string::npos) {
    add_count++;
    pos += 13;
  }

  pos = 0;
  while ((pos = hlo_str.find("stablehlo.dot", pos)) != std::string::npos) {
    dot_count++;
    pos += 13;
  }

  EXPECT_EQ(add_count, 10) << "Expected 10 add operations in concatenated IR";
  EXPECT_EQ(dot_count, 1) << "Expected 1 dot operation in concatenated IR";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

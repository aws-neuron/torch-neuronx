#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/concatenation/IrNode.h"
#include "torch_neuronx/csrc/core/concatenation/OpConcatUtils.h"

using namespace torch_neuronx;

class IrNodeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup common test data
  }

  std::vector<uint8_t> createDummyIr() { return {0x01, 0x02, 0x03, 0x04}; }

  void* createPtr() {
    static int counter = 0;
    return reinterpret_cast<void*>(++counter);
  }

  std::vector<uint8_t> createValidMLIR() {
    std::string mlir = R"(
module @main {
  func.func public @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}
)";
    return std::vector<uint8_t>(mlir.begin(), mlir.end());
  }
};

TEST_F(IrNodeTest, IrNodeBasicConstructor) {
  auto ir = createDummyIr();
  std::vector<void*> inputs = {createPtr(), createPtr()};
  std::vector<void*> outputs = {createPtr()};

  IrNode node("test_op", "test_cache_key", std::move(ir), IrNodeType::STABLEHLO, std::move(inputs),
              std::move(outputs), false);

  EXPECT_EQ(node.op_name, "test_op");
  EXPECT_EQ(node.cache_key, "test_cache_key");
  EXPECT_EQ(node.ir_type, IrNodeType::STABLEHLO);
  EXPECT_EQ(node.inputs.size(), 2);
  EXPECT_EQ(node.outputs.size(), 1);
  EXPECT_EQ(node.status, IrNodeStatus::CREATED);
  EXPECT_FALSE(node.has_collectives);
}

TEST_F(IrNodeTest, StableHloNodeConstructorWithValidIr) {
  auto ir = createValidMLIR();
  std::vector<void*> inputs = {createPtr()};
  std::vector<void*> outputs = {createPtr()};

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
  StableHloNode node("mlir_op", "mlir_cache_key", std::move(ir), std::move(inputs),
                     std::move(outputs), false, &context);

  EXPECT_EQ(node.op_name, "mlir_op");
  EXPECT_EQ(node.cache_key, "mlir_cache_key");
  EXPECT_NE(node.module.get(), nullptr);
}

TEST_F(IrNodeTest, StableHloNodeModuleConstructor) {
  // Create MLIR context and module
  auto context = std::make_unique<mlir::MLIRContext>();
  context->getOrLoadDialect<mlir::func::FuncDialect>();
  context->getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto mlir = createValidMLIR();
  auto mlir_string = std::string(mlir.begin(), mlir.end());
  auto module = std::make_shared<mlir::OwningOpRef<mlir::ModuleOp>>(
      std::move(mlir::parseSourceString<mlir::ModuleOp>(mlir_string, context.get())));
  ASSERT_TRUE(module);

  std::vector<void*> inputs = {createPtr()};
  std::vector<void*> outputs = {createPtr()};

  StableHloNode node("module_op", "module_cache_key", module, std::move(inputs), std::move(outputs),
                     true);

  EXPECT_EQ(node.op_name, "module_op");
  EXPECT_EQ(node.cache_key, "module_cache_key");
  EXPECT_EQ(node.ir_type, IrNodeType::STABLEHLO);
  EXPECT_TRUE(node.has_collectives);
  EXPECT_NE(node.module.get(), nullptr);

  // Verify ir_serialized was populated from module
  EXPECT_FALSE(node.ir_serialized.empty());
}

TEST_F(IrNodeTest, IrNodeStatusEnum) {
  auto ir = createDummyIr();
  std::vector<void*> inputs = {createPtr()};
  std::vector<void*> outputs = {createPtr()};

  IrNode node("test_op", "test_cache_key", std::move(ir), IrNodeType::STABLEHLO, std::move(inputs),
              std::move(outputs), false);

  // Test initial status
  EXPECT_EQ(node.status, IrNodeStatus::CREATED);

  // Test status transitions
  node.status = IrNodeStatus::SUBMITTED_FOR_EXECUTION;
  EXPECT_EQ(node.status, IrNodeStatus::SUBMITTED_FOR_EXECUTION);

  node.status = IrNodeStatus::EXECUTING;
  EXPECT_EQ(node.status, IrNodeStatus::EXECUTING);

  node.status = IrNodeStatus::EXECUTED;
  EXPECT_EQ(node.status, IrNodeStatus::EXECUTED);

  node.status = IrNodeStatus::FAILED;
  EXPECT_EQ(node.status, IrNodeStatus::FAILED);
}

TEST_F(IrNodeTest, IrNodeTypeEnum) {
  auto ir = createValidMLIR();
  std::vector<void*> inputs = {createPtr()};
  std::vector<void*> outputs = {createPtr()};

  // Test STABLEHLO type
  IrNode stablehlo_node("op", "key", std::move(ir), IrNodeType::STABLEHLO,
                        std::vector<void*>(inputs), std::vector<void*>(outputs), false);
  EXPECT_EQ(stablehlo_node.ir_type, IrNodeType::STABLEHLO);

  // Test HLO type
  IrNode hlo_node("op", "key", std::move(ir), IrNodeType::HLO, std::vector<void*>(inputs),
                  std::vector<void*>(outputs), false);
  EXPECT_EQ(hlo_node.ir_type, IrNodeType::HLO);

  // Test OTHERS type
  IrNode other_node("op", "key", std::move(ir), IrNodeType::OTHERS, std::vector<void*>(inputs),
                    std::vector<void*>(outputs), false);
  EXPECT_EQ(other_node.ir_type, IrNodeType::OTHERS);
}

TEST_F(IrNodeTest, StableHloNodeEmptyInputsOutputs) {
  auto ir = createValidMLIR();
  std::vector<void*> empty_inputs;
  std::vector<void*> empty_outputs;

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
  StableHloNode node("empty_op", "empty_cache_key", std::move(ir), std::move(empty_inputs),
                     std::move(empty_outputs), false, &context);

  EXPECT_TRUE(node.inputs.empty());
  EXPECT_TRUE(node.outputs.empty());
  EXPECT_EQ(node.ir_type, IrNodeType::STABLEHLO);
}

TEST_F(IrNodeTest, StableHloNodeLargeInputsOutputs) {
  auto ir = createValidMLIR();
  std::vector<void*> large_inputs;
  std::vector<void*> large_outputs;

  // Create 100 inputs and outputs
  for (int i = 0; i < 100; ++i) {
    large_inputs.push_back(createPtr());
    large_outputs.push_back(createPtr());
  }

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
  StableHloNode node("large_op", "large_cache_key", std::move(ir), std::move(large_inputs),
                     std::move(large_outputs), false, &context);

  EXPECT_EQ(node.inputs.size(), 100);
  EXPECT_EQ(node.outputs.size(), 100);
}

TEST_F(IrNodeTest, StableHloNodeInvalidMLIR) {
  std::string invalid_mlir = "invalid mlir syntax";
  std::vector<uint8_t> ir(invalid_mlir.begin(), invalid_mlir.end());
  std::vector<void*> inputs = {createPtr()};
  std::vector<void*> outputs = {createPtr()};

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
  // Constructor should throw exception for invalid MLIR
  EXPECT_THROW(StableHloNode node("invalid_op", "invalid_cache_key", std::move(ir),
                                  std::move(inputs), std::move(outputs), false, &context),
               std::exception);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

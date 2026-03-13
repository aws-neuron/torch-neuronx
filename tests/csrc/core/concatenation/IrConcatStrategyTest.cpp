#include <gtest/gtest.h>

#include <memory>

#include "torch_neuronx/csrc/core/concatenation/IrConcatStrategy.h"

using namespace torch_neuronx;

class MatMulToMatMulStrategyTest : public ::testing::Test {
 protected:
  void SetUp() override { strategy = std::make_unique<MatMulToMatMulStrategy>(); }

  std::unique_ptr<MatMulToMatMulStrategy> strategy;
};

TEST_F(MatMulToMatMulStrategyTest, MatMulOperations) {
  EXPECT_TRUE(strategy->IsFusibleBoundaryOperation("aten::linear"));
  EXPECT_TRUE(strategy->IsFusibleBoundaryOperation("aten::linear_backward"));
  EXPECT_TRUE(strategy->IsFusibleBoundaryOperation("aten::matmul"));
  EXPECT_TRUE(strategy->IsFusibleBoundaryOperation("aten::matmul_backward"));
  EXPECT_TRUE(strategy->IsFusibleBoundaryOperation("aten::mm"));
  EXPECT_TRUE(strategy->IsFusibleBoundaryOperation("aten::bmm"));
  EXPECT_TRUE(strategy->IsFusibleBoundaryOperation("aten::addmm"));
  EXPECT_TRUE(strategy->IsFusibleBoundaryOperation("aten::baddbmm"));
  EXPECT_TRUE(strategy->IsFusibleBoundaryOperation("aten::addbmm"));
  EXPECT_TRUE(strategy->IsFusibleBoundaryOperation("aten::conv2d"));
}

TEST_F(MatMulToMatMulStrategyTest, NkiKernelOperations) {
  EXPECT_TRUE(strategy->IsFusibleBoundaryOperation("nki_kernel_global"));
}

TEST_F(MatMulToMatMulStrategyTest, NonMatMulOperations) {
  EXPECT_FALSE(strategy->IsFusibleBoundaryOperation("aten::add"));
  EXPECT_FALSE(strategy->IsFusibleBoundaryOperation("aten::relu"));
  EXPECT_FALSE(strategy->IsFusibleBoundaryOperation("stablehlo.transpose"));
  EXPECT_FALSE(strategy->IsFusibleBoundaryOperation("unknown_op"));
}

TEST_F(MatMulToMatMulStrategyTest, EmptyString) {
  EXPECT_FALSE(strategy->IsFusibleBoundaryOperation(""));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

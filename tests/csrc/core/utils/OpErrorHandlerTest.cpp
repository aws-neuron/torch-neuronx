#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "tests/csrc/mocks/MockKernelExecution.h"
#include "torch_neuronx/csrc/core/utils/CPUFallbackContext.h"
#include "torch_neuronx/csrc/core/utils/OpErrorHandler.h"
#include "torch_neuronx/csrc/core/utils/TensorContext.h"

using namespace at::neuron;
namespace mock = at::neuron::testing;

namespace {

// Build a minimal OperationContext with the given op name. The cpu_fallback
// context starts empty; tests populate input_metadata/tensor_metadata directly.
std::unique_ptr<OperationContext> MakeOp(const std::string& op_name) {
  auto kernel = std::make_unique<mock::MockNeuronKernelExecution>(op_name);
  EXPECT_CALL(*kernel, RequiresCompilation()).WillRepeatedly(::testing::Return(false));
  EXPECT_CALL(*kernel, ValidateImpl()).WillRepeatedly(::testing::Return(true));
  return std::make_unique<OperationContext>(std::move(kernel), "stack");
}

}  // namespace

class OpErrorHandlerTest : public ::testing::Test {
 protected:
  // The invoker is a process-global; clear it before and after each test so
  // tests don't leak state into one another.
  void SetUp() override { OpErrorHandler::RegisterInvoker({}); }
  void TearDown() override { OpErrorHandler::RegisterInvoker({}); }
};

// --- OpErrorHandler::SurfaceException plumbing -------------------------------

TEST_F(OpErrorHandlerTest, SurfaceExceptionNullOpIsNoOp) {
  bool called = false;
  OpErrorHandler::RegisterInvoker([&](const std::string&, OpErrorContext&) { called = true; });
  EXPECT_NO_THROW(OpErrorHandler::SurfaceException(nullptr));
  EXPECT_FALSE(called);
}

TEST_F(OpErrorHandlerTest, SurfaceExceptionNoInvokerIsNoOp) {
  // No invoker registered (cleared in SetUp): even a fault-carrying op is a no-op.
  auto op = MakeOp("aten::scatter");
  op->cpu_fallback_context.input_metadata.emplace_back(TensorMarker{});
  EXPECT_NO_THROW(OpErrorHandler::SurfaceException(op.get()));
}

TEST_F(OpErrorHandlerTest, SurfaceExceptionEmptyContextDoesNotInvoke) {
  bool called = false;
  OpErrorHandler::RegisterInvoker([&](const std::string&, OpErrorContext&) { called = true; });
  auto op = MakeOp("aten::scatter");  // cpu_fallback_context left empty
  OpErrorHandler::SurfaceException(op.get());
  EXPECT_FALSE(called);
}

TEST_F(OpErrorHandlerTest, SurfaceExceptionInvokesHandlerWithOpName) {
  std::string seen;
  OpErrorHandler::RegisterInvoker([&](const std::string& name, OpErrorContext&) { seen = name; });
  auto op = MakeOp("aten::_index_put_impl_");
  op->cpu_fallback_context.input_metadata.emplace_back(TensorMarker{});
  OpErrorHandler::SurfaceException(op.get());
  EXPECT_EQ(seen, "aten::_index_put_impl_");
}

TEST_F(OpErrorHandlerTest, SurfaceExceptionPropagatesHandlerException) {
  OpErrorHandler::RegisterInvoker([](const std::string&, OpErrorContext&) {
    throw std::runtime_error("index 10 is out of bounds for dimension 0 with size 5");
  });
  auto op = MakeOp("aten::scatter");
  op->cpu_fallback_context.input_metadata.emplace_back(TensorMarker{});
  try {
    OpErrorHandler::SurfaceException(op.get());
    FAIL() << "expected the handler's exception to propagate";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ(e.what(), "index 10 is out of bounds for dimension 0 with size 5");
  }
}

// --- OpErrorContext accessors (metadata only, no device transfer) ------------

TEST_F(OpErrorHandlerTest, OpErrorContextShapeScalarAndTensorOffset) {
  auto op = MakeOp("aten::scatter");
  auto& c = op->cpu_fallback_context;
  // scatter-like positional layout: (self: tensor, dim: scalar, index: tensor)
  c.input_metadata.emplace_back(TensorMarker{});
  c.input_metadata.emplace_back(ScalarValue{c10::IValue(static_cast<int64_t>(2))});
  c.input_metadata.emplace_back(TensorMarker{});
  // tensor_metadata holds only the tensors, in order: self, index
  c.tensor_metadata.push_back(TensorContext::FromTensor(torch::ones({4, 5})));
  c.tensor_metadata.push_back(TensorContext::FromTensor(torch::ones({6, 7})));

  OpErrorContext ec(op.get());
  EXPECT_EQ(ec.Shape(0), (std::vector<int64_t>{4, 5}));  // self
  EXPECT_EQ(ec.Scalar(1).toInt(), 2);                    // dim
  // arg 2 (index) maps to tensor slot 1 — the scalar arg consumes no tensor slot.
  EXPECT_EQ(ec.Shape(2), (std::vector<int64_t>{6, 7}));
}

TEST_F(OpErrorHandlerTest, OpErrorContextTypeMismatchThrows) {
  auto op = MakeOp("aten::scatter");
  auto& c = op->cpu_fallback_context;
  c.input_metadata.emplace_back(TensorMarker{});
  c.input_metadata.emplace_back(ScalarValue{c10::IValue(static_cast<int64_t>(0))});
  c.tensor_metadata.push_back(TensorContext::FromTensor(torch::ones({3})));

  OpErrorContext ec(op.get());
  EXPECT_THROW(ec.Shape(1), c10::Error);   // arg 1 is a scalar, not a tensor
  EXPECT_THROW(ec.Scalar(0), c10::Error);  // arg 0 is a tensor, not a scalar
  EXPECT_THROW(ec.Tensor(1), c10::Error);  // arg 1 is a scalar, not a tensor
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

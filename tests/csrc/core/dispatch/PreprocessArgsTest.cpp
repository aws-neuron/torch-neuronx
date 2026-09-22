/**
 * Unit tests for PreprocessArgs.
 *
 * Only tests that don't require a Neuron device are included here.
 * Tests involving scalar→tensor conversion or device transfer require
 * PrivateUse1 device availability and belong in integration tests.
 */

#include <gtest/gtest.h>

#include "torch_neuronx/csrc/core/dispatch/PreprocessArgs.h"

namespace torch_neuronx {
namespace dispatch {
namespace {

static c10::FunctionSchema AddSchema() {
  return c10::Dispatcher::singleton().findSchemaOrThrow("aten::add", "Tensor").schema();
}

TEST(PreprocessArgsTest, StaticArgnumsSkipped) {
  // For aten::add, arg[2] is the alpha scalar.
  // With static_argnums={2}, it should remain an int IValue.
  std::vector<c10::IValue> args = {torch::zeros({2}), torch::ones({2}), c10::IValue(1)};
  std::unordered_set<size_t> static_args = {2};

  PreprocessArgs(args, AddSchema(), static_args, /*skip_contiguous=*/false);

  // Alpha at index 2 should still be an int, not converted to tensor
  EXPECT_TRUE(args[2].isInt());
  EXPECT_EQ(args[2].toInt(), 1);
}

}  // namespace
}  // namespace dispatch
}  // namespace torch_neuronx

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <torch/torch.h>

#include "torch_neuronx/csrc/core/utils/PythonTypeConverter.h"

namespace py = pybind11;

class PythonTypeConverterTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // Initialize Python interpreter once for all tests
    py::initialize_interpreter();
  }

  static void TearDownTestSuite() {
    // Finalize Python interpreter after all tests
    py::finalize_interpreter();
  }
};

TEST_F(PythonTypeConverterTest, ConvertNone) {
  py::list py_list;
  py_list.append(py::none());

  auto result = at::neuron::utils::ConvertStaticArgsToIValues(py_list);

  ASSERT_EQ(result.size(), 1);
  EXPECT_TRUE(result[0].isNone());
}

TEST_F(PythonTypeConverterTest, ConvertBool) {
  py::list py_list;
  py_list.append(py::bool_(true));
  py_list.append(py::bool_(false));

  auto result = at::neuron::utils::ConvertStaticArgsToIValues(py_list);

  ASSERT_EQ(result.size(), 2);
  EXPECT_TRUE(result[0].isBool());
  EXPECT_EQ(result[0].toBool(), true);
  EXPECT_TRUE(result[1].isBool());
  EXPECT_EQ(result[1].toBool(), false);
}

TEST_F(PythonTypeConverterTest, ConvertInt) {
  py::list py_list;
  py_list.append(py::int_(42));
  py_list.append(py::int_(-100));

  auto result = at::neuron::utils::ConvertStaticArgsToIValues(py_list);

  ASSERT_EQ(result.size(), 2);
  EXPECT_TRUE(result[0].isInt());
  EXPECT_EQ(result[0].toInt(), 42);
  EXPECT_TRUE(result[1].isInt());
  EXPECT_EQ(result[1].toInt(), -100);
}

TEST_F(PythonTypeConverterTest, ConvertFloat) {
  py::list py_list;
  py_list.append(py::float_(3.14));
  py_list.append(py::float_(-2.5));

  auto result = at::neuron::utils::ConvertStaticArgsToIValues(py_list);

  ASSERT_EQ(result.size(), 2);
  EXPECT_TRUE(result[0].isDouble());
  EXPECT_DOUBLE_EQ(result[0].toDouble(), 3.14);
  EXPECT_TRUE(result[1].isDouble());
  EXPECT_DOUBLE_EQ(result[1].toDouble(), -2.5);
}

// TODO: Add integration test for tensor conversion
// ConvertTensor test removed due to segfault with py::cast and THPVariable

TEST_F(PythonTypeConverterTest, ConvertTuple) {
  py::list py_list;
  py::tuple py_tuple = py::make_tuple(1, 2, 3);
  py_list.append(py_tuple);

  auto result = at::neuron::utils::ConvertStaticArgsToIValues(py_list);

  ASSERT_EQ(result.size(), 1);
  EXPECT_TRUE(result[0].isIntList());
  auto int_list = result[0].toIntList();
  ASSERT_EQ(int_list.size(), 3);
  EXPECT_EQ(int_list[0], 1);
  EXPECT_EQ(int_list[1], 2);
  EXPECT_EQ(int_list[2], 3);
}

TEST_F(PythonTypeConverterTest, ConvertList) {
  py::list py_list;
  py::list inner_list;
  inner_list.append(py::int_(10));
  inner_list.append(py::int_(20));
  py_list.append(inner_list);

  auto result = at::neuron::utils::ConvertStaticArgsToIValues(py_list);

  ASSERT_EQ(result.size(), 1);
  EXPECT_TRUE(result[0].isIntList());
  auto int_list = result[0].toIntList();
  ASSERT_EQ(int_list.size(), 2);
  EXPECT_EQ(int_list[0], 10);
  EXPECT_EQ(int_list[1], 20);
}

TEST_F(PythonTypeConverterTest, ConvertMixedTypes) {
  py::list py_list;
  py_list.append(py::none());
  py_list.append(py::bool_(true));
  py_list.append(py::int_(42));
  py_list.append(py::float_(3.14));
  py_list.append(py::make_tuple(1, 2));

  auto result = at::neuron::utils::ConvertStaticArgsToIValues(py_list);

  ASSERT_EQ(result.size(), 5);
  EXPECT_TRUE(result[0].isNone());
  EXPECT_TRUE(result[1].isBool());
  EXPECT_EQ(result[1].toBool(), true);
  EXPECT_TRUE(result[2].isInt());
  EXPECT_EQ(result[2].toInt(), 42);
  EXPECT_TRUE(result[3].isDouble());
  EXPECT_DOUBLE_EQ(result[3].toDouble(), 3.14);
  EXPECT_TRUE(result[4].isIntList());
}

TEST_F(PythonTypeConverterTest, ConvertBoolList) {
  py::list py_list;
  py::list bool_list;
  bool_list.append(py::bool_(true));
  bool_list.append(py::bool_(false));
  bool_list.append(py::bool_(true));
  py_list.append(bool_list);

  auto result = at::neuron::utils::ConvertStaticArgsToIValues(py_list);

  ASSERT_EQ(result.size(), 1);
  EXPECT_TRUE(result[0].isBoolList());
  auto list = result[0].toBoolList();
  ASSERT_EQ(list.size(), 3);
  EXPECT_EQ(list[0], true);
  EXPECT_EQ(list[1], false);
  EXPECT_EQ(list[2], true);
}

TEST_F(PythonTypeConverterTest, ConvertFloatList) {
  py::list py_list;
  py::list float_list;
  float_list.append(py::float_(1.5));
  float_list.append(py::float_(2.5));
  py_list.append(float_list);

  auto result = at::neuron::utils::ConvertStaticArgsToIValues(py_list);

  ASSERT_EQ(result.size(), 1);
  EXPECT_TRUE(result[0].isDoubleList());
  auto list = result[0].toDoubleList();
  ASSERT_EQ(list.size(), 2);
  EXPECT_DOUBLE_EQ(list[0], 1.5);
  EXPECT_DOUBLE_EQ(list[1], 2.5);
}

TEST_F(PythonTypeConverterTest, ConvertMixedTypeList) {
  py::list py_list;
  py::list mixed_list;
  mixed_list.append(py::int_(1));
  mixed_list.append(py::float_(2.5));
  mixed_list.append(py::bool_(true));
  py_list.append(mixed_list);

  auto result = at::neuron::utils::ConvertStaticArgsToIValues(py_list);

  ASSERT_EQ(result.size(), 1);
  EXPECT_TRUE(result[0].isTuple());
  auto tuple = result[0].toTuple();
  ASSERT_EQ(tuple->elements().size(), 3);
  EXPECT_TRUE(tuple->elements()[0].isInt());
  EXPECT_EQ(tuple->elements()[0].toInt(), 1);
  EXPECT_TRUE(tuple->elements()[1].isDouble());
  EXPECT_DOUBLE_EQ(tuple->elements()[1].toDouble(), 2.5);
  EXPECT_TRUE(tuple->elements()[2].isBool());
  EXPECT_EQ(tuple->elements()[2].toBool(), true);
}

TEST_F(PythonTypeConverterTest, ConvertEmptyList) {
  py::list py_list;

  auto result = at::neuron::utils::ConvertStaticArgsToIValues(py_list);

  EXPECT_EQ(result.size(), 0);
}

TEST_F(PythonTypeConverterTest, ConvertString) {
  py::list py_list;
  py_list.append(py::str("constant"));
  py_list.append(py::str("reflect"));

  auto result = at::neuron::utils::ConvertStaticArgsToIValues(py_list);

  ASSERT_EQ(result.size(), 2);
  EXPECT_TRUE(result[0].isString());
  EXPECT_EQ(result[0].toStringRef(), "constant");
  EXPECT_TRUE(result[1].isString());
  EXPECT_EQ(result[1].toStringRef(), "reflect");
}

TEST_F(PythonTypeConverterTest, ConvertEmptySequence) {
  py::list py_list;
  py::list empty_list;
  py::tuple empty_tuple = py::tuple();
  py_list.append(empty_list);
  py_list.append(empty_tuple);

  auto result = at::neuron::utils::ConvertStaticArgsToIValues(py_list);

  ASSERT_EQ(result.size(), 2);
  // Empty sequences should convert to empty int64 lists
  EXPECT_TRUE(result[0].isIntList());
  EXPECT_EQ(result[0].toIntList().size(), 0);
  EXPECT_TRUE(result[1].isIntList());
  EXPECT_EQ(result[1].toIntList().size(), 0);
}

TEST_F(PythonTypeConverterTest, ConvertMixedTypeListWithString) {
  py::list py_list;
  py::list mixed_list;
  mixed_list.append(py::int_(42));
  mixed_list.append(py::str("hello"));
  mixed_list.append(py::float_(3.14));
  mixed_list.append(py::bool_(true));
  py_list.append(mixed_list);

  auto result = at::neuron::utils::ConvertStaticArgsToIValues(py_list);

  ASSERT_EQ(result.size(), 1);
  EXPECT_TRUE(result[0].isTuple());
  auto tuple = result[0].toTuple();
  ASSERT_EQ(tuple->elements().size(), 4);
  EXPECT_TRUE(tuple->elements()[0].isInt());
  EXPECT_EQ(tuple->elements()[0].toInt(), 42);
  EXPECT_TRUE(tuple->elements()[1].isString());
  EXPECT_EQ(tuple->elements()[1].toStringRef(), "hello");
  EXPECT_TRUE(tuple->elements()[2].isDouble());
  EXPECT_DOUBLE_EQ(tuple->elements()[2].toDouble(), 3.14);
  EXPECT_TRUE(tuple->elements()[3].isBool());
  EXPECT_EQ(tuple->elements()[3].toBool(), true);
}

// ============================================================================
// ConvertToCPUFallbackContext Tests
// ============================================================================

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_None) {
  py::object none_obj = py::none();

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(none_obj);

  EXPECT_TRUE(result.IsEmpty());
  EXPECT_FALSE(result.HasData());
}

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_EmptyDict) {
  py::dict empty_dict;

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(empty_dict);

  EXPECT_TRUE(result.IsEmpty());
  EXPECT_FALSE(result.HasData());
}

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_NonDictObject) {
  py::list not_a_dict;

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(not_a_dict);

  EXPECT_TRUE(result.IsEmpty());
  EXPECT_FALSE(result.HasData());
}

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_WithStaticArgs) {
  py::dict ctx_dict;
  py::list static_args;
  static_args.append(py::int_(42));
  static_args.append(py::float_(3.14));
  static_args.append(py::bool_(true));

  py::list static_positions;
  static_positions.append(py::int_(0));
  static_positions.append(py::int_(2));
  static_positions.append(py::int_(4));

  ctx_dict["static_args"] = static_args;
  ctx_dict["static_positions"] = static_positions;
  ctx_dict["list_positions"] = py::list();
  ctx_dict["list_lengths"] = py::list();

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(ctx_dict);

  EXPECT_FALSE(result.IsEmpty());
  EXPECT_TRUE(result.HasData());

  ASSERT_EQ(result.static_args.size(), 3);
  EXPECT_EQ(result.static_args[0].toInt(), 42);
  EXPECT_DOUBLE_EQ(result.static_args[1].toDouble(), 3.14);
  EXPECT_EQ(result.static_args[2].toBool(), true);

  ASSERT_EQ(result.static_arg_positions.size(), 3);
  EXPECT_EQ(result.static_arg_positions[0], 0);
  EXPECT_EQ(result.static_arg_positions[1], 2);
  EXPECT_EQ(result.static_arg_positions[2], 4);

  EXPECT_TRUE(result.list_arg_positions.empty());
  EXPECT_TRUE(result.list_arg_lengths.empty());
}

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_WithListArgs) {
  py::dict ctx_dict;
  py::list list_positions;
  list_positions.append(py::int_(1));
  list_positions.append(py::int_(3));

  py::list list_lengths;
  list_lengths.append(py::int_(3));
  list_lengths.append(py::int_(2));

  ctx_dict["static_args"] = py::list();
  ctx_dict["static_positions"] = py::list();
  ctx_dict["list_positions"] = list_positions;
  ctx_dict["list_lengths"] = list_lengths;

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(ctx_dict);

  EXPECT_FALSE(result.IsEmpty());
  EXPECT_TRUE(result.HasData());

  EXPECT_TRUE(result.static_args.empty());
  EXPECT_TRUE(result.static_arg_positions.empty());

  ASSERT_EQ(result.list_arg_positions.size(), 2);
  EXPECT_EQ(result.list_arg_positions[0], 1);
  EXPECT_EQ(result.list_arg_positions[1], 3);

  ASSERT_EQ(result.list_arg_lengths.size(), 2);
  EXPECT_EQ(result.list_arg_lengths[0], 3);
  EXPECT_EQ(result.list_arg_lengths[1], 2);
}

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_FullContext) {
  py::dict ctx_dict;

  py::list static_args;
  static_args.append(py::int_(100));
  static_args.append(py::str("axis"));

  py::list static_positions;
  static_positions.append(py::int_(0));
  static_positions.append(py::int_(3));

  py::list list_positions;
  list_positions.append(py::int_(1));

  py::list list_lengths;
  list_lengths.append(py::int_(5));

  ctx_dict["static_args"] = static_args;
  ctx_dict["static_positions"] = static_positions;
  ctx_dict["list_positions"] = list_positions;
  ctx_dict["list_lengths"] = list_lengths;

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(ctx_dict);

  EXPECT_FALSE(result.IsEmpty());
  EXPECT_TRUE(result.HasData());

  ASSERT_EQ(result.static_args.size(), 2);
  EXPECT_EQ(result.static_args[0].toInt(), 100);
  EXPECT_EQ(result.static_args[1].toStringRef(), "axis");

  ASSERT_EQ(result.static_arg_positions.size(), 2);
  EXPECT_EQ(result.static_arg_positions[0], 0);
  EXPECT_EQ(result.static_arg_positions[1], 3);

  ASSERT_EQ(result.list_arg_positions.size(), 1);
  EXPECT_EQ(result.list_arg_positions[0], 1);

  ASSERT_EQ(result.list_arg_lengths.size(), 1);
  EXPECT_EQ(result.list_arg_lengths[0], 5);
}

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_MissingKeys) {
  // Dict with only some keys - should handle gracefully
  py::dict ctx_dict;
  py::list static_args;
  static_args.append(py::int_(42));
  ctx_dict["static_args"] = static_args;
  // Missing: static_positions, list_positions, list_lengths

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(ctx_dict);

  // Should have static_args but empty positions/lengths
  ASSERT_EQ(result.static_args.size(), 1);
  EXPECT_EQ(result.static_args[0].toInt(), 42);
  EXPECT_TRUE(result.static_arg_positions.empty());
  EXPECT_TRUE(result.list_arg_positions.empty());
  EXPECT_TRUE(result.list_arg_lengths.empty());
}

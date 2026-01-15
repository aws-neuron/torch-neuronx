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

// ============================================================================
// ConvertToInputMetadata Tests
// ============================================================================

TEST_F(PythonTypeConverterTest, ConvertToInputMetadata_None) {
  std::vector<at::neuron::TensorContext> tensor_metadata;
  std::vector<void*> tensor_data_ptrs;

  auto result =
      at::neuron::utils::ConvertToInputMetadata(py::none(), tensor_metadata, tensor_data_ptrs);

  EXPECT_TRUE(at::neuron::IsScalar(result));
  EXPECT_TRUE(at::neuron::GetScalarValue(result).isNone());
  EXPECT_TRUE(tensor_metadata.empty());
  EXPECT_TRUE(tensor_data_ptrs.empty());
}

TEST_F(PythonTypeConverterTest, ConvertToInputMetadata_Bool) {
  std::vector<at::neuron::TensorContext> tensor_metadata;
  std::vector<void*> tensor_data_ptrs;

  auto result =
      at::neuron::utils::ConvertToInputMetadata(py::bool_(true), tensor_metadata, tensor_data_ptrs);

  EXPECT_TRUE(at::neuron::IsScalar(result));
  EXPECT_TRUE(at::neuron::GetScalarValue(result).isBool());
  EXPECT_EQ(at::neuron::GetScalarValue(result).toBool(), true);
  EXPECT_TRUE(tensor_metadata.empty());
}

TEST_F(PythonTypeConverterTest, ConvertToInputMetadata_Int) {
  std::vector<at::neuron::TensorContext> tensor_metadata;
  std::vector<void*> tensor_data_ptrs;

  auto result =
      at::neuron::utils::ConvertToInputMetadata(py::int_(42), tensor_metadata, tensor_data_ptrs);

  EXPECT_TRUE(at::neuron::IsScalar(result));
  EXPECT_TRUE(at::neuron::GetScalarValue(result).isInt());
  EXPECT_EQ(at::neuron::GetScalarValue(result).toInt(), 42);
}

TEST_F(PythonTypeConverterTest, ConvertToInputMetadata_Double) {
  std::vector<at::neuron::TensorContext> tensor_metadata;
  std::vector<void*> tensor_data_ptrs;

  auto result = at::neuron::utils::ConvertToInputMetadata(py::float_(3.14), tensor_metadata,
                                                          tensor_data_ptrs);

  EXPECT_TRUE(at::neuron::IsScalar(result));
  EXPECT_TRUE(at::neuron::GetScalarValue(result).isDouble());
  EXPECT_DOUBLE_EQ(at::neuron::GetScalarValue(result).toDouble(), 3.14);
}

TEST_F(PythonTypeConverterTest, ConvertToInputMetadata_String) {
  std::vector<at::neuron::TensorContext> tensor_metadata;
  std::vector<void*> tensor_data_ptrs;

  auto result = at::neuron::utils::ConvertToInputMetadata(py::str("hello"), tensor_metadata,
                                                          tensor_data_ptrs);

  EXPECT_TRUE(at::neuron::IsScalar(result));
  EXPECT_TRUE(at::neuron::GetScalarValue(result).isString());
  EXPECT_EQ(at::neuron::GetScalarValue(result).toStringRef(), "hello");
}

TEST_F(PythonTypeConverterTest, ConvertToInputMetadata_IntList) {
  std::vector<at::neuron::TensorContext> tensor_metadata;
  std::vector<void*> tensor_data_ptrs;

  py::list int_list;
  int_list.append(py::int_(1));
  int_list.append(py::int_(2));
  int_list.append(py::int_(3));

  auto result =
      at::neuron::utils::ConvertToInputMetadata(int_list, tensor_metadata, tensor_data_ptrs);

  EXPECT_TRUE(at::neuron::IsScalar(result));
  EXPECT_TRUE(at::neuron::GetScalarValue(result).isIntList());
  auto list = at::neuron::GetScalarValue(result).toIntList();
  EXPECT_EQ(list.size(), 3);
  EXPECT_EQ(list[0], 1);
  EXPECT_EQ(list[1], 2);
  EXPECT_EQ(list[2], 3);
}

TEST_F(PythonTypeConverterTest, ConvertToInputMetadata_BoolList) {
  std::vector<at::neuron::TensorContext> tensor_metadata;
  std::vector<void*> tensor_data_ptrs;

  py::list bool_list;
  bool_list.append(py::bool_(true));
  bool_list.append(py::bool_(false));

  auto result =
      at::neuron::utils::ConvertToInputMetadata(bool_list, tensor_metadata, tensor_data_ptrs);

  EXPECT_TRUE(at::neuron::IsScalar(result));
  EXPECT_TRUE(at::neuron::GetScalarValue(result).isBoolList());
}

TEST_F(PythonTypeConverterTest, ConvertToInputMetadata_DoubleList) {
  std::vector<at::neuron::TensorContext> tensor_metadata;
  std::vector<void*> tensor_data_ptrs;

  py::list double_list;
  double_list.append(py::float_(1.5));
  double_list.append(py::float_(2.5));

  auto result =
      at::neuron::utils::ConvertToInputMetadata(double_list, tensor_metadata, tensor_data_ptrs);

  EXPECT_TRUE(at::neuron::IsScalar(result));
  EXPECT_TRUE(at::neuron::GetScalarValue(result).isDoubleList());
}

TEST_F(PythonTypeConverterTest, ConvertToInputMetadata_MixedTuple) {
  std::vector<at::neuron::TensorContext> tensor_metadata;
  std::vector<void*> tensor_data_ptrs;

  py::list mixed_list;
  mixed_list.append(py::int_(1));
  mixed_list.append(py::str("hello"));
  mixed_list.append(py::float_(3.14));

  auto result =
      at::neuron::utils::ConvertToInputMetadata(mixed_list, tensor_metadata, tensor_data_ptrs);

  EXPECT_TRUE(at::neuron::IsScalar(result));
  EXPECT_TRUE(at::neuron::GetScalarValue(result).isTuple());
}

TEST_F(PythonTypeConverterTest, ConvertToInputMetadata_EmptyList) {
  std::vector<at::neuron::TensorContext> tensor_metadata;
  std::vector<void*> tensor_data_ptrs;

  py::list empty_list;

  auto result =
      at::neuron::utils::ConvertToInputMetadata(empty_list, tensor_metadata, tensor_data_ptrs);

  EXPECT_TRUE(at::neuron::IsScalar(result));
  EXPECT_TRUE(at::neuron::GetScalarValue(result).isIntList());
  EXPECT_EQ(at::neuron::GetScalarValue(result).toIntList().size(), 0);
}

// ============================================================================
// ConvertToCPUFallbackContext Tests
// ============================================================================

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_None) {
  py::object none_obj = py::none();

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(none_obj);

  EXPECT_TRUE(result.IsEmpty());
}

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_EmptyDict) {
  py::dict empty_dict;

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(empty_dict);

  EXPECT_TRUE(result.IsEmpty());
}

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_NonDictObject) {
  py::list not_a_dict;

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(not_a_dict);

  EXPECT_TRUE(result.IsEmpty());
}

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_WithMixedInputs) {
  py::dict ctx_dict;

  py::list original_inputs;
  original_inputs.append(py::int_(42));
  original_inputs.append(py::float_(3.14));
  original_inputs.append(py::bool_(true));

  ctx_dict["original_inputs"] = original_inputs;

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(ctx_dict);

  EXPECT_FALSE(result.IsEmpty());

  ASSERT_EQ(result.input_metadata.size(), 3);

  EXPECT_TRUE(at::neuron::IsScalar(result.input_metadata[0]));
  EXPECT_EQ(at::neuron::GetScalarValue(result.input_metadata[0]).toInt(), 42);

  EXPECT_TRUE(at::neuron::IsScalar(result.input_metadata[1]));
  EXPECT_DOUBLE_EQ(at::neuron::GetScalarValue(result.input_metadata[1]).toDouble(), 3.14);

  EXPECT_TRUE(at::neuron::IsScalar(result.input_metadata[2]));
  EXPECT_EQ(at::neuron::GetScalarValue(result.input_metadata[2]).toBool(), true);
}

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_FullContext) {
  py::dict ctx_dict;

  py::list original_inputs;
  original_inputs.append(py::int_(100));
  original_inputs.append(py::str("axis"));
  original_inputs.append(py::int_(999));

  ctx_dict["original_inputs"] = original_inputs;

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(ctx_dict);

  EXPECT_FALSE(result.IsEmpty());

  ASSERT_EQ(result.input_metadata.size(), 3);

  EXPECT_TRUE(at::neuron::IsScalar(result.input_metadata[0]));
  EXPECT_EQ(at::neuron::GetScalarValue(result.input_metadata[0]).toInt(), 100);

  EXPECT_TRUE(at::neuron::IsScalar(result.input_metadata[1]));
  EXPECT_EQ(at::neuron::GetScalarValue(result.input_metadata[1]).toStringRef(), "axis");

  EXPECT_TRUE(at::neuron::IsScalar(result.input_metadata[2]));
  EXPECT_EQ(at::neuron::GetScalarValue(result.input_metadata[2]).toInt(), 999);
}

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_OnlyInputs) {
  py::dict ctx_dict;
  py::list original_inputs;
  original_inputs.append(py::int_(42));
  ctx_dict["original_inputs"] = original_inputs;

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(ctx_dict);

  ASSERT_EQ(result.input_metadata.size(), 1);
  EXPECT_TRUE(at::neuron::IsScalar(result.input_metadata[0]));
  EXPECT_EQ(at::neuron::GetScalarValue(result.input_metadata[0]).toInt(), 42);
  EXPECT_FALSE(result.IsEmpty());
}

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_WithNestedLists) {
  py::dict ctx_dict;

  py::list inner_list;
  inner_list.append(py::int_(10));
  inner_list.append(py::int_(20));

  py::list original_inputs;
  original_inputs.append(py::int_(5));
  original_inputs.append(inner_list);
  original_inputs.append(py::str("test"));

  ctx_dict["original_inputs"] = original_inputs;

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(ctx_dict);

  EXPECT_FALSE(result.IsEmpty());
  ASSERT_EQ(result.input_metadata.size(), 3);

  EXPECT_TRUE(at::neuron::IsScalar(result.input_metadata[0]));
  EXPECT_EQ(at::neuron::GetScalarValue(result.input_metadata[0]).toInt(), 5);

  EXPECT_TRUE(at::neuron::IsScalar(result.input_metadata[1]));
  EXPECT_TRUE(at::neuron::GetScalarValue(result.input_metadata[1]).isIntList());
  EXPECT_EQ(at::neuron::GetScalarValue(result.input_metadata[1]).toIntList().size(), 2);

  EXPECT_TRUE(at::neuron::IsScalar(result.input_metadata[2]));
  EXPECT_EQ(at::neuron::GetScalarValue(result.input_metadata[2]).toStringRef(), "test");
}

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_WithKwargs) {
  py::dict ctx_dict;

  py::list original_inputs;
  original_inputs.append(py::int_(42));
  ctx_dict["original_inputs"] = original_inputs;

  py::dict original_kwargs;
  original_kwargs["alpha"] = py::float_(0.5);
  original_kwargs["dim"] = py::int_(1);
  ctx_dict["original_kwargs"] = original_kwargs;

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(ctx_dict);

  EXPECT_FALSE(result.IsEmpty());
  ASSERT_EQ(result.input_metadata.size(), 1);
  EXPECT_EQ(result.original_kwargs.size(), 2);
  EXPECT_TRUE(result.original_kwargs.contains("alpha"));
  EXPECT_TRUE(result.original_kwargs.contains("dim"));
}

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_WithBoolList) {
  py::dict ctx_dict;

  py::list bool_list;
  bool_list.append(py::bool_(true));
  bool_list.append(py::bool_(false));

  py::list original_inputs;
  original_inputs.append(bool_list);
  ctx_dict["original_inputs"] = original_inputs;

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(ctx_dict);

  EXPECT_FALSE(result.IsEmpty());
  ASSERT_EQ(result.input_metadata.size(), 1);
  EXPECT_TRUE(at::neuron::IsScalar(result.input_metadata[0]));
  EXPECT_TRUE(at::neuron::GetScalarValue(result.input_metadata[0]).isBoolList());
}

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_WithDoubleList) {
  py::dict ctx_dict;

  py::list double_list;
  double_list.append(py::float_(1.5));
  double_list.append(py::float_(2.5));

  py::list original_inputs;
  original_inputs.append(double_list);
  ctx_dict["original_inputs"] = original_inputs;

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(ctx_dict);

  EXPECT_FALSE(result.IsEmpty());
  ASSERT_EQ(result.input_metadata.size(), 1);
  EXPECT_TRUE(at::neuron::IsScalar(result.input_metadata[0]));
  EXPECT_TRUE(at::neuron::GetScalarValue(result.input_metadata[0]).isDoubleList());
}

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_WithMixedTuple) {
  py::dict ctx_dict;

  py::list mixed_list;
  mixed_list.append(py::int_(1));
  mixed_list.append(py::str("hello"));
  mixed_list.append(py::float_(3.14));

  py::list original_inputs;
  original_inputs.append(mixed_list);
  ctx_dict["original_inputs"] = original_inputs;

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(ctx_dict);

  EXPECT_FALSE(result.IsEmpty());
  ASSERT_EQ(result.input_metadata.size(), 1);
  EXPECT_TRUE(at::neuron::IsScalar(result.input_metadata[0]));
  EXPECT_TRUE(at::neuron::GetScalarValue(result.input_metadata[0]).isTuple());
}

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_WithNoneInput) {
  py::dict ctx_dict;

  py::list original_inputs;
  original_inputs.append(py::none());
  original_inputs.append(py::int_(42));
  ctx_dict["original_inputs"] = original_inputs;

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(ctx_dict);

  EXPECT_FALSE(result.IsEmpty());
  ASSERT_EQ(result.input_metadata.size(), 2);

  EXPECT_TRUE(at::neuron::IsScalar(result.input_metadata[0]));
  EXPECT_TRUE(at::neuron::GetScalarValue(result.input_metadata[0]).isNone());

  EXPECT_TRUE(at::neuron::IsScalar(result.input_metadata[1]));
  EXPECT_EQ(at::neuron::GetScalarValue(result.input_metadata[1]).toInt(), 42);
}

TEST_F(PythonTypeConverterTest, ConvertToCPUFallbackContext_EmptySequence) {
  py::dict ctx_dict;

  py::list empty_list;
  py::list original_inputs;
  original_inputs.append(empty_list);
  ctx_dict["original_inputs"] = original_inputs;

  auto result = at::neuron::utils::ConvertToCPUFallbackContext(ctx_dict);

  EXPECT_FALSE(result.IsEmpty());
  ASSERT_EQ(result.input_metadata.size(), 1);
  EXPECT_TRUE(at::neuron::IsScalar(result.input_metadata[0]));
  EXPECT_TRUE(at::neuron::GetScalarValue(result.input_metadata[0]).isIntList());
  EXPECT_EQ(at::neuron::GetScalarValue(result.input_metadata[0]).toIntList().size(), 0);
}

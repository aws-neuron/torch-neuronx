#pragma once

#include <pybind11/pybind11.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/torch.h>

#include <vector>

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/utils/CPUFallbackContext.h"

namespace py = pybind11;

namespace at::neuron::utils {

/**
 * @brief Converts Python objects to PyTorch C++ IValue representations.
 *
 * This utility function bridges Python and C++ by converting a list of Python objects
 * into their corresponding PyTorch IValue representations. This is essential for
 * CPU fallback operations where Python arguments need to be passed to C++ PyTorch
 * operations.
 *
 * Supported Python types and their conversions:
 * - None -> empty IValue
 * - bool -> c10::IValue(bool)
 * - int -> c10::IValue(int64_t)
 * - float -> c10::IValue(double)
 * - str -> c10::IValue(std::string)
 * - torch.Tensor -> c10::IValue(at::Tensor)
 * - torch.dtype -> c10::IValue(c10::ScalarType)
 * - list/tuple of homogeneous types -> c10::List<T>
 * - list/tuple of mixed types -> c10::ivalue::Tuple
 *
 * @param py_list A Python list containing static arguments to be converted.
 *                Each element will be inspected and converted to its appropriate IValue type.
 *
 * @return std::vector<c10::IValue> A vector of IValues corresponding to the input Python objects.
 *         The order of elements is preserved from the input list.
 *
 * @note For sequences (lists/tuples), the function attempts to create homogeneous typed lists
 *       (c10::List<bool>, c10::List<int64_t>, or c10::List<double>) when all elements are
 *       of the same type. For mixed-type sequences, it creates a c10::ivalue::Tuple.
 *
 * @note Empty sequences are converted to empty c10::List<int64_t> by default.
 *
 * @note Unsupported types are converted to empty IValue (None equivalent).
 *
 * @see c10::IValue for details on PyTorch's internal value representation.
 */
inline std::vector<c10::IValue> ConvertArgsToIValues(const py::list& py_list) {
  std::vector<c10::IValue> result;
  result.reserve(py_list.size());

  for (size_t i = 0; i < py_list.size(); ++i) {
    py::object obj = py_list[i];

    // Handle None type - converts to empty IValue
    if (obj.is_none()) {
      result.push_back(c10::IValue());
    }
    // Handle Python bool - must check before int since bool is subclass of int in Python
    else if (py::isinstance<py::bool_>(obj)) {
      result.push_back(obj.cast<bool>());
    }
    // Handle Python int - converts to int64_t for PyTorch compatibility
    else if (py::isinstance<py::int_>(obj)) {
      result.push_back(obj.cast<int64_t>());
    }
    // Handle Python float - converts to double for PyTorch compatibility
    else if (py::isinstance<py::float_>(obj)) {
      result.push_back(obj.cast<double>());
    }
    // Handle Python string
    else if (py::isinstance<py::str>(obj)) {
      result.push_back(obj.cast<std::string>());
    }
    // Handle PyTorch Tensor objects
    else if (THPVariable_Check(obj.ptr())) {
      result.push_back(THPVariable_Unpack(obj.ptr()));
    }
    // Handle PyTorch dtype objects
    else if (THPDtype_Check(obj.ptr())) {
      auto dtype_obj = reinterpret_cast<THPDtype*>(obj.ptr());
      result.push_back(dtype_obj->scalar_type);
    }
    // Handle sequences (lists and tuples)
    else if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
      auto seq = obj.cast<py::sequence>();

      // Handle empty sequences - default to empty int64 list
      if (seq.size() == 0) {
        result.push_back(c10::List<int64_t>());
      } else {
        // Determine if sequence contains homogeneous types
        // This enables more efficient typed list representations in PyTorch
        py::object first = seq[0];
        bool all_tensor = THPVariable_Check(first.ptr());
        bool all_bool = !all_tensor && py::isinstance<py::bool_>(first);
        bool all_float = !all_tensor && !all_bool && py::isinstance<py::float_>(first);
        bool all_int = !all_tensor && !all_bool && !all_float && py::isinstance<py::int_>(first);

        // Verify all elements match the first element's type
        for (size_t j = 1; j < seq.size() && (all_tensor || all_bool || all_float || all_int);
             ++j) {
          py::object item = seq[j];
          if (all_tensor && !THPVariable_Check(item.ptr())) all_tensor = false;
          if (all_bool && !py::isinstance<py::bool_>(item)) all_bool = false;
          if (all_float && !py::isinstance<py::float_>(item)) all_float = false;
          if (all_int && !py::isinstance<py::int_>(item)) all_int = false;
        }

        // Convert to typed list if homogeneous
        if (all_tensor) {
          c10::List<at::Tensor> list;
          for (auto item : seq) {
            list.push_back(THPVariable_Unpack(item.ptr()));
          }
          result.push_back(list);
        } else if (all_bool) {
          c10::List<bool> list;
          for (auto item : seq) {
            list.push_back(item.cast<bool>());
          }
          result.push_back(list);
        } else if (all_float) {
          c10::List<double> list;
          for (auto item : seq) {
            list.push_back(item.cast<double>());
          }
          result.push_back(list);
        } else if (all_int) {
          c10::List<int64_t> list;
          for (auto item : seq) {
            list.push_back(item.cast<int64_t>());
          }
          result.push_back(list);
        } else {
          // Mixed types - convert to heterogeneous tuple of IValues
          // This handles cases like [1, "hello", 3.14, True] and nested lists
          std::vector<c10::IValue> tuple_elements;
          for (auto item : seq) {
            if (py::isinstance<py::bool_>(item)) {
              tuple_elements.push_back(item.cast<bool>());
            } else if (py::isinstance<py::int_>(item)) {
              tuple_elements.push_back(item.cast<int64_t>());
            } else if (py::isinstance<py::float_>(item)) {
              tuple_elements.push_back(item.cast<double>());
            } else if (py::isinstance<py::str>(item)) {
              tuple_elements.push_back(item.cast<std::string>());
            } else if (py::isinstance<py::tuple>(item) || py::isinstance<py::list>(item)) {
              // Recursively handle nested sequences
              py::list nested_list;
              nested_list.append(item);
              auto nested_result = ConvertArgsToIValues(nested_list);
              if (!nested_result.empty()) {
                tuple_elements.push_back(nested_result[0]);
              } else {
                tuple_elements.push_back(c10::IValue());
              }
            } else {
              // Unsupported type within sequence - convert to None
              tuple_elements.push_back(c10::IValue());
            }
          }
          result.push_back(c10::ivalue::Tuple::create(tuple_elements));
        }
      }
    }
    // Fallback for unsupported types - convert to None
    else {
      result.push_back(c10::IValue());
    }
  }

  return result;
}

/**
 * @brief Recursively extracts tensor metadata and data pointers from IValue.
 *
 * Handles nested lists of tensors by recursively traversing the structure.
 *
 * @param input The IValue to process (tensor, tensor list, or other)
 * @param tensor_metadata Output vector for tensor metadata
 * @param tensor_data_ptrs Output vector for tensor data pointers
 */
inline void ExtractTensorMetadataRecursive(const c10::IValue& input,
                                           std::vector<at::neuron::TensorContext>& tensor_metadata,
                                           std::vector<void*>& tensor_data_ptrs) {
  if (input.isTensor()) {
    const auto& tensor = input.toTensor();
    tensor_metadata.push_back(at::neuron::TensorContext::FromTensor(tensor));
    tensor_data_ptrs.push_back(tensor.data_ptr());
  } else if (input.isTensorList()) {
    c10::List<at::Tensor> tensor_list = input.toTensorList();
    for (size_t i = 0; i < tensor_list.size(); ++i) {
      const auto& tensor = tensor_list.get(i);
      tensor_metadata.push_back(at::neuron::TensorContext::FromTensor(tensor));
      tensor_data_ptrs.push_back(tensor.data_ptr());
    }
  } else if (input.isList()) {
    // Handle nested generic lists (e.g., List<IValue>)
    c10::List<c10::IValue> list = input.toList();
    for (size_t i = 0; i < list.size(); ++i) {
      ExtractTensorMetadataRecursive(list.get(i), tensor_metadata, tensor_data_ptrs);
    }
  } else if (input.isTuple()) {
    // Handle tuples containing tensors
    auto tuple = input.toTuple();
    for (const auto& elem : tuple->elements()) {
      ExtractTensorMetadataRecursive(elem, tensor_metadata, tensor_data_ptrs);
    }
  }
  // Scalars and other types are ignored
}

/**
 * @brief Converts a Python dictionary to CPUFallbackContext.
 *
 * This function converts a Python dictionary containing CPU fallback metadata
 * into a C++ CPUFallbackContext struct. The dictionary is expected to have:
 * - 'original_inputs': list of ALL original input values (tensors + scalars, lists preserved)
 *
 * The function extracts tensor_metadata and tensor_data_ptrs from original_inputs.
 * Lists of tensors are preserved as c10::List<at::Tensor> in original_inputs.
 *
 * @param context A Python object (dict or None) containing the fallback context.
 *                If None or empty, returns a default-constructed CPUFallbackContext.
 *
 * @return CPUFallbackContext The converted context struct with metadata extracted.
 */
inline CPUFallbackContext ConvertToCPUFallbackContext(const py::object& context) {
  // Handle None or empty context
  if (context.is_none()) {
    return CPUFallbackContext();
  }

  // Ensure it's a dictionary
  if (!py::isinstance<py::dict>(context)) {
    return CPUFallbackContext();
  }

  py::dict ctx_dict = context.cast<py::dict>();

  // Extract original_inputs and build tensor_metadata and data_ptrs
  std::vector<c10::IValue> original_inputs;
  std::vector<at::neuron::TensorContext> tensor_metadata;
  std::vector<void*> tensor_data_ptrs;

  if (ctx_dict.contains("original_inputs")) {
    py::object inputs_obj = ctx_dict["original_inputs"];

    if (py::isinstance<py::list>(inputs_obj)) {
      py::list inputs_list = inputs_obj.cast<py::list>();

      original_inputs = ConvertArgsToIValues(inputs_list);

      // Reserve space for efficiency
      tensor_metadata.reserve(original_inputs.size());
      tensor_data_ptrs.reserve(original_inputs.size());

      // Recursively extract tensor metadata from all inputs (handles nested lists)
      for (const auto& input : original_inputs) {
        ExtractTensorMetadataRecursive(input, tensor_metadata, tensor_data_ptrs);
      }
    }
  }

  // Create context with original_inputs
  CPUFallbackContext ctx(std::move(original_inputs));

  // Populate extracted metadata and pointers
  ctx.tensor_metadata = std::move(tensor_metadata);
  ctx.tensor_data_ptrs = std::move(tensor_data_ptrs);

  // Extract original_kwargs directly into GenericDict
  if (ctx_dict.contains("original_kwargs")) {
    py::object kwargs_obj = ctx_dict["original_kwargs"];
    if (py::isinstance<py::dict>(kwargs_obj)) {
      py::dict kwargs_dict = kwargs_obj.cast<py::dict>();
      for (auto item : kwargs_dict) {
        std::string key = item.first.cast<std::string>();
        py::object value = item.second.cast<py::object>();

        // Convert Python value to IValue
        c10::IValue ivalue_key(key);
        c10::IValue ivalue_val;

        if (value.is_none()) {
          ivalue_val = c10::IValue();
        } else if (THPVariable_Check(value.ptr())) {
          ivalue_val = THPVariable_Unpack(value.ptr());
        } else if (py::isinstance<py::bool_>(value)) {
          ivalue_val = value.cast<bool>();
        } else if (py::isinstance<py::int_>(value)) {
          ivalue_val = value.cast<int64_t>();
        } else if (py::isinstance<py::float_>(value)) {
          ivalue_val = value.cast<double>();
        } else if (py::isinstance<py::str>(value)) {
          ivalue_val = value.cast<std::string>();
        }

        ctx.original_kwargs.insert(ivalue_key, ivalue_val);
      }
    }
  }

  return ctx;
}

}  // namespace at::neuron::utils

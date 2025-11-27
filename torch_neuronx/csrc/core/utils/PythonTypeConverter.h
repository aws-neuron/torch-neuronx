#pragma once

#include <pybind11/pybind11.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/torch.h>

#include <vector>

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
inline std::vector<c10::IValue> ConvertStaticArgsToIValues(const py::list& py_list) {
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
        bool all_bool = py::isinstance<py::bool_>(first);
        bool all_float = !all_bool && py::isinstance<py::float_>(first);
        bool all_int = !all_bool && !all_float && py::isinstance<py::int_>(first);

        // Verify all elements match the first element's type
        for (size_t j = 1; j < seq.size() && (all_bool || all_float || all_int); ++j) {
          py::object item = seq[j];
          if (all_bool && !py::isinstance<py::bool_>(item)) all_bool = false;
          if (all_float && !py::isinstance<py::float_>(item)) all_float = false;
          if (all_int && !py::isinstance<py::int_>(item)) all_int = false;
        }

        // Convert to typed list if homogeneous
        if (all_bool) {
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
          // This handles cases like [1, "hello", 3.14, True]
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
 * @brief Converts a Python dictionary to CPUFallbackContext.
 *
 * This function converts a Python dictionary containing CPU fallback metadata
 * into a C++ CPUFallbackContext struct. The dictionary is expected to have
 * the following keys:
 * - 'static_args': list of static argument values
 * - 'static_positions': list of int positions for static args
 * - 'list_positions': list of int positions for list arguments
 * - 'list_lengths': list of int lengths for each list argument
 *
 * @param context A Python object (dict or None) containing the fallback context.
 *                If None or empty, returns a default-constructed CPUFallbackContext.
 *
 * @return CPUFallbackContext The converted context struct.
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

  // Extract static_args
  std::vector<c10::IValue> static_args;
  if (ctx_dict.contains("static_args")) {
    py::object args_obj = ctx_dict["static_args"];
    if (py::isinstance<py::list>(args_obj)) {
      static_args = ConvertStaticArgsToIValues(args_obj.cast<py::list>());
    }
  }

  // Extract static_positions
  std::vector<int64_t> static_positions;
  if (ctx_dict.contains("static_positions")) {
    py::object pos_obj = ctx_dict["static_positions"];
    if (py::isinstance<py::list>(pos_obj)) {
      py::list pos_list = pos_obj.cast<py::list>();
      static_positions.reserve(pos_list.size());
      for (size_t i = 0; i < pos_list.size(); ++i) {
        static_positions.push_back(pos_list[i].cast<int64_t>());
      }
    }
  }

  // Extract list_positions
  std::vector<int64_t> list_positions;
  if (ctx_dict.contains("list_positions")) {
    py::object pos_obj = ctx_dict["list_positions"];
    if (py::isinstance<py::list>(pos_obj)) {
      py::list pos_list = pos_obj.cast<py::list>();
      list_positions.reserve(pos_list.size());
      for (size_t i = 0; i < pos_list.size(); ++i) {
        list_positions.push_back(pos_list[i].cast<int64_t>());
      }
    }
  }

  // Extract list_lengths
  std::vector<int64_t> list_lengths;
  if (ctx_dict.contains("list_lengths")) {
    py::object len_obj = ctx_dict["list_lengths"];
    if (py::isinstance<py::list>(len_obj)) {
      py::list len_list = len_obj.cast<py::list>();
      list_lengths.reserve(len_list.size());
      for (size_t i = 0; i < len_list.size(); ++i) {
        list_lengths.push_back(len_list[i].cast<int64_t>());
      }
    }
  }

  return CPUFallbackContext(std::move(static_args), std::move(static_positions),
                            std::move(list_positions), std::move(list_lengths));
}

}  // namespace at::neuron::utils

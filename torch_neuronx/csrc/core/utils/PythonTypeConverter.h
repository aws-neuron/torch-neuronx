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
 * @brief Extracts tensor metadata and data pointers from a tensor.
 *
 * Helper function to populate tensor_metadata and tensor_data_ptrs vectors.
 */
inline void ExtractTensorInfo(const at::Tensor& tensor,
                              std::vector<at::neuron::TensorContext>& tensor_metadata,
                              std::vector<void*>& tensor_data_ptrs) {
  tensor_metadata.push_back(at::neuron::TensorContext::FromTensor(tensor));
  tensor_data_ptrs.push_back(tensor.data_ptr());
}

/**
 * @brief Converts a Python object to InputMetadata and extracts tensor info if applicable.
 *
 * This function inspects a Python object and returns:
 * - TensorMarker: for single tensors (metadata extracted to output vectors)
 * - TensorListMarker: for tensor lists (metadata extracted to output vectors)
 * - ScalarValue: for all non-tensor values (None, int, double, bool, string, lists, etc.)
 *
 * @param obj The Python object to convert
 * @param tensor_metadata Output vector for tensor metadata (populated for tensor types)
 * @param tensor_data_ptrs Output vector for tensor data pointers (populated for tensor types)
 * @return InputMetadata describing the input
 */
inline InputMetadata ConvertToInputMetadata(py::object obj,
                                            std::vector<at::neuron::TensorContext>& tensor_metadata,
                                            std::vector<void*>& tensor_data_ptrs) {
  // Handle None type
  if (obj.is_none()) {
    return ScalarValue(c10::IValue());
  }

  // Handle PyTorch Tensor - extract metadata but DON'T store the tensor IValue
  if (THPVariable_Check(obj.ptr())) {
    at::Tensor tensor = THPVariable_Unpack(obj.ptr());
    ExtractTensorInfo(tensor, tensor_metadata, tensor_data_ptrs);
    return TensorMarker{};
  }

  // Handle Python bool - must check before int since bool is subclass of int
  if (py::isinstance<py::bool_>(obj)) {
    return ScalarValue(c10::IValue(obj.cast<bool>()));
  }

  // Handle Python int
  if (py::isinstance<py::int_>(obj)) {
    return ScalarValue(c10::IValue(obj.cast<int64_t>()));
  }

  // Handle Python float
  if (py::isinstance<py::float_>(obj)) {
    return ScalarValue(c10::IValue(obj.cast<double>()));
  }

  // Handle Python string
  if (py::isinstance<py::str>(obj)) {
    return ScalarValue(c10::IValue(obj.cast<std::string>()));
  }

  // Handle PyTorch dtype
  if (THPDtype_Check(obj.ptr())) {
    auto dtype_obj = reinterpret_cast<THPDtype*>(obj.ptr());
    return ScalarValue(c10::IValue(dtype_obj->scalar_type));
  }

  // Handle sequences (lists and tuples)
  if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
    auto seq = obj.cast<py::sequence>();
    size_t seq_size = seq.size();

    // Handle empty sequences
    if (seq_size == 0) {
      return ScalarValue(c10::IValue(c10::List<int64_t>()));
    }

    // Check first element to determine sequence type
    py::object first = seq[0];

    // Check if all elements are tensors
    bool all_tensor = THPVariable_Check(first.ptr());
    if (all_tensor) {
      for (size_t i = 1; i < seq_size; ++i) {
        if (!THPVariable_Check(seq[i].ptr())) {
          all_tensor = false;
          break;
        }
      }
    }

    if (all_tensor) {
      // TensorList - extract metadata for each tensor
      for (size_t i = 0; i < seq_size; ++i) {
        at::Tensor tensor = THPVariable_Unpack(seq[i].ptr());
        ExtractTensorInfo(tensor, tensor_metadata, tensor_data_ptrs);
      }
      return TensorListMarker(seq_size);
    }

    // Check for homogeneous scalar lists
    bool all_bool = py::isinstance<py::bool_>(first);
    bool all_int = !all_bool && py::isinstance<py::int_>(first);
    bool all_float = !all_bool && !all_int && py::isinstance<py::float_>(first);

    for (size_t i = 1; i < seq_size && (all_bool || all_int || all_float); ++i) {
      py::object item = seq[i];
      if (all_bool && !py::isinstance<py::bool_>(item)) all_bool = false;
      if (all_int && !py::isinstance<py::int_>(item)) all_int = false;
      if (all_float && !py::isinstance<py::float_>(item)) all_float = false;
    }

    if (all_bool) {
      c10::List<bool> list;
      for (auto item : seq) {
        list.push_back(item.cast<bool>());
      }
      return ScalarValue(c10::IValue(list));
    }

    if (all_int) {
      c10::List<int64_t> list;
      for (auto item : seq) {
        list.push_back(item.cast<int64_t>());
      }
      return ScalarValue(c10::IValue(list));
    }

    if (all_float) {
      c10::List<double> list;
      for (auto item : seq) {
        list.push_back(item.cast<double>());
      }
      return ScalarValue(c10::IValue(list));
    }

    // Mixed types - convert to tuple
    std::vector<c10::IValue> tuple_elements;
    tuple_elements.reserve(seq_size);
    for (auto item : seq) {
      if (py::isinstance<py::bool_>(item)) {
        tuple_elements.push_back(item.cast<bool>());
      } else if (py::isinstance<py::int_>(item)) {
        tuple_elements.push_back(item.cast<int64_t>());
      } else if (py::isinstance<py::float_>(item)) {
        tuple_elements.push_back(item.cast<double>());
      } else if (py::isinstance<py::str>(item)) {
        tuple_elements.push_back(item.cast<std::string>());
      } else if (item.is_none()) {
        tuple_elements.push_back(c10::IValue());
      } else {
        // Unsupported nested type - store as None
        tuple_elements.push_back(c10::IValue());
      }
    }
    return ScalarValue(c10::IValue(c10::ivalue::Tuple::create(tuple_elements)));
  }

  // Unsupported type - store as None
  return ScalarValue(c10::IValue());
}

/**
 * @brief Converts a Python dictionary to CPUFallbackContext.
 *
 * This function converts a Python dictionary containing CPU fallback metadata
 * into a C++ CPUFallbackContext struct.
 *
 * Tensors are NOT stored as c10::IValue.
 * Instead, we store:
 * - InputMetadata for each position (TensorMarker/TensorListMarker/ScalarValue)
 * - tensor_metadata and tensor_data_ptrs for tensor reconstruction
 *
 * @param context A Python object (dict or None) containing the fallback context.
 *                Expected keys: 'original_inputs', 'original_kwargs'
 * @return CPUFallbackContext The converted context struct.
 */
inline CPUFallbackContext ConvertToCPUFallbackContext(const py::object& context) {
  CPUFallbackContext ctx;

  // Handle None or empty context
  if (context.is_none()) {
    return ctx;
  }

  // Ensure it's a dictionary
  if (!py::isinstance<py::dict>(context)) {
    return ctx;
  }

  py::dict ctx_dict = context.cast<py::dict>();

  // Process original_inputs - convert each to InputMetadata
  if (ctx_dict.contains("original_inputs")) {
    py::object inputs_obj = ctx_dict["original_inputs"];

    if (py::isinstance<py::list>(inputs_obj)) {
      py::list inputs_list = inputs_obj.cast<py::list>();
      ctx.input_metadata.reserve(inputs_list.size());

      for (size_t i = 0; i < inputs_list.size(); ++i) {
        py::object obj = inputs_list[i];
        InputMetadata meta = ConvertToInputMetadata(obj, ctx.tensor_metadata, ctx.tensor_data_ptrs);
        ctx.input_metadata.push_back(std::move(meta));
      }
    }
  }

  // Process original_kwargs - store scalar values, track tensor kwargs separately
  if (ctx_dict.contains("original_kwargs")) {
    py::object kwargs_obj = ctx_dict["original_kwargs"];
    if (py::isinstance<py::dict>(kwargs_obj)) {
      py::dict kwargs_dict = kwargs_obj.cast<py::dict>();
      for (auto item : kwargs_dict) {
        std::string key = item.first.cast<std::string>();
        py::object value = item.second.cast<py::object>();

        c10::IValue ivalue_val;

        if (value.is_none()) {
          ivalue_val = c10::IValue();
        } else if (THPVariable_Check(value.ptr())) {
          // Tensor in kwargs - extract metadata and track the kwarg name
          at::Tensor tensor = THPVariable_Unpack(value.ptr());
          ExtractTensorInfo(tensor, ctx.tensor_metadata, ctx.tensor_data_ptrs);
          // Track this kwarg name so we can map it to cpu_inputs later
          ctx.tensor_kwarg_names.push_back(key);
          // Store None as placeholder - actual tensor comes from cpu_inputs
          ivalue_val = c10::IValue();
          TORCH_NEURONX_DEBUG("Tensor kwarg tracked", "key=", key,
                              "tensor_kwarg_index=", ctx.tensor_kwarg_names.size() - 1);
        } else if (py::isinstance<py::bool_>(value)) {
          ivalue_val = value.cast<bool>();
        } else if (py::isinstance<py::int_>(value)) {
          ivalue_val = value.cast<int64_t>();
        } else if (py::isinstance<py::float_>(value)) {
          ivalue_val = value.cast<double>();
        } else if (py::isinstance<py::str>(value)) {
          ivalue_val = value.cast<std::string>();
        } else if (THPDtype_Check(value.ptr())) {
          auto dtype_obj = reinterpret_cast<THPDtype*>(value.ptr());
          ivalue_val = c10::IValue(dtype_obj->scalar_type);
        }

        ctx.original_kwargs.insert(key, ivalue_val);
      }
    }
  }

  TORCH_NEURONX_DEBUG("ConvertToCPUFallbackContext completed",
                      "input_count=", ctx.input_metadata.size(),
                      "tensor_count=", ctx.tensor_metadata.size(),
                      "tensor_kwarg_count=", ctx.tensor_kwarg_names.size(),
                      "kwargs_count=", ctx.original_kwargs.size());

  return ctx;
}

}  // namespace at::neuron::utils

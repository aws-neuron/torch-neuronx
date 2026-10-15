/**
 * TypeConversion.cpp - Schema-driven type conversion implementation
 */

#include "torch_neuronx/csrc/utils/TypeConversion.h"

#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/autograd/python_variable.h>

#include <sstream>
#include <stdexcept>

namespace torch_neuronx {
namespace utils {

// Forward declaration for recursive calls
py::object IValueToPyObjectWithSchema(const c10::IValue& ivalue, const c10::TypePtr& real_type);

py::object ScalarTypeToPyDtype(c10::ScalarType scalar_type) {
  // Get torch module and access dtype attributes
  py::object torch_module = py::module::import("torch");

  switch (scalar_type) {
    case c10::ScalarType::Float:
      return torch_module.attr("float32");
    case c10::ScalarType::Double:
      return torch_module.attr("float64");
    case c10::ScalarType::Half:
      return torch_module.attr("float16");
    case c10::ScalarType::BFloat16:
      return torch_module.attr("bfloat16");
    case c10::ScalarType::Int:
      return torch_module.attr("int32");
    case c10::ScalarType::Long:
      return torch_module.attr("int64");
    case c10::ScalarType::Short:
      return torch_module.attr("int16");
    case c10::ScalarType::Char:
      return torch_module.attr("int8");
    case c10::ScalarType::Byte:
      return torch_module.attr("uint8");
    case c10::ScalarType::Bool:
      return torch_module.attr("bool");
    case c10::ScalarType::ComplexFloat:
      return torch_module.attr("complex64");
    case c10::ScalarType::ComplexDouble:
      return torch_module.attr("complex128");
    case c10::ScalarType::QInt8:
      return torch_module.attr("qint8");
    case c10::ScalarType::QUInt8:
      return torch_module.attr("quint8");
    case c10::ScalarType::QInt32:
      return torch_module.attr("qint32");
    case c10::ScalarType::Float8_e5m2:
      return torch_module.attr("float8_e5m2");
    case c10::ScalarType::Float8_e4m3fn:
      return torch_module.attr("float8_e4m3fn");
    default:
      throw std::runtime_error("Unsupported ScalarType: " +
                               std::to_string(static_cast<int>(scalar_type)));
  }
}

py::object LayoutToPyLayout(c10::Layout layout) {
  py::object torch_module = py::module::import("torch");

  switch (layout) {
    case c10::Layout::Strided:
      return torch_module.attr("strided");
    case c10::Layout::Sparse:
      return torch_module.attr("sparse_coo");
    case c10::Layout::SparseCsr:
      return torch_module.attr("sparse_csr");
    case c10::Layout::SparseCsc:
      return torch_module.attr("sparse_csc");
    case c10::Layout::SparseBsr:
      return torch_module.attr("sparse_bsr");
    case c10::Layout::SparseBsc:
      return torch_module.attr("sparse_bsc");
    default:
      throw std::runtime_error("Unsupported Layout: " + std::to_string(static_cast<int>(layout)));
  }
}

py::object MemoryFormatToPyMemoryFormat(c10::MemoryFormat memory_format) {
  py::object torch_module = py::module::import("torch");

  switch (memory_format) {
    case c10::MemoryFormat::Contiguous:
      return torch_module.attr("contiguous_format");
    case c10::MemoryFormat::Preserve:
      return torch_module.attr("preserve_format");
    case c10::MemoryFormat::ChannelsLast:
      return torch_module.attr("channels_last");
    case c10::MemoryFormat::ChannelsLast3d:
      return torch_module.attr("channels_last_3d");
    default:
      throw std::runtime_error("Unsupported MemoryFormat: " +
                               std::to_string(static_cast<int>(memory_format)));
  }
}

py::object DeviceToPyDevice(const c10::Device& device) {
  py::object torch_module = py::module::import("torch");
  // Create torch.device from string representation
  return torch_module.attr("device")(device.str());
}

std::string TypeKindToString(c10::TypeKind kind) {
  switch (kind) {
    case c10::TypeKind::TensorType:
      return "TensorType";
    case c10::TypeKind::IntType:
      return "IntType";
    case c10::TypeKind::FloatType:
      return "FloatType";
    case c10::TypeKind::BoolType:
      return "BoolType";
    case c10::TypeKind::StringType:
      return "StringType";
    case c10::TypeKind::NoneType:
      return "NoneType";
    case c10::TypeKind::OptionalType:
      return "OptionalType";
    case c10::TypeKind::ListType:
      return "ListType";
    case c10::TypeKind::TupleType:
      return "TupleType";
    case c10::TypeKind::DictType:
      return "DictType";
    case c10::TypeKind::DeviceObjType:
      return "DeviceObjType";
    case c10::TypeKind::ScalarTypeType:
      return "ScalarTypeType";
    case c10::TypeKind::LayoutType:
      return "LayoutType";
    case c10::TypeKind::MemoryFormatType:
      return "MemoryFormatType";
    case c10::TypeKind::NumberType:
      return "NumberType";
    case c10::TypeKind::AnyType:
      return "AnyType";
    case c10::TypeKind::ClassType:
      return "ClassType";
    case c10::TypeKind::SymIntType:
      return "SymIntType";
    case c10::TypeKind::SymFloatType:
      return "SymFloatType";
    case c10::TypeKind::SymBoolType:
      return "SymBoolType";
    default:
      return "Unknown(" + std::to_string(static_cast<int>(kind)) + ")";
  }
}

py::object IValueToPyObjectWithSchema(const c10::IValue& ivalue, const c10::TypePtr& real_type) {
  // Handle None first - applies to any type
  if (ivalue.isNone()) {
    return py::none();
  }

  switch (real_type->kind()) {
    // =========== Special Semantic Types ===========
    case c10::TypeKind::ScalarTypeType: {
      // IValue stores dtype as int64_t, convert to torch.dtype
      int64_t int_val;
      if (ivalue.isInt()) {
        int_val = ivalue.toInt();
      } else if (ivalue.isScalar()) {
        int_val = ivalue.toScalar().toLong();
      } else {
        throw std::runtime_error("ScalarTypeType expects int or scalar IValue, got: " +
                                 std::string(ivalue.tagKind()));
      }
      auto scalar_type = static_cast<c10::ScalarType>(int_val);
      return ScalarTypeToPyDtype(scalar_type);
    }

    case c10::TypeKind::LayoutType: {
      int64_t int_val;
      if (ivalue.isInt()) {
        int_val = ivalue.toInt();
      } else if (ivalue.isScalar()) {
        int_val = ivalue.toScalar().toLong();
      } else {
        throw std::runtime_error("LayoutType expects int or scalar IValue, got: " +
                                 std::string(ivalue.tagKind()));
      }
      auto layout = static_cast<c10::Layout>(int_val);
      return LayoutToPyLayout(layout);
    }

    case c10::TypeKind::MemoryFormatType: {
      int64_t int_val;
      if (ivalue.isInt()) {
        int_val = ivalue.toInt();
      } else if (ivalue.isScalar()) {
        int_val = ivalue.toScalar().toLong();
      } else {
        throw std::runtime_error("MemoryFormatType expects int or scalar IValue, got: " +
                                 std::string(ivalue.tagKind()));
      }
      auto memory_format = static_cast<c10::MemoryFormat>(int_val);
      return MemoryFormatToPyMemoryFormat(memory_format);
    }

    case c10::TypeKind::DeviceObjType: {
      if (ivalue.isDevice()) {
        return DeviceToPyDevice(ivalue.toDevice());
      } else if (ivalue.isString()) {
        // Device can be passed as string
        py::object torch_module = py::module::import("torch");
        return torch_module.attr("device")(ivalue.toStringRef());
      } else {
        throw std::runtime_error("DeviceObjType expects Device or string IValue, got: " +
                                 std::string(ivalue.tagKind()));
      }
    }

    // =========== Tensor Type ===========
    case c10::TypeKind::TensorType: {
      if (!ivalue.isTensor()) {
        throw std::runtime_error("TensorType expects Tensor IValue, got: " +
                                 std::string(ivalue.tagKind()));
      }
      // Use standard tensor conversion - preserves tensor as-is
      return py::cast(ivalue.toTensor());
    }

    // =========== Primitive Types ===========
    case c10::TypeKind::IntType:
    case c10::TypeKind::SymIntType: {
      if (ivalue.isInt()) {
        return py::cast(ivalue.toInt());
      } else if (ivalue.isScalar()) {
        return py::cast(ivalue.toScalar().toLong());
      } else if (ivalue.isSymInt()) {
        // Guard and extract actual int
        return py::cast(ivalue.toSymInt().guard_int(__FILE__, __LINE__));
      }
      throw std::runtime_error("IntType expects int/scalar IValue, got: " +
                               std::string(ivalue.tagKind()));
    }

    case c10::TypeKind::FloatType:
    case c10::TypeKind::SymFloatType: {
      if (ivalue.isDouble()) {
        return py::cast(ivalue.toDouble());
      } else if (ivalue.isScalar()) {
        return py::cast(ivalue.toScalar().toDouble());
      } else if (ivalue.isSymFloat()) {
        return py::cast(ivalue.toSymFloat().guard_float(__FILE__, __LINE__));
      }
      throw std::runtime_error("FloatType expects double/scalar IValue, got: " +
                               std::string(ivalue.tagKind()));
    }

    case c10::TypeKind::BoolType:
    case c10::TypeKind::SymBoolType: {
      if (ivalue.isBool()) {
        return py::cast(ivalue.toBool());
      } else if (ivalue.isSymBool()) {
        return py::cast(ivalue.toSymBool().guard_bool(__FILE__, __LINE__));
      }
      throw std::runtime_error("BoolType expects bool IValue, got: " +
                               std::string(ivalue.tagKind()));
    }

    case c10::TypeKind::StringType: {
      if (!ivalue.isString()) {
        throw std::runtime_error("StringType expects string IValue, got: " +
                                 std::string(ivalue.tagKind()));
      }
      return py::cast(ivalue.toStringRef());
    }

    case c10::TypeKind::NoneType: {
      return py::none();
    }

    // =========== NumberType (union of int/float/complex) ===========
    case c10::TypeKind::NumberType: {
      if (ivalue.isInt()) {
        return py::cast(ivalue.toInt());
      } else if (ivalue.isDouble()) {
        return py::cast(ivalue.toDouble());
      } else if (ivalue.isScalar()) {
        auto scalar = ivalue.toScalar();
        if (scalar.isIntegral(false)) {
          return py::cast(scalar.toLong());
        } else if (scalar.isFloatingPoint()) {
          return py::cast(scalar.toDouble());
        } else if (scalar.isComplex()) {
          return py::cast(scalar.toComplexDouble());
        }
      } else if (ivalue.isComplexDouble()) {
        return py::cast(static_cast<std::complex<double>>(ivalue.toComplexDouble()));
      }
      // Fall back to standard conversion
      return torch::jit::toPyObject(ivalue);
    }

    // =========== Container Types ===========
    case c10::TypeKind::OptionalType: {
      // Already handled None above, so we have a value
      auto elem_type = real_type->cast<c10::OptionalType>()->getElementType();
      return IValueToPyObjectWithSchema(ivalue, elem_type);
    }

    case c10::TypeKind::ListType: {
      if (!ivalue.isList()) {
        throw std::runtime_error("ListType expects List IValue, got: " +
                                 std::string(ivalue.tagKind()));
      }
      auto list = ivalue.toList();
      auto elem_type = real_type->cast<c10::ListType>()->getElementType();
      py::list py_list;
      for (size_t i = 0; i < list.size(); ++i) {
        py_list.append(IValueToPyObjectWithSchema(list.get(i), elem_type));
      }
      return py_list;
    }

    case c10::TypeKind::TupleType: {
      if (!ivalue.isTuple()) {
        throw std::runtime_error("TupleType expects Tuple IValue, got: " +
                                 std::string(ivalue.tagKind()));
      }
      auto tuple = ivalue.toTuple();
      auto tuple_type = real_type->cast<c10::TupleType>();
      const auto& elem_types = tuple_type->elements();
      const auto& elements = tuple->elements();

      if (elem_types.size() != elements.size()) {
        throw std::runtime_error("Tuple size mismatch: schema expects " +
                                 std::to_string(elem_types.size()) + ", got " +
                                 std::to_string(elements.size()));
      }

      py::tuple py_tuple(elements.size());
      for (size_t i = 0; i < elements.size(); ++i) {
        py_tuple[i] = IValueToPyObjectWithSchema(elements[i], elem_types[i]);
      }
      return py_tuple;
    }

    case c10::TypeKind::DictType: {
      if (!ivalue.isGenericDict()) {
        throw std::runtime_error("DictType expects Dict IValue, got: " +
                                 std::string(ivalue.tagKind()));
      }
      auto dict = ivalue.toGenericDict();
      auto dict_type = real_type->cast<c10::DictType>();
      auto key_type = dict_type->getKeyType();
      auto value_type = dict_type->getValueType();

      py::dict py_dict;
      for (const auto& pair : dict) {
        py::object py_key = IValueToPyObjectWithSchema(pair.key(), key_type);
        py::object py_value = IValueToPyObjectWithSchema(pair.value(), value_type);
        py_dict[py_key] = py_value;
      }
      return py_dict;
    }

    // =========== Fallback ===========
    case c10::TypeKind::AnyType:
    default: {
      // For unknown or Any types, use standard PyTorch conversion
      return torch::jit::toPyObject(ivalue);
    }
  }
}

void BuildPyArgsFromSchema(const c10::FunctionSchema& schema, const std::vector<c10::IValue>& stack,
                           py::tuple& out_args, py::dict& out_kwargs) {
  const auto& schema_args = schema.arguments();

  if (stack.size() != schema_args.size()) {
    std::ostringstream oss;
    oss << "Stack size (" << stack.size() << ") doesn't match schema arguments ("
        << schema_args.size() << ") for " << schema.name();
    throw std::runtime_error(oss.str());
  }

  // First pass: count positional args
  size_t num_positional = 0;
  for (const auto& arg : schema_args) {
    if (!arg.kwarg_only()) {
      ++num_positional;
    }
  }

  // Build outputs
  out_args = py::tuple(num_positional);
  out_kwargs = py::dict();

  size_t pos_idx = 0;
  for (size_t i = 0; i < stack.size(); ++i) {
    const auto& arg = schema_args[i];
    const auto& ivalue = stack[i];

    // Use real_type() to get semantic type for conversion
    py::object py_obj = IValueToPyObjectWithSchema(ivalue, arg.real_type());

    if (arg.kwarg_only()) {
      out_kwargs[py::str(arg.name())] = py_obj;
    } else {
      out_args[pos_idx++] = py_obj;
    }
  }
}

}  // namespace utils
}  // namespace torch_neuronx

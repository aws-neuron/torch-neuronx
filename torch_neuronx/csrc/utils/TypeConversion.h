/**
 * TypeConversion.h - Schema-driven type conversion between IValue and PyObject
 *
 * This module provides schema-aware conversion of PyTorch IValues to Python
 * objects. Unlike torch::jit::toPyObject(), which only looks at the IValue's
 * runtime tag, these functions use the FunctionSchema's type information to
 * correctly convert values.
 *
 * Key insight: The same IValue (e.g., int64_t) can represent different semantic
 * types (dtype, layout, memory_format, plain int). The FunctionSchema's
 * Argument::real_type() tells us the intended semantic type.
 *
 * Example:
 *   - Schema says arg is ScalarTypeType -> IValue int64_t -> torch.dtype
 *   - Schema says arg is IntType -> IValue int64_t -> Python int
 */

#pragma once

#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <string>
#include <vector>

namespace py = pybind11;

namespace torch_neuronx {
namespace utils {

/**
 * Convert an IValue to a Python object using the schema's type information.
 *
 * This function handles special cases where the IValue's runtime type differs
 * from its semantic type:
 *   - ScalarTypeType: int64_t -> torch.dtype
 *   - LayoutType: int8_t -> torch.layout
 *   - MemoryFormatType: int8_t -> torch.memory_format
 *   - DeviceObjType: Device -> torch.device
 *   - OptionalType: Recursively convert element
 *   - ListType: Recursively convert elements
 *
 * @param ivalue The IValue to convert
 * @param real_type The semantic type from FunctionSchema (Argument::real_type())
 * @return py::object The converted Python object
 * @throws std::runtime_error if conversion fails
 */
py::object IValueToPyObjectWithSchema(const c10::IValue& ivalue, const c10::TypePtr& real_type);

/**
 * Convert a c10::ScalarType enum value to a torch.dtype Python object.
 *
 * @param scalar_type The ScalarType enum value
 * @return py::object A torch.dtype object
 */
py::object ScalarTypeToPyDtype(c10::ScalarType scalar_type);

/**
 * Convert a c10::Layout enum value to a torch.layout Python object.
 *
 * @param layout The Layout enum value
 * @return py::object A torch.layout object
 */
py::object LayoutToPyLayout(c10::Layout layout);

/**
 * Convert a c10::MemoryFormat enum value to a torch.memory_format Python object.
 *
 * @param memory_format The MemoryFormat enum value
 * @return py::object A torch.memory_format object
 */
py::object MemoryFormatToPyMemoryFormat(c10::MemoryFormat memory_format);

/**
 * Convert a c10::Device to a torch.device Python object.
 *
 * @param device The Device struct
 * @return py::object A torch.device object
 */
py::object DeviceToPyDevice(const c10::Device& device);

/**
 * Build Python args (tuple) and kwargs (dict) from a stack of IValues
 * using the FunctionSchema for type-correct conversion.
 *
 * This function:
 * 1. Iterates through schema arguments
 * 2. Converts each IValue using the argument's real_type()
 * 3. Places positional args in a tuple, kwarg_only args in a dict
 *
 * @param schema The function schema describing the arguments
 * @param stack The IValue stack (from dispatcher)
 * @param out_args Output: tuple of positional arguments
 * @param out_kwargs Output: dict of keyword arguments
 * @throws std::runtime_error if conversion fails or stack size mismatches
 */
void BuildPyArgsFromSchema(const c10::FunctionSchema& schema, const std::vector<c10::IValue>& stack,
                           py::tuple& out_args, py::dict& out_kwargs);

/**
 * Get the name of a TypeKind for debugging purposes.
 */
std::string TypeKindToString(c10::TypeKind kind);

}  // namespace utils
}  // namespace torch_neuronx

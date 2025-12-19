#include "TypeUtils.h"

#include <stdexcept>

#include "mlir/IR/Builders.h"

namespace torch_neuronx {
namespace type_utils {

mlir::Type stringToMlirType(mlir::OpBuilder& builder, const std::string& element_type) {
  // Boolean types
  if (element_type == "i1") {
    return builder.getI1Type();
  }
  // Signed integer types
  else if (element_type == "i8") {
    return builder.getIntegerType(8);
  } else if (element_type == "i16") {
    return builder.getIntegerType(16);
  } else if (element_type == "i32") {
    return builder.getIntegerType(32);
  } else if (element_type == "i64") {
    return builder.getIntegerType(64);
  }
  // Unsigned integer types
  else if (element_type == "ui8") {
    return builder.getIntegerType(8, /*isSigned=*/false);
  } else if (element_type == "ui16") {
    return builder.getIntegerType(16, /*isSigned=*/false);
  } else if (element_type == "ui32") {
    return builder.getIntegerType(32, /*isSigned=*/false);
  } else if (element_type == "ui64") {
    return builder.getIntegerType(64, /*isSigned=*/false);
  }
  // Floating point types
  else if (element_type == "f16") {
    return builder.getF16Type();
  } else if (element_type == "f32") {
    return builder.getF32Type();
  } else if (element_type == "f64") {
    return builder.getF64Type();
  } else if (element_type == "bf16") {
    return builder.getBF16Type();
  }
  // Complex types
  else if (element_type == "complex<f16>") {
    return mlir::ComplexType::get(builder.getF16Type());
  } else if (element_type == "complex<f32>") {
    return mlir::ComplexType::get(builder.getF32Type());
  } else if (element_type == "complex<f64>") {
    return mlir::ComplexType::get(builder.getF64Type());
  } else {
    throw std::invalid_argument("Unsupported element type: " + element_type);
  }
}

}  // namespace type_utils
}  // namespace torch_neuronx

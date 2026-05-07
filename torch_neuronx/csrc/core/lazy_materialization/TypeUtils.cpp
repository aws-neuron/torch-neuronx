#include "TypeUtils.h"

#include <stdexcept>

namespace c10_neuron {
namespace lazy {

std::string ScalarTypeToElementTypeString(c10::ScalarType scalar_type) {
  switch (scalar_type) {
    case c10::ScalarType::Byte:
      return "ui8";
    case c10::ScalarType::Char:
      return "i8";
    case c10::ScalarType::Short:
      return "i16";
    case c10::ScalarType::Int:
      return "i32";
    case c10::ScalarType::Long:
      return "i64";
    case c10::ScalarType::Half:
      return "f16";
    case c10::ScalarType::Float:
      return "f32";
    case c10::ScalarType::Double:
      return "f64";
    case c10::ScalarType::ComplexHalf:
      return "complex<f16>";
    case c10::ScalarType::ComplexFloat:
      return "complex<f32>";
    case c10::ScalarType::ComplexDouble:
      return "complex<f64>";
    case c10::ScalarType::Bool:
      return "i1";
    case c10::ScalarType::QInt8:
      return "i8";  // Quantized int8 represented as i8
    case c10::ScalarType::QUInt8:
      return "ui8";  // Quantized uint8 represented as ui8
    case c10::ScalarType::QInt32:
      return "i32";  // Quantized int32 represented as i32
    case c10::ScalarType::BFloat16:
      return "bf16";
    case c10::ScalarType::QUInt4x2:
      return "ui8";  // Packed 4-bit quantized values represented as ui8
    case c10::ScalarType::QUInt2x4:
      return "ui8";  // Packed 2-bit quantized values represented as ui8
    case c10::ScalarType::Bits1x8:
      return "ui8";  // 1-bit values packed in ui8
    case c10::ScalarType::Bits2x4:
      return "ui8";  // 2-bit values packed in ui8
    case c10::ScalarType::Bits4x2:
      return "ui8";  // 4-bit values packed in ui8
    case c10::ScalarType::Bits8:
      return "ui8";  // 8-bit values as ui8
    case c10::ScalarType::Bits16:
      return "ui16";  // 16-bit values as ui16
    case c10::ScalarType::Float8_e5m2:
      return "f8e5m2";
    case c10::ScalarType::Float8_e4m3fn:
      return "f8e4m3fn";
    case c10::ScalarType::Float8_e5m2fnuz:
      return "f8e5m2fnuz";
    case c10::ScalarType::Float8_e4m3fnuz:
      return "f8e4m3fnuz";
    case c10::ScalarType::UInt16:
      return "ui16";
    case c10::ScalarType::UInt32:
      return "ui32";
    case c10::ScalarType::UInt64:
      return "ui64";
    case c10::ScalarType::Float8_e8m0fnu:
      return "f8e8m0fnu";
    case c10::ScalarType::Float4_e2m1fn_x2:
      return "f4e2m1fn_x2";
    default:
      throw std::runtime_error("Unsupported element type for transformation");
  }
}

}  // namespace lazy
}  // namespace c10_neuron

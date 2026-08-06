// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL:  func.func @torch.aten.view_as_complex(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[2,4,8,2],f32>) -> !torch.vtensor<[2,4,8],complex<f32>> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[2,4,8,2],f32> -> tensor<2x4x8x2xf32>
// CHECK:         %[[T1:.*]] = stablehlo.slice %[[T0]] [0:2, 0:4, 0:8, 0:1] : (tensor<2x4x8x2xf32>) -> tensor<2x4x8x1xf32>
// CHECK:         %[[T2:.*]] = stablehlo.slice %[[T0]] [0:2, 0:4, 0:8, 1:2] : (tensor<2x4x8x2xf32>) -> tensor<2x4x8x1xf32>
// CHECK:         %[[T3:.*]] = stablehlo.reshape %[[T1]] : (tensor<2x4x8x1xf32>) -> tensor<2x4x8xf32>
// CHECK:         %[[T4:.*]] = stablehlo.reshape %[[T2]] : (tensor<2x4x8x1xf32>) -> tensor<2x4x8xf32>
// CHECK:         %[[T5:.*]] = stablehlo.complex %[[T3]], %[[T4]] : tensor<2x4x8xcomplex<f32>>
// CHECK:         %[[T6:.*]] = torch_c.from_builtin_tensor %[[T5]] : tensor<2x4x8xcomplex<f32>> -> !torch.vtensor<[2,4,8],complex<f32>>
// CHECK:         return %[[T6]] : !torch.vtensor<[2,4,8],complex<f32>>
func.func @torch.aten.view_as_complex(%arg0: !torch.vtensor<[2,4,8,2],f32>) -> !torch.vtensor<[2,4,8],complex<f32>> {
  %0 = torch.aten.view_as_complex %arg0 : !torch.vtensor<[2,4,8,2],f32> -> !torch.vtensor<[2,4,8],complex<f32>>
  return %0 : !torch.vtensor<[2,4,8],complex<f32>>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.view_as_real(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[2,4,8],complex<f32>>) -> !torch.vtensor<[2,4,8,2],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[2,4,8],complex<f32>> -> tensor<2x4x8xcomplex<f32>>
// CHECK:         %[[T1:.*]] = stablehlo.real %[[T0]] : (tensor<2x4x8xcomplex<f32>>) -> tensor<2x4x8xf32>
// CHECK:         %[[T2:.*]] = stablehlo.imag %[[T0]] : (tensor<2x4x8xcomplex<f32>>) -> tensor<2x4x8xf32>
// CHECK:         %[[T3:.*]] = stablehlo.reshape %[[T1]] : (tensor<2x4x8xf32>) -> tensor<2x4x8x1xf32>
// CHECK:         %[[T4:.*]] = stablehlo.reshape %[[T2]] : (tensor<2x4x8xf32>) -> tensor<2x4x8x1xf32>
// CHECK:         %[[T5:.*]] = stablehlo.concatenate %[[T3]], %[[T4]], dim = 3 : (tensor<2x4x8x1xf32>, tensor<2x4x8x1xf32>) -> tensor<2x4x8x2xf32>
// CHECK:         %[[T6:.*]] = torch_c.from_builtin_tensor %[[T5]] : tensor<2x4x8x2xf32> -> !torch.vtensor<[2,4,8,2],f32>
// CHECK:         return %[[T6]] : !torch.vtensor<[2,4,8,2],f32>
func.func @torch.aten.view_as_real(%arg0: !torch.vtensor<[2,4,8],complex<f32>>) -> !torch.vtensor<[2,4,8,2],f32> {
  %0 = torch.aten.view_as_real %arg0 : !torch.vtensor<[2,4,8],complex<f32>> -> !torch.vtensor<[2,4,8,2],f32>
  return %0 : !torch.vtensor<[2,4,8,2],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.mul.Tensor$complex(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[2,4,8],complex<f32>>, %[[ARG1:.*]]: !torch.vtensor<[2,4,8],complex<f32>>) -> !torch.vtensor<[2,4,8],complex<f32>> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[2,4,8],complex<f32>> -> tensor<2x4x8xcomplex<f32>>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[2,4,8],complex<f32>> -> tensor<2x4x8xcomplex<f32>>
// CHECK:         %[[T2:.*]] = chlo.broadcast_multiply %[[T1]], %[[T0]] : (tensor<2x4x8xcomplex<f32>>, tensor<2x4x8xcomplex<f32>>) -> tensor<2x4x8xcomplex<f32>>
// CHECK:         %[[T3:.*]] = torch_c.from_builtin_tensor %[[T2]] : tensor<2x4x8xcomplex<f32>> -> !torch.vtensor<[2,4,8],complex<f32>>
// CHECK:         return %[[T3]] : !torch.vtensor<[2,4,8],complex<f32>>
func.func @torch.aten.mul.Tensor$complex(%arg0: !torch.vtensor<[2,4,8],complex<f32>>, %arg1: !torch.vtensor<[2,4,8],complex<f32>>) -> !torch.vtensor<[2,4,8],complex<f32>> {
  %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[2,4,8],complex<f32>>, !torch.vtensor<[2,4,8],complex<f32>> -> !torch.vtensor<[2,4,8],complex<f32>>
  return %0 : !torch.vtensor<[2,4,8],complex<f32>>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.real(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[2,4,8],complex<f32>>) -> !torch.vtensor<[2,4,8],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[2,4,8],complex<f32>> -> tensor<2x4x8xcomplex<f32>>
// CHECK:         %[[T1:.*]] = stablehlo.real %[[T0]] : (tensor<2x4x8xcomplex<f32>>) -> tensor<2x4x8xf32>
// CHECK:         %[[T2:.*]] = torch_c.from_builtin_tensor %[[T1]] : tensor<2x4x8xf32> -> !torch.vtensor<[2,4,8],f32>
// CHECK:         return %[[T2]] : !torch.vtensor<[2,4,8],f32>
func.func @torch.aten.real(%arg0: !torch.vtensor<[2,4,8],complex<f32>>) -> !torch.vtensor<[2,4,8],f32> {
  %0 = torch.aten.real %arg0 : !torch.vtensor<[2,4,8],complex<f32>> -> !torch.vtensor<[2,4,8],f32>
  return %0 : !torch.vtensor<[2,4,8],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.imag(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[2,4,8],complex<f32>>) -> !torch.vtensor<[2,4,8],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[2,4,8],complex<f32>> -> tensor<2x4x8xcomplex<f32>>
// CHECK:         %[[T1:.*]] = stablehlo.imag %[[T0]] : (tensor<2x4x8xcomplex<f32>>) -> tensor<2x4x8xf32>
// CHECK:         %[[T2:.*]] = torch_c.from_builtin_tensor %[[T1]] : tensor<2x4x8xf32> -> !torch.vtensor<[2,4,8],f32>
// CHECK:         return %[[T2]] : !torch.vtensor<[2,4,8],f32>
func.func @torch.aten.imag(%arg0: !torch.vtensor<[2,4,8],complex<f32>>) -> !torch.vtensor<[2,4,8],f32> {
  %0 = torch.aten.imag %arg0 : !torch.vtensor<[2,4,8],complex<f32>> -> !torch.vtensor<[2,4,8],f32>
  return %0 : !torch.vtensor<[2,4,8],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.complex(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[2,4,8],f32>, %[[ARG1:.*]]: !torch.vtensor<[2,4,8],f32>) -> !torch.vtensor<[2,4,8],complex<f32>> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[2,4,8],f32> -> tensor<2x4x8xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[2,4,8],f32> -> tensor<2x4x8xf32>
// CHECK:         %[[T2:.*]] = stablehlo.complex %[[T1]], %[[T0]] : tensor<2x4x8xcomplex<f32>>
// CHECK:         %[[T3:.*]] = torch_c.from_builtin_tensor %[[T2]] : tensor<2x4x8xcomplex<f32>> -> !torch.vtensor<[2,4,8],complex<f32>>
// CHECK:         return %[[T3]] : !torch.vtensor<[2,4,8],complex<f32>>
func.func @torch.aten.complex(%arg0: !torch.vtensor<[2,4,8],f32>, %arg1: !torch.vtensor<[2,4,8],f32>) -> !torch.vtensor<[2,4,8],complex<f32>> {
  %0 = torch.aten.complex %arg0, %arg1 : !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,8],f32> -> !torch.vtensor<[2,4,8],complex<f32>>
  return %0 : !torch.vtensor<[2,4,8],complex<f32>>
}

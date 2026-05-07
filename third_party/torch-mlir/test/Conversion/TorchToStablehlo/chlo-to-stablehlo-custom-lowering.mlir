// RUN: torch-mlir-opt -convert-chlo-to-stablehlo-custom-lowering -split-input-file %s | FileCheck %s

// Test scalar + scalar -> scalar
// CHECK-LABEL: func.func @chlo_broadcast_add_scalar_scalar
// CHECK-SAME:    (%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32>
// CHECK:         %[[ADD:.*]] = stablehlo.add %arg0, %arg1 : tensor<f32>
// CHECK:         return %[[ADD]] : tensor<f32>
func.func @chlo_broadcast_add_scalar_scalar(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// Test vector + scalar -> vector
// CHECK-LABEL: func.func @chlo_broadcast_multiply_vector_scalar
// CHECK-SAME:    (%arg0: tensor<8xf64>, %arg1: tensor<f64>) -> tensor<8xf64>
// CHECK:         %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f64>) -> tensor<8xf64>
// CHECK:         %[[MUL:.*]] = stablehlo.multiply %arg0, %[[BROADCAST]] : tensor<8xf64>
// CHECK:         return %[[MUL]] : tensor<8xf64>
func.func @chlo_broadcast_multiply_vector_scalar(%arg0: tensor<8xf64>, %arg1: tensor<f64>) -> tensor<8xf64> {
  %0 = chlo.broadcast_multiply %arg0, %arg1 : (tensor<8xf64>, tensor<f64>) -> tensor<8xf64>
  return %0 : tensor<8xf64>
}

// -----

// Test matrix + vector -> matrix (broadcast vector along dim 1)
// CHECK-LABEL: func.func @chlo_broadcast_subtract_matrix_vector
// CHECK-SAME:    (%arg0: tensor<3x7xf32>, %arg1: tensor<7xf32>) -> tensor<3x7xf32>
// CHECK:         %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<7xf32>) -> tensor<3x7xf32>
// CHECK:         %[[SUB:.*]] = stablehlo.subtract %arg0, %[[BROADCAST]] : tensor<3x7xf32>
// CHECK:         return %[[SUB]] : tensor<3x7xf32>
func.func @chlo_broadcast_subtract_matrix_vector(%arg0: tensor<3x7xf32>, %arg1: tensor<7xf32>) -> tensor<3x7xf32> {
  %0 = chlo.broadcast_subtract %arg0, %arg1 : (tensor<3x7xf32>, tensor<7xf32>) -> tensor<3x7xf32>
  return %0 : tensor<3x7xf32>
}

// -----

// Test matrix + matrix -> matrix (broadcast each matrix along 1 dim)
// CHECK-LABEL: func.func @chlo_broadcast_subtract_matrix_vector
// CHECK-SAME:    (%arg0: tensor<1x7xf32>, %arg1: tensor<7x1xf32>) -> tensor<7x7xf32>
// CHECK:         %[[BROADCAST_0:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<7x7xf32>
// CHECK:         %[[BROADCAST_1:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<7x1xf32>) -> tensor<7x7xf32>
// CHECK:         %[[SUB:.*]] = stablehlo.subtract %[[BROADCAST_0]], %[[BROADCAST_1]] : tensor<7x7xf32>
// CHECK:         return %[[SUB]] : tensor<7x7xf32>
func.func @chlo_broadcast_subtract_matrix_vector(%arg0: tensor<1x7xf32>, %arg1: tensor<7x1xf32>) -> tensor<7x7xf32> {
  %0 = chlo.broadcast_subtract %arg0, %arg1 : (tensor<1x7xf32>, tensor<7x1xf32>) -> tensor<7x7xf32>
  return %0 : tensor<7x7xf32>
}

// -----

// Test 3D tensor + matrix -> 3D tensor (broadcast matrix along dims [1,2])
// CHECK-LABEL: func.func @chlo_broadcast_divide_3d_matrix
// CHECK-SAME:    (%arg0: tensor<2x4x5xf16>, %arg1: tensor<4x5xf16>) -> tensor<2x4x5xf16>
// CHECK:         %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<4x5xf16>) -> tensor<2x4x5xf16>
// CHECK:         %[[DIV:.*]] = stablehlo.divide %arg0, %[[BROADCAST]] : tensor<2x4x5xf16>
// CHECK:         return %[[DIV]] : tensor<2x4x5xf16>
func.func @chlo_broadcast_divide_3d_matrix(%arg0: tensor<2x4x5xf16>, %arg1: tensor<4x5xf16>) -> tensor<2x4x5xf16> {
  %0 = chlo.broadcast_divide %arg0, %arg1 : (tensor<2x4x5xf16>, tensor<4x5xf16>) -> tensor<2x4x5xf16>
  return %0 : tensor<2x4x5xf16>
}

// -----

// Test 4D tensor + scalar -> 4D tensor
// CHECK-LABEL: func.func @chlo_broadcast_max_4d_scalar
// CHECK-SAME:    (%arg0: tensor<2x3x4x5xf64>, %arg1: tensor<f64>) -> tensor<2x3x4x5xf64>
// CHECK:         %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f64>) -> tensor<2x3x4x5xf64>
// CHECK:         %[[MAX:.*]] = stablehlo.maximum %arg0, %[[BROADCAST]] : tensor<2x3x4x5xf64>
// CHECK:         return %[[MAX]] : tensor<2x3x4x5xf64>
func.func @chlo_broadcast_max_4d_scalar(%arg0: tensor<2x3x4x5xf64>, %arg1: tensor<f64>) -> tensor<2x3x4x5xf64> {
  %0 = chlo.broadcast_maximum %arg0, %arg1 : (tensor<2x3x4x5xf64>, tensor<f64>) -> tensor<2x3x4x5xf64>
  return %0 : tensor<2x3x4x5xf64>
}

// -----

// Test vector + vector -> vector (same shape)
// CHECK-LABEL: func.func @chlo_broadcast_min_vector_vector
// CHECK-SAME:    (%arg0: tensor<6xi32>, %arg1: tensor<6xi32>) -> tensor<6xi32>
// CHECK:         %[[MIN:.*]] = stablehlo.minimum %arg0, %arg1 : tensor<6xi32>
// CHECK:         return %[[MIN]] : tensor<6xi32>
func.func @chlo_broadcast_min_vector_vector(%arg0: tensor<6xi32>, %arg1: tensor<6xi32>) -> tensor<6xi32> {
  %0 = chlo.broadcast_minimum %arg0, %arg1 : (tensor<6xi32>, tensor<6xi32>) -> tensor<6xi32>
  return %0 : tensor<6xi32>
}

// -----

// Test compare operation: matrix + scalar -> matrix
// CHECK-LABEL: func.func @chlo_broadcast_compare_matrix_scalar
// CHECK-SAME:    (%arg0: tensor<5x8xf32>, %arg1: tensor<f32>) -> tensor<5x8xi1>
// CHECK:         %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<5x8xf32>
// CHECK:         %[[COMPARE:.*]] = stablehlo.compare  GT, %arg0, %[[BROADCAST]],  FLOAT : (tensor<5x8xf32>, tensor<5x8xf32>) -> tensor<5x8xi1>
// CHECK:         return %[[COMPARE]] : tensor<5x8xi1>
func.func @chlo_broadcast_compare_matrix_scalar(%arg0: tensor<5x8xf32>, %arg1: tensor<f32>) -> tensor<5x8xi1> {
  %0 = chlo.broadcast_compare %arg0, %arg1 {comparison_direction = #chlo<comparison_direction GT>, compare_type = #chlo<comparison_type FLOAT>} : (tensor<5x8xf32>, tensor<f32>) -> tensor<5x8xi1>
  return %0 : tensor<5x8xi1>
}

// -----

// Test compare operation: 3D + vector -> 3D
// CHECK-LABEL: func.func @chlo_broadcast_compare_3d_vector
// CHECK-SAME:    (%arg0: tensor<3x4x6xi64>, %arg1: tensor<6xi64>) -> tensor<3x4x6xi1>
// CHECK:         %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<6xi64>) -> tensor<3x4x6xi64>
// CHECK:         %[[COMPARE:.*]] = stablehlo.compare  LE, %arg0, %[[BROADCAST]],  SIGNED : (tensor<3x4x6xi64>, tensor<3x4x6xi64>) -> tensor<3x4x6xi1>
// CHECK:         return %[[COMPARE]] : tensor<3x4x6xi1>
func.func @chlo_broadcast_compare_3d_vector(%arg0: tensor<3x4x6xi64>, %arg1: tensor<6xi64>) -> tensor<3x4x6xi1> {
  %0 = chlo.broadcast_compare %arg0, %arg1 {comparison_direction = #chlo<comparison_direction LE>, compare_type = #chlo<comparison_type SIGNED>} : (tensor<3x4x6xi64>, tensor<6xi64>) -> tensor<3x4x6xi1>
  return %0 : tensor<3x4x6xi1>
}

// -----

// Test select operation: scalar predicate, vector operands
// CHECK-LABEL: func.func @chlo_broadcast_select_scalar_pred_vector
// CHECK-SAME:    (%arg0: tensor<i1>, %arg1: tensor<7xf16>, %arg2: tensor<f16>) -> tensor<7xf16>
// CHECK:         %[[PRED_BROADCAST:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<i1>) -> tensor<7xi1>
// CHECK:         %[[RHS_BROADCAST:.*]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f16>) -> tensor<7xf16>
// CHECK:         %[[SELECT:.*]] = stablehlo.select %[[PRED_BROADCAST]], %arg1, %[[RHS_BROADCAST]] : tensor<7xi1>, tensor<7xf16>
// CHECK:         return %[[SELECT]] : tensor<7xf16>
func.func @chlo_broadcast_select_scalar_pred_vector(%arg0: tensor<i1>, %arg1: tensor<7xf16>, %arg2: tensor<f16>) -> tensor<7xf16> {
  %0 = chlo.broadcast_select %arg0, %arg1, %arg2 : (tensor<i1>, tensor<7xf16>, tensor<f16>) -> tensor<7xf16>
  return %0 : tensor<7xf16>
}

// -----

// Test select operation: 3D predicate, matrix + scalar operands -> 3D
// CHECK-LABEL: func.func @chlo_broadcast_select_3d_pred_matrix_scalar
// CHECK-SAME:    (%arg0: tensor<2x5x9xi1>, %arg1: tensor<5x9xf64>, %arg2: tensor<f64>) -> tensor<2x5x9xf64>
// CHECK:         %[[LHS_BROADCAST:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<5x9xf64>) -> tensor<2x5x9xf64>
// CHECK:         %[[RHS_BROADCAST:.*]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<2x5x9xf64>
// CHECK:         %[[SELECT:.*]] = stablehlo.select %arg0, %[[LHS_BROADCAST]], %[[RHS_BROADCAST]] : tensor<2x5x9xi1>, tensor<2x5x9xf64>
// CHECK:         return %[[SELECT]] : tensor<2x5x9xf64>
func.func @chlo_broadcast_select_3d_pred_matrix_scalar(%arg0: tensor<2x5x9xi1>, %arg1: tensor<5x9xf64>, %arg2: tensor<f64>) -> tensor<2x5x9xf64> {
  %0 = chlo.broadcast_select %arg0, %arg1, %arg2 : (tensor<2x5x9xi1>, tensor<5x9xf64>, tensor<f64>) -> tensor<2x5x9xf64>
  return %0 : tensor<2x5x9xf64>
}

// -----

// Test complex operation: matrix + scalar -> matrix
// CHECK-LABEL: func.func @chlo_broadcast_complex_matrix_scalar
// CHECK-SAME:    (%arg0: tensor<4x6xf32>, %arg1: tensor<f32>) -> tensor<4x6xcomplex<f32>>
// CHECK:         %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<4x6xf32>
// CHECK:         %[[COMPLEX:.*]] = stablehlo.complex %arg0, %[[BROADCAST]] : tensor<4x6xcomplex<f32>>
// CHECK:         return %[[COMPLEX]] : tensor<4x6xcomplex<f32>>
func.func @chlo_broadcast_complex_matrix_scalar(%arg0: tensor<4x6xf32>, %arg1: tensor<f32>) -> tensor<4x6xcomplex<f32>> {
  %0 = chlo.broadcast_complex %arg0, %arg1 : (tensor<4x6xf32>, tensor<f32>) -> tensor<4x6xcomplex<f32>>
  return %0 : tensor<4x6xcomplex<f32>>
}

// -----

// Test integer operation: 3D + vector -> 3D
// CHECK-LABEL: func.func @chlo_broadcast_and_3d_vector
// CHECK-SAME:    (%arg0: tensor<3x7x2xi8>, %arg1: tensor<2xi8>) -> tensor<3x7x2xi8>
// CHECK:         %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<2xi8>) -> tensor<3x7x2xi8>
// CHECK:         %[[AND:.*]] = stablehlo.and %arg0, %[[BROADCAST]] : tensor<3x7x2xi8>
// CHECK:         return %[[AND]] : tensor<3x7x2xi8>
func.func @chlo_broadcast_and_3d_vector(%arg0: tensor<3x7x2xi8>, %arg1: tensor<2xi8>) -> tensor<3x7x2xi8> {
  %0 = chlo.broadcast_and %arg0, %arg1 : (tensor<3x7x2xi8>, tensor<2xi8>) -> tensor<3x7x2xi8>
  return %0 : tensor<3x7x2xi8>
}

// -----

// Test power operation: 4D + vector -> 4D
// CHECK-LABEL: func.func @chlo_broadcast_pow_4d_vector
// CHECK-SAME:    (%arg0: tensor<2x3x4x8xf32>, %arg1: tensor<8xf32>) -> tensor<2x3x4x8xf32>
// CHECK:         %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<8xf32>) -> tensor<2x3x4x8xf32>
// CHECK:         %[[POW:.*]] = stablehlo.power %arg0, %[[BROADCAST]] : tensor<2x3x4x8xf32>
// CHECK:         return %[[POW]] : tensor<2x3x4x8xf32>
func.func @chlo_broadcast_pow_4d_vector(%arg0: tensor<2x3x4x8xf32>, %arg1: tensor<8xf32>) -> tensor<2x3x4x8xf32> {
  %0 = chlo.broadcast_power %arg0, %arg1 : (tensor<2x3x4x8xf32>, tensor<8xf32>) -> tensor<2x3x4x8xf32>
  return %0 : tensor<2x3x4x8xf32>
}

// -----

// Test remainder operation: matrix + matrix -> matrix (same rank, same shape)
// CHECK-LABEL: func.func @chlo_broadcast_rem_matrix_matrix
// CHECK-SAME:    (%arg0: tensor<9x11xi16>, %arg1: tensor<9x11xi16>) -> tensor<9x11xi16>
// CHECK:         %[[REM:.*]] = stablehlo.remainder %arg0, %arg1 : tensor<9x11xi16>
// CHECK:         return %[[REM]] : tensor<9x11xi16>
func.func @chlo_broadcast_rem_matrix_matrix(%arg0: tensor<9x11xi16>, %arg1: tensor<9x11xi16>) -> tensor<9x11xi16> {
  %0 = chlo.broadcast_remainder %arg0, %arg1 : (tensor<9x11xi16>, tensor<9x11xi16>) -> tensor<9x11xi16>
  return %0 : tensor<9x11xi16>
}

// -----

// Test shift operation: 3D + matrix -> 3D
// CHECK-LABEL: func.func @chlo_broadcast_shift_3d_matrix
// CHECK-SAME:    (%arg0: tensor<5x6x7xi32>, %arg1: tensor<6x7xi32>) -> tensor<5x6x7xi32>
// CHECK:         %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<6x7xi32>) -> tensor<5x6x7xi32>
// CHECK:         %[[SHIFT:.*]] = stablehlo.shift_right_logical %arg0, %[[BROADCAST]] : tensor<5x6x7xi32>
// CHECK:         return %[[SHIFT]] : tensor<5x6x7xi32>
func.func @chlo_broadcast_shift_3d_matrix(%arg0: tensor<5x6x7xi32>, %arg1: tensor<6x7xi32>) -> tensor<5x6x7xi32> {
  %0 = chlo.broadcast_shift_right_logical %arg0, %arg1 : (tensor<5x6x7xi32>, tensor<6x7xi32>) -> tensor<5x6x7xi32>
  return %0 : tensor<5x6x7xi32>
}

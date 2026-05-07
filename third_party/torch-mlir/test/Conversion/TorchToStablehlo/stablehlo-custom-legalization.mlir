// RUN: torch-mlir-opt -stablehlo-custom-legalization -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @scatter_i1_indices_upcasting(%arg0: tensor<2x4xi64>, %arg1: tensor<1xi32>, %arg2: tensor<1x4xi64>) -> tensor<2x4xi64>
// CHECK: %[[INPUT_CONVERT:.*]] = stablehlo.convert %arg0 : (tensor<2x4xi64>) -> tensor<2x4xf32>
// CHECK: %[[UPDATE_CONVERT:.*]] = stablehlo.convert %arg2 : (tensor<1x4xi64>) -> tensor<1x4xf32>
// CHECK: %[[SCATTER:.*]] = "stablehlo.scatter"(%[[INPUT_CONVERT]], %arg1, %[[UPDATE_CONVERT]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK: ^bb0(%[[ARG3:.*]]: tensor<f32>, %[[ARG4:.*]]: tensor<f32>):
// CHECK:   stablehlo.return %[[ARG4]] : tensor<f32>
// CHECK: }) : (tensor<2x4xf32>, tensor<1xi32>, tensor<1x4xf32>) -> tensor<2x4xf32>
// CHECK: %[[RESULT_CONVERT:.*]] = stablehlo.convert %[[SCATTER]] : (tensor<2x4xf32>) -> tensor<2x4xi64>
// CHECK: return %[[RESULT_CONVERT]] : tensor<2x4xi64>
func.func @scatter_i1_indices_upcasting(%arg0: tensor<2x4xi64>, %arg1: tensor<1xi32>, %arg2: tensor<1x4xi64>) -> tensor<2x4xi64> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%arg3: tensor<i64>, %arg4: tensor<i64>):
    stablehlo.return %arg4 : tensor<i64>
  }) : (tensor<2x4xi64>, tensor<1xi32>, tensor<1x4xi64>) -> tensor<2x4xi64>
  return %0 : tensor<2x4xi64>
}

// -----

// CHECK-LABEL: func.func @scatter_f32_no_conversion(%arg0: tensor<2x4xf32>, %arg1: tensor<1xi32>, %arg2: tensor<1x4xf32>) -> tensor<2x4xf32>
// CHECK: %[[SCATTER:.*]] = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK: ^bb0(%[[ARG3:.*]]: tensor<f32>, %[[ARG4:.*]]: tensor<f32>):
// CHECK:   stablehlo.return %[[ARG4]] : tensor<f32>
// CHECK: }) : (tensor<2x4xf32>, tensor<1xi32>, tensor<1x4xf32>) -> tensor<2x4xf32>
// CHECK: return %[[SCATTER]] : tensor<2x4xf32>
func.func @scatter_f32_no_conversion(%arg0: tensor<2x4xf32>, %arg1: tensor<1xi32>, %arg2: tensor<1x4xf32>) -> tensor<2x4xf32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    stablehlo.return %arg4 : tensor<f32>
  }) : (tensor<2x4xf32>, tensor<1xi32>, tensor<1x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

// Test: i1 function arg normalized in-body (signature stays i1, convert+compare inserted)
// CHECK-LABEL: func.func @bool_arg_to_f32(%arg0: tensor<4xi1>) -> tensor<4xf32>
// CHECK: %[[TO_I8:.*]] = stablehlo.convert %arg0 : (tensor<4xi1>) -> tensor<4xi8>
// CHECK: %[[SCALAR_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i8>
// CHECK: %[[ZERO:.*]] = stablehlo.broadcast_in_dim %[[SCALAR_ZERO]], dims = [] : (tensor<i8>) -> tensor<4xi8>
// CHECK: %[[NORM:.*]] = stablehlo.compare NE, %[[TO_I8]], %[[ZERO]] : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi1>
// CHECK: %[[RESULT:.*]] = stablehlo.convert %[[NORM]] : (tensor<4xi1>) -> tensor<4xf32>
// CHECK: return %[[RESULT]] : tensor<4xf32>
func.func @bool_arg_to_f32(%arg0: tensor<4xi1>) -> tensor<4xf32> {
  %0 = stablehlo.convert %arg0 : (tensor<4xi1>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// Test: i1 function arg normalized in-body (bool -> i32)
// CHECK-LABEL: func.func @bool_arg_to_i32(%arg0: tensor<4xi1>) -> tensor<4xi32>
// CHECK: %[[TO_I8:.*]] = stablehlo.convert %arg0 : (tensor<4xi1>) -> tensor<4xi8>
// CHECK: %[[SCALAR_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i8>
// CHECK: %[[ZERO:.*]] = stablehlo.broadcast_in_dim %[[SCALAR_ZERO]], dims = [] : (tensor<i8>) -> tensor<4xi8>
// CHECK: %[[NORM:.*]] = stablehlo.compare NE, %[[TO_I8]], %[[ZERO]] : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi1>
// CHECK: %[[RESULT:.*]] = stablehlo.convert %[[NORM]] : (tensor<4xi1>) -> tensor<4xi32>
// CHECK: return %[[RESULT]] : tensor<4xi32>
func.func @bool_arg_to_i32(%arg0: tensor<4xi1>) -> tensor<4xi32> {
  %0 = stablehlo.convert %arg0 : (tensor<4xi1>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// Test: 2D i1 arg normalization (signature stays i1)
// CHECK-LABEL: func.func @bool_arg_2d(%arg0: tensor<2x3xi1>) -> tensor<2x3xf32>
// CHECK: %[[TO_I8:.*]] = stablehlo.convert %arg0 : (tensor<2x3xi1>) -> tensor<2x3xi8>
// CHECK: %[[SCALAR_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i8>
// CHECK: %[[ZERO:.*]] = stablehlo.broadcast_in_dim %[[SCALAR_ZERO]], dims = [] : (tensor<i8>) -> tensor<2x3xi8>
// CHECK: %[[NORM:.*]] = stablehlo.compare NE, %[[TO_I8]], %[[ZERO]] : (tensor<2x3xi8>, tensor<2x3xi8>) -> tensor<2x3xi1>
// CHECK: %[[RESULT:.*]] = stablehlo.convert %[[NORM]] : (tensor<2x3xi1>) -> tensor<2x3xf32>
// CHECK: return %[[RESULT]] : tensor<2x3xf32>
func.func @bool_arg_2d(%arg0: tensor<2x3xi1>) -> tensor<2x3xf32> {
  %0 = stablehlo.convert %arg0 : (tensor<2x3xi1>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

// Test: non-bool args are not modified
// CHECK-LABEL: func.func @non_bool_arg_unchanged(%arg0: tensor<4xf32>) -> tensor<4xi32>
// CHECK-NOT: stablehlo.compare
// CHECK: %[[RESULT:.*]] = stablehlo.convert %arg0 : (tensor<4xf32>) -> tensor<4xi32>
// CHECK: return %[[RESULT]] : tensor<4xi32>
func.func @non_bool_arg_unchanged(%arg0: tensor<4xf32>) -> tensor<4xi32> {
  %0 = stablehlo.convert %arg0 : (tensor<4xf32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// Test: multiple args, only i1 ones are normalized (signature stays i1)
// CHECK-LABEL: func.func @mixed_args(%arg0: tensor<4xi1>, %arg1: tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[TO_I8:.*]] = stablehlo.convert %arg0 : (tensor<4xi1>) -> tensor<4xi8>
// CHECK: %[[SCALAR_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i8>
// CHECK: %[[ZERO:.*]] = stablehlo.broadcast_in_dim %[[SCALAR_ZERO]], dims = [] : (tensor<i8>) -> tensor<4xi8>
// CHECK: %[[NORM:.*]] = stablehlo.compare NE, %[[TO_I8]], %[[ZERO]] : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi1>
func.func @mixed_args(%arg0: tensor<4xi1>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.convert %arg0 : (tensor<4xi1>) -> tensor<4xf32>
  %1 = stablehlo.add %0, %arg1 : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// Test: i1 arg used directly (no convert) still gets normalized
// CHECK-LABEL: func.func @bool_arg_no_convert(%arg0: tensor<4xi1>) -> tensor<4xi1>
// CHECK: %[[TO_I8:.*]] = stablehlo.convert %arg0 : (tensor<4xi1>) -> tensor<4xi8>
// CHECK: %[[SCALAR_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i8>
// CHECK: %[[ZERO:.*]] = stablehlo.broadcast_in_dim %[[SCALAR_ZERO]], dims = [] : (tensor<i8>) -> tensor<4xi8>
// CHECK: %[[NORM:.*]] = stablehlo.compare NE, %[[TO_I8]], %[[ZERO]] : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi1>
// CHECK: return %[[NORM]] : tensor<4xi1>
func.func @bool_arg_no_convert(%arg0: tensor<4xi1>) -> tensor<4xi1> {
  return %arg0 : tensor<4xi1>
}

// -----

// Test: scalar (0-d) i1 arg normalization
// CHECK-LABEL: func.func @bool_arg_scalar(%arg0: tensor<i1>) -> tensor<i1>
// CHECK: %[[TO_I8:.*]] = stablehlo.convert %arg0 : (tensor<i1>) -> tensor<i8>
// CHECK: %[[SCALAR_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i8>
// CHECK: %[[ZERO:.*]] = stablehlo.broadcast_in_dim %[[SCALAR_ZERO]], dims = [] : (tensor<i8>) -> tensor<i8>
// CHECK: %[[NORM:.*]] = stablehlo.compare NE, %[[TO_I8]], %[[ZERO]] : (tensor<i8>, tensor<i8>) -> tensor<i1>
// CHECK: return %[[NORM]] : tensor<i1>
func.func @bool_arg_scalar(%arg0: tensor<i1>) -> tensor<i1> {
  return %arg0 : tensor<i1>
}

// -----

// Test: bool output type is preserved (not changed to i8), no normalization on non-bool args
// CHECK-LABEL: func.func @bool_output_preserved(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1>
// CHECK-NOT: stablehlo.convert %arg0 : (tensor<4xf32>) -> tensor<4xi8>
// CHECK: %[[CMP:.*]] = stablehlo.compare NE, %arg0, %arg1, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// CHECK: return %[[CMP]] : tensor<4xi1>
func.func @bool_output_preserved(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
  %0 = stablehlo.compare NE, %arg0, %arg1, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

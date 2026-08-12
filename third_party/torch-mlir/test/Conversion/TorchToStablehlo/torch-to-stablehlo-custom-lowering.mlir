// RUN: torch-mlir-opt -convert-torch-to-neuron-custom-lowering -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @arith_constant_i1_tensor() -> tensor<i1>
// CHECK: %[[C0:.*]] = stablehlo.constant dense<true> : tensor<i1>
// CHECK: %[[C1:.*]] = stablehlo.constant dense<true> : tensor<i1>
// CHECK: return %[[C1]] : tensor<i1>
func.func @arith_constant_i1_tensor() -> tensor<i1> {
  %0 = arith.constant true
  %1 = stablehlo.constant dense<true> : tensor<i1>
  return %1 : tensor<i1>
}

// -----

// CHECK-LABEL: func.func @arith_constant_index_to_i64() -> tensor<i64>
// CHECK: %[[C0:.*]] = stablehlo.constant dense<42> : tensor<i64>
// CHECK: %[[C1:.*]] = stablehlo.constant dense<42> : tensor<i64>
// CHECK: return %[[C1]] : tensor<i64>
func.func @arith_constant_index_to_i64() -> tensor<i64> {
  %0 = arith.constant 42 : index
  %1 = stablehlo.constant dense<42> : tensor<i64>
  return %1 : tensor<i64>
}

// -----

// CHECK-LABEL: func.func @arith_cmpi_boolean_eq(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1>
// CHECK: %[[C0:.*]] = stablehlo.compare EQ, %arg0, %arg1, UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
// CHECK: return %[[C0]] : tensor<i1>
func.func @arith_cmpi_boolean_eq(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
  %0 = arith.cmpi eq, %arg0, %arg1 : tensor<i1>
  return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: func.func @arith_cmpi_integer_slt(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i1>
// CHECK: %[[C0:.*]] = stablehlo.compare LT, %arg0, %arg1, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK: return %[[C0]] : tensor<i1>
func.func @arith_cmpi_integer_slt(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i1> {
  %0 = arith.cmpi slt, %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: func.func @arith_select_boolean_mask(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32>
// CHECK: %[[S0:.*]] = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
// CHECK: return %[[S0]] : tensor<f32>
func.func @arith_select_boolean_mask(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %0 = arith.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func.func @arith_select_vector_mask(%arg0: tensor<4xi1>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[S0:.*]] = stablehlo.select %arg0, %arg1, %arg2 : tensor<4xi1>, tensor<4xf32>
// CHECK: return %[[S0]] : tensor<4xf32>
func.func @arith_select_vector_mask(%arg0: tensor<4xi1>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.select %arg0, %arg1, %arg2 : tensor<4xi1>, tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @arith_addi_boolean(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1>
// CHECK: %[[A0:.*]] = stablehlo.add %arg0, %arg1 : tensor<i1>
// CHECK: return %[[A0]] : tensor<i1>
func.func @arith_addi_boolean(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
  %0 = arith.addi %arg0, %arg1 : tensor<i1>
  return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: func.func @arith_maxsi_minsi(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>)
// CHECK: %[[MAX:.*]] = stablehlo.maximum %arg0, %arg1 : tensor<i32>
// CHECK: %[[MIN:.*]] = stablehlo.minimum %arg0, %arg1 : tensor<i32>
// CHECK: return %[[MAX]], %[[MIN]] : tensor<i32>, tensor<i32>
func.func @arith_maxsi_minsi(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0 = arith.maxsi %arg0, %arg1 : tensor<i32>
  %1 = arith.minsi %arg0, %arg1 : tensor<i32>
  return %0, %1 : tensor<i32>, tensor<i32>
}

// -----

// CHECK-LABEL: func.func @tensor_from_elements_boolean(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<2x2xi1>
// CHECK: %[[FALSE:.*]] = stablehlo.constant dense<false> : tensor<i1>
// CHECK: %[[TRUE:.*]] = stablehlo.constant dense<true> : tensor<i1>
// CHECK: %[[CONST:.*]] = stablehlo.constant dense<{{\[\[}}false, true{{\]}}, {{\[}}true, false{{\]\]}}> : tensor<2x2xi1>
// CHECK: return %[[CONST]] : tensor<2x2xi1>
func.func @tensor_from_elements_boolean(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<2x2xi1> {
  %0 = arith.constant false
  %1 = arith.constant true
  %2 = tensor.from_elements %0, %1, %1, %0 : tensor<2x2xi1>
  return %2 : tensor<2x2xi1>
}

// -----

// CHECK-LABEL: func.func @tensor_dim_extract(%arg0: tensor<4x4xi1>) -> tensor<i1>
// CHECK: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK: %[[C1:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK: %[[C4:.*]] = stablehlo.constant dense<4> : tensor<i64>
// CHECK: %[[SLICE:.*]] = stablehlo.dynamic_slice %arg0, %[[C1]], %[[C1]], sizes = [1, 1] : (tensor<4x4xi1>, tensor<i64>, tensor<i64>) -> tensor<1x1xi1>
// CHECK: %[[FALSE:.*]] = stablehlo.constant dense<false> : tensor<i1>
// CHECK: return %[[FALSE]] : tensor<i1>
func.func @tensor_dim_extract(%arg0: tensor<4x4xi1>) -> tensor<i1> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<4x4xi1>
  %1 = tensor.extract %arg0[%c1, %c1] : tensor<4x4xi1>
  %2 = stablehlo.constant dense<false> : tensor<i1>
  return %2 : tensor<i1>
}

// -----

// CHECK-LABEL: func.func @tensor_from_elements_index_cast(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<4x1xi1>
// CHECK: %[[C0:.*]] = stablehlo.constant dense<false> : tensor<i1>
// CHECK: %[[C1:.*]] = stablehlo.constant dense<true> : tensor<i1>
// CHECK: %[[C2:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK: %[[C3:.*]] = stablehlo.constant dense<true> : tensor<i1>
// CHECK: %[[RESHAPE0:.*]] = stablehlo.reshape %[[C0]] : (tensor<i1>) -> tensor<1xi1>
// CHECK: %[[RESHAPE1:.*]] = stablehlo.reshape %[[C1]] : (tensor<i1>) -> tensor<1xi1>
// CHECK: %[[RESHAPE2:.*]] = stablehlo.reshape %[[C3]] : (tensor<i1>) -> tensor<1xi1>
// CHECK: %[[RESHAPE3:.*]] = stablehlo.reshape %[[C0]] : (tensor<i1>) -> tensor<1xi1>
// CHECK: %[[CONCAT:.*]] = stablehlo.concatenate %[[RESHAPE0]], %[[RESHAPE1]], %[[RESHAPE2]], %[[RESHAPE3]], dim = 0 : (tensor<1xi1>, tensor<1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<4xi1>
// CHECK: %[[RESHAPE4:.*]] = stablehlo.reshape %[[CONCAT]] : (tensor<4xi1>) -> tensor<4x1xi1>
// CHECK: return %[[RESHAPE4]] : tensor<4x1xi1>
func.func @tensor_from_elements_index_cast(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<4x1xi1> {
  %0 = arith.constant false
  %1 = arith.constant true
  %2 = arith.constant 1 : index
  %3 = arith.index_cast %2 : index to i1
  %4 = tensor.from_elements %0, %1, %3, %0 : tensor<4x1xi1>
  return %4 : tensor<4x1xi1>
}

// -----

// CHECK-LABEL:   func.func @reshape_static(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<24xf32>) -> tensor<6x4xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<6> : tensor<i32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<4> : tensor<i32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.constant dense<[6, 4]> : tensor<2xi32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<24xf32>) -> tensor<6x4xf32>
// CHECK:           return %[[VAL_4]] : tensor<6x4xf32>
module {
  func.func @reshape_static(%arg0: tensor<24xf32>) -> tensor<6x4xf32> {
    %c6 = arith.constant 6 : i32
    %c4 = arith.constant 4 : i32
    %shape = tensor.from_elements %c6, %c4 : tensor<2xi32>
    %result = tensor.reshape %arg0(%shape) : (tensor<24xf32>, tensor<2xi32>) -> tensor<6x4xf32>
    return %result : tensor<6x4xf32>
  }
}

// -----

// CHECK-LABEL:   func.func @multi_concat(
// CHECK-SAME:                            %[[VAL_0:.*]]: tensor<2x2xf32>, %[[VAL_1:.*]]: tensor<2x2xf32>, %[[VAL_2:.*]]: tensor<2x2xf32>, %[[VAL_3:.*]]: tensor<2x2xf32>) -> tensor<4x4xf32> {
// CHECK:           %[[VAL_4:.*]] = stablehlo.concatenate %[[VAL_0]], %[[VAL_1]], dim = 1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.concatenate %[[VAL_2]], %[[VAL_3]], dim = 1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.concatenate %[[VAL_4]], %[[VAL_5]], dim = 0 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
// CHECK:           return %[[VAL_6]] : tensor<4x4xf32>
module {
  func.func @multi_concat(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>, %arg3: tensor<2x2xf32>) -> tensor<4x4xf32> {
    %row1 = tensor.concat dim(1) %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
    %row2 = tensor.concat dim(1) %arg2, %arg3 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
    %result = tensor.concat dim(0) %row1, %row2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
    return %result : tensor<4x4xf32>
  }
}

// -----

// CHECK-LABEL:   func.func @bitcast_and_reshape(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<8xf32>) -> tensor<2x4xi32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.bitcast_convert %[[VAL_0]] : (tensor<8xf32>) -> tensor<8xi32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<2> : tensor<i32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.constant dense<4> : tensor<i32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.constant dense<[2, 4]> : tensor<2xi32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.reshape %[[VAL_1]] : (tensor<8xi32>) -> tensor<2x4xi32>
// CHECK:           return %[[VAL_5]] : tensor<2x4xi32>
module {
  func.func @bitcast_and_reshape(%arg0: tensor<8xf32>) -> tensor<2x4xi32> {
    %0 = tensor.bitcast %arg0 : tensor<8xf32> to tensor<8xi32>
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %shape = tensor.from_elements %c2, %c4 : tensor<2xi32>
    %1 = tensor.reshape %0(%shape) : (tensor<8xi32>, tensor<2xi32>) -> tensor<2x4xi32>
    return %1 : tensor<2x4xi32>
  }
}

// -----

// CHECK-LABEL:   func.func @empty_multi_use() -> (tensor<4x4xf32>, tensor<16xf32>) {
// CHECK:           %[[VAL_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<16> : tensor<i32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<16> : tensor<1xi32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<4x4xf32>) -> tensor<16xf32>
// CHECK:           return %[[VAL_0]], %[[VAL_3]] : tensor<4x4xf32>, tensor<16xf32>
module {
  func.func @empty_multi_use() -> (tensor<4x4xf32>, tensor<16xf32>) {
    %empty = tensor.empty() : tensor<4x4xf32>
    %c16 = arith.constant 16 : i32
    %shape = tensor.from_elements %c16 : tensor<1xi32>
    %reshaped = tensor.reshape %empty(%shape) : (tensor<4x4xf32>, tensor<1xi32>) -> tensor<16xf32>
    return %empty, %reshaped : tensor<4x4xf32>, tensor<16xf32>
  }
}

// -----

// CHECK-LABEL:   func.func @extract_slice_static(
// CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<8x16x32xf32>) -> tensor<4x8x16xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [2:6, 4:12, 8:24] : (tensor<8x16x32xf32>) -> tensor<4x8x16xf32>
// CHECK:           return %[[VAL_1]] : tensor<4x8x16xf32>
module {
  func.func @extract_slice_static(%arg0: tensor<8x16x32xf32>) -> tensor<4x8x16xf32> {
    %slice = tensor.extract_slice %arg0[2, 4, 8] [4, 8, 16] [1, 1, 1] : tensor<8x16x32xf32> to tensor<4x8x16xf32>
    return %slice : tensor<4x8x16xf32>
  }
}

// -----

// CHECK-LABEL:   func.func @extract_slice_dynamic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<8x16x32xf32>) -> tensor<4x8x16xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<4> : tensor<i64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.constant dense<8> : tensor<i64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.dynamic_slice %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], sizes = [4, 8, 16] : (tensor<8x16x32xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x8x16xf32>
// CHECK:           return %[[VAL_4]] : tensor<4x8x16xf32>
module {
  func.func @extract_slice_dynamic(%arg0: tensor<8x16x32xf32>) -> tensor<4x8x16xf32> {
    %c0 = arith.constant 2 : index
    %slice = tensor.extract_slice %arg0[%c0, 4, 8] [4, 8, 16] [1, 1, 1] : tensor<8x16x32xf32> to tensor<4x8x16xf32>
    return %slice : tensor<4x8x16xf32>
  }
}

// -----

// CHECK-LABEL:   func.func @insert_dynamic(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<8x16x4xf32>) -> tensor<8x16x4xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<2.000000e+02> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.reshape %[[VAL_2]] : (tensor<f32>) -> tensor<1x1x1xf32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.dynamic_update_slice %[[VAL_0]], %[[VAL_3]], %[[VAL_1]], %[[VAL_1]], %[[VAL_1]] : (tensor<8x16x4xf32>, tensor<1x1x1xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<8x16x4xf32>
// CHECK:           return %[[VAL_4]] : tensor<8x16x4xf32>
module {
  func.func @insert_dynamic(%arg0: tensor<8x16x4xf32>) -> tensor<8x16x4xf32> {
    %c1 = arith.constant 0 : index
    %c2 = arith.constant 200.0 : f32
    %1 = tensor.insert %c2 into %arg0[%c1, %c1, %c1] :  tensor<8x16x4xf32>
    return %1 : tensor<8x16x4xf32>
  }
}

// -----

// CHECK-LABEL:   func.func @splat_op(
// CHECK-SAME:                        %[[VAL_0:.*]]: tensor<f32>) -> tensor<8x16xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.broadcast_in_dim %[[VAL_0]], dims = [] : (tensor<f32>) -> tensor<8x16xf32>
// CHECK:           return %[[VAL_1]] : tensor<8x16xf32>
module {
  func.func @splat_op(%arg0: tensor<f32>) -> tensor<8x16xf32> {
    %c = tensor.extract %arg0 [] : tensor<f32>
    %t = tensor.splat %c : tensor<8x16xf32>
    return %t : tensor<8x16xf32>
  }
}

// -----

// CHECK-LABEL:   func.func @insert_slice_static(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<1x16x4xf32>, %[[VAL_1:.*]]: tensor<8x16x4xf32>) -> tensor<8x16x4xf32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.dynamic_update_slice %[[VAL_1]], %[[VAL_0]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : (tensor<8x16x4xf32>, tensor<1x16x4xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<8x16x4xf32>
// CHECK:           return %[[VAL_5]] : tensor<8x16x4xf32>
module {
  func.func @insert_slice_static(%arg0: tensor<1x16x4xf32>, %arg1: tensor<8x16x4xf32>) -> tensor<8x16x4xf32> {
    %1 = tensor.insert_slice %arg0 into %arg1[0, 0, 0][1, 16, 4][1, 1, 1] : tensor<1x16x4xf32> into tensor<8x16x4xf32>
    return %1 : tensor<8x16x4xf32>
  }
}

// -----

module {
  func.func @tensor_rank_static(%arg0: tensor<4x8xf32>) -> tensor<i64> {
    %rank = tensor.rank %arg0 : tensor<4x8xf32>
    %rank_i64 = arith.index_cast %rank : index to i64
    %result = arith.constant dense<0> : tensor<i64>
    return %result : tensor<i64>
  }
  // CHECK-LABEL:   func.func @tensor_rank_static(
  // CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<4x8xf32>) -> tensor<i64> {
  // CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<2> : tensor<i64>
  // CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK:           return %[[VAL_2]] : tensor<i64>
  // CHECK:         }

  func.func @tensor_rank_scalar(%arg0: tensor<f32>) -> tensor<i64> {
    %rank = tensor.rank %arg0 : tensor<f32>
    %rank_i64 = arith.index_cast %rank : index to i64
    %result = arith.constant dense<0> : tensor<i64>
    return %result : tensor<i64>
  }
  // CHECK-LABEL:   func.func @tensor_rank_scalar(
  // CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<f32>) -> tensor<i64> {
  // CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK:           return %[[VAL_2]] : tensor<i64>
  // CHECK:         }
}

// -----

module {
  // CHECK-LABEL:   func.func @tensor_cast_chain(
  // CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK:           return %[[VAL_0]] : tensor<4x8xf32>
  // CHECK:         }
  func.func @tensor_cast_chain(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %0 = tensor.cast %arg0 : tensor<4x8xf32> to tensor<4x8xf32>
    %1 = tensor.cast %0 : tensor<4x8xf32> to tensor<4x8xf32>
    %2 = tensor.cast %1 : tensor<4x8xf32> to tensor<4x8xf32>
    return %2 : tensor<4x8xf32>
  }

  // CHECK-LABEL:   func.func @tensor_cast_with_ops(
  // CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<4x8xf32>,
  // CHECK-SAME:                                    %[[VAL_1:.*]]: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK:           %[[VAL_2:.*]] = stablehlo.add %[[VAL_0]], %[[VAL_1]] : tensor<4x8xf32>
  // CHECK:           return %[[VAL_2]] : tensor<4x8xf32>
  // CHECK:         }
  func.func @tensor_cast_with_ops(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %0 = tensor.cast %arg0 : tensor<4x8xf32> to tensor<4x8xf32>
    %1 = arith.addf %0, %arg1 : tensor<4x8xf32>
    return %1 : tensor<4x8xf32>
  }

  // CHECK-LABEL:   func.func @tensor_cast_1d(
  // CHECK-SAME:                              %[[VAL_0:.*]]: tensor<16xf32>) -> tensor<16xf32> {
  // CHECK:           return %[[VAL_0]] : tensor<16xf32>
  // CHECK:         }
  func.func @tensor_cast_1d(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %0 = tensor.cast %arg0 : tensor<16xf32> to tensor<16xf32>
    return %0 : tensor<16xf32>
  }

  // CHECK-LABEL:   func.func @tensor_cast_scalar(
  // CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<f32>) -> tensor<f32> {
  // CHECK:           return %[[VAL_0]] : tensor<f32>
  // CHECK:         }
  func.func @tensor_cast_scalar(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = tensor.cast %arg0 : tensor<f32> to tensor<f32>
    return %0 : tensor<f32>
  }
}

// -----

module {
// CHECK-LABEL:   func.func @collapse_4d_to_2d(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<2x3x4x5xf32>) -> tensor<6x20xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<2x3x4x5xf32>) -> tensor<6x20xf32>
// CHECK:           return %[[VAL_1]] : tensor<6x20xf32>
// CHECK:         }
func.func @collapse_4d_to_2d(%arg0: tensor<2x3x4x5xf32>) -> tensor<6x20xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<2x3x4x5xf32> into tensor<6x20xf32>
  return %0 : tensor<6x20xf32>
}

// CHECK-LABEL:   func.func @collapse_4d_to_3d(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<2x3x4x5xf32>) -> tensor<2x12x5xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<2x3x4x5xf32>) -> tensor<2x12x5xf32>
// CHECK:           return %[[VAL_1]] : tensor<2x12x5xf32>
// CHECK:         }
func.func @collapse_4d_to_3d(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x12x5xf32> {
  %0 = tensor.collapse_shape %arg0 [[0], [1, 2], [3]] : tensor<2x3x4x5xf32> into tensor<2x12x5xf32>
  return %0 : tensor<2x12x5xf32>
}

// CHECK-LABEL:   func.func @collapse_5d_to_3d(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<2x3x4x5x6xf32>) -> tensor<6x20x6xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<2x3x4x5x6xf32>) -> tensor<6x20x6xf32>
// CHECK:           return %[[VAL_1]] : tensor<6x20x6xf32>
// CHECK:         }
func.func @collapse_5d_to_3d(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<6x20x6xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3], [4]] : tensor<2x3x4x5x6xf32> into tensor<6x20x6xf32>
  return %0 : tensor<6x20x6xf32>
}

// CHECK-LABEL:   func.func @collapse_i32(
// CHECK-SAME:                            %[[VAL_0:.*]]: tensor<4x8xi32>) -> tensor<32xi32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<4x8xi32>) -> tensor<32xi32>
// CHECK:           return %[[VAL_1]] : tensor<32xi32>
// CHECK:         }
func.func @collapse_i32(%arg0: tensor<4x8xi32>) -> tensor<32xi32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<4x8xi32> into tensor<32xi32>
  return %0 : tensor<32xi32>
}
}

// -----

module {
// CHECK-LABEL:   func.func @expand_1d_to_2d(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<32xf32>) -> tensor<4x8xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<32xf32>) -> tensor<4x8xf32>
// CHECK:           return %[[VAL_1]] : tensor<4x8xf32>
// CHECK:         }
func.func @expand_1d_to_2d(%arg0: tensor<32xf32>) -> tensor<4x8xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1]] output_shape [4, 8] : tensor<32xf32> into tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL:   func.func @expand_2d_to_4d(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<6x20xf32>) -> tensor<2x3x4x5xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<6x20xf32>) -> tensor<2x3x4x5xf32>
// CHECK:           return %[[VAL_1]] : tensor<2x3x4x5xf32>
// CHECK:         }
func.func @expand_2d_to_4d(%arg0: tensor<6x20xf32>) -> tensor<2x3x4x5xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [2, 3, 4, 5] : tensor<6x20xf32> into tensor<2x3x4x5xf32>
  return %0 : tensor<2x3x4x5xf32>
}

// CHECK-LABEL:   func.func @expand_3d_to_4d(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<2x12x5xf32>) -> tensor<2x3x4x5xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<2x12x5xf32>) -> tensor<2x3x4x5xf32>
// CHECK:           return %[[VAL_1]] : tensor<2x3x4x5xf32>
// CHECK:         }
func.func @expand_3d_to_4d(%arg0: tensor<2x12x5xf32>) -> tensor<2x3x4x5xf32> {
  %0 = tensor.expand_shape %arg0 [[0], [1, 2], [3]] output_shape [2, 3, 4, 5] : tensor<2x12x5xf32> into tensor<2x3x4x5xf32>
  return %0 : tensor<2x3x4x5xf32>
}

// CHECK-LABEL:   func.func @expand_i32(
// CHECK-SAME:                          %[[VAL_0:.*]]: tensor<32xi32>) -> tensor<4x8xi32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<32xi32>) -> tensor<4x8xi32>
// CHECK:           return %[[VAL_1]] : tensor<4x8xi32>
// CHECK:         }
func.func @expand_i32(%arg0: tensor<32xi32>) -> tensor<4x8xi32> {
  %0 = tensor.expand_shape %arg0 [[0, 1]] output_shape [4, 8] : tensor<32xi32> into tensor<4x8xi32>
  return %0 : tensor<4x8xi32>
}
}

// -----

module {
// CHECK-LABEL:   func.func @collapse_expand_roundtrip(
// CHECK-SAME:                                         %[[VAL_0:.*]]: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<2x3x4xf32>) -> tensor<24xf32>
// CHECK:           return %[[VAL_0]] : tensor<2x3x4xf32>
// CHECK:         }
func.func @collapse_expand_roundtrip(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<2x3x4xf32> into tensor<24xf32>
  %1 = tensor.expand_shape %0 [[0, 1, 2]] output_shape [2, 3, 4] : tensor<24xf32> into tensor<2x3x4xf32>
  return %1 : tensor<2x3x4xf32>
}

// CHECK-LABEL:   func.func @multiple_collapse(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<2x3x4x5xf32>) -> tensor<120xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<2x3x4x5xf32>) -> tensor<6x20xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.reshape %[[VAL_1]] : (tensor<6x20xf32>) -> tensor<120xf32>
// CHECK:           return %[[VAL_2]] : tensor<120xf32>
// CHECK:         }
func.func @multiple_collapse(%arg0: tensor<2x3x4x5xf32>) -> tensor<120xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<2x3x4x5xf32> into tensor<6x20xf32>
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<6x20xf32> into tensor<120xf32>
  return %1 : tensor<120xf32>
}

// CHECK-LABEL:   func.func @multiple_expand(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<120xf32>) -> tensor<2x3x4x5xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<120xf32>) -> tensor<6x20xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.reshape %[[VAL_1]] : (tensor<6x20xf32>) -> tensor<2x3x4x5xf32>
// CHECK:           return %[[VAL_2]] : tensor<2x3x4x5xf32>
// CHECK:         }
func.func @multiple_expand(%arg0: tensor<120xf32>) -> tensor<2x3x4x5xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1]] output_shape [6, 20] : tensor<120xf32> into tensor<6x20xf32>
  %1 = tensor.expand_shape %0 [[0, 1], [2, 3]] output_shape [2, 3, 4, 5] : tensor<6x20xf32> into tensor<2x3x4x5xf32>
  return %1 : tensor<2x3x4x5xf32>
}

// CHECK-LABEL:   func.func @collapse_expand_different(
// CHECK-SAME:                                         %[[VAL_0:.*]]: tensor<2x3x4xf32>) -> tensor<3x8xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<2x3x4xf32>) -> tensor<24xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.reshape %[[VAL_1]] : (tensor<24xf32>) -> tensor<3x8xf32>
// CHECK:           return %[[VAL_2]] : tensor<3x8xf32>
// CHECK:         }
func.func @collapse_expand_different(%arg0: tensor<2x3x4xf32>) -> tensor<3x8xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<2x3x4xf32> into tensor<24xf32>
  %1 = tensor.expand_shape %0 [[0, 1]] output_shape [3, 8] : tensor<24xf32> into tensor<3x8xf32>
  return %1 : tensor<3x8xf32>
}
}

// -----

module {
// CHECK-LABEL:   func.func @pad_negative(
// CHECK-SAME:                            %[[VAL_0:.*]]: tensor<4x8xf32>) -> tensor<6x10xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.pad %[[VAL_0]], %[[VAL_1]], low = [1, 1], high = [1, 1], interior = [0, 0] : (tensor<4x8xf32>, tensor<f32>) -> tensor<6x10xf32>
// CHECK:           return %[[VAL_2]] : tensor<6x10xf32>
// CHECK:         }
func.func @pad_negative(%arg0: tensor<4x8xf32>) -> tensor<6x10xf32> {
  %pad_value = arith.constant -1.0 : f32
  %0 = tensor.pad %arg0 low[1, 1] high[1, 1] {
  ^bb0(%arg1: index, %arg2: index):
    tensor.yield %pad_value : f32
  } : tensor<4x8xf32> to tensor<6x10xf32>
  return %0 : tensor<6x10xf32>
}

// CHECK-LABEL:   func.func @pad_with_tensor_arithmetic(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tensor<4x8xf32>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: tensor<f32>) -> tensor<6x10xf32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.add %[[VAL_1]], %[[VAL_2]] : tensor<f32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.pad %[[VAL_0]], %[[VAL_3]], low = [1, 1], high = [1, 1], interior = [0, 0] : (tensor<4x8xf32>, tensor<f32>) -> tensor<6x10xf32>
// CHECK:           return %[[VAL_4]] : tensor<6x10xf32>
// CHECK:         }
func.func @pad_with_tensor_arithmetic(%arg0: tensor<4x8xf32>, %pad_val: tensor<f32>) -> tensor<6x10xf32> {
  %c1 = arith.constant dense<2.0> : tensor<f32>
  %added = arith.addf %pad_val, %c1 : tensor<f32>
  %scalar = tensor.extract %added[] : tensor<f32>
  %0 = tensor.pad %arg0 low[1, 1] high[1, 1] {
  ^bb0(%arg1_idx: index, %arg2_idx: index):
    tensor.yield %scalar : f32
  } : tensor<4x8xf32> to tensor<6x10xf32>
  return %0 : tensor<6x10xf32>
}

// CHECK-LABEL:   func.func @pad_with_complex_tensor_ops(
// CHECK-SAME:                                           %[[VAL_0:.*]]: tensor<4x8xf32>,
// CHECK-SAME:                                           %[[VAL_1:.*]]: tensor<f32>) -> tensor<6x10xf32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.constant dense<3.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.add %[[VAL_1]], %[[VAL_2]] : tensor<f32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.multiply %[[VAL_4]], %[[VAL_3]] : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.pad %[[VAL_0]], %[[VAL_5]], low = [1, 1], high = [1, 1], interior = [0, 0] : (tensor<4x8xf32>, tensor<f32>) -> tensor<6x10xf32>
// CHECK:           return %[[VAL_6]] : tensor<6x10xf32>
// CHECK:         }
func.func @pad_with_complex_tensor_ops(%arg0: tensor<4x8xf32>, %pad_val: tensor<f32>) -> tensor<6x10xf32> {
  %c1 = arith.constant dense<2.0> : tensor<f32>
  %c2 = arith.constant dense<3.0> : tensor<f32>
  %add1 = arith.addf %pad_val, %c1 : tensor<f32>
  %mul = arith.mulf %add1, %c2 : tensor<f32>
  %scalar = tensor.extract %mul[] : tensor<f32>
  %0 = tensor.pad %arg0 low[1, 1] high[1, 1] {
  ^bb0(%arg1_idx: index, %arg2_idx: index):
    tensor.yield %scalar : f32
  } : tensor<4x8xf32> to tensor<6x10xf32>
  return %0 : tensor<6x10xf32>
}

// CHECK-LABEL:   func.func @pad_nofold_true_same_shape(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tensor<4x8xf32>) -> tensor<4x8xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.pad %[[VAL_0]], %[[VAL_1]], low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.pad %[[VAL_2]], %[[VAL_1]], low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
// CHECK:           return %[[VAL_3]] : tensor<4x8xf32>
// CHECK:         }
func.func @pad_nofold_true_same_shape(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %pad_value = arith.constant 0.0 : f32
  %0 = tensor.pad %arg0 nofold low[0, 0] high[0, 0] {
  ^bb0(%arg1_idx: index, %arg2_idx: index):
    tensor.yield %pad_value : f32
  } : tensor<4x8xf32> to tensor<4x8xf32>
  %1 = tensor.pad %0 nofold low[0, 0] high[0, 0] {
  ^bb0(%arg1_idx: index, %arg2_idx: index):
    tensor.yield %pad_value : f32
  } : tensor<4x8xf32> to tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// CHECK-LABEL:   func.func @pad_default_nofold_same_shape(
// CHECK-SAME:                                             %[[VAL_0:.*]]: tensor<4x8xf32>) -> tensor<4x8xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           return %[[VAL_0]] : tensor<4x8xf32>
// CHECK:         }
func.func @pad_default_nofold_same_shape(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %pad_value = arith.constant 0.0 : f32
  %0 = tensor.pad %arg0 low[0, 0] high[0, 0] {
  ^bb0(%arg1_idx: index, %arg2_idx: index):
    tensor.yield %pad_value : f32
  } : tensor<4x8xf32> to tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
}

// -----

// arith.constant scalar index -> stablehlo.constant tensor<i64>
// CHECK-LABEL: func.func @arith_constant_index_scalar
// CHECK-SAME:  () -> tensor<i64> {
// CHECK:         %[[C:.*]] = stablehlo.constant dense<42> : tensor<i64>
// CHECK:         %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:         return %[[C0]] : tensor<i64>
// CHECK:       }
func.func @arith_constant_index_scalar() -> tensor<i64> {
  %0 = arith.constant 42 : index
  %1 = stablehlo.constant dense<0> : tensor<i64>
  return %1 : tensor<i64>
}

// -----

// arith.constant scalar index with different value
// CHECK-LABEL: func.func @arith_constant_index_scalar_99
// CHECK-SAME:  () -> tensor<i64> {
// CHECK:         %[[C:.*]] = stablehlo.constant dense<99> : tensor<i64>
// CHECK:         %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:         return %[[C0]] : tensor<i64>
// CHECK:       }
func.func @arith_constant_index_scalar_99() -> tensor<i64> {
  %0 = arith.constant 99 : index
  %1 = stablehlo.constant dense<0> : tensor<i64>
  return %1 : tensor<i64>
}

// -----

// arith.constant non-index i32 unchanged (negative test)
// CHECK-LABEL: func.func @arith_constant_non_index
// CHECK-SAME:  () -> tensor<i32> {
// CHECK:         %[[C:.*]] = stablehlo.constant dense<7> : tensor<i32>
// CHECK:         %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK:         return %[[C0]] : tensor<i32>
// CHECK:       }
func.func @arith_constant_non_index() -> tensor<i32> {
  %0 = arith.constant 7 : i32
  %1 = stablehlo.constant dense<0> : tensor<i32>
  return %1 : tensor<i32>
}

// -----

// stablehlo op with non-index operands passes through
// CHECK-LABEL: func.func @stablehlo_legal_non_index
// CHECK-SAME:  (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<4xf32>, %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1xi64>) -> tensor<4xf32> {
// CHECK:         %[[R:.*]] = stablehlo.dynamic_broadcast_in_dim %[[ARG0]], %[[ARG1]], dims = [0] : (tensor<4xf32>, tensor<1xi64>) -> tensor<4xf32>
// CHECK:         return %[[R]] : tensor<4xf32>
// CHECK:       }
func.func @stablehlo_legal_non_index(%arg0: tensor<4xf32>, %arg1: tensor<1xi64>) -> tensor<4xf32> {
  %0 = stablehlo.dynamic_broadcast_in_dim %arg0, %arg1, dims = [0] : (tensor<4xf32>, tensor<1xi64>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// tensor.dim on static shape produces constant (dim result consumed internally)
// CHECK-LABEL: func.func @tensor_dim_static
// CHECK-SAME:  (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<4x8xf32>) -> tensor<i1> {
// CHECK:         %[[C4:.*]] = stablehlo.constant dense<4> : tensor<i64>
// CHECK:         %[[FALSE:.*]] = stablehlo.constant dense<false> : tensor<i1>
// CHECK:         return %[[FALSE]] : tensor<i1>
// CHECK:       }
func.func @tensor_dim_static(%arg0: tensor<4x8xf32>) -> tensor<i1> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<4x8xf32>
  %result = stablehlo.constant dense<false> : tensor<i1>
  return %result : tensor<i1>
}

// -----

// tensor.dim on dynamic shape uses get_dimension_size
// CHECK-LABEL: func.func @tensor_dim_dynamic
// CHECK-SAME:  (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x8xf32>) -> tensor<i1> {
// CHECK:         %[[DIM:.*]] = stablehlo.get_dimension_size %[[ARG0]], dim = 0 : (tensor<?x8xf32>) -> tensor<i32>
// CHECK:         %[[CVT:.*]] = stablehlo.convert %[[DIM]] : (tensor<i32>) -> tensor<i64>
// CHECK:         %[[FALSE:.*]] = stablehlo.constant dense<false> : tensor<i1>
// CHECK:         return %[[FALSE]] : tensor<i1>
// CHECK:       }
func.func @tensor_dim_dynamic(%arg0: tensor<?x8xf32>) -> tensor<i1> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x8xf32>
  %result = stablehlo.constant dense<false> : tensor<i1>
  return %result : tensor<i1>
}

// -----

// tensor.dim on second dynamic dimension
// CHECK-LABEL: func.func @tensor_dim_dynamic_dim1
// CHECK-SAME:  (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<4x?xf32>) -> tensor<i1> {
// CHECK:         %[[DIM:.*]] = stablehlo.get_dimension_size %[[ARG0]], dim = 1 : (tensor<4x?xf32>) -> tensor<i32>
// CHECK:         %[[CVT:.*]] = stablehlo.convert %[[DIM]] : (tensor<i32>) -> tensor<i64>
// CHECK:         %[[FALSE:.*]] = stablehlo.constant dense<false> : tensor<i1>
// CHECK:         return %[[FALSE]] : tensor<i1>
// CHECK:       }
func.func @tensor_dim_dynamic_dim1(%arg0: tensor<4x?xf32>) -> tensor<i1> {
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c1 : tensor<4x?xf32>
  %result = stablehlo.constant dense<false> : tensor<i1>
  return %result : tensor<i1>
}

// -----

// tensor.cast identity (same type) is no-op
// CHECK-LABEL: func.func @tensor_cast_identity
// CHECK-SAME:  (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         return %[[ARG0]] : tensor<?x?xf32>
// CHECK:       }
func.func @tensor_cast_identity(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tensor.cast %arg0 : tensor<?x?xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// tensor.cast to fully static shape uses reshape
// CHECK-LABEL: func.func @tensor_cast_to_static
// CHECK-SAME:  (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>) -> tensor<4x8xf32> {
// CHECK:         %[[RES:.*]] = stablehlo.reshape %[[ARG0]] : (tensor<?x?xf32>) -> tensor<4x8xf32>
// CHECK:         return %[[RES]] : tensor<4x8xf32>
// CHECK:       }
func.func @tensor_cast_to_static(%arg0: tensor<?x?xf32>) -> tensor<4x8xf32> {
  %0 = tensor.cast %arg0 : tensor<?x?xf32> to tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

// tensor.cast shape refinement uses dynamic_reshape
// CHECK-LABEL: func.func @tensor_cast_shape_refinement
// CHECK-SAME:  (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>) -> tensor<1x?x?xf32> {
// CHECK:         stablehlo.dynamic_reshape %[[ARG0]]
// CHECK-SAME:    -> tensor<1x?x?xf32>
// CHECK:       }
func.func @tensor_cast_shape_refinement(%arg0: tensor<?x?x?xf32>) -> tensor<1x?x?xf32> {
  %0 = tensor.cast %arg0 : tensor<?x?x?xf32> to tensor<1x?x?xf32>
  return %0 : tensor<1x?x?xf32>
}

// -----

// shape.shape_of on rank-2 tensor
// CHECK-LABEL: func.func @shape_shape_of_rank2
// CHECK-SAME:  (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>) -> tensor<2xi64> {
// CHECK:         %[[D0:.*]] = stablehlo.get_dimension_size %[[ARG0]], dim = 0 : (tensor<?x?xf32>) -> tensor<i32>
// CHECK:         %[[D0_I64:.*]] = stablehlo.convert %[[D0]] : (tensor<i32>) -> tensor<i64>
// CHECK:         %[[D0_R:.*]] = stablehlo.reshape %[[D0_I64]] : (tensor<i64>) -> tensor<1xi64>
// CHECK:         %[[D1:.*]] = stablehlo.get_dimension_size %[[ARG0]], dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
// CHECK:         %[[D1_I64:.*]] = stablehlo.convert %[[D1]] : (tensor<i32>) -> tensor<i64>
// CHECK:         %[[D1_R:.*]] = stablehlo.reshape %[[D1_I64]] : (tensor<i64>) -> tensor<1xi64>
// CHECK:         %[[RESULT:.*]] = stablehlo.concatenate %[[D0_R]], %[[D1_R]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:       }
func.func @shape_shape_of_rank2(%arg0: tensor<?x?xf32>) -> tensor<2xindex> {
  %0 = shape.shape_of %arg0 : tensor<?x?xf32> -> tensor<2xindex>
  return %0 : tensor<2xindex>
}

// -----

// shape.shape_of on rank-3 tensor
// CHECK-LABEL: func.func @shape_shape_of_rank3
// CHECK-SAME:  (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>) -> tensor<3xi64> {
// CHECK:         stablehlo.get_dimension_size %[[ARG0]], dim = 0
// CHECK:         stablehlo.get_dimension_size %[[ARG0]], dim = 1
// CHECK:         stablehlo.get_dimension_size %[[ARG0]], dim = 2
// CHECK:         %[[RESULT:.*]] = stablehlo.concatenate {{.*}}, dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:       }
func.func @shape_shape_of_rank3(%arg0: tensor<?x?x?xf32>) -> tensor<3xindex> {
  %0 = shape.shape_of %arg0 : tensor<?x?x?xf32> -> tensor<3xindex>
  return %0 : tensor<3xindex>
}

// -----

// shape.num_elements reduces shape to scalar product
// CHECK-LABEL: func.func @shape_num_elements
// CHECK-SAME:  (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<3xi64>) -> tensor<i64> {
// CHECK:         %[[INIT:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK:         %[[REDUCE:.*]] = stablehlo.reduce(%{{.*}} init: %[[INIT]])
// CHECK:       }
func.func @shape_num_elements(%arg0: tensor<3xindex>) -> index {
  %0 = shape.num_elements %arg0 : tensor<3xindex> -> index
  return %0 : index
}

// -----

// shape.num_elements on rank-2 shape
// CHECK-LABEL: func.func @shape_num_elements_rank2
// CHECK-SAME:  (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<2xi64>) -> tensor<i64> {
// CHECK:         %[[INIT:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK:         %[[REDUCE:.*]] = stablehlo.reduce(%{{.*}} init: %[[INIT]])
// CHECK:       }
func.func @shape_num_elements_rank2(%arg0: tensor<2xindex>) -> index {
  %0 = shape.num_elements %arg0 : tensor<2xindex> -> index
  return %0 : index
}

// -----

// function signature conversion - i64 arg becomes tensor<i64>
// CHECK-LABEL: func.func @func_sig_i64_arg
// CHECK-SAME:  (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<i64>) -> tensor<i64> {
// CHECK:         %[[C:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:         return %[[C]] : tensor<i64>
// CHECK:       }
func.func @func_sig_i64_arg(%arg0: i64) -> tensor<i64> {
  %0 = stablehlo.constant dense<0> : tensor<i64>
  return %0 : tensor<i64>
}

// -----

// function signature conversion - f64 arg becomes tensor<f64>
// CHECK-LABEL: func.func @func_sig_f64_arg
// CHECK-SAME:  (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<f64>) -> tensor<f64> {
// CHECK:         %[[C:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:         return %[[C]] : tensor<f64>
// CHECK:       }
func.func @func_sig_f64_arg(%arg0: f64) -> tensor<f64> {
  %0 = stablehlo.constant dense<0.0> : tensor<f64>
  return %0 : tensor<f64>
}

// -----

// function with index arg gets converted to tensor<i64>
// CHECK-LABEL: func.func @func_sig_index_arg
// CHECK-SAME:  (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<i64>) -> tensor<i64> {
// CHECK:         %[[C:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:         return %[[C]] : tensor<i64>
// CHECK:       }
func.func @func_sig_index_arg(%arg0: index) -> tensor<i64> {
  %0 = stablehlo.constant dense<0> : tensor<i64>
  return %0 : tensor<i64>
}

// -----

// CHECK-LABEL: func.func @math_ipowi
// CHECK-SAME:  (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<i64>, %[[ARG1:[a-zA-Z0-9_]+]]: tensor<i64>) -> tensor<i64> {
// CHECK:         %[[RES:.*]] = stablehlo.power %[[ARG0]], %[[ARG1]] : tensor<i64>
// CHECK:         return %[[RES]] : tensor<i64>
// CHECK:       }
func.func @math_ipowi(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
  %0 = math.ipowi %arg0, %arg1 : tensor<i64>
  return %0 : tensor<i64>
}

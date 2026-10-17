// RUN: torch-mlir-opt -torch-custom-legalization -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @index_put_boolean_mask_splat_false
// CHECK: %[[FALSE_MASK:.*]] = torch.vtensor.literal(dense<false> : tensor<3x4xi1>) : !torch.vtensor<[3,4],i1>
// CHECK: %[[VALUE:.*]] = torch.vtensor.literal(dense<1.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
// CHECK: %[[INT1:.*]] = torch.constant.int 1
// CHECK: %[[FALSE:.*]] = torch.constant.bool false
// CHECK: %[[INT3:.*]] = torch.constant.int 3
// CHECK: %[[INT4:.*]] = torch.constant.int 4
// CHECK: %[[SHAPE_LIST:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[EXPANDED_MASK:.*]] = torch.aten.expand %[[FALSE_MASK]], %[[SHAPE_LIST]], %[[FALSE]] : !torch.vtensor<[3,4],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,4],i1>
// CHECK: %[[RESHAPE_LIST:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_VALUE:.*]] = torch.aten.reshape %[[VALUE]], %[[RESHAPE_LIST]] : !torch.vtensor<[],f32>, !torch.list<int> -> !torch.vtensor<[1,1],f32>
// CHECK: %[[EXPANDED_VALUE:.*]] = torch.aten.expand %[[RESHAPED_VALUE]], %[[SHAPE_LIST]], %[[FALSE]] : !torch.vtensor<[1,1],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,4],f32>
// CHECK: %[[RESULT:.*]] = torch.aten.where.self %[[EXPANDED_MASK]], %[[EXPANDED_VALUE]], %arg0 : !torch.vtensor<[3,4],i1>, !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[3,4],f32>
func.func @index_put_boolean_mask_splat_false(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %false = torch.constant.bool false
  %0 = torch.vtensor.literal(dense<false> : tensor<3x4xi1>) : !torch.vtensor<[3,4],i1>
  %1 = torch.vtensor.literal(dense<1.0> : tensor<f32>) : !torch.vtensor<[],f32>
  %2 = torch.prim.ListConstruct %0 : (!torch.vtensor<[3,4],i1>) -> !torch.list<optional<vtensor>>
  %3 = torch.aten.index_put %arg0, %2, %1, %false : !torch.vtensor<[3,4],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[],f32>, !torch.bool -> !torch.vtensor<[3,4],f32>
  return %3 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: func.func @index_put_boolean_mask_splat_true
// CHECK: %[[TRUE_MASK:.*]] = torch.vtensor.literal(dense<true> : tensor<2x3xi1>) : !torch.vtensor<[2,3],i1>
// CHECK: %[[VALUE:.*]] = torch.vtensor.literal(dense<2.500000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
// CHECK: %[[INT1:.*]] = torch.constant.int 1
// CHECK: %[[FALSE:.*]] = torch.constant.bool false
// CHECK: %[[INT2:.*]] = torch.constant.int 2
// CHECK: %[[INT3:.*]] = torch.constant.int 3
// CHECK: %[[SHAPE_LIST:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[EXPANDED_MASK:.*]] = torch.aten.expand %[[TRUE_MASK]], %[[SHAPE_LIST]], %[[FALSE]] : !torch.vtensor<[2,3],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,3],i1>
// CHECK: %[[RESHAPE_LIST:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_VALUE:.*]] = torch.aten.reshape %[[VALUE]], %[[RESHAPE_LIST]] : !torch.vtensor<[],f32>, !torch.list<int> -> !torch.vtensor<[1,1],f32>
// CHECK: %[[EXPANDED_VALUE:.*]] = torch.aten.expand %[[RESHAPED_VALUE]], %[[SHAPE_LIST]], %[[FALSE]] : !torch.vtensor<[1,1],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,3],f32>
// CHECK: %[[RESULT:.*]] = torch.aten.where.self %[[EXPANDED_MASK]], %[[EXPANDED_VALUE]], %arg0 : !torch.vtensor<[2,3],i1>, !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,3],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[2,3],f32>
func.func @index_put_boolean_mask_splat_true(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %false = torch.constant.bool false
  %0 = torch.vtensor.literal(dense<true> : tensor<2x3xi1>) : !torch.vtensor<[2,3],i1>
  %1 = torch.vtensor.literal(dense<2.5> : tensor<f32>) : !torch.vtensor<[],f32>
  %2 = torch.prim.ListConstruct %0 : (!torch.vtensor<[2,3],i1>) -> !torch.list<optional<vtensor>>
  %3 = torch.aten.index_put %arg0, %2, %1, %false : !torch.vtensor<[2,3],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[],f32>, !torch.bool -> !torch.vtensor<[2,3],f32>
  return %3 : !torch.vtensor<[2,3],f32>
}

// -----

// CHECK-LABEL: func.func @index_put_boolean_mask_values_splat
// CHECK: %[[MASK:.*]] = torch.vtensor.literal(dense<{{\[\[}}true, false], [false, true]]> : tensor<2x2xi1>) : !torch.vtensor<[2,2],i1>
// CHECK: %[[VALUE:.*]] = torch.vtensor.literal(dense<3.140000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
// CHECK: %[[INT1:.*]] = torch.constant.int 1
// CHECK: %[[FALSE:.*]] = torch.constant.bool false
// CHECK: %[[INT2:.*]] = torch.constant.int 2
// CHECK: %[[SHAPE_LIST:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[EXPANDED_MASK:.*]] = torch.aten.expand %[[MASK]], %[[SHAPE_LIST]], %[[FALSE]] : !torch.vtensor<[2,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,2],i1>
// CHECK: %[[RESHAPE_LIST:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_VALUE:.*]] = torch.aten.reshape %[[VALUE]], %[[RESHAPE_LIST]] : !torch.vtensor<[],f32>, !torch.list<int> -> !torch.vtensor<[1,1],f32>
// CHECK: %[[EXPANDED_VALUE:.*]] = torch.aten.expand %[[RESHAPED_VALUE]], %[[SHAPE_LIST]], %[[FALSE]] : !torch.vtensor<[1,1],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,2],f32>
// CHECK: %[[RESULT:.*]] = torch.aten.where.self %[[EXPANDED_MASK]], %[[EXPANDED_VALUE]], %arg0 : !torch.vtensor<[2,2],i1>, !torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32> -> !torch.vtensor<[2,2],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[2,2],f32>
func.func @index_put_boolean_mask_values_splat(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,2],f32> {
  %false = torch.constant.bool false
  %0 = torch.vtensor.literal(dense<[[true, false], [false, true]]> : tensor<2x2xi1>) : !torch.vtensor<[2,2],i1>
  %1 = torch.vtensor.literal(dense<3.14> : tensor<f32>) : !torch.vtensor<[],f32>
  %2 = torch.prim.ListConstruct %0 : (!torch.vtensor<[2,2],i1>) -> !torch.list<optional<vtensor>>
  %3 = torch.aten.index_put %arg0, %2, %1, %false : !torch.vtensor<[2,2],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[],f32>, !torch.bool -> !torch.vtensor<[2,2],f32>
  return %3 : !torch.vtensor<[2,2],f32>
}

// -----

// CHECK-LABEL: func.func @index_put_boolean_mask_values_dense
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[FLATTENED_MASK:.*]] = torch.vtensor.literal(dense<[true, false, false, true, true, false]> : tensor<6xi1>) : !torch.vtensor<[6],i1>
// CHECK: %[[INT3:.*]] = torch.constant.int 3
// CHECK: %[[INT0:.*]] = torch.constant.int 0
// CHECK: %[[INT1:.*]] = torch.constant.int 1
// CHECK: %[[FALSE:.*]] = torch.constant.bool false
// CHECK: %[[MASK:.*]] = torch.vtensor.literal(dense<{{\[\[}}true, false], [false, true], [true, false]]> : tensor<3x2xi1>) : !torch.vtensor<[3,2],i1>
// CHECK: %[[VALUES:.*]] = torch.vtensor.literal(dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>) : !torch.vtensor<[3],f32>
// CHECK: %[[INT2:.*]] = torch.constant.int 2
// CHECK: %[[NUMEL:.*]] = torch.aten.numel %[[MASK]] : !torch.vtensor<[3,2],i1> -> !torch.int
// CHECK: %[[DTYPE_CONVERTED:.*]] = torch.aten.to.dtype %[[FLATTENED_MASK]], %[[INT3]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[6],i1>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[6],si32>
// CHECK: %[[CUMSUM:.*]] = torch.aten.cumsum %[[DTYPE_CONVERTED]], %[[INT0]], %[[NONE]] : !torch.vtensor<[6],si32>, !torch.int, !torch.none -> !torch.vtensor<[6],si32>
// CHECK: %[[SUB:.*]] = torch.aten.sub.Scalar %[[CUMSUM]], %[[INT1]], %[[INT1]] : !torch.vtensor<[6],si32>, !torch.int, !torch.int -> !torch.vtensor<[6],si32>
// CHECK: %[[CLAMP:.*]] = torch.aten.clamp_min %[[SUB]], %[[INT0]] : !torch.vtensor<[6],si32>, !torch.int -> !torch.vtensor<[6],si32>
// CHECK: %[[BROADCAST_SHAPE:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT2]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[ARANGE_ROWS:.*]] = torch.aten.arange.start_step %[[INT0]], %[[INT3]], %[[INT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3],si32>
// CHECK: %[[RESHAPE_ROWS_SHAPE:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT1]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_ROWS:.*]] = torch.aten.reshape %[[ARANGE_ROWS]], %[[RESHAPE_ROWS_SHAPE]] : !torch.vtensor<[3],si32>, !torch.list<int> -> !torch.vtensor<[3,1,1],si32>
// CHECK: %[[BROADCASTED_ROWS:.*]] = torch.aten.broadcast_to %[[RESHAPED_ROWS]], %[[BROADCAST_SHAPE]] : !torch.vtensor<[3,1,1],si32>, !torch.list<int> -> !torch.vtensor<[3,2,1],si32>
// CHECK: %[[ARANGE_COLS:.*]] = torch.aten.arange.start_step %[[INT0]], %[[INT2]], %[[INT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2],si32>
// CHECK: %[[RESHAPE_COLS_SHAPE:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT2]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_COLS:.*]] = torch.aten.reshape %[[ARANGE_COLS]], %[[RESHAPE_COLS_SHAPE]] : !torch.vtensor<[2],si32>, !torch.list<int> -> !torch.vtensor<[1,2,1],si32>
// CHECK: %[[BROADCASTED_COLS:.*]] = torch.aten.broadcast_to %[[RESHAPED_COLS]], %[[BROADCAST_SHAPE]] : !torch.vtensor<[1,2,1],si32>, !torch.list<int> -> !torch.vtensor<[3,2,1],si32>
// CHECK: %[[INDICES_LIST:.*]] = torch.prim.ListConstruct %[[BROADCASTED_ROWS]], %[[BROADCASTED_COLS]] : (!torch.vtensor<[3,2,1],si32>, !torch.vtensor<[3,2,1],si32>) -> !torch.list<vtensor>
// CHECK: %[[INDICES_CAT:.*]] = torch.aten.cat %[[INDICES_LIST]], %[[INT2]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[3,2,2],si32>
// CHECK: %[[NUMEL_SHAPE_1:.*]] = torch.prim.ListConstruct %[[NUMEL]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[NUMEL_SHAPE_2:.*]] = torch.prim.ListConstruct %[[NUMEL]], %[[INT2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[FLATTENED_INDICES:.*]] = torch.aten.flatten.using_ints %[[INDICES_CAT]], %[[INT0]], %[[INT1]] : !torch.vtensor<[3,2,2],si32>, !torch.int, !torch.int -> !torch.vtensor<[6,2],si32>
// CHECK: %[[RESHAPED_MASK:.*]] = torch.aten.reshape %[[DTYPE_CONVERTED]], %[[NUMEL_SHAPE_1]] : !torch.vtensor<[6],si32>, !torch.list<int> -> !torch.vtensor<[6,1],si32>
// CHECK: %[[BROADCASTED_MASK:.*]] = torch.aten.broadcast_to %[[RESHAPED_MASK]], %[[NUMEL_SHAPE_2]] : !torch.vtensor<[6,1],si32>, !torch.list<int> -> !torch.vtensor<[6,2],si32>
// CHECK: %[[MASKED_INDICES:.*]] = torch.aten.mul.Tensor %[[FLATTENED_INDICES]], %[[BROADCASTED_MASK]] : !torch.vtensor<[6,2],si32>, !torch.vtensor<[6,2],si32> -> !torch.vtensor<[6,2],si32>
// CHECK: %[[RESHAPED_VALUES_INDEX:.*]] = torch.aten.reshape %[[CLAMP]], %[[NUMEL_SHAPE_1]] : !torch.vtensor<[6],si32>, !torch.list<int> -> !torch.vtensor<[6,1],si32>
// CHECK: %[[BROADCASTED_VALUES_INDEX:.*]] = torch.aten.broadcast_to %[[RESHAPED_VALUES_INDEX]], %[[NUMEL_SHAPE_2]] : !torch.vtensor<[6,1],si32>, !torch.list<int> -> !torch.vtensor<[6,2],si32>
// CHECK: %[[ZEROS_LIKE:.*]] = torch.aten.zeros_like %[[MASKED_INDICES]], %[[INT3]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.vtensor<[6,2],si32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[6,2],si32>
// CHECK: %[[SCATTER_ADD:.*]] = torch.aten.scatter_add %[[ZEROS_LIKE]], %[[INT0]], %[[BROADCASTED_VALUES_INDEX]], %[[MASKED_INDICES]] : !torch.vtensor<[6,2],si32>, !torch.int, !torch.vtensor<[6,2],si32>, !torch.vtensor<[6,2],si32> -> !torch.vtensor<[6,2],si32>
// CHECK: %[[NARROW:.*]] = torch.aten.narrow %[[SCATTER_ADD]], %[[INT0]], %[[INT0]], %[[INT1]] : !torch.vtensor<[6,2],si32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2],si32>
// CHECK: %[[INDEX_LIST:.*]] = torch.prim.ListConstruct %[[NARROW]] : (!torch.vtensor<[1,2],si32>) -> !torch.list<vtensor>
// CHECK: %[[RESULT:.*]] = torch.aten.index_put %arg0, %[[INDEX_LIST]], %[[VALUES]], %[[FALSE]] : !torch.vtensor<[3,2],f32>, !torch.list<vtensor>, !torch.vtensor<[3],f32>, !torch.bool -> !torch.vtensor<[3,2],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[3,2],f32>
func.func @index_put_boolean_mask_values_dense(%arg0: !torch.vtensor<[3,2],f32>) -> !torch.vtensor<[3,2],f32> {
  %false = torch.constant.bool false
  %0 = torch.vtensor.literal(dense<[[true, false], [false, true], [true, false]]> : tensor<3x2xi1>) : !torch.vtensor<[3,2],i1>
  %1 = torch.vtensor.literal(dense<[1.0, 2.0, 3.0]> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %2 = torch.prim.ListConstruct %0 : (!torch.vtensor<[3,2],i1>) -> !torch.list<optional<vtensor>>
  %3 = torch.aten.index_put %arg0, %2, %1, %false : !torch.vtensor<[3,2],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[3],f32>, !torch.bool -> !torch.vtensor<[3,2],f32>
  return %3 : !torch.vtensor<[3,2],f32>
}

// -----

// CHECK-LABEL: func.func @index_put_mixed_indices_should_fail
// This should remain unchanged (not be legalized) because it has mixed boolean and integer indices
// CHECK: %[[FALSE:.*]] = torch.constant.bool false
// CHECK: %[[BOOL_MASK:.*]] = torch.vtensor.literal(dense<[true, false, true, false]> : tensor<4xi1>) : !torch.vtensor<[4],i1>
// CHECK: %[[INT_INDICES:.*]] = torch.vtensor.literal(dense<[0, 1, 2, 3]> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
// CHECK: %[[VALUE:.*]] = torch.vtensor.literal(dense<5.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
// CHECK: %[[INDICES_LIST:.*]] = torch.prim.ListConstruct %[[BOOL_MASK]], %[[INT_INDICES]] : (!torch.vtensor<[4],i1>, !torch.vtensor<[4],si64>) -> !torch.list<optional<vtensor>>
// CHECK: %[[RESULT:.*]] = torch.aten.index_put %arg0, %[[INDICES_LIST]], %[[VALUE]], %[[FALSE]] : !torch.vtensor<[4,4],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[],f32>, !torch.bool -> !torch.vtensor<[4,4],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[4,4],f32>
func.func @index_put_mixed_indices_should_fail(%arg0: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
  %false = torch.constant.bool false
  %0 = torch.vtensor.literal(dense<[true, false, true, false]> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  %1 = torch.vtensor.literal(dense<[0, 1, 2, 3]> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %2 = torch.vtensor.literal(dense<5.0> : tensor<f32>) : !torch.vtensor<[],f32>
  %3 = torch.prim.ListConstruct %0, %1 : (!torch.vtensor<[4],i1>, !torch.vtensor<[4],si64>) -> !torch.list<optional<vtensor>>
  %4 = torch.aten.index_put %arg0, %3, %2, %false : !torch.vtensor<[4,4],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[],f32>, !torch.bool -> !torch.vtensor<[4,4],f32>
  return %4 : !torch.vtensor<[4,4],f32>
}

// -----

// CHECK-LABEL: func.func @index_put_boolean_mask_regular_case
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[FLATTENED_MASK:.*]] = torch.vtensor.literal(dense<[true, false, true, false, true, false]> : tensor<6xi1>) : !torch.vtensor<[6],i1>
// CHECK: %[[INT3:.*]] = torch.constant.int 3
// CHECK: %[[INT0:.*]] = torch.constant.int 0
// CHECK: %[[INT1:.*]] = torch.constant.int 1
// CHECK: %[[FALSE:.*]] = torch.constant.bool false
// CHECK: %[[MASK:.*]] = torch.vtensor.literal(dense<{{\[\[}}true, false, true], [false, true, false]]> : tensor<2x3xi1>) : !torch.vtensor<[2,3],i1>
// CHECK: %[[VALUES:.*]] = torch.vtensor.literal(dense<[1.000000e+01, 2.000000e+01, 3.000000e+01, 4.000000e+01]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
// CHECK: %[[INT2:.*]] = torch.constant.int 2
// CHECK: %[[NUMEL:.*]] = torch.aten.numel %[[MASK]] : !torch.vtensor<[2,3],i1> -> !torch.int
// CHECK: %[[DTYPE_CONVERTED:.*]] = torch.aten.to.dtype %[[FLATTENED_MASK]], %[[INT3]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[6],i1>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[6],si32>
// CHECK: %[[CUMSUM:.*]] = torch.aten.cumsum %[[DTYPE_CONVERTED]], %[[INT0]], %[[NONE]] : !torch.vtensor<[6],si32>, !torch.int, !torch.none -> !torch.vtensor<[6],si32>
// CHECK: %[[SUB:.*]] = torch.aten.sub.Scalar %[[CUMSUM]], %[[INT1]], %[[INT1]] : !torch.vtensor<[6],si32>, !torch.int, !torch.int -> !torch.vtensor<[6],si32>
// CHECK: %[[CLAMP:.*]] = torch.aten.clamp_min %[[SUB]], %[[INT0]] : !torch.vtensor<[6],si32>, !torch.int -> !torch.vtensor<[6],si32>
// CHECK: %[[BROADCAST_SHAPE:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT3]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[ARANGE_ROWS:.*]] = torch.aten.arange.start_step %[[INT0]], %[[INT2]], %[[INT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2],si32>
// CHECK: %[[RESHAPE_ROWS_SHAPE:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT1]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_ROWS:.*]] = torch.aten.reshape %[[ARANGE_ROWS]], %[[RESHAPE_ROWS_SHAPE]] : !torch.vtensor<[2],si32>, !torch.list<int> -> !torch.vtensor<[2,1,1],si32>
// CHECK: %[[BROADCASTED_ROWS:.*]] = torch.aten.broadcast_to %[[RESHAPED_ROWS]], %[[BROADCAST_SHAPE]] : !torch.vtensor<[2,1,1],si32>, !torch.list<int> -> !torch.vtensor<[2,3,1],si32>
// CHECK: %[[ARANGE_COLS:.*]] = torch.aten.arange.start_step %[[INT0]], %[[INT3]], %[[INT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3],si32>
// CHECK: %[[RESHAPE_COLS_SHAPE:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT3]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_COLS:.*]] = torch.aten.reshape %[[ARANGE_COLS]], %[[RESHAPE_COLS_SHAPE]] : !torch.vtensor<[3],si32>, !torch.list<int> -> !torch.vtensor<[1,3,1],si32>
// CHECK: %[[BROADCASTED_COLS:.*]] = torch.aten.broadcast_to %[[RESHAPED_COLS]], %[[BROADCAST_SHAPE]] : !torch.vtensor<[1,3,1],si32>, !torch.list<int> -> !torch.vtensor<[2,3,1],si32>
// CHECK: %[[INDICES_LIST:.*]] = torch.prim.ListConstruct %[[BROADCASTED_ROWS]], %[[BROADCASTED_COLS]] : (!torch.vtensor<[2,3,1],si32>, !torch.vtensor<[2,3,1],si32>) -> !torch.list<vtensor>
// CHECK: %[[INDICES_CAT:.*]] = torch.aten.cat %[[INDICES_LIST]], %[[INT2]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[2,3,2],si32>
// CHECK: %[[NUMEL_SHAPE_1:.*]] = torch.prim.ListConstruct %[[NUMEL]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[NUMEL_SHAPE_2:.*]] = torch.prim.ListConstruct %[[NUMEL]], %[[INT2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[FLATTENED_INDICES:.*]] = torch.aten.flatten.using_ints %[[INDICES_CAT]], %[[INT0]], %[[INT1]] : !torch.vtensor<[2,3,2],si32>, !torch.int, !torch.int -> !torch.vtensor<[6,2],si32>
// CHECK: %[[RESHAPED_MASK:.*]] = torch.aten.reshape %[[DTYPE_CONVERTED]], %[[NUMEL_SHAPE_1]] : !torch.vtensor<[6],si32>, !torch.list<int> -> !torch.vtensor<[6,1],si32>
// CHECK: %[[BROADCASTED_MASK:.*]] = torch.aten.broadcast_to %[[RESHAPED_MASK]], %[[NUMEL_SHAPE_2]] : !torch.vtensor<[6,1],si32>, !torch.list<int> -> !torch.vtensor<[6,2],si32>
// CHECK: %[[MASKED_INDICES:.*]] = torch.aten.mul.Tensor %[[FLATTENED_INDICES]], %[[BROADCASTED_MASK]] : !torch.vtensor<[6,2],si32>, !torch.vtensor<[6,2],si32> -> !torch.vtensor<[6,2],si32>
// CHECK: %[[RESHAPED_VALUES_INDEX:.*]] = torch.aten.reshape %[[CLAMP]], %[[NUMEL_SHAPE_1]] : !torch.vtensor<[6],si32>, !torch.list<int> -> !torch.vtensor<[6,1],si32>
// CHECK: %[[BROADCASTED_VALUES_INDEX:.*]] = torch.aten.broadcast_to %[[RESHAPED_VALUES_INDEX]], %[[NUMEL_SHAPE_2]] : !torch.vtensor<[6,1],si32>, !torch.list<int> -> !torch.vtensor<[6,2],si32>
// CHECK: %[[ZEROS_LIKE:.*]] = torch.aten.zeros_like %[[MASKED_INDICES]], %[[INT3]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.vtensor<[6,2],si32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[6,2],si32>
// CHECK: %[[SCATTER_ADD:.*]] = torch.aten.scatter_add %[[ZEROS_LIKE]], %[[INT0]], %[[BROADCASTED_VALUES_INDEX]], %[[MASKED_INDICES]] : !torch.vtensor<[6,2],si32>, !torch.int, !torch.vtensor<[6,2],si32>, !torch.vtensor<[6,2],si32> -> !torch.vtensor<[6,2],si32>
// CHECK: %[[NARROW:.*]] = torch.aten.narrow %[[SCATTER_ADD]], %[[INT0]], %[[INT0]], %[[INT1]] : !torch.vtensor<[6,2],si32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2],si32>
// CHECK: %[[INDEX_LIST:.*]] = torch.prim.ListConstruct %[[NARROW]] : (!torch.vtensor<[1,2],si32>) -> !torch.list<vtensor>
// CHECK: %[[RESULT:.*]] = torch.aten.index_put %arg0, %[[INDEX_LIST]], %[[VALUES]], %[[FALSE]] : !torch.vtensor<[2,3],f32>, !torch.list<vtensor>, !torch.vtensor<[4],f32>, !torch.bool -> !torch.vtensor<[2,3],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[2,3],f32>
func.func @index_put_boolean_mask_regular_case(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %false = torch.constant.bool false
  %0 = torch.vtensor.literal(dense<[[true, false, true], [false, true, false]]> : tensor<2x3xi1>) : !torch.vtensor<[2,3],i1>
  %1 = torch.vtensor.literal(dense<[10.0, 20.0, 30.0, 40.0]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %2 = torch.prim.ListConstruct %0 : (!torch.vtensor<[2,3],i1>) -> !torch.list<optional<vtensor>>
  %3 = torch.aten.index_put %arg0, %2, %1, %false : !torch.vtensor<[2,3],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[4],f32>, !torch.bool -> !torch.vtensor<[2,3],f32>
  return %3 : !torch.vtensor<[2,3],f32>
}

// -----

// CHECK-LABEL: func.func @index_put_boolean_mask_with_accumulate
// CHECK: %[[MASK:.*]] = torch.vtensor.literal(dense<{{\[\[}}true, false, false], [false, true, false], [false, false, true]]> : tensor<3x3xi1>) : !torch.vtensor<[3,3],i1>
// CHECK: %[[VALUE:.*]] = torch.vtensor.literal(dense<1.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
// CHECK: %[[INT1:.*]] = torch.constant.int 1
// CHECK: %[[FALSE:.*]] = torch.constant.bool false
// CHECK: %[[INT3:.*]] = torch.constant.int 3
// CHECK: %[[SHAPE_LIST:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[EXPANDED_MASK:.*]] = torch.aten.expand %[[MASK]], %[[SHAPE_LIST]], %[[FALSE]] : !torch.vtensor<[3,3],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,3],i1>
// CHECK: %[[RESHAPE_LIST:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_VALUE:.*]] = torch.aten.reshape %[[VALUE]], %[[RESHAPE_LIST]] : !torch.vtensor<[],f32>, !torch.list<int> -> !torch.vtensor<[1,1],f32>
// CHECK: %[[EXPANDED_VALUE:.*]] = torch.aten.expand %[[RESHAPED_VALUE]], %[[SHAPE_LIST]], %[[FALSE]] : !torch.vtensor<[1,1],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,3],f32>
// CHECK: %[[ADD_RESULT:.*]] = torch.aten.add.Tensor %arg0, %[[EXPANDED_VALUE]], %[[INT1]] : !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],f32>, !torch.int -> !torch.vtensor<[3,3],f32>
// CHECK: %[[RESULT:.*]] = torch.aten.where.self %[[EXPANDED_MASK]], %[[ADD_RESULT]], %arg0 : !torch.vtensor<[3,3],i1>, !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],f32> -> !torch.vtensor<[3,3],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[3,3],f32>
func.func @index_put_boolean_mask_with_accumulate(%arg0: !torch.vtensor<[3,3],f32>) -> !torch.vtensor<[3,3],f32> {
  %true = torch.constant.bool true
  %0 = torch.vtensor.literal(dense<[[true, false, false], [false, true, false], [false, false, true]]> : tensor<3x3xi1>) : !torch.vtensor<[3,3],i1>
  %1 = torch.vtensor.literal(dense<1.0> : tensor<f32>) : !torch.vtensor<[],f32>
  %2 = torch.prim.ListConstruct %0 : (!torch.vtensor<[3,3],i1>) -> !torch.list<optional<vtensor>>
  %3 = torch.aten.index_put %arg0, %2, %1, %true : !torch.vtensor<[3,3],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[],f32>, !torch.bool -> !torch.vtensor<[3,3],f32>
  return %3 : !torch.vtensor<[3,3],f32>
}

// -----

// CHECK-LABEL: func.func @index_put_impl_boolean_mask
// CHECK: %[[INT4:.*]] = torch.constant.int 4
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[FLATTENED_MASK:.*]] = torch.vtensor.literal(dense<[false, true, false, true, true, false, true, false]> : tensor<8xi1>) : !torch.vtensor<[8],i1>
// CHECK: %[[INT3:.*]] = torch.constant.int 3
// CHECK: %[[INT0:.*]] = torch.constant.int 0
// CHECK: %[[INT1:.*]] = torch.constant.int 1
// CHECK: %[[FALSE:.*]] = torch.constant.bool false
// CHECK: %[[MASK:.*]] = torch.vtensor.literal(dense<{{\[\[}}false, true, false, true], [true, false, true, false]]> : tensor<2x4xi1>) : !torch.vtensor<[2,4],i1>
// CHECK: %[[VALUES:.*]] = torch.vtensor.literal(dense<[7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
// CHECK: %[[INT2:.*]] = torch.constant.int 2
// CHECK: %[[NUMEL:.*]] = torch.aten.numel %[[MASK]] : !torch.vtensor<[2,4],i1> -> !torch.int
// CHECK: %[[DTYPE_CONVERTED:.*]] = torch.aten.to.dtype %[[FLATTENED_MASK]], %[[INT3]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[8],i1>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[8],si32>
// CHECK: %[[CUMSUM:.*]] = torch.aten.cumsum %[[DTYPE_CONVERTED]], %[[INT0]], %[[NONE]] : !torch.vtensor<[8],si32>, !torch.int, !torch.none -> !torch.vtensor<[8],si32>
// CHECK: %[[SUB:.*]] = torch.aten.sub.Scalar %[[CUMSUM]], %[[INT1]], %[[INT1]] : !torch.vtensor<[8],si32>, !torch.int, !torch.int -> !torch.vtensor<[8],si32>
// CHECK: %[[CLAMP:.*]] = torch.aten.clamp_min %[[SUB]], %[[INT0]] : !torch.vtensor<[8],si32>, !torch.int -> !torch.vtensor<[8],si32>
// CHECK: %[[BROADCAST_SHAPE:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT4]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[ARANGE_ROWS:.*]] = torch.aten.arange.start_step %[[INT0]], %[[INT2]], %[[INT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2],si32>
// CHECK: %[[RESHAPE_ROWS_SHAPE:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT1]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_ROWS:.*]] = torch.aten.reshape %[[ARANGE_ROWS]], %[[RESHAPE_ROWS_SHAPE]] : !torch.vtensor<[2],si32>, !torch.list<int> -> !torch.vtensor<[2,1,1],si32>
// CHECK: %[[BROADCASTED_ROWS:.*]] = torch.aten.broadcast_to %[[RESHAPED_ROWS]], %[[BROADCAST_SHAPE]] : !torch.vtensor<[2,1,1],si32>, !torch.list<int> -> !torch.vtensor<[2,4,1],si32>
// CHECK: %[[ARANGE_COLS:.*]] = torch.aten.arange.start_step %[[INT0]], %[[INT4]], %[[INT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[4],si32>
// CHECK: %[[RESHAPE_COLS_SHAPE:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT4]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_COLS:.*]] = torch.aten.reshape %[[ARANGE_COLS]], %[[RESHAPE_COLS_SHAPE]] : !torch.vtensor<[4],si32>, !torch.list<int> -> !torch.vtensor<[1,4,1],si32>
// CHECK: %[[BROADCASTED_COLS:.*]] = torch.aten.broadcast_to %[[RESHAPED_COLS]], %[[BROADCAST_SHAPE]] : !torch.vtensor<[1,4,1],si32>, !torch.list<int> -> !torch.vtensor<[2,4,1],si32>
// CHECK: %[[INDICES_LIST:.*]] = torch.prim.ListConstruct %[[BROADCASTED_ROWS]], %[[BROADCASTED_COLS]] : (!torch.vtensor<[2,4,1],si32>, !torch.vtensor<[2,4,1],si32>) -> !torch.list<vtensor>
// CHECK: %[[INDICES_CAT:.*]] = torch.aten.cat %[[INDICES_LIST]], %[[INT2]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[2,4,2],si32>
// CHECK: %[[NUMEL_SHAPE_1:.*]] = torch.prim.ListConstruct %[[NUMEL]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[NUMEL_SHAPE_2:.*]] = torch.prim.ListConstruct %[[NUMEL]], %[[INT2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[FLATTENED_INDICES:.*]] = torch.aten.flatten.using_ints %[[INDICES_CAT]], %[[INT0]], %[[INT1]] : !torch.vtensor<[2,4,2],si32>, !torch.int, !torch.int -> !torch.vtensor<[8,2],si32>
// CHECK: %[[RESHAPED_MASK:.*]] = torch.aten.reshape %[[DTYPE_CONVERTED]], %[[NUMEL_SHAPE_1]] : !torch.vtensor<[8],si32>, !torch.list<int> -> !torch.vtensor<[8,1],si32>
// CHECK: %[[BROADCASTED_MASK:.*]] = torch.aten.broadcast_to %[[RESHAPED_MASK]], %[[NUMEL_SHAPE_2]] : !torch.vtensor<[8,1],si32>, !torch.list<int> -> !torch.vtensor<[8,2],si32>
// CHECK: %[[MASKED_INDICES:.*]] = torch.aten.mul.Tensor %[[FLATTENED_INDICES]], %[[BROADCASTED_MASK]] : !torch.vtensor<[8,2],si32>, !torch.vtensor<[8,2],si32> -> !torch.vtensor<[8,2],si32>
// CHECK: %[[RESHAPED_VALUES_INDEX:.*]] = torch.aten.reshape %[[CLAMP]], %[[NUMEL_SHAPE_1]] : !torch.vtensor<[8],si32>, !torch.list<int> -> !torch.vtensor<[8,1],si32>
// CHECK: %[[BROADCASTED_VALUES_INDEX:.*]] = torch.aten.broadcast_to %[[RESHAPED_VALUES_INDEX]], %[[NUMEL_SHAPE_2]] : !torch.vtensor<[8,1],si32>, !torch.list<int> -> !torch.vtensor<[8,2],si32>
// CHECK: %[[ZEROS_LIKE:.*]] = torch.aten.zeros_like %[[MASKED_INDICES]], %[[INT3]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.vtensor<[8,2],si32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[8,2],si32>
// CHECK: %[[SCATTER_ADD:.*]] = torch.aten.scatter_add %[[ZEROS_LIKE]], %[[INT0]], %[[BROADCASTED_VALUES_INDEX]], %[[MASKED_INDICES]] : !torch.vtensor<[8,2],si32>, !torch.int, !torch.vtensor<[8,2],si32>, !torch.vtensor<[8,2],si32> -> !torch.vtensor<[8,2],si32>
// CHECK: %[[NARROW:.*]] = torch.aten.narrow %[[SCATTER_ADD]], %[[INT0]], %[[INT0]], %[[INT1]] : !torch.vtensor<[8,2],si32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2],si32>
// CHECK: %[[INDEX_LIST:.*]] = torch.prim.ListConstruct %[[NARROW]] : (!torch.vtensor<[1,2],si32>) -> !torch.list<vtensor>
// CHECK: %[[RESULT:.*]] = torch.aten._index_put_impl %arg0, %[[INDEX_LIST]], %[[VALUES]], %[[FALSE]], %[[FALSE]] : !torch.vtensor<[2,4],f32>, !torch.list<vtensor>, !torch.vtensor<[4],f32>, !torch.bool, !torch.bool -> !torch.vtensor<[2,4],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[2,4],f32>
func.func @index_put_impl_boolean_mask(%arg0: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %false = torch.constant.bool false
  %0 = torch.vtensor.literal(dense<[[false, true, false, true], [true, false, true, false]]> : tensor<2x4xi1>) : !torch.vtensor<[2,4],i1>
  %1 = torch.vtensor.literal(dense<[7.0, 8.0, 9.0, 10.0]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %2 = torch.prim.ListConstruct %0 : (!torch.vtensor<[2,4],i1>) -> !torch.list<optional<vtensor>>
  %3 = torch.aten._index_put_impl %arg0, %2, %1, %false, %false : !torch.vtensor<[2,4],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[4],f32>, !torch.bool, !torch.bool -> !torch.vtensor<[2,4],f32>
  return %3 : !torch.vtensor<[2,4],f32>
}

// -----

// CHECK-LABEL: func.func @index_put_boolean_mask_1d_tensor
// CHECK: %[[INT6:.*]] = torch.constant.int 6
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[INT3:.*]] = torch.constant.int 3
// CHECK: %[[INT0:.*]] = torch.constant.int 0
// CHECK: %[[INT1:.*]] = torch.constant.int 1
// CHECK: %[[FALSE:.*]] = torch.constant.bool false
// CHECK: %[[MASK:.*]] = torch.vtensor.literal(dense<[true, false, true, false, true, false]> : tensor<6xi1>) : !torch.vtensor<[6],i1>
// CHECK: %[[VALUES:.*]] = torch.vtensor.literal(dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>) : !torch.vtensor<[3],f32>
// CHECK: %[[NUMEL:.*]] = torch.aten.numel %[[MASK]] : !torch.vtensor<[6],i1> -> !torch.int
// CHECK: %[[DTYPE_CONVERTED:.*]] = torch.aten.to.dtype %[[MASK]], %[[INT3]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[6],i1>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[6],si32>
// CHECK: %[[CUMSUM:.*]] = torch.aten.cumsum %[[DTYPE_CONVERTED]], %[[INT0]], %[[NONE]] : !torch.vtensor<[6],si32>, !torch.int, !torch.none -> !torch.vtensor<[6],si32>
// CHECK: %[[SUB:.*]] = torch.aten.sub.Scalar %[[CUMSUM]], %[[INT1]], %[[INT1]] : !torch.vtensor<[6],si32>, !torch.int, !torch.int -> !torch.vtensor<[6],si32>
// CHECK: %[[CLAMP:.*]] = torch.aten.clamp_min %[[SUB]], %[[INT0]] : !torch.vtensor<[6],si32>, !torch.int -> !torch.vtensor<[6],si32>
// CHECK: %[[ARANGE:.*]] = torch.aten.arange.start_step %[[INT0]], %[[INT6]], %[[INT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[6],si32>
// CHECK: %[[RESHAPE_SHAPE:.*]] = torch.prim.ListConstruct %[[INT6]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_ARANGE:.*]] = torch.aten.reshape %[[ARANGE]], %[[RESHAPE_SHAPE]] : !torch.vtensor<[6],si32>, !torch.list<int> -> !torch.vtensor<[6,1],si32>
// CHECK: %[[NUMEL_SHAPE_1:.*]] = torch.prim.ListConstruct %[[NUMEL]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[FLATTENED_INDICES:.*]] = torch.aten.flatten.using_ints %[[RESHAPED_ARANGE]], %[[INT0]], %[[INT0]] : !torch.vtensor<[6,1],si32>, !torch.int, !torch.int -> !torch.vtensor<[6,1],si32>
// CHECK: %[[RESHAPED_MASK:.*]] = torch.aten.reshape %[[DTYPE_CONVERTED]], %[[NUMEL_SHAPE_1]] : !torch.vtensor<[6],si32>, !torch.list<int> -> !torch.vtensor<[6,1],si32>
// CHECK: %[[MASKED_INDICES:.*]] = torch.aten.mul.Tensor %[[FLATTENED_INDICES]], %[[RESHAPED_MASK]] : !torch.vtensor<[6,1],si32>, !torch.vtensor<[6,1],si32> -> !torch.vtensor<[6,1],si32>
// CHECK: %[[RESHAPED_VALUES_INDEX:.*]] = torch.aten.reshape %[[CLAMP]], %[[NUMEL_SHAPE_1]] : !torch.vtensor<[6],si32>, !torch.list<int> -> !torch.vtensor<[6,1],si32>
// CHECK: %[[ZEROS_LIKE:.*]] = torch.aten.zeros_like %[[MASKED_INDICES]], %[[INT3]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.vtensor<[6,1],si32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[6,1],si32>
// CHECK: %[[SCATTER_ADD:.*]] = torch.aten.scatter_add %[[ZEROS_LIKE]], %[[INT0]], %[[RESHAPED_VALUES_INDEX]], %[[MASKED_INDICES]] : !torch.vtensor<[6,1],si32>, !torch.int, !torch.vtensor<[6,1],si32>, !torch.vtensor<[6,1],si32> -> !torch.vtensor<[6,1],si32>
// CHECK: %[[NARROW:.*]] = torch.aten.narrow %[[SCATTER_ADD]], %[[INT0]], %[[INT0]], %[[INT1]] : !torch.vtensor<[6,1],si32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1],si32>
// CHECK: %[[INDEX_LIST:.*]] = torch.prim.ListConstruct %[[NARROW]] : (!torch.vtensor<[1,1],si32>) -> !torch.list<vtensor>
// CHECK: %[[RESULT:.*]] = torch.aten.index_put %arg0, %[[INDEX_LIST]], %[[VALUES]], %[[FALSE]] : !torch.vtensor<[6],f32>, !torch.list<vtensor>, !torch.vtensor<[3],f32>, !torch.bool -> !torch.vtensor<[6],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[6],f32>
func.func @index_put_boolean_mask_1d_tensor(%arg0: !torch.vtensor<[6],f32>) -> !torch.vtensor<[6],f32> {
  %false = torch.constant.bool false
  %0 = torch.vtensor.literal(dense<[true, false, true, false, true, false]> : tensor<6xi1>) : !torch.vtensor<[6],i1>
  %1 = torch.vtensor.literal(dense<[1.0, 2.0, 3.0]> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %2 = torch.prim.ListConstruct %0 : (!torch.vtensor<[6],i1>) -> !torch.list<optional<vtensor>>
  %3 = torch.aten.index_put %arg0, %2, %1, %false : !torch.vtensor<[6],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[3],f32>, !torch.bool -> !torch.vtensor<[6],f32>
  return %3 : !torch.vtensor<[6],f32>
}

// -----

// CHECK-LABEL: func.func @index_put_boolean_mask_sparse_pattern
// CHECK: %[[INT4:.*]] = torch.constant.int 4
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[FLATTENED_MASK:.*]] = torch.vtensor.literal(dense<[true, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false]> : tensor<16xi1>) : !torch.vtensor<[16],i1>
// CHECK: %[[INT3:.*]] = torch.constant.int 3
// CHECK: %[[INT0:.*]] = torch.constant.int 0
// CHECK: %[[INT1:.*]] = torch.constant.int 1
// CHECK: %[[FALSE:.*]] = torch.constant.bool false
// CHECK: %[[MASK:.*]] = torch.vtensor.literal(dense<{{\[\[}}true, false, false, false], [false, false, false, false], [false, false, true, false], [false, false, false, false]]> : tensor<4x4xi1>) : !torch.vtensor<[4,4],i1>
// CHECK: %[[VALUES:.*]] = torch.vtensor.literal(dense<[4.200000e+01, 8.400000e+01]> : tensor<2xf32>) : !torch.vtensor<[2],f32>
// CHECK: %[[INT2:.*]] = torch.constant.int 2
// CHECK: %[[NUMEL:.*]] = torch.aten.numel %[[MASK]] : !torch.vtensor<[4,4],i1> -> !torch.int
// CHECK: %[[DTYPE_CONVERTED:.*]] = torch.aten.to.dtype %[[FLATTENED_MASK]], %[[INT3]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[16],i1>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[16],si32>
// CHECK: %[[CUMSUM:.*]] = torch.aten.cumsum %[[DTYPE_CONVERTED]], %[[INT0]], %[[NONE]] : !torch.vtensor<[16],si32>, !torch.int, !torch.none -> !torch.vtensor<[16],si32>
// CHECK: %[[SUB:.*]] = torch.aten.sub.Scalar %[[CUMSUM]], %[[INT1]], %[[INT1]] : !torch.vtensor<[16],si32>, !torch.int, !torch.int -> !torch.vtensor<[16],si32>
// CHECK: %[[CLAMP:.*]] = torch.aten.clamp_min %[[SUB]], %[[INT0]] : !torch.vtensor<[16],si32>, !torch.int -> !torch.vtensor<[16],si32>
// CHECK: %[[BROADCAST_SHAPE:.*]] = torch.prim.ListConstruct %[[INT4]], %[[INT4]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[ARANGE_ROWS:.*]] = torch.aten.arange.start_step %[[INT0]], %[[INT4]], %[[INT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[4],si32>
// CHECK: %[[RESHAPE_ROWS_SHAPE:.*]] = torch.prim.ListConstruct %[[INT4]], %[[INT1]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_ROWS:.*]] = torch.aten.reshape %[[ARANGE_ROWS]], %[[RESHAPE_ROWS_SHAPE]] : !torch.vtensor<[4],si32>, !torch.list<int> -> !torch.vtensor<[4,1,1],si32>
// CHECK: %[[BROADCASTED_ROWS:.*]] = torch.aten.broadcast_to %[[RESHAPED_ROWS]], %[[BROADCAST_SHAPE]] : !torch.vtensor<[4,1,1],si32>, !torch.list<int> -> !torch.vtensor<[4,4,1],si32>
// CHECK: %[[ARANGE_COLS:.*]] = torch.aten.arange.start_step %[[INT0]], %[[INT4]], %[[INT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[4],si32>
// CHECK: %[[RESHAPE_COLS_SHAPE:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT4]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_COLS:.*]] = torch.aten.reshape %[[ARANGE_COLS]], %[[RESHAPE_COLS_SHAPE]] : !torch.vtensor<[4],si32>, !torch.list<int> -> !torch.vtensor<[1,4,1],si32>
// CHECK: %[[BROADCASTED_COLS:.*]] = torch.aten.broadcast_to %[[RESHAPED_COLS]], %[[BROADCAST_SHAPE]] : !torch.vtensor<[1,4,1],si32>, !torch.list<int> -> !torch.vtensor<[4,4,1],si32>
// CHECK: %[[INDICES_LIST:.*]] = torch.prim.ListConstruct %[[BROADCASTED_ROWS]], %[[BROADCASTED_COLS]] : (!torch.vtensor<[4,4,1],si32>, !torch.vtensor<[4,4,1],si32>) -> !torch.list<vtensor>
// CHECK: %[[INDICES_CAT:.*]] = torch.aten.cat %[[INDICES_LIST]], %[[INT2]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[4,4,2],si32>
// CHECK: %[[NUMEL_SHAPE_1:.*]] = torch.prim.ListConstruct %[[NUMEL]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[NUMEL_SHAPE_2:.*]] = torch.prim.ListConstruct %[[NUMEL]], %[[INT2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[FLATTENED_INDICES:.*]] = torch.aten.flatten.using_ints %[[INDICES_CAT]], %[[INT0]], %[[INT1]] : !torch.vtensor<[4,4,2],si32>, !torch.int, !torch.int -> !torch.vtensor<[16,2],si32>
// CHECK: %[[RESHAPED_MASK:.*]] = torch.aten.reshape %[[DTYPE_CONVERTED]], %[[NUMEL_SHAPE_1]] : !torch.vtensor<[16],si32>, !torch.list<int> -> !torch.vtensor<[16,1],si32>
// CHECK: %[[BROADCASTED_MASK:.*]] = torch.aten.broadcast_to %[[RESHAPED_MASK]], %[[NUMEL_SHAPE_2]] : !torch.vtensor<[16,1],si32>, !torch.list<int> -> !torch.vtensor<[16,2],si32>
// CHECK: %[[MASKED_INDICES:.*]] = torch.aten.mul.Tensor %[[FLATTENED_INDICES]], %[[BROADCASTED_MASK]] : !torch.vtensor<[16,2],si32>, !torch.vtensor<[16,2],si32> -> !torch.vtensor<[16,2],si32>
// CHECK: %[[RESHAPED_VALUES_INDEX:.*]] = torch.aten.reshape %[[CLAMP]], %[[NUMEL_SHAPE_1]] : !torch.vtensor<[16],si32>, !torch.list<int> -> !torch.vtensor<[16,1],si32>
// CHECK: %[[BROADCASTED_VALUES_INDEX:.*]] = torch.aten.broadcast_to %[[RESHAPED_VALUES_INDEX]], %[[NUMEL_SHAPE_2]] : !torch.vtensor<[16,1],si32>, !torch.list<int> -> !torch.vtensor<[16,2],si32>
// CHECK: %[[ZEROS_LIKE:.*]] = torch.aten.zeros_like %[[MASKED_INDICES]], %[[INT3]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.vtensor<[16,2],si32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[16,2],si32>
// CHECK: %[[SCATTER_ADD:.*]] = torch.aten.scatter_add %[[ZEROS_LIKE]], %[[INT0]], %[[BROADCASTED_VALUES_INDEX]], %[[MASKED_INDICES]] : !torch.vtensor<[16,2],si32>, !torch.int, !torch.vtensor<[16,2],si32>, !torch.vtensor<[16,2],si32> -> !torch.vtensor<[16,2],si32>
// CHECK: %[[NARROW:.*]] = torch.aten.narrow %[[SCATTER_ADD]], %[[INT0]], %[[INT0]], %[[INT1]] : !torch.vtensor<[16,2],si32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2],si32>
// CHECK: %[[INDEX_LIST:.*]] = torch.prim.ListConstruct %[[NARROW]] : (!torch.vtensor<[1,2],si32>) -> !torch.list<vtensor>
// CHECK: %[[RESULT:.*]] = torch.aten.index_put %arg0, %[[INDEX_LIST]], %[[VALUES]], %[[FALSE]] : !torch.vtensor<[4,4],f32>, !torch.list<vtensor>, !torch.vtensor<[2],f32>, !torch.bool -> !torch.vtensor<[4,4],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[4,4],f32>
func.func @index_put_boolean_mask_sparse_pattern(%arg0: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
  %false = torch.constant.bool false
  %0 = torch.vtensor.literal(dense<[[true, false, false, false], [false, false, false, false], [false, false, true, false], [false, false, false, false]]> : tensor<4x4xi1>) : !torch.vtensor<[4,4],i1>
  %1 = torch.vtensor.literal(dense<[42.0, 84.0]> : tensor<2xf32>) : !torch.vtensor<[2],f32>
  %2 = torch.prim.ListConstruct %0 : (!torch.vtensor<[4,4],i1>) -> !torch.list<optional<vtensor>>
  %3 = torch.aten.index_put %arg0, %2, %1, %false : !torch.vtensor<[4,4],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[2],f32>, !torch.bool -> !torch.vtensor<[4,4],f32>
  return %3 : !torch.vtensor<[4,4],f32>
}

// -----

// CHECK-LABEL: func.func @index_put_boolean_mask_3d_tensor
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[FLATTENED_MASK:.*]] = torch.vtensor.literal(dense<[true, false, false, true, false, true, true, false]> : tensor<8xi1>) : !torch.vtensor<[8],i1>
// CHECK: %[[INT2:.*]] = torch.constant.int 2
// CHECK: %[[INT0:.*]] = torch.constant.int 0
// CHECK: %[[INT1:.*]] = torch.constant.int 1
// CHECK: %[[FALSE:.*]] = torch.constant.bool false
// CHECK: %[[MASK:.*]] = torch.vtensor.literal(dense<{{\[\[\[}}true, false], [false, true]], {{\[\[}}false, true], [true, false]]]> : tensor<2x2x2xi1>) : !torch.vtensor<[2,2,2],i1>
// CHECK: %[[VALUES:.*]] = torch.vtensor.literal(dense<[1.000000e+01, 2.000000e+01, 3.000000e+01, 4.000000e+01]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
// CHECK: %[[INT3:.*]] = torch.constant.int 3
// CHECK: %[[NUMEL:.*]] = torch.aten.numel %[[MASK]] : !torch.vtensor<[2,2,2],i1> -> !torch.int
// CHECK: %[[DTYPE_CONVERTED:.*]] = torch.aten.to.dtype %[[FLATTENED_MASK]], %[[INT3]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[8],i1>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[8],si32>
// CHECK: %[[CUMSUM:.*]] = torch.aten.cumsum %[[DTYPE_CONVERTED]], %[[INT0]], %[[NONE]] : !torch.vtensor<[8],si32>, !torch.int, !torch.none -> !torch.vtensor<[8],si32>
// CHECK: %[[SUB:.*]] = torch.aten.sub.Scalar %[[CUMSUM]], %[[INT1]], %[[INT1]] : !torch.vtensor<[8],si32>, !torch.int, !torch.int -> !torch.vtensor<[8],si32>
// CHECK: %[[CLAMP:.*]] = torch.aten.clamp_min %[[SUB]], %[[INT0]] : !torch.vtensor<[8],si32>, !torch.int -> !torch.vtensor<[8],si32>
// CHECK: %[[BROADCAST_SHAPE:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT2]], %[[INT2]], %[[INT1]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[ARANGE_DIM0:.*]] = torch.aten.arange.start_step %[[INT0]], %[[INT2]], %[[INT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2],si32>
// CHECK: %[[RESHAPE_DIM0_SHAPE:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT1]], %[[INT1]], %[[INT1]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_DIM0:.*]] = torch.aten.reshape %[[ARANGE_DIM0]], %[[RESHAPE_DIM0_SHAPE]] : !torch.vtensor<[2],si32>, !torch.list<int> -> !torch.vtensor<[2,1,1,1],si32>
// CHECK: %[[BROADCASTED_DIM0:.*]] = torch.aten.broadcast_to %[[RESHAPED_DIM0]], %[[BROADCAST_SHAPE]] : !torch.vtensor<[2,1,1,1],si32>, !torch.list<int> -> !torch.vtensor<[2,2,2,1],si32>
// CHECK: %[[ARANGE_DIM1:.*]] = torch.aten.arange.start_step %[[INT0]], %[[INT2]], %[[INT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2],si32>
// CHECK: %[[RESHAPE_DIM1_SHAPE:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT2]], %[[INT1]], %[[INT1]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_DIM1:.*]] = torch.aten.reshape %[[ARANGE_DIM1]], %[[RESHAPE_DIM1_SHAPE]] : !torch.vtensor<[2],si32>, !torch.list<int> -> !torch.vtensor<[1,2,1,1],si32>
// CHECK: %[[BROADCASTED_DIM1:.*]] = torch.aten.broadcast_to %[[RESHAPED_DIM1]], %[[BROADCAST_SHAPE]] : !torch.vtensor<[1,2,1,1],si32>, !torch.list<int> -> !torch.vtensor<[2,2,2,1],si32>
// CHECK: %[[ARANGE_DIM2:.*]] = torch.aten.arange.start_step %[[INT0]], %[[INT2]], %[[INT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2],si32>
// CHECK: %[[RESHAPE_DIM2_SHAPE:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1]], %[[INT2]], %[[INT1]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_DIM2:.*]] = torch.aten.reshape %[[ARANGE_DIM2]], %[[RESHAPE_DIM2_SHAPE]] : !torch.vtensor<[2],si32>, !torch.list<int> -> !torch.vtensor<[1,1,2,1],si32>
// CHECK: %[[BROADCASTED_DIM2:.*]] = torch.aten.broadcast_to %[[RESHAPED_DIM2]], %[[BROADCAST_SHAPE]] : !torch.vtensor<[1,1,2,1],si32>, !torch.list<int> -> !torch.vtensor<[2,2,2,1],si32>
// CHECK: %[[INDICES_LIST:.*]] = torch.prim.ListConstruct %[[BROADCASTED_DIM0]], %[[BROADCASTED_DIM1]], %[[BROADCASTED_DIM2]] : (!torch.vtensor<[2,2,2,1],si32>, !torch.vtensor<[2,2,2,1],si32>, !torch.vtensor<[2,2,2,1],si32>) -> !torch.list<vtensor>
// CHECK: %[[INDICES_CAT:.*]] = torch.aten.cat %[[INDICES_LIST]], %[[INT3]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[2,2,2,3],si32>
// CHECK: %[[NUMEL_SHAPE_1:.*]] = torch.prim.ListConstruct %[[NUMEL]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[NUMEL_SHAPE_3:.*]] = torch.prim.ListConstruct %[[NUMEL]], %[[INT3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[FLATTENED_INDICES:.*]] = torch.aten.flatten.using_ints %[[INDICES_CAT]], %[[INT0]], %[[INT2]] : !torch.vtensor<[2,2,2,3],si32>, !torch.int, !torch.int -> !torch.vtensor<[8,3],si32>
// CHECK: %[[RESHAPED_MASK:.*]] = torch.aten.reshape %[[DTYPE_CONVERTED]], %[[NUMEL_SHAPE_1]] : !torch.vtensor<[8],si32>, !torch.list<int> -> !torch.vtensor<[8,1],si32>
// CHECK: %[[BROADCASTED_MASK:.*]] = torch.aten.broadcast_to %[[RESHAPED_MASK]], %[[NUMEL_SHAPE_3]] : !torch.vtensor<[8,1],si32>, !torch.list<int> -> !torch.vtensor<[8,3],si32>
// CHECK: %[[MASKED_INDICES:.*]] = torch.aten.mul.Tensor %[[FLATTENED_INDICES]], %[[BROADCASTED_MASK]] : !torch.vtensor<[8,3],si32>, !torch.vtensor<[8,3],si32> -> !torch.vtensor<[8,3],si32>
// CHECK: %[[RESHAPED_VALUES_INDEX:.*]] = torch.aten.reshape %[[CLAMP]], %[[NUMEL_SHAPE_1]] : !torch.vtensor<[8],si32>, !torch.list<int> -> !torch.vtensor<[8,1],si32>
// CHECK: %[[BROADCASTED_VALUES_INDEX:.*]] = torch.aten.broadcast_to %[[RESHAPED_VALUES_INDEX]], %[[NUMEL_SHAPE_3]] : !torch.vtensor<[8,1],si32>, !torch.list<int> -> !torch.vtensor<[8,3],si32>
// CHECK: %[[ZEROS_LIKE:.*]] = torch.aten.zeros_like %[[MASKED_INDICES]], %[[INT3]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.vtensor<[8,3],si32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[8,3],si32>
// CHECK: %[[SCATTER_ADD:.*]] = torch.aten.scatter_add %[[ZEROS_LIKE]], %[[INT0]], %[[BROADCASTED_VALUES_INDEX]], %[[MASKED_INDICES]] : !torch.vtensor<[8,3],si32>, !torch.int, !torch.vtensor<[8,3],si32>, !torch.vtensor<[8,3],si32> -> !torch.vtensor<[8,3],si32>
// CHECK: %[[NARROW:.*]] = torch.aten.narrow %[[SCATTER_ADD]], %[[INT0]], %[[INT0]], %[[INT1]] : !torch.vtensor<[8,3],si32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,3],si32>
// CHECK: %[[INDEX_LIST:.*]] = torch.prim.ListConstruct %[[NARROW]] : (!torch.vtensor<[1,3],si32>) -> !torch.list<vtensor>
// CHECK: %[[RESULT:.*]] = torch.aten.index_put %arg0, %[[INDEX_LIST]], %[[VALUES]], %[[FALSE]] : !torch.vtensor<[2,2,2],f32>, !torch.list<vtensor>, !torch.vtensor<[4],f32>, !torch.bool -> !torch.vtensor<[2,2,2],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[2,2,2],f32>
func.func @index_put_boolean_mask_3d_tensor(%arg0: !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[2,2,2],f32> {
  %false = torch.constant.bool false
  %0 = torch.vtensor.literal(dense<[[[true, false], [false, true]], [[false, true], [true, false]]]> : tensor<2x2x2xi1>) : !torch.vtensor<[2,2,2],i1>
  %1 = torch.vtensor.literal(dense<[10.0, 20.0, 30.0, 40.0]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %2 = torch.prim.ListConstruct %0 : (!torch.vtensor<[2,2,2],i1>) -> !torch.list<optional<vtensor>>
  %3 = torch.aten.index_put %arg0, %2, %1, %false : !torch.vtensor<[2,2,2],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[4],f32>, !torch.bool -> !torch.vtensor<[2,2,2],f32>
  return %3 : !torch.vtensor<[2,2,2],f32>
}

// -----

// CHECK-LABEL: func.func @index_put_boolean_mask_single_element
// CHECK: %[[INT5:.*]] = torch.constant.int 5
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[FLATTENED_MASK:.*]] = torch.vtensor.literal(dense<[false, false, false, false, false, false, false, true, false, false, false, false, false, false, false]> : tensor<15xi1>) : !torch.vtensor<[15],i1>
// CHECK: %[[INT3:.*]] = torch.constant.int 3
// CHECK: %[[INT0:.*]] = torch.constant.int 0
// CHECK: %[[INT1:.*]] = torch.constant.int 1
// CHECK: %[[FALSE:.*]] = torch.constant.bool false
// CHECK: %[[MASK:.*]] = torch.vtensor.literal(dense<{{\[\[}}false, false, false], [false, false, false], [false, true, false], [false, false, false], [false, false, false]]> : tensor<5x3xi1>) : !torch.vtensor<[5,3],i1>
// CHECK: %[[VALUES:.*]] = torch.vtensor.literal(dense<9.950000e+01> : tensor<1xf32>) : !torch.vtensor<[1],f32>
// CHECK: %[[INT2:.*]] = torch.constant.int 2
// CHECK: %[[NUMEL:.*]] = torch.aten.numel %[[MASK]] : !torch.vtensor<[5,3],i1> -> !torch.int
// CHECK: %[[DTYPE_CONVERTED:.*]] = torch.aten.to.dtype %[[FLATTENED_MASK]], %[[INT3]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[15],i1>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[15],si32>
// CHECK: %[[CUMSUM:.*]] = torch.aten.cumsum %[[DTYPE_CONVERTED]], %[[INT0]], %[[NONE]] : !torch.vtensor<[15],si32>, !torch.int, !torch.none -> !torch.vtensor<[15],si32>
// CHECK: %[[SUB:.*]] = torch.aten.sub.Scalar %[[CUMSUM]], %[[INT1]], %[[INT1]] : !torch.vtensor<[15],si32>, !torch.int, !torch.int -> !torch.vtensor<[15],si32>
// CHECK: %[[CLAMP:.*]] = torch.aten.clamp_min %[[SUB]], %[[INT0]] : !torch.vtensor<[15],si32>, !torch.int -> !torch.vtensor<[15],si32>
// CHECK: %[[BROADCAST_SHAPE:.*]] = torch.prim.ListConstruct %[[INT5]], %[[INT3]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[ARANGE_ROWS:.*]] = torch.aten.arange.start_step %[[INT0]], %[[INT5]], %[[INT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[5],si32>
// CHECK: %[[RESHAPE_ROWS_SHAPE:.*]] = torch.prim.ListConstruct %[[INT5]], %[[INT1]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_ROWS:.*]] = torch.aten.reshape %[[ARANGE_ROWS]], %[[RESHAPE_ROWS_SHAPE]] : !torch.vtensor<[5],si32>, !torch.list<int> -> !torch.vtensor<[5,1,1],si32>
// CHECK: %[[BROADCASTED_ROWS:.*]] = torch.aten.broadcast_to %[[RESHAPED_ROWS]], %[[BROADCAST_SHAPE]] : !torch.vtensor<[5,1,1],si32>, !torch.list<int> -> !torch.vtensor<[5,3,1],si32>
// CHECK: %[[ARANGE_COLS:.*]] = torch.aten.arange.start_step %[[INT0]], %[[INT3]], %[[INT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3],si32>
// CHECK: %[[RESHAPE_COLS_SHAPE:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT3]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_COLS:.*]] = torch.aten.reshape %[[ARANGE_COLS]], %[[RESHAPE_COLS_SHAPE]] : !torch.vtensor<[3],si32>, !torch.list<int> -> !torch.vtensor<[1,3,1],si32>
// CHECK: %[[BROADCASTED_COLS:.*]] = torch.aten.broadcast_to %[[RESHAPED_COLS]], %[[BROADCAST_SHAPE]] : !torch.vtensor<[1,3,1],si32>, !torch.list<int> -> !torch.vtensor<[5,3,1],si32>
// CHECK: %[[INDICES_LIST:.*]] = torch.prim.ListConstruct %[[BROADCASTED_ROWS]], %[[BROADCASTED_COLS]] : (!torch.vtensor<[5,3,1],si32>, !torch.vtensor<[5,3,1],si32>) -> !torch.list<vtensor>
// CHECK: %[[INDICES_CAT:.*]] = torch.aten.cat %[[INDICES_LIST]], %[[INT2]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[5,3,2],si32>
// CHECK: %[[NUMEL_SHAPE_1:.*]] = torch.prim.ListConstruct %[[NUMEL]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[NUMEL_SHAPE_2:.*]] = torch.prim.ListConstruct %[[NUMEL]], %[[INT2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[FLATTENED_INDICES:.*]] = torch.aten.flatten.using_ints %[[INDICES_CAT]], %[[INT0]], %[[INT1]] : !torch.vtensor<[5,3,2],si32>, !torch.int, !torch.int -> !torch.vtensor<[15,2],si32>
// CHECK: %[[RESHAPED_MASK:.*]] = torch.aten.reshape %[[DTYPE_CONVERTED]], %[[NUMEL_SHAPE_1]] : !torch.vtensor<[15],si32>, !torch.list<int> -> !torch.vtensor<[15,1],si32>
// CHECK: %[[BROADCASTED_MASK:.*]] = torch.aten.broadcast_to %[[RESHAPED_MASK]], %[[NUMEL_SHAPE_2]] : !torch.vtensor<[15,1],si32>, !torch.list<int> -> !torch.vtensor<[15,2],si32>
// CHECK: %[[MASKED_INDICES:.*]] = torch.aten.mul.Tensor %[[FLATTENED_INDICES]], %[[BROADCASTED_MASK]] : !torch.vtensor<[15,2],si32>, !torch.vtensor<[15,2],si32> -> !torch.vtensor<[15,2],si32>
// CHECK: %[[RESHAPED_VALUES_INDEX:.*]] = torch.aten.reshape %[[CLAMP]], %[[NUMEL_SHAPE_1]] : !torch.vtensor<[15],si32>, !torch.list<int> -> !torch.vtensor<[15,1],si32>
// CHECK: %[[BROADCASTED_VALUES_INDEX:.*]] = torch.aten.broadcast_to %[[RESHAPED_VALUES_INDEX]], %[[NUMEL_SHAPE_2]] : !torch.vtensor<[15,1],si32>, !torch.list<int> -> !torch.vtensor<[15,2],si32>
// CHECK: %[[ZEROS_LIKE:.*]] = torch.aten.zeros_like %[[MASKED_INDICES]], %[[INT3]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.vtensor<[15,2],si32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[15,2],si32>
// CHECK: %[[SCATTER_ADD:.*]] = torch.aten.scatter_add %[[ZEROS_LIKE]], %[[INT0]], %[[BROADCASTED_VALUES_INDEX]], %[[MASKED_INDICES]] : !torch.vtensor<[15,2],si32>, !torch.int, !torch.vtensor<[15,2],si32>, !torch.vtensor<[15,2],si32> -> !torch.vtensor<[15,2],si32>
// CHECK: %[[NARROW:.*]] = torch.aten.narrow %[[SCATTER_ADD]], %[[INT0]], %[[INT0]], %[[INT1]] : !torch.vtensor<[15,2],si32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2],si32>
// CHECK: %[[INDEX_LIST:.*]] = torch.prim.ListConstruct %[[NARROW]] : (!torch.vtensor<[1,2],si32>) -> !torch.list<vtensor>
// CHECK: %[[RESULT:.*]] = torch.aten.index_put %arg0, %[[INDEX_LIST]], %[[VALUES]], %[[FALSE]] : !torch.vtensor<[5,3],f32>, !torch.list<vtensor>, !torch.vtensor<[1],f32>, !torch.bool -> !torch.vtensor<[5,3],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[5,3],f32>
func.func @index_put_boolean_mask_single_element(%arg0: !torch.vtensor<[5,3],f32>) -> !torch.vtensor<[5,3],f32> {
  %false = torch.constant.bool false
  %0 = torch.vtensor.literal(dense<[[false, false, false], [false, false, false], [false, true, false], [false, false, false], [false, false, false]]> : tensor<5x3xi1>) : !torch.vtensor<[5,3],i1>
  %1 = torch.vtensor.literal(dense<[99.5]> : tensor<1xf32>) : !torch.vtensor<[1],f32>
  %2 = torch.prim.ListConstruct %0 : (!torch.vtensor<[5,3],i1>) -> !torch.list<optional<vtensor>>
  %3 = torch.aten.index_put %arg0, %2, %1, %false : !torch.vtensor<[5,3],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[1],f32>, !torch.bool -> !torch.vtensor<[5,3],f32>
  return %3 : !torch.vtensor<[5,3],f32>
}

// -----

// CHECK-LABEL: func.func @unsafe_index_put_hacked_twin_boolean_mask
// CHECK: %[[INT4:.*]] = torch.constant.int 4
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[FLATTENED_MASK:.*]] = torch.vtensor.literal(dense<[false, true, true, false, true, false, false, true, false, false, true, true]> : tensor<12xi1>) : !torch.vtensor<[12],i1>
// CHECK: %[[INT3:.*]] = torch.constant.int 3
// CHECK: %[[INT0:.*]] = torch.constant.int 0
// CHECK: %[[INT1:.*]] = torch.constant.int 1
// CHECK: %[[FALSE:.*]] = torch.constant.bool false
// CHECK: %[[MASK:.*]] = torch.vtensor.literal(dense<{{\[\[}}false, true, true, false], [true, false, false, true], [false, false, true, true]]> : tensor<3x4xi1>) : !torch.vtensor<[3,4],i1>
// CHECK: %[[VALUES:.*]] = torch.vtensor.literal(dense<[1.100000e+01, 2.200000e+01, 3.300000e+01, 4.400000e+01, 5.500000e+01, 6.600000e+01]> : tensor<6xf32>) : !torch.vtensor<[6],f32>
// CHECK: %[[INT2:.*]] = torch.constant.int 2
// CHECK: %[[NUMEL:.*]] = torch.aten.numel %[[MASK]] : !torch.vtensor<[3,4],i1> -> !torch.int
// CHECK: %[[DTYPE_CONVERTED:.*]] = torch.aten.to.dtype %[[FLATTENED_MASK]], %[[INT3]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[12],i1>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[12],si32>
// CHECK: %[[CUMSUM:.*]] = torch.aten.cumsum %[[DTYPE_CONVERTED]], %[[INT0]], %[[NONE]] : !torch.vtensor<[12],si32>, !torch.int, !torch.none -> !torch.vtensor<[12],si32>
// CHECK: %[[SUB:.*]] = torch.aten.sub.Scalar %[[CUMSUM]], %[[INT1]], %[[INT1]] : !torch.vtensor<[12],si32>, !torch.int, !torch.int -> !torch.vtensor<[12],si32>
// CHECK: %[[CLAMP:.*]] = torch.aten.clamp_min %[[SUB]], %[[INT0]] : !torch.vtensor<[12],si32>, !torch.int -> !torch.vtensor<[12],si32>
// CHECK: %[[BROADCAST_SHAPE:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT4]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[ARANGE_ROWS:.*]] = torch.aten.arange.start_step %[[INT0]], %[[INT3]], %[[INT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3],si32>
// CHECK: %[[RESHAPE_ROWS_SHAPE:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT1]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_ROWS:.*]] = torch.aten.reshape %[[ARANGE_ROWS]], %[[RESHAPE_ROWS_SHAPE]] : !torch.vtensor<[3],si32>, !torch.list<int> -> !torch.vtensor<[3,1,1],si32>
// CHECK: %[[BROADCASTED_ROWS:.*]] = torch.aten.broadcast_to %[[RESHAPED_ROWS]], %[[BROADCAST_SHAPE]] : !torch.vtensor<[3,1,1],si32>, !torch.list<int> -> !torch.vtensor<[3,4,1],si32>
// CHECK: %[[ARANGE_COLS:.*]] = torch.aten.arange.start_step %[[INT0]], %[[INT4]], %[[INT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[4],si32>
// CHECK: %[[RESHAPE_COLS_SHAPE:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT4]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_COLS:.*]] = torch.aten.reshape %[[ARANGE_COLS]], %[[RESHAPE_COLS_SHAPE]] : !torch.vtensor<[4],si32>, !torch.list<int> -> !torch.vtensor<[1,4,1],si32>
// CHECK: %[[BROADCASTED_COLS:.*]] = torch.aten.broadcast_to %[[RESHAPED_COLS]], %[[BROADCAST_SHAPE]] : !torch.vtensor<[1,4,1],si32>, !torch.list<int> -> !torch.vtensor<[3,4,1],si32>
// CHECK: %[[INDICES_LIST:.*]] = torch.prim.ListConstruct %[[BROADCASTED_ROWS]], %[[BROADCASTED_COLS]] : (!torch.vtensor<[3,4,1],si32>, !torch.vtensor<[3,4,1],si32>) -> !torch.list<vtensor>
// CHECK: %[[INDICES_CAT:.*]] = torch.aten.cat %[[INDICES_LIST]], %[[INT2]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[3,4,2],si32>
// CHECK: %[[NUMEL_SHAPE_1:.*]] = torch.prim.ListConstruct %[[NUMEL]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[NUMEL_SHAPE_2:.*]] = torch.prim.ListConstruct %[[NUMEL]], %[[INT2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[FLATTENED_INDICES:.*]] = torch.aten.flatten.using_ints %[[INDICES_CAT]], %[[INT0]], %[[INT1]] : !torch.vtensor<[3,4,2],si32>, !torch.int, !torch.int -> !torch.vtensor<[12,2],si32>
// CHECK: %[[RESHAPED_MASK:.*]] = torch.aten.reshape %[[DTYPE_CONVERTED]], %[[NUMEL_SHAPE_1]] : !torch.vtensor<[12],si32>, !torch.list<int> -> !torch.vtensor<[12,1],si32>
// CHECK: %[[BROADCASTED_MASK:.*]] = torch.aten.broadcast_to %[[RESHAPED_MASK]], %[[NUMEL_SHAPE_2]] : !torch.vtensor<[12,1],si32>, !torch.list<int> -> !torch.vtensor<[12,2],si32>
// CHECK: %[[MASKED_INDICES:.*]] = torch.aten.mul.Tensor %[[FLATTENED_INDICES]], %[[BROADCASTED_MASK]] : !torch.vtensor<[12,2],si32>, !torch.vtensor<[12,2],si32> -> !torch.vtensor<[12,2],si32>
// CHECK: %[[RESHAPED_VALUES_INDEX:.*]] = torch.aten.reshape %[[CLAMP]], %[[NUMEL_SHAPE_1]] : !torch.vtensor<[12],si32>, !torch.list<int> -> !torch.vtensor<[12,1],si32>
// CHECK: %[[BROADCASTED_VALUES_INDEX:.*]] = torch.aten.broadcast_to %[[RESHAPED_VALUES_INDEX]], %[[NUMEL_SHAPE_2]] : !torch.vtensor<[12,1],si32>, !torch.list<int> -> !torch.vtensor<[12,2],si32>
// CHECK: %[[ZEROS_LIKE:.*]] = torch.aten.zeros_like %[[MASKED_INDICES]], %[[INT3]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.vtensor<[12,2],si32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[12,2],si32>
// CHECK: %[[SCATTER_ADD:.*]] = torch.aten.scatter_add %[[ZEROS_LIKE]], %[[INT0]], %[[BROADCASTED_VALUES_INDEX]], %[[MASKED_INDICES]] : !torch.vtensor<[12,2],si32>, !torch.int, !torch.vtensor<[12,2],si32>, !torch.vtensor<[12,2],si32> -> !torch.vtensor<[12,2],si32>
// CHECK: %[[NARROW:.*]] = torch.aten.narrow %[[SCATTER_ADD]], %[[INT0]], %[[INT0]], %[[INT1]] : !torch.vtensor<[12,2],si32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2],si32>
// CHECK: %[[INDEX_LIST:.*]] = torch.prim.ListConstruct %[[NARROW]] : (!torch.vtensor<[1,2],si32>) -> !torch.list<vtensor>
// CHECK: %[[RESULT:.*]] = torch.aten._unsafe_index_put.hacked_twin %arg0, %[[INDEX_LIST]], %[[VALUES]], %[[FALSE]] : !torch.vtensor<[3,4],f32>, !torch.list<vtensor>, !torch.vtensor<[6],f32>, !torch.bool -> !torch.vtensor<[3,4],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[3,4],f32>
func.func @unsafe_index_put_hacked_twin_boolean_mask(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %false = torch.constant.bool false
  %0 = torch.vtensor.literal(dense<[[false, true, true, false], [true, false, false, true], [false, false, true, true]]> : tensor<3x4xi1>) : !torch.vtensor<[3,4],i1>
  %1 = torch.vtensor.literal(dense<[11.0, 22.0, 33.0, 44.0, 55.0, 66.0]> : tensor<6xf32>) : !torch.vtensor<[6],f32>
  %2 = torch.prim.ListConstruct %0 : (!torch.vtensor<[3,4],i1>) -> !torch.list<vtensor>
  %3 = torch.aten._unsafe_index_put.hacked_twin %arg0, %2, %1, %false : !torch.vtensor<[3,4],f32>, !torch.list<vtensor>, !torch.vtensor<[6],f32>, !torch.bool -> !torch.vtensor<[3,4],f32>
  return %3 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: func.func @index_put_boolean_mask_accumulate_diagonal_2d
// CHECK: %[[MASK:.*]] = torch.vtensor.literal(dense<{{\[\[}}true, false, false, false], [false, true, false, false], [false, false, true, false], [false, false, false, true]]> : tensor<4x4xi1>) : !torch.vtensor<[4,4],i1>
// CHECK: %[[VALUE:.*]] = torch.vtensor.literal(dense<5.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
// CHECK: %[[INT1:.*]] = torch.constant.int 1
// CHECK: %[[FALSE:.*]] = torch.constant.bool false
// CHECK: %[[INT4:.*]] = torch.constant.int 4
// CHECK: %[[SHAPE_LIST:.*]] = torch.prim.ListConstruct %[[INT4]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[EXPANDED_MASK:.*]] = torch.aten.expand %[[MASK]], %[[SHAPE_LIST]], %[[FALSE]] : !torch.vtensor<[4,4],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,4],i1>
// CHECK: %[[RESHAPE_LIST:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_VALUE:.*]] = torch.aten.reshape %[[VALUE]], %[[RESHAPE_LIST]] : !torch.vtensor<[],f32>, !torch.list<int> -> !torch.vtensor<[1,1],f32>
// CHECK: %[[EXPANDED_VALUE:.*]] = torch.aten.expand %[[RESHAPED_VALUE]], %[[SHAPE_LIST]], %[[FALSE]] : !torch.vtensor<[1,1],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,4],f32>
// CHECK: %[[ADD_RESULT:.*]] = torch.aten.add.Tensor %arg0, %[[EXPANDED_VALUE]], %[[INT1]] : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f32>, !torch.int -> !torch.vtensor<[4,4],f32>
// CHECK: %[[RESULT:.*]] = torch.aten.where.self %[[EXPANDED_MASK]], %[[ADD_RESULT]], %arg0 : !torch.vtensor<[4,4],i1>, !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f32> -> !torch.vtensor<[4,4],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[4,4],f32>
func.func @index_put_boolean_mask_accumulate_diagonal_2d(%arg0: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
  %true = torch.constant.bool true
  %0 = torch.vtensor.literal(dense<[[true, false, false, false], [false, true, false, false], [false, false, true, false], [false, false, false, true]]> : tensor<4x4xi1>) : !torch.vtensor<[4,4],i1>
  %1 = torch.vtensor.literal(dense<5.0> : tensor<f32>) : !torch.vtensor<[],f32>
  %2 = torch.prim.ListConstruct %0 : (!torch.vtensor<[4,4],i1>) -> !torch.list<optional<vtensor>>
  %3 = torch.aten.index_put %arg0, %2, %1, %true : !torch.vtensor<[4,4],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[],f32>, !torch.bool -> !torch.vtensor<[4,4],f32>
  return %3 : !torch.vtensor<[4,4],f32>
}

// -----

// CHECK-LABEL: func.func @index_put_boolean_mask_accumulate_checkerboard_pattern
// CHECK: %[[MASK:.*]] = torch.vtensor.literal(dense<{{\[\[}}true, false, true], [false, true, false], [true, false, true]]> : tensor<3x3xi1>) : !torch.vtensor<[3,3],i1>
// CHECK: %[[VALUE:.*]] = torch.vtensor.literal(dense<2.500000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
// CHECK: %[[INT1:.*]] = torch.constant.int 1
// CHECK: %[[FALSE:.*]] = torch.constant.bool false
// CHECK: %[[INT3:.*]] = torch.constant.int 3
// CHECK: %[[SHAPE_LIST:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[EXPANDED_MASK:.*]] = torch.aten.expand %[[MASK]], %[[SHAPE_LIST]], %[[FALSE]] : !torch.vtensor<[3,3],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,3],i1>
// CHECK: %[[RESHAPE_LIST:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_VALUE:.*]] = torch.aten.reshape %[[VALUE]], %[[RESHAPE_LIST]] : !torch.vtensor<[],f32>, !torch.list<int> -> !torch.vtensor<[1,1],f32>
// CHECK: %[[EXPANDED_VALUE:.*]] = torch.aten.expand %[[RESHAPED_VALUE]], %[[SHAPE_LIST]], %[[FALSE]] : !torch.vtensor<[1,1],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,3],f32>
// CHECK: %[[ADD_RESULT:.*]] = torch.aten.add.Tensor %arg0, %[[EXPANDED_VALUE]], %[[INT1]] : !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],f32>, !torch.int -> !torch.vtensor<[3,3],f32>
// CHECK: %[[RESULT:.*]] = torch.aten.where.self %[[EXPANDED_MASK]], %[[ADD_RESULT]], %arg0 : !torch.vtensor<[3,3],i1>, !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],f32> -> !torch.vtensor<[3,3],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[3,3],f32>
func.func @index_put_boolean_mask_accumulate_checkerboard_pattern(%arg0: !torch.vtensor<[3,3],f32>) -> !torch.vtensor<[3,3],f32> {
  %true = torch.constant.bool true
  %0 = torch.vtensor.literal(dense<[[true, false, true], [false, true, false], [true, false, true]]> : tensor<3x3xi1>) : !torch.vtensor<[3,3],i1>
  %1 = torch.vtensor.literal(dense<2.5> : tensor<f32>) : !torch.vtensor<[],f32>
  %2 = torch.prim.ListConstruct %0 : (!torch.vtensor<[3,3],i1>) -> !torch.list<optional<vtensor>>
  %3 = torch.aten.index_put %arg0, %2, %1, %true : !torch.vtensor<[3,3],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[],f32>, !torch.bool -> !torch.vtensor<[3,3],f32>
  return %3 : !torch.vtensor<[3,3],f32>
}

// -----

// CHECK-LABEL: func.func @index_put_impl_boolean_mask_accumulate_corners
// CHECK: %[[MASK:.*]] = torch.vtensor.literal(dense<{{\[\[}}true, false, false, true], [false, false, false, false], [false, false, false, false], [true, false, false, true]]> : tensor<4x4xi1>) : !torch.vtensor<[4,4],i1>
// CHECK: %[[VALUE:.*]] = torch.vtensor.literal(dense<7.6999998> : tensor<f32>) : !torch.vtensor<[],f32>
// CHECK: %[[INT1:.*]] = torch.constant.int 1
// CHECK: %[[FALSE:.*]] = torch.constant.bool false
// CHECK: %[[INT4:.*]] = torch.constant.int 4
// CHECK: %[[SHAPE_LIST:.*]] = torch.prim.ListConstruct %[[INT4]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[EXPANDED_MASK:.*]] = torch.aten.expand %[[MASK]], %[[SHAPE_LIST]], %[[FALSE]] : !torch.vtensor<[4,4],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,4],i1>
// CHECK: %[[RESHAPE_LIST:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESHAPED_VALUE:.*]] = torch.aten.reshape %[[VALUE]], %[[RESHAPE_LIST]] : !torch.vtensor<[],f32>, !torch.list<int> -> !torch.vtensor<[1,1],f32>
// CHECK: %[[EXPANDED_VALUE:.*]] = torch.aten.expand %[[RESHAPED_VALUE]], %[[SHAPE_LIST]], %[[FALSE]] : !torch.vtensor<[1,1],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,4],f32>
// CHECK: %[[ADD_RESULT:.*]] = torch.aten.add.Tensor %arg0, %[[EXPANDED_VALUE]], %[[INT1]] : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f32>, !torch.int -> !torch.vtensor<[4,4],f32>
// CHECK: %[[RESULT:.*]] = torch.aten.where.self %[[EXPANDED_MASK]], %[[ADD_RESULT]], %arg0 : !torch.vtensor<[4,4],i1>, !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f32> -> !torch.vtensor<[4,4],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[4,4],f32>
func.func @index_put_impl_boolean_mask_accumulate_corners(%arg0: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
  %true = torch.constant.bool true
  %0 = torch.vtensor.literal(dense<[[true, false, false, true], [false, false, false, false], [false, false, false, false], [true, false, false, true]]> : tensor<4x4xi1>) : !torch.vtensor<[4,4],i1>
  %1 = torch.vtensor.literal(dense<7.7> : tensor<f32>) : !torch.vtensor<[],f32>
  %2 = torch.prim.ListConstruct %0 : (!torch.vtensor<[4,4],i1>) -> !torch.list<optional<vtensor>>
  %3 = torch.aten._index_put_impl %arg0, %2, %1, %true, %true : !torch.vtensor<[4,4],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[],f32>, !torch.bool, !torch.bool -> !torch.vtensor<[4,4],f32>
  return %3 : !torch.vtensor<[4,4],f32>
}

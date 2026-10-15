// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.index_select$basic(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,4],f32>, %[[ARG1:.*]]: !torch.vtensor<[2],si64>) -> !torch.vtensor<[2,4],f32> {
// CHECK-DAG:     %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,4],f32> -> tensor<?x4xf32>
// CHECK-DAG:     %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[2],si64> -> tensor<2xi64>
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x4xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[T4:.*]] = tensor.from_elements %[[C1_I64]], %[[T3]] : tensor<2xi64>
// CHECK:         %[[T5:.*]] = "stablehlo.dynamic_gather"(%[[T0]], %[[T1]], %[[T4]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false}> : (tensor<?x4xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x4xf32>
// CHECK:         %[[T6:.*]] = stablehlo.convert %[[T5]] : tensor<2x4xf32>
// CHECK:         %[[T7:.*]] = torch_c.from_builtin_tensor %[[T6]] : tensor<2x4xf32> -> !torch.vtensor<[2,4],f32>
// CHECK:         return %[[T7]] : !torch.vtensor<[2,4],f32>
func.func @torch.aten.index_select$basic(%arg0: !torch.vtensor<[?,4],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.vtensor<[2,4],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.index_select %arg0, %int0, %arg1 : !torch.vtensor<[?,4],f32>, !torch.int, !torch.vtensor<[2],si64> -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL:  func.func @torch.aten.embedding$basic(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>, %[[ARG1:.*]]: !torch.vtensor<[?],si64>) -> !torch.vtensor<[?,?],f32> {
// CHECK-DAG:     %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK-DAG:     %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?],si64> -> tensor<?xi64>
// CHECK:         %[[FALSE:.*]] = torch.constant.bool false
// CHECK:         %[[INT:.*]]-1 = torch.constant.int -1
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[T4:.*]] = tensor.from_elements %[[C1_I64]], %[[T3]] : tensor<2xi64>
// CHECK:         %[[T5:.*]] = "stablehlo.dynamic_gather"(%[[T0]], %[[T1]], %[[T4]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false}> : (tensor<?x?xf32>, tensor<?xi64>, tensor<2xi64>) -> tensor<?x?xf32>
// CHECK:         %[[T6:.*]] = stablehlo.convert %[[T5]] : tensor<?x?xf32>
// CHECK:         %[[T7:.*]] = torch_c.from_builtin_tensor %[[T6]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[T7]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.embedding$basic(%weight: !torch.vtensor<[?,?],f32>, %indices: !torch.vtensor<[?], si64>) -> !torch.vtensor<[?,?],f32> {
  %false = torch.constant.bool false
  %int-1 = torch.constant.int -1
  %ret = torch.aten.embedding %weight, %indices, %int-1, %false, %false : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?], si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[?,?],f32>
  return %ret: !torch.vtensor<[?,?],f32>
}

// CHECK-LABEL:  func.func @torch.aten.embedding$rank_two_indices(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>, %[[ARG1:.*]]: !torch.vtensor<[?,1],si64>) -> !torch.vtensor<[?,1,?],f32> {
// CHECK-DAG:     %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK-DAG:     %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?,1],si64> -> tensor<?x1xi64>
// CHECK:         %[[FALSE:.*]] = torch.constant.bool false
// CHECK:         %[[INT:.*]]-1 = torch.constant.int -1
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[T4:.*]] = tensor.from_elements %[[C1_I64]], %[[T3]] : tensor<2xi64>
// CHECK:         %[[T5:.*]] = "stablehlo.dynamic_gather"(%[[T0]], %[[T1]], %[[T4]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false}> : (tensor<?x?xf32>, tensor<?x1xi64>, tensor<2xi64>) -> tensor<?x1x?xf32>
// CHECK:         %[[T6:.*]] = stablehlo.convert %[[T5]] : tensor<?x1x?xf32>
// CHECK:         %[[T7:.*]] = torch_c.from_builtin_tensor %[[T6]] : tensor<?x1x?xf32> -> !torch.vtensor<[?,1,?],f32>
// CHECK:         return %[[T7]] : !torch.vtensor<[?,1,?],f32>
func.func @torch.aten.embedding$rank_two_indices(%weight: !torch.vtensor<[?,?],f32>, %indices: !torch.vtensor<[?,1], si64>) -> !torch.vtensor<[?,1,?],f32> {
  %false = torch.constant.bool false
  %int-1 = torch.constant.int -1
  %ret = torch.aten.embedding %weight, %indices, %int-1, %false, %false : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,1], si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[?,1,?],f32>
  return %ret: !torch.vtensor<[?,1,?],f32>
}

// -----

// CHECK-LABEL:   func.func @aten.embedding_bag.padding_idx$sum_reduction(
// CHECK-SAME:                                                            %[[VAL_0:.*]]: !torch.vtensor<[3],si64>,
// CHECK-SAME:                                                            %[[VAL_1:.*]]: !torch.vtensor<[10,8],f32>,
// CHECK-SAME:                                                            %[[VAL_2:.*]]: !torch.vtensor<[5],si64>) -> !torch.vtensor<[3,8],f32> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3],si64> -> tensor<3xi64>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[5],si64> -> tensor<5xi64>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[10,8],f32> -> tensor<10x8xf32>
// CHECK:           %[[VAL_6:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_8:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_9:.*]] = torch.constant.none
// CHECK:           %[[VAL_10:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_11:.*]] = torch.constant.none
// CHECK:           %[[VAL_12:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_13:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_14:.*]] = tensor.dim %[[VAL_5]], %[[VAL_13]] : tensor<10x8xf32>
// CHECK:           %[[VAL_15:.*]] = arith.index_cast %[[VAL_14]] : index to i64
// CHECK:           %[[VAL_16:.*]] = tensor.from_elements %[[VAL_12]], %[[VAL_15]] : tensor<2xi64>
// CHECK:           %[[VAL_17:.*]] = "stablehlo.dynamic_gather"(%[[VAL_5]], %[[VAL_4]], %[[VAL_16]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false}> : (tensor<10x8xf32>, tensor<5xi64>, tensor<2xi64>) -> tensor<5x8xf32>
// CHECK:           %[[VAL_18:.*]] = stablehlo.iota dim = 0 : tensor<5xi64>
// CHECK:           %[[VAL_19:.*]] = stablehlo.broadcast_in_dim %[[VAL_3]], dims = [0] : (tensor<3xi64>) -> tensor<3x5xi64>
// CHECK:           %[[VAL_20:.*]] = stablehlo.broadcast_in_dim %[[VAL_18]], dims = [1] : (tensor<5xi64>) -> tensor<3x5xi64>
// CHECK:           %[[VAL_21:.*]] = stablehlo.compare  GE, %[[VAL_20]], %[[VAL_19]] : (tensor<3x5xi64>, tensor<3x5xi64>) -> tensor<3x5xi1>
// CHECK:           %[[VAL_22:.*]] = stablehlo.convert %[[VAL_21]] : (tensor<3x5xi1>) -> tensor<3x5xi64>
// CHECK:           %[[VAL_23:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_24:.*]] = stablehlo.reduce(%[[VAL_22]] init: %[[VAL_23]]) applies stablehlo.add across dimensions = [0] : (tensor<3x5xi64>, tensor<i64>) -> tensor<5xi64>
// CHECK:           %[[VAL_25:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK:           %[[VAL_26:.*]] = chlo.broadcast_subtract %[[VAL_24]], %[[VAL_25]] : (tensor<5xi64>, tensor<i64>) -> tensor<5xi64>
// CHECK:           %[[VAL_27:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_28:.*]] = stablehlo.broadcast_in_dim %[[VAL_27]], dims = [] : (tensor<f32>) -> tensor<3x8xf32>
// CHECK:           %[[VAL_29:.*]] = stablehlo.reshape %[[VAL_26]] : (tensor<5xi64>) -> tensor<5x1xi64>
// CHECK:           %[[VAL_30:.*]] = "stablehlo.scatter"(%[[VAL_28]], %[[VAL_29]], %[[VAL_17]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK:           ^bb0(%[[VAL_31:.*]]: tensor<f32>, %[[VAL_32:.*]]: tensor<f32>):
// CHECK:             %[[VAL_33:.*]] = stablehlo.add %[[VAL_31]], %[[VAL_32]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_33]] : tensor<f32>
// CHECK:           }) : (tensor<3x8xf32>, tensor<5x1xi64>, tensor<5x8xf32>) -> tensor<3x8xf32>
// CHECK:           %[[VAL_34:.*]] = torch_c.from_builtin_tensor %[[VAL_35:.*]] : tensor<3x8xf32> -> !torch.vtensor<[3,8],f32>
// CHECK:           %[[VAL_36:.*]] = stablehlo.constant dense<0> : tensor<0xi64>
// CHECK:           %[[VAL_37:.*]] = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK:           %[[VAL_38:.*]] = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK:           return %[[VAL_34]] : !torch.vtensor<[3,8],f32>
// CHECK:         }
func.func @aten.embedding_bag.padding_idx$sum_reduction(%arg0: !torch.vtensor<[3],si64>, %arg1: !torch.vtensor<[10,8],f32>, %arg2: !torch.vtensor<[5],si64>) -> !torch.vtensor<[3,8],f32> {
    %false = torch.constant.bool false
    %int0 = torch.constant.int 0
    %false_0 = torch.constant.bool false
    %none = torch.constant.none
    %false_1 = torch.constant.bool false
    %none_2 = torch.constant.none
    %result0, %result1, %result2, %result3 = torch.aten.embedding_bag.padding_idx %arg1, %arg2, %arg0, %false, %int0, %false_0, %none, %false_1, %none_2 : !torch.vtensor<[10,8],f32>, !torch.vtensor<[5],si64>, !torch.vtensor<[3],si64>, !torch.bool, !torch.int, !torch.bool, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[3,8],f32>, !torch.vtensor<[0],si64>, !torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>
    return %result0 : !torch.vtensor<[3,8],f32>
}

// -----

// CHECK-LABEL:   func.func @aten.embedding_bag.padding_idx$mean_reduction(
// CHECK-SAME:                                                             %[[VAL_0:.*]]: !torch.vtensor<[2],si64>,
// CHECK-SAME:                                                             %[[VAL_1:.*]]: !torch.vtensor<[30,64],bf16>,
// CHECK-SAME:                                                             %[[VAL_2:.*]]: !torch.vtensor<[5],si64>) -> !torch.vtensor<[2,64],bf16> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2],si64> -> tensor<2xi64>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[5],si64> -> tensor<5xi64>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[30,64],bf16> -> tensor<30x64xbf16>
// CHECK:           %[[VAL_6:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_8:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_9:.*]] = torch.constant.none
// CHECK:           %[[VAL_10:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_11:.*]] = torch.constant.none
// CHECK:           %[[VAL_12:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_13:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_14:.*]] = tensor.dim %[[VAL_5]], %[[VAL_13]] : tensor<30x64xbf16>
// CHECK:           %[[VAL_15:.*]] = arith.index_cast %[[VAL_14]] : index to i64
// CHECK:           %[[VAL_16:.*]] = tensor.from_elements %[[VAL_12]], %[[VAL_15]] : tensor<2xi64>
// CHECK:           %[[VAL_17:.*]] = "stablehlo.dynamic_gather"(%[[VAL_5]], %[[VAL_4]], %[[VAL_16]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false}> : (tensor<30x64xbf16>, tensor<5xi64>, tensor<2xi64>) -> tensor<5x64xbf16>
// CHECK:           %[[VAL_18:.*]] = stablehlo.iota dim = 0 : tensor<5xi64>
// CHECK:           %[[VAL_19:.*]] = stablehlo.broadcast_in_dim %[[VAL_3]], dims = [0] : (tensor<2xi64>) -> tensor<2x5xi64>
// CHECK:           %[[VAL_20:.*]] = stablehlo.broadcast_in_dim %[[VAL_18]], dims = [1] : (tensor<5xi64>) -> tensor<2x5xi64>
// CHECK:           %[[VAL_21:.*]] = stablehlo.compare  GE, %[[VAL_20]], %[[VAL_19]] : (tensor<2x5xi64>, tensor<2x5xi64>) -> tensor<2x5xi1>
// CHECK:           %[[VAL_22:.*]] = stablehlo.convert %[[VAL_21]] : (tensor<2x5xi1>) -> tensor<2x5xi64>
// CHECK:           %[[VAL_23:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_24:.*]] = stablehlo.reduce(%[[VAL_22]] init: %[[VAL_23]]) applies stablehlo.add across dimensions = [0] : (tensor<2x5xi64>, tensor<i64>) -> tensor<5xi64>
// CHECK:           %[[VAL_25:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK:           %[[VAL_26:.*]] = chlo.broadcast_subtract %[[VAL_24]], %[[VAL_25]] : (tensor<5xi64>, tensor<i64>) -> tensor<5xi64>
// CHECK:           %[[VAL_27:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// CHECK:           %[[VAL_28:.*]] = stablehlo.broadcast_in_dim %[[VAL_27]], dims = [] : (tensor<bf16>) -> tensor<2x64xbf16>
// CHECK:           %[[VAL_29:.*]] = stablehlo.reshape %[[VAL_26]] : (tensor<5xi64>) -> tensor<5x1xi64>
// CHECK:           %[[VAL_30:.*]] = "stablehlo.scatter"(%[[VAL_28]], %[[VAL_29]], %[[VAL_17]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK:           ^bb0(%[[VAL_31:.*]]: tensor<bf16>, %[[VAL_32:.*]]: tensor<bf16>):
// CHECK:             %[[VAL_33:.*]] = stablehlo.add %[[VAL_31]], %[[VAL_32]] : tensor<bf16>
// CHECK:             stablehlo.return %[[VAL_33]] : tensor<bf16>
// CHECK:           }) : (tensor<2x64xbf16>, tensor<5x1xi64>, tensor<5x64xbf16>) -> tensor<2x64xbf16>
// CHECK:           %[[VAL_34:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
// CHECK:           %[[VAL_35:.*]] = stablehlo.broadcast_in_dim %[[VAL_34]], dims = [] : (tensor<bf16>) -> tensor<5xbf16>
// CHECK:           %[[VAL_36:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// CHECK:           %[[VAL_37:.*]] = stablehlo.broadcast_in_dim %[[VAL_36]], dims = [] : (tensor<bf16>) -> tensor<2xbf16>
// CHECK:           %[[VAL_38:.*]] = "stablehlo.scatter"(%[[VAL_37]], %[[VAL_29]], %[[VAL_35]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK:           ^bb0(%[[VAL_39:.*]]: tensor<bf16>, %[[VAL_40:.*]]: tensor<bf16>):
// CHECK:             %[[VAL_41:.*]] = stablehlo.add %[[VAL_39]], %[[VAL_40]] : tensor<bf16>
// CHECK:             stablehlo.return %[[VAL_41]] : tensor<bf16>
// CHECK:           }) : (tensor<2xbf16>, tensor<5x1xi64>, tensor<5xbf16>) -> tensor<2xbf16>
// CHECK:           %[[VAL_42:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
// CHECK:           %[[VAL_43:.*]] = chlo.broadcast_maximum %[[VAL_44:.*]], %[[VAL_42]] : (tensor<2xbf16>, tensor<bf16>) -> tensor<2xbf16>
// CHECK:           %[[VAL_45:.*]] = stablehlo.broadcast_in_dim %[[VAL_43]], dims = [0] : (tensor<2xbf16>) -> tensor<2x64xbf16>
// CHECK:           %[[VAL_46:.*]] = stablehlo.divide %[[VAL_47:.*]], %[[VAL_45]] : tensor<2x64xbf16>
// CHECK:           %[[VAL_48:.*]] = torch_c.from_builtin_tensor %[[VAL_46]] : tensor<2x64xbf16> -> !torch.vtensor<[2,64],bf16>
// CHECK:           %[[VAL_49:.*]] = stablehlo.constant dense<0> : tensor<5xi64>
// CHECK:           %[[VAL_50:.*]] = stablehlo.constant dense<0> : tensor<2xi64>
// CHECK:           %[[VAL_51:.*]] = stablehlo.constant dense<0> : tensor<2xi64>
// CHECK:           return %[[VAL_48]] : !torch.vtensor<[2,64],bf16>
// CHECK:         }
func.func @aten.embedding_bag.padding_idx$mean_reduction(%arg0: !torch.vtensor<[2],si64>, %arg1: !torch.vtensor<[30,64],bf16>, %arg2: !torch.vtensor<[5],si64>) -> !torch.vtensor<[2,64],bf16> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %false_0 = torch.constant.bool false
  %none = torch.constant.none
  %false_1 = torch.constant.bool false
  %none_2 = torch.constant.none
  %result0, %result1, %result2, %result3 = torch.aten.embedding_bag.padding_idx %arg1, %arg2, %arg0, %false, %int1, %false_0, %none, %false_1, %none_2 : !torch.vtensor<[30,64],bf16>, !torch.vtensor<[5],si64>, !torch.vtensor<[2],si64>, !torch.bool, !torch.int, !torch.bool, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[2,64],bf16>, !torch.vtensor<[5],si64>, !torch.vtensor<[2],si64>, !torch.vtensor<[2],si64>
  return %result0 : !torch.vtensor<[2,64],bf16>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.index_select$empty_indices(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[4,8],f32>, %[[ARG1:.*]]: !torch.vtensor<[0],si64>) -> !torch.vtensor<[0,8],f32> {
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<0x8xf32>
// CHECK:         %[[CONVERT:.*]] = stablehlo.convert %[[CST]] : tensor<0x8xf32>
// CHECK:         %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[CONVERT]] : tensor<0x8xf32> -> !torch.vtensor<[0,8],f32>
// CHECK:         return %[[RESULT]] : !torch.vtensor<[0,8],f32>
func.func @torch.aten.index_select$empty_indices(%arg0: !torch.vtensor<[4,8],f32>, %arg1: !torch.vtensor<[0],si64>) -> !torch.vtensor<[0,8],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.index_select %arg0, %int0, %arg1 : !torch.vtensor<[4,8],f32>, !torch.int, !torch.vtensor<[0],si64> -> !torch.vtensor<[0,8],f32>
  return %0 : !torch.vtensor<[0,8],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.embedding$empty_indices(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[10,8],f32>, %[[ARG1:.*]]: !torch.vtensor<[0],si64>) -> !torch.vtensor<[0,8],f32> {
// CHECK:         %[[FALSE:.*]] = torch.constant.bool false
// CHECK:         %[[INT_NEG1:.*]] = torch.constant.int -1
// CHECK:         %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<0x8xf32>
// CHECK:         %[[CONVERT:.*]] = stablehlo.convert %[[CST]] : tensor<0x8xf32>
// CHECK:         %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[CONVERT]] : tensor<0x8xf32> -> !torch.vtensor<[0,8],f32>
// CHECK:         return %[[RESULT]] : !torch.vtensor<[0,8],f32>
func.func @torch.aten.embedding$empty_indices(%weight: !torch.vtensor<[10,8],f32>, %indices: !torch.vtensor<[0], si64>) -> !torch.vtensor<[0,8],f32> {
  %false = torch.constant.bool false
  %int-1 = torch.constant.int -1
  %ret = torch.aten.embedding %weight, %indices, %int-1, %false, %false : !torch.vtensor<[10,8],f32>, !torch.vtensor<[0], si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[0,8],f32>
  return %ret: !torch.vtensor<[0,8],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.index_select$empty_indices_dim1(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[4,8],f32>, %[[ARG1:.*]]: !torch.vtensor<[0],si64>) -> !torch.vtensor<[4,0],f32> {
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<4x0xf32>
// CHECK:         %[[CONVERT:.*]] = stablehlo.convert %[[CST]] : tensor<4x0xf32>
// CHECK:         %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[CONVERT]] : tensor<4x0xf32> -> !torch.vtensor<[4,0],f32>
// CHECK:         return %[[RESULT]] : !torch.vtensor<[4,0],f32>
func.func @torch.aten.index_select$empty_indices_dim1(%arg0: !torch.vtensor<[4,8],f32>, %arg1: !torch.vtensor<[0],si64>) -> !torch.vtensor<[4,0],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.index_select %arg0, %int1, %arg1 : !torch.vtensor<[4,8],f32>, !torch.int, !torch.vtensor<[0],si64> -> !torch.vtensor<[4,0],f32>
  return %0 : !torch.vtensor<[4,0],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.gather$rank0(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[],bf16>, %[[ARG1:.*]]: !torch.vtensor<[],si32>) -> !torch.vtensor<[],bf16> {
// CHECK:         return %[[ARG0]] : !torch.vtensor<[],bf16>
func.func @torch.aten.gather$rank0(%arg0: !torch.vtensor<[],bf16>, %arg1: !torch.vtensor<[],si32>) -> !torch.vtensor<[],bf16> {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %0 = torch.aten.gather %arg0, %int0, %arg1, %false : !torch.vtensor<[],bf16>, !torch.int, !torch.vtensor<[],si32>, !torch.bool -> !torch.vtensor<[],bf16>
  return %0 : !torch.vtensor<[],bf16>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.index_select$rank0(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[],f32>, %[[ARG1:.*]]: !torch.vtensor<[],si64>) -> !torch.vtensor<[],f32> {
// CHECK:         return %[[ARG0]] : !torch.vtensor<[],f32>
func.func @torch.aten.index_select$rank0(%arg0: !torch.vtensor<[],f32>, %arg1: !torch.vtensor<[],si64>) -> !torch.vtensor<[],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.index_select %arg0, %int0, %arg1 : !torch.vtensor<[],f32>, !torch.int, !torch.vtensor<[],si64> -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.index.Tensor_hacked_twin$dynamic_trailing(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[4,?,8],f32>, %[[ARG1:.*]]: !torch.vtensor<[2],si64>) -> !torch.vtensor<[2,?,8],f32> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[4,?,8],f32> -> tensor<4x?x8xf32>
// CHECK:           %[[IDX:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[2],si64> -> tensor<2xi64>
// CHECK:           %[[RESHAPE:.*]] = stablehlo.reshape %[[IDX]] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK:           %[[ONE:.*]] = arith.constant 1 : i64
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIM1:.*]] = tensor.dim %[[INPUT]], %[[C1]] : tensor<4x?x8xf32>
// CHECK:           %[[DIM1_I64:.*]] = arith.index_cast %[[DIM1]] : index to i64
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[DIM2:.*]] = tensor.dim %[[INPUT]], %[[C2]] : tensor<4x?x8xf32>
// CHECK:           %[[DIM2_I64:.*]] = arith.index_cast %[[DIM2]] : index to i64
// CHECK:           %[[SLICE_SIZES:.*]] = tensor.from_elements %[[ONE]], %[[DIM1_I64]], %[[DIM2_I64]] : tensor<3xi64>
// CHECK:           %[[RESULT:.*]] = "stablehlo.dynamic_gather"(%[[INPUT]], %[[RESHAPE]], %[[SLICE_SIZES]])
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<2x?x8xf32> -> !torch.vtensor<[2,?,8],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[2,?,8],f32>
func.func @torch.aten.index.Tensor_hacked_twin$dynamic_trailing(%arg0: !torch.vtensor<[4,?,8],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.vtensor<[2,?,8],f32> {
  %0 = torch.prim.ListConstruct %arg1 : (!torch.vtensor<[2],si64>) -> !torch.list<vtensor>
  %1 = torch.aten.index.Tensor_hacked_twin %arg0, %0 : !torch.vtensor<[4,?,8],f32>, !torch.list<vtensor> -> !torch.vtensor<[2,?,8],f32>
  return %1 : !torch.vtensor<[2,?,8],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.index.Tensor_hacked_twin$dynamic_all_trailing(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[4,?,?],f32>, %[[ARG1:.*]]: !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,?,?],f32> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[4,?,?],f32> -> tensor<4x?x?xf32>
// CHECK:           %[[IDX:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[3],si64> -> tensor<3xi64>
// CHECK:           %[[RESHAPE:.*]] = stablehlo.reshape %[[IDX]] : (tensor<3xi64>) -> tensor<3x1xi64>
// CHECK:           %[[ONE:.*]] = arith.constant 1 : i64
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIM1:.*]] = tensor.dim %[[INPUT]], %[[C1]] : tensor<4x?x?xf32>
// CHECK:           %[[DIM1_I64:.*]] = arith.index_cast %[[DIM1]] : index to i64
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[DIM2:.*]] = tensor.dim %[[INPUT]], %[[C2]] : tensor<4x?x?xf32>
// CHECK:           %[[DIM2_I64:.*]] = arith.index_cast %[[DIM2]] : index to i64
// CHECK:           %[[SLICE_SIZES:.*]] = tensor.from_elements %[[ONE]], %[[DIM1_I64]], %[[DIM2_I64]] : tensor<3xi64>
// CHECK:           %[[RESULT:.*]] = "stablehlo.dynamic_gather"(%[[INPUT]], %[[RESHAPE]], %[[SLICE_SIZES]])
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<3x?x?xf32> -> !torch.vtensor<[3,?,?],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[3,?,?],f32>
func.func @torch.aten.index.Tensor_hacked_twin$dynamic_all_trailing(%arg0: !torch.vtensor<[4,?,?],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,?,?],f32> {
  %0 = torch.prim.ListConstruct %arg1 : (!torch.vtensor<[3],si64>) -> !torch.list<vtensor>
  %1 = torch.aten.index.Tensor_hacked_twin %arg0, %0 : !torch.vtensor<[4,?,?],f32>, !torch.list<vtensor> -> !torch.vtensor<[3,?,?],f32>
  return %1 : !torch.vtensor<[3,?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.index.Tensor_hacked_twin$multi_index_dynamic(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[4,4,?],f32>, %[[ARG1:.*]]: !torch.vtensor<[2],si64>, %[[ARG2:.*]]: !torch.vtensor<[2],si64>) -> !torch.vtensor<[2,?],f32> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[4,4,?],f32> -> tensor<4x4x?xf32>
// CHECK:           %[[IDX0:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[2],si64> -> tensor<2xi64>
// CHECK:           %[[IDX1:.*]] = torch_c.to_builtin_tensor %[[ARG2]] : !torch.vtensor<[2],si64> -> tensor<2xi64>
// CHECK:           %[[IDX0_R:.*]] = stablehlo.reshape %[[IDX0]] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK:           %[[IDX1_R:.*]] = stablehlo.reshape %[[IDX1]] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK:           %[[INDICES:.*]] = stablehlo.concatenate %[[IDX0_R]], %[[IDX1_R]], dim = 1 : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<2x2xi64>
// CHECK:           %[[C1:.*]] = arith.constant 1 : i64
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[DIM2:.*]] = tensor.dim %[[INPUT]], %[[C2]] : tensor<4x4x?xf32>
// CHECK:           %[[DIM2_I64:.*]] = arith.index_cast %[[DIM2]] : index to i64
// CHECK:           %[[SLICE_SIZES:.*]] = tensor.from_elements %[[C1]], %[[C1]], %[[DIM2_I64]] : tensor<3xi64>
// CHECK:           %[[RESULT:.*]] = "stablehlo.dynamic_gather"(%[[INPUT]], %[[INDICES]], %[[SLICE_SIZES]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false}> : (tensor<4x4x?xf32>, tensor<2x2xi64>, tensor<3xi64>) -> tensor<2x?xf32>
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<2x?xf32> -> !torch.vtensor<[2,?],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[2,?],f32>
// CHECK:         }
func.func @torch.aten.index.Tensor_hacked_twin$multi_index_dynamic(%arg0: !torch.vtensor<[4,4,?],f32>, %arg1: !torch.vtensor<[2],si64>, %arg2: !torch.vtensor<[2],si64>) -> !torch.vtensor<[2,?],f32> {
  %0 = torch.prim.ListConstruct %arg1, %arg2 : (!torch.vtensor<[2],si64>, !torch.vtensor<[2],si64>) -> !torch.list<vtensor>
  %1 = torch.aten.index.Tensor_hacked_twin %arg0, %0 : !torch.vtensor<[4,4,?],f32>, !torch.list<vtensor> -> !torch.vtensor<[2,?],f32>
  return %1 : !torch.vtensor<[2,?],f32>
}
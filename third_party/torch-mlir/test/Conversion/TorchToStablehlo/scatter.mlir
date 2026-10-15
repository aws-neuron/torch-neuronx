// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @forward(
// CHECK-SAME:                      %[[ARG_0:.*]]: !torch.vtensor<[?,?],si64>, %[[ARG_1:.*]]: !torch.vtensor<[?,?],si64>, %[[ARG_2:.*]]: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[?,?],si64> {
// CHECK-DAG:      %[[VAR_0:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
// CHECK-DAG:      %[[VAR_1:.*]] = torch_c.to_builtin_tensor %[[ARG_1]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
// CHECK-DAG:      %[[VAR_2:.*]] = torch_c.to_builtin_tensor %[[ARG_2]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
// CHECK:          %int0 = torch.constant.int 0
// CHECK:          %[[INDEX_0:.*]] = arith.constant 0 : index
// CHECK:          %[[DIM_0:.*]] = tensor.dim %[[VAR_1]], %[[INDEX_0]] : tensor<?x?xi64>
// CHECK:          %[[INDEX_1:.*]] = arith.constant 1 : index
// CHECK:          %[[DIM_1:.*]] = tensor.dim %1, %[[INDEX_1]] : tensor<?x?xi64>
// CHECK:          %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:          %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:          %[[FE_:.*]] = tensor.from_elements %[[CONSTANT_0]], %[[CONSTANT_0]] : tensor<2xindex>
// CHECK:          %[[FE_1:.*]] = tensor.from_elements %[[CONSTANT_1]], %[[CONSTANT_1]] : tensor<2xindex>
// CHECK:          %[[FE_2:.*]] = tensor.from_elements %[[DIM_0]], %[[DIM_1]] : tensor<2xindex>
// CHECK:          %[[VAR_5:.*]] = stablehlo.real_dynamic_slice %[[VAR_2]], %[[FE_]], %[[FE_2]], %[[FE_1]] : (tensor<?x?xi64>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<?x?xi64>
// CHECK:          %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK:          %[[FE_3:.*]] = tensor.from_elements %[[DIM_0]], %[[DIM_1]], %[[CONSTANT_2]] : tensor<3xindex>
// CHECK:          %[[VAR_6:.*]] = stablehlo.dynamic_reshape %1, %[[FE_3]] : (tensor<?x?xi64>, tensor<3xindex>) -> tensor<?x?x1xi64>
// CHECK:          %[[VAR_7:.*]] = stablehlo.dynamic_iota %[[FE_3]], dim = 1 : (tensor<3xindex>) -> tensor<?x?x1xi64>
// CHECK:          %[[VAR_8:.*]] = stablehlo.concatenate %[[VAR_6]], %[[VAR_7]], dim = 2 : (tensor<?x?x1xi64>, tensor<?x?x1xi64>) -> tensor<?x?x2xi64>
// CHECK:          %[[VAR_9:.*]] = "stablehlo.scatter"(%[[VAR_0]], %[[VAR_8]], %[[VAR_5]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK:          ^bb0(%arg3: tensor<i64>, %[[ARG_4:.*]]: tensor<i64>):
// CHECK:            stablehlo.return %[[ARG_4]] : tensor<i64>
// CHECK:          }) : (tensor<?x?xi64>, tensor<?x?x2xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>
// CHECK:          %[[VAR_10:.*]] = torch_c.from_builtin_tensor %[[VAR_9]] : tensor<?x?xi64> -> !torch.vtensor<[?,?],si64>
// CHECK:          return %[[VAR_10]] : !torch.vtensor<[?,?],si64>
func.func @forward(%arg0: !torch.vtensor<[?,?],si64>, %arg1: !torch.vtensor<[?,?],si64>, %arg2: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[?,?],si64> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.scatter.src %arg0, %int0, %arg1, %arg2 : !torch.vtensor<[?,?],si64>, !torch.int, !torch.vtensor<[?,?],si64>, !torch.vtensor<[?,?],si64> -> !torch.vtensor<[?,?],si64>
  return %0 : !torch.vtensor<[?,?],si64>
}

// CHECK-LABEL:   func.func @test_scatter_reduce_sum(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !torch.vtensor<[5,6],f32>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: !torch.vtensor<[3,6],f32>,
// CHECK-SAME:                                       %[[VAL_2:.*]]: !torch.vtensor<[3,6],si64>) -> !torch.vtensor<[5,6],f32> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[3,6],f32> -> tensor<3x6xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[3,6],si64> -> tensor<3x6xi64>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[5,6],f32> -> tensor<5x6xf32>
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_7:.*]] = torch.constant.str "sum"
// CHECK:           %[[VAL_8:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_10:.*]] = tensor.dim %[[VAL_4]], %[[VAL_9]] : tensor<3x6xi64>
// CHECK:           %[[VAL_11:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_12:.*]] = tensor.dim %[[VAL_4]], %[[VAL_11]] : tensor<3x6xi64>
// CHECK:           %[[VAL_13:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_14:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_15:.*]] = tensor.from_elements %[[VAL_13]], %[[VAL_13]] : tensor<2xindex>
// CHECK:           %[[VAL_16:.*]] = tensor.from_elements %[[VAL_14]], %[[VAL_14]] : tensor<2xindex>
// CHECK:           %[[VAL_17:.*]] = tensor.from_elements %[[VAL_10]], %[[VAL_12]] : tensor<2xindex>
// CHECK:           %[[VAL_18:.*]] = stablehlo.real_dynamic_slice %[[VAL_3]], %[[VAL_15]], %[[VAL_17]], %[[VAL_16]] : (tensor<3x6xf32>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<3x6xf32>
// CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_20:.*]] = tensor.from_elements %[[VAL_10]], %[[VAL_12]], %[[VAL_19]] : tensor<3xindex>
// CHECK:           %[[VAL_21:.*]] = stablehlo.dynamic_reshape %[[VAL_4]], %[[VAL_20]] : (tensor<3x6xi64>, tensor<3xindex>) -> tensor<3x6x1xi64>
// CHECK:           %[[VAL_22:.*]] = stablehlo.dynamic_iota %[[VAL_20]], dim = 1 : (tensor<3xindex>) -> tensor<3x6x1xi64>
// CHECK:           %[[VAL_23:.*]] = stablehlo.concatenate %[[VAL_21]], %[[VAL_22]], dim = 2 : (tensor<3x6x1xi64>, tensor<3x6x1xi64>) -> tensor<3x6x2xi64>
// CHECK:           %[[VAL_24:.*]] = "stablehlo.scatter"(%[[VAL_5]], %[[VAL_23]], %[[VAL_18]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK:           ^bb0(%[[VAL_25:.*]]: tensor<f32>, %[[VAL_26:.*]]: tensor<f32>):
// CHECK:             %[[VAL_27:.*]] = stablehlo.add %[[VAL_25]], %[[VAL_26]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_27]] : tensor<f32>
// CHECK:           }) : (tensor<5x6xf32>, tensor<3x6x2xi64>, tensor<3x6xf32>) -> tensor<5x6xf32>
// CHECK:           %[[VAL_28:.*]] = torch_c.from_builtin_tensor %[[VAL_29:.*]] : tensor<5x6xf32> -> !torch.vtensor<[5,6],f32>
// CHECK:           return %[[VAL_28]] : !torch.vtensor<[5,6],f32>
// CHECK:         }
func.func @test_scatter_reduce_sum(%arg0: !torch.vtensor<[5,6],f32>, %arg1: !torch.vtensor<[3,6],f32>, %arg2: !torch.vtensor<[3,6],si64>) -> !torch.vtensor<[5,6],f32> {
    %int0 = torch.constant.int 0
    %str = torch.constant.str "sum"
    %true = torch.constant.bool true
    %0 = torch.aten.scatter_reduce.two %arg0, %int0, %arg2, %arg1, %str, %true : !torch.vtensor<[5,6],f32>, !torch.int, !torch.vtensor<[3,6],si64>, !torch.vtensor<[3,6],f32>, !torch.str, !torch.bool -> !torch.vtensor<[5,6],f32>
    return %0 : !torch.vtensor<[5,6],f32>
}

// CHECK-LABEL:   func.func @test_scatter_reduce_prod(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[3,4,5],f32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: !torch.vtensor<[3,6,5],f32>,
// CHECK-SAME:                                        %[[VAL_2:.*]]: !torch.vtensor<[3,6,5],si64>) -> !torch.vtensor<[3,4,5],f32> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[3,6,5],f32> -> tensor<3x6x5xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[3,6,5],si64> -> tensor<3x6x5xi64>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4,5],f32> -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_7:.*]] = torch.constant.str "prod"
// CHECK:           %[[VAL_8:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_10:.*]] = tensor.dim %[[VAL_4]], %[[VAL_9]] : tensor<3x6x5xi64>
// CHECK:           %[[VAL_11:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_12:.*]] = tensor.dim %[[VAL_4]], %[[VAL_11]] : tensor<3x6x5xi64>
// CHECK:           %[[VAL_13:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_14:.*]] = tensor.dim %[[VAL_4]], %[[VAL_13]] : tensor<3x6x5xi64>
// CHECK:           %[[VAL_15:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_16:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_17:.*]] = tensor.from_elements %[[VAL_15]], %[[VAL_15]], %[[VAL_15]] : tensor<3xindex>
// CHECK:           %[[VAL_18:.*]] = tensor.from_elements %[[VAL_16]], %[[VAL_16]], %[[VAL_16]] : tensor<3xindex>
// CHECK:           %[[VAL_19:.*]] = tensor.from_elements %[[VAL_10]], %[[VAL_12]], %[[VAL_14]] : tensor<3xindex>
// CHECK:           %[[VAL_20:.*]] = stablehlo.real_dynamic_slice %[[VAL_3]], %[[VAL_17]], %[[VAL_19]], %[[VAL_18]] : (tensor<3x6x5xf32>, tensor<3xindex>, tensor<3xindex>, tensor<3xindex>) -> tensor<3x6x5xf32>
// CHECK:           %[[VAL_21:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_22:.*]] = tensor.from_elements %[[VAL_10]], %[[VAL_12]], %[[VAL_14]], %[[VAL_21]] : tensor<4xindex>
// CHECK:           %[[VAL_23:.*]] = stablehlo.dynamic_iota %[[VAL_22]], dim = 0 : (tensor<4xindex>) -> tensor<3x6x5x1xi64>
// CHECK:           %[[VAL_24:.*]] = stablehlo.dynamic_reshape %[[VAL_4]], %[[VAL_22]] : (tensor<3x6x5xi64>, tensor<4xindex>) -> tensor<3x6x5x1xi64>
// CHECK:           %[[VAL_25:.*]] = stablehlo.dynamic_iota %[[VAL_22]], dim = 2 : (tensor<4xindex>) -> tensor<3x6x5x1xi64>
// CHECK:           %[[VAL_26:.*]] = stablehlo.concatenate %[[VAL_23]], %[[VAL_24]], %[[VAL_25]], dim = 3 : (tensor<3x6x5x1xi64>, tensor<3x6x5x1xi64>, tensor<3x6x5x1xi64>) -> tensor<3x6x5x3xi64>
// CHECK:           %[[VAL_27:.*]] = "chlo.constant_like"(%[[VAL_20]]) <{value = 1.000000e+00 : f32}> : (tensor<3x6x5xf32>) -> tensor<3x6x5xf32>
// CHECK:           %[[VAL_28:.*]] = "stablehlo.scatter"(%[[VAL_5]], %[[VAL_26]], %[[VAL_27]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1, 2], scatter_dims_to_operand_dims = [0, 1, 2], index_vector_dim = 3>, unique_indices = false}> ({
// CHECK:           ^bb0(%[[VAL_29:.*]]: tensor<f32>, %[[VAL_30:.*]]: tensor<f32>):
// CHECK:             stablehlo.return %[[VAL_30]] : tensor<f32>
// CHECK:           }) : (tensor<3x4x5xf32>, tensor<3x6x5x3xi64>, tensor<3x6x5xf32>) -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_31:.*]] = "stablehlo.scatter"(%[[VAL_32:.*]], %[[VAL_26]], %[[VAL_20]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1, 2], scatter_dims_to_operand_dims = [0, 1, 2], index_vector_dim = 3>, unique_indices = false}> ({
// CHECK:           ^bb0(%[[VAL_33:.*]]: tensor<f32>, %[[VAL_34:.*]]: tensor<f32>):
// CHECK:             %[[VAL_35:.*]] = stablehlo.multiply %[[VAL_33]], %[[VAL_34]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_35]] : tensor<f32>
// CHECK:           }) : (tensor<3x4x5xf32>, tensor<3x6x5x3xi64>, tensor<3x6x5xf32>) -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_36:.*]] = torch_c.from_builtin_tensor %[[VAL_37:.*]] : tensor<3x4x5xf32> -> !torch.vtensor<[3,4,5],f32>
// CHECK:           return %[[VAL_36]] : !torch.vtensor<[3,4,5],f32>
// CHECK:         }
func.func @test_scatter_reduce_prod(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,6,5],f32>, %arg2: !torch.vtensor<[3,6,5],si64>) -> !torch.vtensor<[3,4,5],f32> {
    %int1 = torch.constant.int 1
    %str = torch.constant.str "prod"
    %false = torch.constant.bool false
    %0 = torch.aten.scatter_reduce.two %arg0, %int1, %arg2, %arg1, %str, %false : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.vtensor<[3,6,5],si64>, !torch.vtensor<[3,6,5],f32>, !torch.str, !torch.bool -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL:   func.func @test_scatter_reduce_mean(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[8],f32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: !torch.vtensor<[12],f32>,
// CHECK-SAME:                                        %[[VAL_2:.*]]: !torch.vtensor<[12],si64>) -> !torch.vtensor<[8],f32> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[12],f32> -> tensor<12xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[12],si64> -> tensor<12xi64>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[8],f32> -> tensor<8xf32>
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_7:.*]] = torch.constant.str "mean"
// CHECK:           %[[VAL_8:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_10:.*]] = tensor.dim %[[VAL_4]], %[[VAL_9]] : tensor<12xi64>
// CHECK:           %[[VAL_11:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_13:.*]] = tensor.from_elements %[[VAL_11]] : tensor<1xindex>
// CHECK:           %[[VAL_14:.*]] = tensor.from_elements %[[VAL_12]] : tensor<1xindex>
// CHECK:           %[[VAL_15:.*]] = tensor.from_elements %[[VAL_10]] : tensor<1xindex>
// CHECK:           %[[VAL_16:.*]] = stablehlo.real_dynamic_slice %[[VAL_3]], %[[VAL_13]], %[[VAL_15]], %[[VAL_14]] : (tensor<12xf32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<12xf32>
// CHECK:           %[[VAL_17:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_18:.*]] = tensor.from_elements %[[VAL_10]], %[[VAL_17]] : tensor<2xindex>
// CHECK:           %[[VAL_19:.*]] = stablehlo.dynamic_reshape %[[VAL_4]], %[[VAL_18]] : (tensor<12xi64>, tensor<2xindex>) -> tensor<12x1xi64>
// CHECK:           %[[VAL_20:.*]] = stablehlo.concatenate %[[VAL_19]], dim = 1 : (tensor<12x1xi64>) -> tensor<12x1xi64>
// CHECK:           %[[VAL_21:.*]] = "chlo.constant_like"(%[[VAL_16]]) <{value = 0.000000e+00 : f32}> : (tensor<12xf32>) -> tensor<12xf32>
// CHECK:           %[[VAL_22:.*]] = "stablehlo.scatter"(%[[VAL_5]], %[[VAL_20]], %[[VAL_21]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK:           ^bb0(%[[VAL_23:.*]]: tensor<f32>, %[[VAL_24:.*]]: tensor<f32>):
// CHECK:             stablehlo.return %[[VAL_24]] : tensor<f32>
// CHECK:           }) : (tensor<8xf32>, tensor<12x1xi64>, tensor<12xf32>) -> tensor<8xf32>
// CHECK:           %[[VAL_25:.*]] = "stablehlo.scatter"(%[[VAL_26:.*]], %[[VAL_20]], %[[VAL_16]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK:           ^bb0(%[[VAL_27:.*]]: tensor<f32>, %[[VAL_28:.*]]: tensor<f32>):
// CHECK:             %[[VAL_29:.*]] = stablehlo.add %[[VAL_27]], %[[VAL_28]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_29]] : tensor<f32>
// CHECK:           }) : (tensor<8xf32>, tensor<12x1xi64>, tensor<12xf32>) -> tensor<8xf32>
// CHECK:           %[[VAL_30:.*]] = "chlo.constant_like"(%[[VAL_16]]) <{value = 1.000000e+00 : f32}> : (tensor<12xf32>) -> tensor<12xf32>
// CHECK:           %[[VAL_31:.*]] = "chlo.constant_like"(%[[VAL_5]]) <{value = 0.000000e+00 : f32}> : (tensor<8xf32>) -> tensor<8xf32>
// CHECK:           %[[VAL_32:.*]] = "stablehlo.scatter"(%[[VAL_31]], %[[VAL_20]], %[[VAL_30]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK:           ^bb0(%[[VAL_33:.*]]: tensor<f32>, %[[VAL_34:.*]]: tensor<f32>):
// CHECK:             %[[VAL_35:.*]] = stablehlo.add %[[VAL_33]], %[[VAL_34]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_35]] : tensor<f32>
// CHECK:           }) : (tensor<8xf32>, tensor<12x1xi64>, tensor<12xf32>) -> tensor<8xf32>
// CHECK:           %[[VAL_36:.*]] = "chlo.constant_like"(%[[VAL_37:.*]]) <{value = 1.000000e+00 : f32}> : (tensor<8xf32>) -> tensor<8xf32>
// CHECK:           %[[VAL_38:.*]] = stablehlo.maximum %[[VAL_37]], %[[VAL_36]] : tensor<8xf32>
// CHECK:           %[[VAL_39:.*]] = stablehlo.divide %[[VAL_40:.*]], %[[VAL_38]] : tensor<8xf32>
// CHECK:           %[[VAL_41:.*]] = torch_c.from_builtin_tensor %[[VAL_39]] : tensor<8xf32> -> !torch.vtensor<[8],f32>
// CHECK:           return %[[VAL_41]] : !torch.vtensor<[8],f32>
// CHECK:         }
func.func @test_scatter_reduce_mean(%arg0: !torch.vtensor<[8],f32>, %arg1: !torch.vtensor<[12],f32>, %arg2: !torch.vtensor<[12],si64>) -> !torch.vtensor<[8],f32> {
    %int0 = torch.constant.int 0
    %str = torch.constant.str "mean"
    %false = torch.constant.bool false
    %0 = torch.aten.scatter_reduce.two %arg0, %int0, %arg2, %arg1, %str, %false : !torch.vtensor<[8],f32>, !torch.int, !torch.vtensor<[12],si64>, !torch.vtensor<[12],f32>, !torch.str, !torch.bool -> !torch.vtensor<[8],f32>
    return %0 : !torch.vtensor<[8],f32>
}

// CHECK-LABEL:   func.func @test_scatter_reduce_amax(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[3,4,5],f32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: !torch.vtensor<[3,4,10],f32>,
// CHECK-SAME:                                        %[[VAL_2:.*]]: !torch.vtensor<[3,4,10],si64>) -> !torch.vtensor<[3,4,5],f32> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[3,4,10],f32> -> tensor<3x4x10xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[3,4,10],si64> -> tensor<3x4x10xi64>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4,5],f32> -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_7:.*]] = torch.constant.str "amax"
// CHECK:           %[[VAL_8:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_10:.*]] = tensor.dim %[[VAL_4]], %[[VAL_9]] : tensor<3x4x10xi64>
// CHECK:           %[[VAL_11:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_12:.*]] = tensor.dim %[[VAL_4]], %[[VAL_11]] : tensor<3x4x10xi64>
// CHECK:           %[[VAL_13:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_14:.*]] = tensor.dim %[[VAL_4]], %[[VAL_13]] : tensor<3x4x10xi64>
// CHECK:           %[[VAL_15:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_16:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_17:.*]] = tensor.from_elements %[[VAL_15]], %[[VAL_15]], %[[VAL_15]] : tensor<3xindex>
// CHECK:           %[[VAL_18:.*]] = tensor.from_elements %[[VAL_16]], %[[VAL_16]], %[[VAL_16]] : tensor<3xindex>
// CHECK:           %[[VAL_19:.*]] = tensor.from_elements %[[VAL_10]], %[[VAL_12]], %[[VAL_14]] : tensor<3xindex>
// CHECK:           %[[VAL_20:.*]] = stablehlo.real_dynamic_slice %[[VAL_3]], %[[VAL_17]], %[[VAL_19]], %[[VAL_18]] : (tensor<3x4x10xf32>, tensor<3xindex>, tensor<3xindex>, tensor<3xindex>) -> tensor<3x4x10xf32>
// CHECK:           %[[VAL_21:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_22:.*]] = tensor.from_elements %[[VAL_10]], %[[VAL_12]], %[[VAL_14]], %[[VAL_21]] : tensor<4xindex>
// CHECK:           %[[VAL_23:.*]] = stablehlo.dynamic_iota %[[VAL_22]], dim = 0 : (tensor<4xindex>) -> tensor<3x4x10x1xi64>
// CHECK:           %[[VAL_24:.*]] = stablehlo.dynamic_iota %[[VAL_22]], dim = 1 : (tensor<4xindex>) -> tensor<3x4x10x1xi64>
// CHECK:           %[[VAL_25:.*]] = stablehlo.dynamic_reshape %[[VAL_4]], %[[VAL_22]] : (tensor<3x4x10xi64>, tensor<4xindex>) -> tensor<3x4x10x1xi64>
// CHECK:           %[[VAL_26:.*]] = stablehlo.concatenate %[[VAL_23]], %[[VAL_24]], %[[VAL_25]], dim = 3 : (tensor<3x4x10x1xi64>, tensor<3x4x10x1xi64>, tensor<3x4x10x1xi64>) -> tensor<3x4x10x3xi64>
// CHECK:           %[[VAL_27:.*]] = "chlo.constant_like"(%[[VAL_20]]) <{value = 0xFF800000 : f32}> : (tensor<3x4x10xf32>) -> tensor<3x4x10xf32>
// CHECK:           %[[VAL_28:.*]] = "stablehlo.scatter"(%[[VAL_5]], %[[VAL_26]], %[[VAL_27]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1, 2], scatter_dims_to_operand_dims = [0, 1, 2], index_vector_dim = 3>, unique_indices = false}> ({
// CHECK:           ^bb0(%[[VAL_29:.*]]: tensor<f32>, %[[VAL_30:.*]]: tensor<f32>):
// CHECK:             stablehlo.return %[[VAL_30]] : tensor<f32>
// CHECK:           }) : (tensor<3x4x5xf32>, tensor<3x4x10x3xi64>, tensor<3x4x10xf32>) -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_31:.*]] = "stablehlo.scatter"(%[[VAL_32:.*]], %[[VAL_26]], %[[VAL_20]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1, 2], scatter_dims_to_operand_dims = [0, 1, 2], index_vector_dim = 3>, unique_indices = false}> ({
// CHECK:           ^bb0(%[[VAL_33:.*]]: tensor<f32>, %[[VAL_34:.*]]: tensor<f32>):
// CHECK:             %[[VAL_35:.*]] = stablehlo.maximum %[[VAL_33]], %[[VAL_34]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_35]] : tensor<f32>
// CHECK:           }) : (tensor<3x4x5xf32>, tensor<3x4x10x3xi64>, tensor<3x4x10xf32>) -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_36:.*]] = torch_c.from_builtin_tensor %[[VAL_37:.*]] : tensor<3x4x5xf32> -> !torch.vtensor<[3,4,5],f32>
// CHECK:           return %[[VAL_36]] : !torch.vtensor<[3,4,5],f32>
// CHECK:         }
func.func @test_scatter_reduce_amax(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,10],f32>, %arg2: !torch.vtensor<[3,4,10],si64>) -> !torch.vtensor<[3,4,5],f32> {
    %int2 = torch.constant.int 2
    %str = torch.constant.str "amax"
    %false = torch.constant.bool false
    %0 = torch.aten.scatter_reduce.two %arg0, %int2, %arg2, %arg1, %str, %false : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.vtensor<[3,4,10],si64>, !torch.vtensor<[3,4,10],f32>, !torch.str, !torch.bool -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL:   func.func @test_scatter_reduce_amin(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[3,4,5],f32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: !torch.vtensor<[3,10,5],f32>,
// CHECK-SAME:                                        %[[VAL_2:.*]]: !torch.vtensor<[3,10,5],si64>) -> !torch.vtensor<[3,4,5],f32> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[3,10,5],f32> -> tensor<3x10x5xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[3,10,5],si64> -> tensor<3x10x5xi64>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4,5],f32> -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_7:.*]] = torch.constant.str "amin"
// CHECK:           %[[VAL_8:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_10:.*]] = tensor.dim %[[VAL_4]], %[[VAL_9]] : tensor<3x10x5xi64>
// CHECK:           %[[VAL_11:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_12:.*]] = tensor.dim %[[VAL_4]], %[[VAL_11]] : tensor<3x10x5xi64>
// CHECK:           %[[VAL_13:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_14:.*]] = tensor.dim %[[VAL_4]], %[[VAL_13]] : tensor<3x10x5xi64>
// CHECK:           %[[VAL_15:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_16:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_17:.*]] = tensor.from_elements %[[VAL_15]], %[[VAL_15]], %[[VAL_15]] : tensor<3xindex>
// CHECK:           %[[VAL_18:.*]] = tensor.from_elements %[[VAL_16]], %[[VAL_16]], %[[VAL_16]] : tensor<3xindex>
// CHECK:           %[[VAL_19:.*]] = tensor.from_elements %[[VAL_10]], %[[VAL_12]], %[[VAL_14]] : tensor<3xindex>
// CHECK:           %[[VAL_20:.*]] = stablehlo.real_dynamic_slice %[[VAL_3]], %[[VAL_17]], %[[VAL_19]], %[[VAL_18]] : (tensor<3x10x5xf32>, tensor<3xindex>, tensor<3xindex>, tensor<3xindex>) -> tensor<3x10x5xf32>
// CHECK:           %[[VAL_21:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_22:.*]] = tensor.from_elements %[[VAL_10]], %[[VAL_12]], %[[VAL_14]], %[[VAL_21]] : tensor<4xindex>
// CHECK:           %[[VAL_23:.*]] = stablehlo.dynamic_iota %[[VAL_22]], dim = 0 : (tensor<4xindex>) -> tensor<3x10x5x1xi64>
// CHECK:           %[[VAL_24:.*]] = stablehlo.dynamic_reshape %[[VAL_4]], %[[VAL_22]] : (tensor<3x10x5xi64>, tensor<4xindex>) -> tensor<3x10x5x1xi64>
// CHECK:           %[[VAL_25:.*]] = stablehlo.dynamic_iota %[[VAL_22]], dim = 2 : (tensor<4xindex>) -> tensor<3x10x5x1xi64>
// CHECK:           %[[VAL_26:.*]] = stablehlo.concatenate %[[VAL_23]], %[[VAL_24]], %[[VAL_25]], dim = 3 : (tensor<3x10x5x1xi64>, tensor<3x10x5x1xi64>, tensor<3x10x5x1xi64>) -> tensor<3x10x5x3xi64>
// CHECK:           %[[VAL_27:.*]] = "stablehlo.scatter"(%[[VAL_5]], %[[VAL_26]], %[[VAL_20]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1, 2], scatter_dims_to_operand_dims = [0, 1, 2], index_vector_dim = 3>, unique_indices = false}> ({
// CHECK:           ^bb0(%[[VAL_28:.*]]: tensor<f32>, %[[VAL_29:.*]]: tensor<f32>):
// CHECK:             %[[VAL_30:.*]] = stablehlo.minimum %[[VAL_28]], %[[VAL_29]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_30]] : tensor<f32>
// CHECK:           }) : (tensor<3x4x5xf32>, tensor<3x10x5x3xi64>, tensor<3x10x5xf32>) -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_31:.*]] = torch_c.from_builtin_tensor %[[VAL_32:.*]] : tensor<3x4x5xf32> -> !torch.vtensor<[3,4,5],f32>
// CHECK:           return %[[VAL_31]] : !torch.vtensor<[3,4,5],f32>
// CHECK:         }
func.func @test_scatter_reduce_amin(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,10,5],f32>, %arg2: !torch.vtensor<[3,10,5],si64>) -> !torch.vtensor<[3,4,5],f32> {
    %int1 = torch.constant.int 1
    %str = torch.constant.str "amin"
    %true = torch.constant.bool true
    %0 = torch.aten.scatter_reduce.two %arg0, %int1, %arg2, %arg1, %str, %true : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.vtensor<[3,10,5],si64>, !torch.vtensor<[3,10,5],f32>, !torch.str, !torch.bool -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL:  func.func @test_scatter_add_rank0(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[],bf16>, %[[ARG1:.*]]: !torch.vtensor<[],si32>, %[[ARG2:.*]]: !torch.vtensor<[],bf16>) -> !torch.vtensor<[],bf16> {
// CHECK-DAG:      %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG2]] : !torch.vtensor<[],bf16> -> tensor<bf16>
// CHECK-DAG:      %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[],bf16> -> tensor<bf16>
// CHECK:          %[[T2:.*]] = stablehlo.add %[[T1]], %[[T0]] : tensor<bf16>
// CHECK:          %[[T3:.*]] = torch_c.from_builtin_tensor %[[T2]] : tensor<bf16> -> !torch.vtensor<[],bf16>
// CHECK:          return %[[T3]] : !torch.vtensor<[],bf16>
func.func @test_scatter_add_rank0(%arg0: !torch.vtensor<[],bf16>, %arg1: !torch.vtensor<[],si32>, %arg2: !torch.vtensor<[],bf16>) -> !torch.vtensor<[],bf16> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.scatter_add %arg0, %int0, %arg1, %arg2 : !torch.vtensor<[],bf16>, !torch.int, !torch.vtensor<[],si32>, !torch.vtensor<[],bf16> -> !torch.vtensor<[],bf16>
  return %0 : !torch.vtensor<[],bf16>
}

// -----

// CHECK-LABEL:  func.func @test_scatter_src_rank0(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[],f32>, %[[ARG1:.*]]: !torch.vtensor<[],si64>, %[[ARG2:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
// CHECK:          return %[[ARG2]] : !torch.vtensor<[],f32>
func.func @test_scatter_src_rank0(%arg0: !torch.vtensor<[],f32>, %arg1: !torch.vtensor<[],si64>, %arg2: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.scatter.src %arg0, %int0, %arg1, %arg2 : !torch.vtensor<[],f32>, !torch.int, !torch.vtensor<[],si64>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL:  func.func @test_scatter_reduce_amax_rank0(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[],f32>, %[[ARG1:.*]]: !torch.vtensor<[],f32>, %[[ARG2:.*]]: !torch.vtensor<[],si64>) -> !torch.vtensor<[],f32> {
// CHECK-DAG:      %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK-DAG:      %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:          %[[T2:.*]] = stablehlo.maximum %[[T1]], %[[T0]] : tensor<f32>
// CHECK:          %[[T3:.*]] = torch_c.from_builtin_tensor %[[T2]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:          return %[[T3]] : !torch.vtensor<[],f32>
func.func @test_scatter_reduce_amax_rank0(%arg0: !torch.vtensor<[],f32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],si64>) -> !torch.vtensor<[],f32> {
  %int0 = torch.constant.int 0
  %str = torch.constant.str "amax"
  %true = torch.constant.bool true
  %0 = torch.aten.scatter_reduce.two %arg0, %int0, %arg2, %arg1, %str, %true : !torch.vtensor<[],f32>, !torch.int, !torch.vtensor<[],si64>, !torch.vtensor<[],f32>, !torch.str, !torch.bool -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL:  func.func @test_scatter_reduce_sum_rank0_no_include_self(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[],f32>, %[[ARG1:.*]]: !torch.vtensor<[],f32>, %[[ARG2:.*]]: !torch.vtensor<[],si64>) -> !torch.vtensor<[],f32> {
// CHECK:          return %[[ARG1]] : !torch.vtensor<[],f32>
func.func @test_scatter_reduce_sum_rank0_no_include_self(%arg0: !torch.vtensor<[],f32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],si64>) -> !torch.vtensor<[],f32> {
  %int0 = torch.constant.int 0
  %str = torch.constant.str "sum"
  %false = torch.constant.bool false
  %0 = torch.aten.scatter_reduce.two %arg0, %int0, %arg2, %arg1, %str, %false : !torch.vtensor<[],f32>, !torch.int, !torch.vtensor<[],si64>, !torch.vtensor<[],f32>, !torch.str, !torch.bool -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// Static input, static src, constant start/end/step
// CHECK-LABEL: func.func @torch.aten.slice_scatter$all_static(
// CHECK-SAME:      %[[INPUT:.*]]: !torch.vtensor<[8,4],f32>, %[[SRC:.*]]: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[8,4],f32>
// CHECK:         %[[INDICES:.*]] = stablehlo.constant dense<{{\[\[}}2], [3]]> : tensor<2x1xi64>
// CHECK:         "stablehlo.scatter"(%{{.*}}, %[[INDICES]], %{{.*}})
func.func @torch.aten.slice_scatter$all_static(%arg0: !torch.vtensor<[8,4],f32>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[8,4],f32> {
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %int4 = torch.constant.int 4
  %int1 = torch.constant.int 1
  %0 = torch.aten.slice_scatter %arg0, %arg1, %int0, %int2, %int4, %int1 : !torch.vtensor<[8,4],f32>, !torch.vtensor<[2,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[8,4],f32>
  return %0 : !torch.vtensor<[8,4],f32>
}

// -----

// Static input, static src, constant start/end with step=2
// CHECK-LABEL: func.func @torch.aten.slice_scatter$static_step2(
// CHECK-SAME:      %[[INPUT:.*]]: !torch.vtensor<[8,4],f32>, %[[SRC:.*]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[8,4],f32>
// CHECK:         %[[INDICES:.*]] = stablehlo.constant dense<{{\[\[}}0], [2], [4]]> : tensor<3x1xi64>
// CHECK:         "stablehlo.scatter"(%{{.*}}, %[[INDICES]], %{{.*}})
func.func @torch.aten.slice_scatter$static_step2(%arg0: !torch.vtensor<[8,4],f32>, %arg1: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[8,4],f32> {
  %int0 = torch.constant.int 0
  %int0_start = torch.constant.int 0
  %int6 = torch.constant.int 6
  %int2 = torch.constant.int 2
  %0 = torch.aten.slice_scatter %arg0, %arg1, %int0, %int0_start, %int6, %int2 : !torch.vtensor<[8,4],f32>, !torch.vtensor<[3,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[8,4],f32>
  return %0 : !torch.vtensor<[8,4],f32>
}

// -----

// Dynamic input, static src, dynamic start, step=1
// CHECK-LABEL: func.func @torch.aten.slice_scatter$dynamic_input_dynamic_start(
// CHECK-SAME:      %[[INPUT:.*]]: !torch.vtensor<[?,4],f32>, %[[SRC:.*]]: !torch.vtensor<[1,4],f32>, %[[START:.*]]: !torch.int) -> !torch.vtensor<[?,4],f32>
// CHECK:         stablehlo.iota dim = 0 : tensor<1xi64>
// CHECK:         chlo.broadcast_mul
// CHECK:         chlo.broadcast_add
// CHECK:         stablehlo.reshape
// CHECK:         "stablehlo.scatter"
func.func @torch.aten.slice_scatter$dynamic_input_dynamic_start(%arg0: !torch.vtensor<[?,4],f32>, %arg1: !torch.vtensor<[1,4],f32>, %arg2: !torch.int) -> !torch.vtensor<[?,4],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int9223372036854775807 = torch.constant.int 9223372036854775807
  %0 = torch.aten.slice_scatter %arg0, %arg1, %int0, %arg2, %int9223372036854775807, %int1 : !torch.vtensor<[?,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,4],f32>
  return %0 : !torch.vtensor<[?,4],f32>
}

// -----

// Dynamic input, dynamic src, dynamic start, step=1
// CHECK-LABEL: func.func @torch.aten.slice_scatter$all_dynamic_step1(
// CHECK-SAME:      %[[INPUT:.*]]: !torch.vtensor<[?,?],f32>, %[[SRC:.*]]: !torch.vtensor<[?,?],f32>, %[[START:.*]]: !torch.int) -> !torch.vtensor<[?,?],f32>
// CHECK:         stablehlo.dynamic_iota
// CHECK:         chlo.broadcast_mul
// CHECK:         chlo.broadcast_add
// CHECK:         stablehlo.dynamic_reshape
// CHECK:         "stablehlo.scatter"
func.func @torch.aten.slice_scatter$all_dynamic_step1(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>, %arg2: !torch.int) -> !torch.vtensor<[?,?],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int9223372036854775807 = torch.constant.int 9223372036854775807
  %0 = torch.aten.slice_scatter %arg0, %arg1, %int0, %arg2, %int9223372036854775807, %int1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// Dynamic input, dynamic src, dynamic start, step=2
// CHECK-LABEL: func.func @torch.aten.slice_scatter$all_dynamic_step2(
// CHECK-SAME:      %[[INPUT:.*]]: !torch.vtensor<[?,?],f32>, %[[SRC:.*]]: !torch.vtensor<[?,?],f32>, %[[START:.*]]: !torch.int) -> !torch.vtensor<[?,?],f32>
// CHECK:         stablehlo.dynamic_iota
// CHECK:         chlo.broadcast_mul
// CHECK:         chlo.broadcast_add
// CHECK:         stablehlo.dynamic_reshape
// CHECK:         "stablehlo.scatter"
func.func @torch.aten.slice_scatter$all_dynamic_step2(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>, %arg2: !torch.int) -> !torch.vtensor<[?,?],f32> {
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %int9223372036854775807 = torch.constant.int 9223372036854775807
  %0 = torch.aten.slice_scatter %arg0, %arg1, %int0, %arg2, %int9223372036854775807, %int2 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// Static input, static src, dynamic start, dim=1
// CHECK-LABEL: func.func @torch.aten.slice_scatter$dynamic_start_dim1(
// CHECK-SAME:      %[[INPUT:.*]]: !torch.vtensor<[4,8],f32>, %[[SRC:.*]]: !torch.vtensor<[4,3],f32>, %[[START:.*]]: !torch.int) -> !torch.vtensor<[4,8],f32>
// CHECK:         stablehlo.iota dim = 0 : tensor<3xi64>
// CHECK:         chlo.broadcast_mul
// CHECK:         chlo.broadcast_add
// CHECK:         stablehlo.reshape
// CHECK:         "stablehlo.scatter"
func.func @torch.aten.slice_scatter$dynamic_start_dim1(%arg0: !torch.vtensor<[4,8],f32>, %arg1: !torch.vtensor<[4,3],f32>, %arg2: !torch.int) -> !torch.vtensor<[4,8],f32> {
  %int1 = torch.constant.int 1
  %int9223372036854775807 = torch.constant.int 9223372036854775807
  %0 = torch.aten.slice_scatter %arg0, %arg1, %int1, %arg2, %int9223372036854775807, %int1 : !torch.vtensor<[4,8],f32>, !torch.vtensor<[4,3],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,8],f32>
  return %0 : !torch.vtensor<[4,8],f32>
}

// -----

// Dynamic start and step, dynamic src dim
// CHECK-LABEL: func.func @torch.aten.slice_scatter$dynamic_start_and_step(
// CHECK-SAME:      %[[INPUT:.*]]: !torch.vtensor<[8,4],f32>, %[[SRC:.*]]: !torch.vtensor<[?,4],f32>, %[[START:.*]]: !torch.int, %[[STEP:.*]]: !torch.int) -> !torch.vtensor<[8,4],f32>
// CHECK:         %[[STEP_I64:.*]] = torch_c.to_i64 %[[STEP]]
// CHECK:         %[[START_I64:.*]] = torch_c.to_i64 %[[START]]
// CHECK:         %[[SRC_T:.*]] = torch_c.to_builtin_tensor %[[SRC]] : !torch.vtensor<[?,4],f32> -> tensor<?x4xf32>
// CHECK:         %[[INPUT_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[8,4],f32> -> tensor<8x4xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[SRC_T]], %[[C0]] : tensor<?x4xf32>
// CHECK:         %[[N:.*]] = arith.index_cast %[[DIM0]] : index to i64
// CHECK:         %[[SHAPE:.*]] = tensor.from_elements %[[N]] : tensor<1xi64>
// CHECK:         %[[IOTA:.*]] = stablehlo.dynamic_iota %[[SHAPE]], dim = 0 : (tensor<1xi64>) -> tensor<?xi64>
// CHECK:         %[[STEP_SCALAR:.*]] = tensor.from_elements %[[STEP_I64]] : tensor<i64>
// CHECK:         %[[SCALED:.*]] = chlo.broadcast_multiply %[[IOTA]], %[[STEP_SCALAR]] : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
// CHECK:         %[[IS_NEG:.*]] = arith.cmpi slt, %[[START_I64]], %{{.*}} : i64
// CHECK:         %[[INPUT_DIM:.*]] = tensor.dim %[[INPUT_T]], %{{.*}} : tensor<8x4xf32>
// CHECK:         %[[DIM_I64:.*]] = arith.index_cast %[[INPUT_DIM]] : index to i64
// CHECK:         %[[ADD:.*]] = arith.addi %[[START_I64]], %[[DIM_I64]] : i64
// CHECK:         %[[START_NORM:.*]] = arith.select %[[IS_NEG]], %[[ADD]], %[[START_I64]] : i64
// CHECK:         %[[START_SCALAR:.*]] = tensor.from_elements %[[START_NORM]] : tensor<i64>
// CHECK:         %[[INDICES:.*]] = chlo.broadcast_add %[[SCALED]], %[[START_SCALAR]] : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
// CHECK:         %[[C0_2:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0_2:.*]] = tensor.dim %[[INDICES]], %[[C0_2]] : tensor<?xi64>
// CHECK:         %[[N2:.*]] = arith.index_cast %[[DIM0_2]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : i64
// CHECK:         %[[RESHAPE_SHAPE:.*]] = tensor.from_elements %[[N2]], %[[C1]] : tensor<2xi64>
// CHECK:         %[[SCATTER_IDX:.*]] = stablehlo.dynamic_reshape %[[INDICES]], %[[RESHAPE_SHAPE]] : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
// CHECK:         %[[RESULT:.*]] = "stablehlo.scatter"(%[[INPUT_T]], %[[SCATTER_IDX]], %[[SRC_T]])
// CHECK:         %[[OUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<8x4xf32> -> !torch.vtensor<[8,4],f32>
// CHECK:         return %[[OUT]] : !torch.vtensor<[8,4],f32>
func.func @torch.aten.slice_scatter$dynamic_start_and_step(%arg0: !torch.vtensor<[8,4],f32>, %arg1: !torch.vtensor<[?,4],f32>, %arg2: !torch.int, %arg3: !torch.int) -> !torch.vtensor<[8,4],f32> {
  %int0 = torch.constant.int 0
  %int9223372036854775807 = torch.constant.int 9223372036854775807
  %0 = torch.aten.slice_scatter %arg0, %arg1, %int0, %arg2, %int9223372036854775807, %arg3 : !torch.vtensor<[8,4],f32>, !torch.vtensor<[?,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[8,4],f32>
  return %0 : !torch.vtensor<[8,4],f32>
}

// -----

// Constant start/end/step but dynamic input dim — must use runtime iota path
// (without this guard, end=INT64_MAX would cause an unbounded loop)
// CHECK-LABEL: func.func @torch.aten.slice_scatter$const_params_dynamic_dim(
// CHECK-SAME:      %[[INPUT:.*]]: !torch.vtensor<[?,4],f32>, %[[SRC:.*]]: !torch.vtensor<[?,4],f32>) -> !torch.vtensor<[?,4],f32>
// CHECK:         %[[SRC_T:.*]] = torch_c.to_builtin_tensor %[[SRC]] : !torch.vtensor<[?,4],f32> -> tensor<?x4xf32>
// CHECK:         %[[INPUT_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[?,4],f32> -> tensor<?x4xf32>
// CHECK:         %[[DIM:.*]] = tensor.dim %[[SRC_T]], %{{.*}} : tensor<?x4xf32>
// CHECK:         %[[N:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:         %[[SHAPE:.*]] = tensor.from_elements %[[N]] : tensor<1xi64>
// CHECK:         %[[IOTA:.*]] = stablehlo.dynamic_iota %[[SHAPE]], dim = 0 : (tensor<1xi64>) -> tensor<?xi64>
// CHECK:         %[[STEP:.*]] = tensor.from_elements %{{.*}} : tensor<i64>
// CHECK:         %[[SCALED:.*]] = chlo.broadcast_multiply %[[IOTA]], %[[STEP]] : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
// CHECK:         %[[IS_NEG:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : i64
// CHECK:         %[[INPUT_DIM:.*]] = tensor.dim %[[INPUT_T]], %{{.*}} : tensor<?x4xf32>
// CHECK:         %[[DIM_I64:.*]] = arith.index_cast %[[INPUT_DIM]] : index to i64
// CHECK:         %[[START_NORM:.*]] = arith.select %[[IS_NEG]], %{{.*}}, %{{.*}} : i64
// CHECK:         %[[START_T:.*]] = tensor.from_elements %[[START_NORM]] : tensor<i64>
// CHECK:         %[[INDICES:.*]] = chlo.broadcast_add %[[SCALED]], %[[START_T]] : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
// CHECK:         %[[SCATTER_IDX:.*]] = stablehlo.dynamic_reshape %[[INDICES]], %{{.*}} : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
// CHECK:         %[[RESULT:.*]] = "stablehlo.scatter"(%[[INPUT_T]], %[[SCATTER_IDX]], %[[SRC_T]])
// CHECK:         %[[OUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<?x4xf32> -> !torch.vtensor<[?,4],f32>
// CHECK:         return %[[OUT]] : !torch.vtensor<[?,4],f32>
func.func @torch.aten.slice_scatter$const_params_dynamic_dim(%arg0: !torch.vtensor<[?,4],f32>, %arg1: !torch.vtensor<[?,4],f32>) -> !torch.vtensor<[?,4],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int9223372036854775807 = torch.constant.int 9223372036854775807
  %0 = torch.aten.slice_scatter %arg0, %arg1, %int0, %int0, %int9223372036854775807, %int1 : !torch.vtensor<[?,4],f32>, !torch.vtensor<[?,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,4],f32>
  return %0 : !torch.vtensor<[?,4],f32>
}

// -----

// Negative start with dynamic input dim — start normalization applied at runtime
// CHECK-LABEL: func.func @torch.aten.slice_scatter$negative_start_dynamic_dim(
// CHECK-SAME:      %[[INPUT:.*]]: !torch.vtensor<[?,4],f32>, %[[SRC:.*]]: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[?,4],f32>
// CHECK:         %[[SRC_T:.*]] = torch_c.to_builtin_tensor %[[SRC]] : !torch.vtensor<[2,4],f32> -> tensor<2x4xf32>
// CHECK:         %[[INPUT_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[?,4],f32> -> tensor<?x4xf32>
// CHECK:         %[[IOTA:.*]] = stablehlo.iota dim = 0 : tensor<2xi64>
// CHECK:         %[[STEP:.*]] = tensor.from_elements %{{.*}} : tensor<i64>
// CHECK:         %[[SCALED:.*]] = chlo.broadcast_multiply %[[IOTA]], %[[STEP]] : (tensor<2xi64>, tensor<i64>) -> tensor<2xi64>
// CHECK:         %[[IS_NEG:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : i64
// CHECK:         %[[INPUT_DIM:.*]] = tensor.dim %[[INPUT_T]], %{{.*}} : tensor<?x4xf32>
// CHECK:         %[[DIM_I64:.*]] = arith.index_cast %[[INPUT_DIM]] : index to i64
// CHECK:         %[[ADD:.*]] = arith.addi %{{.*}}, %[[DIM_I64]] : i64
// CHECK:         %[[START_NORM:.*]] = arith.select %[[IS_NEG]], %[[ADD]], %{{.*}} : i64
// CHECK:         %[[START_T:.*]] = tensor.from_elements %[[START_NORM]] : tensor<i64>
// CHECK:         %[[INDICES:.*]] = chlo.broadcast_add %[[SCALED]], %[[START_T]] : (tensor<2xi64>, tensor<i64>) -> tensor<2xi64>
// CHECK:         %[[SCATTER_IDX:.*]] = stablehlo.reshape %[[INDICES]] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK:         %[[RESULT:.*]] = "stablehlo.scatter"(%[[INPUT_T]], %[[SCATTER_IDX]], %[[SRC_T]])
// CHECK:         %[[OUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<?x4xf32> -> !torch.vtensor<[?,4],f32>
// CHECK:         return %[[OUT]] : !torch.vtensor<[?,4],f32>
func.func @torch.aten.slice_scatter$negative_start_dynamic_dim(%arg0: !torch.vtensor<[?,4],f32>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[?,4],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int9223372036854775807 = torch.constant.int 9223372036854775807
  %int_neg2 = torch.constant.int -2
  %0 = torch.aten.slice_scatter %arg0, %arg1, %int0, %int_neg2, %int9223372036854775807, %int1 : !torch.vtensor<[?,4],f32>, !torch.vtensor<[2,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,4],f32>
  return %0 : !torch.vtensor<[?,4],f32>
}

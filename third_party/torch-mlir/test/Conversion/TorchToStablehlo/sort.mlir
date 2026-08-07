// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL:   func.func @torch.aten.sort(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[4,5],f32>) -> (!torch.vtensor<[4,5],f32>, !torch.vtensor<[4,5],si64>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,5],f32> -> tensor<4x5xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_4:.*]] = stablehlo.iota dim = 1 : tensor<4x5xi64>
// CHECK:           %[[VAL_5:.*]]:2 = "stablehlo.sort"(%[[VAL_1]], %[[VAL_4]]) <{dimension = 1 : i64, is_stable = false}> ({
// CHECK:           ^bb0(%[[VAL_6:.*]]: tensor<f32>, %[[VAL_7:.*]]: tensor<f32>, %[[VAL_8:.*]]: tensor<i64>, %[[VAL_9:.*]]: tensor<i64>):
// CHECK:             %[[VAL_10:.*]] = stablehlo.compare  GT, %[[VAL_6]], %[[VAL_7]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             stablehlo.return %[[VAL_10]] : tensor<i1>
// CHECK:           }) : (tensor<4x5xf32>, tensor<4x5xi64>) -> (tensor<4x5xf32>, tensor<4x5xi64>)
// CHECK:           %[[VAL_11:.*]] = torch_c.from_builtin_tensor %[[VAL_12:.*]]#1 : tensor<4x5xi64> -> !torch.vtensor<[4,5],si64>
// CHECK:           %[[VAL_13:.*]] = torch_c.from_builtin_tensor %[[VAL_12]]#0 : tensor<4x5xf32> -> !torch.vtensor<[4,5],f32>
// CHECK:           return %[[VAL_13]], %[[VAL_11]] : !torch.vtensor<[4,5],f32>, !torch.vtensor<[4,5],si64>
// CHECK:         }
func.func @torch.aten.sort(%arg0: !torch.vtensor<[4,5],f32>) -> (!torch.vtensor<[4,5],f32>, !torch.vtensor<[4,5],si64>) {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %values, %indices = torch.aten.sort %arg0, %int1, %true : !torch.vtensor<[4,5],f32>, !torch.int, !torch.bool -> !torch.vtensor<[4,5],f32>, !torch.vtensor<[4,5],si64>
    return %values, %indices : !torch.vtensor<[4,5],f32>, !torch.vtensor<[4,5],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.sort$dynamic(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[?,5],f32>) -> (!torch.vtensor<[?,5],f32>, !torch.vtensor<[?,5],si64>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,5],f32> -> tensor<?x5xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool true
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIM0:.*]] = tensor.dim %[[VAL_1]], %[[C0]] : tensor<?x5xf32>
// CHECK:           %[[DIM0_I64:.*]] = arith.index_cast %[[DIM0]] : index to i64
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIM1:.*]] = tensor.dim %[[VAL_1]], %[[C1]] : tensor<?x5xf32>
// CHECK:           %[[DIM1_I64:.*]] = arith.index_cast %[[DIM1]] : index to i64
// CHECK:           %[[SHAPE:.*]] = tensor.from_elements %[[DIM0_I64]], %[[DIM1_I64]] : tensor<2xi64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.dynamic_iota %[[SHAPE]], dim = 1 : (tensor<2xi64>) -> tensor<?x5xi64>
// CHECK:           %[[VAL_5:.*]]:2 = "stablehlo.sort"(%[[VAL_1]], %[[VAL_4]]) <{dimension = 1 : i64, is_stable = false}> ({
// CHECK:           ^bb0(%[[VAL_6:.*]]: tensor<f32>, %[[VAL_7:.*]]: tensor<f32>, %[[VAL_8:.*]]: tensor<i64>, %[[VAL_9:.*]]: tensor<i64>):
// CHECK:             %[[VAL_10:.*]] = stablehlo.compare  GT, %[[VAL_6]], %[[VAL_7]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             stablehlo.return %[[VAL_10]] : tensor<i1>
// CHECK:           }) : (tensor<?x5xf32>, tensor<?x5xi64>) -> (tensor<?x5xf32>, tensor<?x5xi64>)
// CHECK:           %[[VAL_11:.*]] = torch_c.from_builtin_tensor %[[VAL_12:.*]]#1 : tensor<?x5xi64> -> !torch.vtensor<[?,5],si64>
// CHECK:           %[[VAL_13:.*]] = torch_c.from_builtin_tensor %[[VAL_12]]#0 : tensor<?x5xf32> -> !torch.vtensor<[?,5],f32>
// CHECK:           return %[[VAL_13]], %[[VAL_11]] : !torch.vtensor<[?,5],f32>, !torch.vtensor<[?,5],si64>
func.func @torch.aten.sort$dynamic(%arg0: !torch.vtensor<[?,5],f32>) -> (!torch.vtensor<[?,5],f32>, !torch.vtensor<[?,5],si64>) {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %values, %indices = torch.aten.sort %arg0, %int1, %true : !torch.vtensor<[?,5],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?,5],f32>, !torch.vtensor<[?,5],si64>
    return %values, %indices : !torch.vtensor<[?,5],f32>, !torch.vtensor<[?,5],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.sort$dynamic_ascending(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],si64>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool false
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIM0:.*]] = tensor.dim %[[VAL_1]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[DIM0_I64:.*]] = arith.index_cast %[[DIM0]] : index to i64
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIM1:.*]] = tensor.dim %[[VAL_1]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[DIM1_I64:.*]] = arith.index_cast %[[DIM1]] : index to i64
// CHECK:           %[[SHAPE:.*]] = tensor.from_elements %[[DIM0_I64]], %[[DIM1_I64]] : tensor<2xi64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.dynamic_iota %[[SHAPE]], dim = 0 : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:           %[[VAL_5:.*]]:2 = "stablehlo.sort"(%[[VAL_1]], %[[VAL_4]]) <{dimension = 0 : i64, is_stable = false}> ({
// CHECK:           ^bb0(%[[VAL_6:.*]]: tensor<f32>, %[[VAL_7:.*]]: tensor<f32>, %[[VAL_8:.*]]: tensor<i64>, %[[VAL_9:.*]]: tensor<i64>):
// CHECK:             %[[VAL_10:.*]] = stablehlo.compare  LT, %[[VAL_6]], %[[VAL_7]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             stablehlo.return %[[VAL_10]] : tensor<i1>
// CHECK:           }) : (tensor<?x?xf32>, tensor<?x?xi64>) -> (tensor<?x?xf32>, tensor<?x?xi64>)
// CHECK:           %[[VAL_11:.*]] = torch_c.from_builtin_tensor %[[VAL_12:.*]]#1 : tensor<?x?xi64> -> !torch.vtensor<[?,?],si64>
// CHECK:           %[[VAL_13:.*]] = torch_c.from_builtin_tensor %[[VAL_12]]#0 : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_13]], %[[VAL_11]] : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],si64>
func.func @torch.aten.sort$dynamic_ascending(%arg0: !torch.vtensor<[?,?],f32>) -> (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],si64>) {
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %values, %indices = torch.aten.sort %arg0, %int0, %false : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],si64>
    return %values, %indices : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],si64>
}

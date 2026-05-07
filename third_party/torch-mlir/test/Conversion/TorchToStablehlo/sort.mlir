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

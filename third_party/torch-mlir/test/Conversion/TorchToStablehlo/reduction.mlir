// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s

// -----

func.func @torch.aten.prod.intdim(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  // CHECK-LABEL: @torch.aten.prod.intdim(
  // CHECK: %[[VAL_0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %none = torch.constant.none
  // CHECK: %[[VAL_1:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %[[VAL_2:.*]] = stablehlo.reduce(%[[VAL_0]] init: %[[VAL_1]]) applies stablehlo.multiply across dimensions = [1] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?xf32>
  %0 = torch.aten.prod.dim_int %arg0, %int1, %false, %none : !torch.vtensor<[?,?,?,?],f32>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?],f32>
  // CHECK: %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
  // CHECK: return %[[VAL_3]] : !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// -----

func.func @torch.aten.prod.intdim_negative_dim(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  // CHECK-LABEL: @torch.aten.prod.intdim_negative_dim(
  // CHECK: %[[VAL_0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
  %int-1 = torch.constant.int -1
  %false = torch.constant.bool false
  %none = torch.constant.none
  // CHECK: %[[VAL_1:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %[[VAL_2:.*]] = stablehlo.reduce(%[[VAL_0]] init: %[[VAL_1]]) applies stablehlo.multiply across dimensions = [3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?xf32>
  %0 = torch.aten.prod.dim_int %arg0, %int-1, %false, %none : !torch.vtensor<[?,?,?,?],f32>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?],f32>
  // CHECK: %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
  // CHECK: return %[[VAL_3]] : !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.max.dim$rank0(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[],si32>) -> (!torch.vtensor<[],si32>, !torch.vtensor<[],si64>) {
// CHECK:         %[[IDX:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:         %[[IDX_TORCH:.*]] = torch_c.from_builtin_tensor %[[IDX]] : tensor<i64> -> !torch.vtensor<[],si64>
// CHECK:         return %[[ARG0]], %[[IDX_TORCH]] : !torch.vtensor<[],si32>, !torch.vtensor<[],si64>
func.func @torch.aten.max.dim$rank0(%arg0: !torch.vtensor<[],si32>) -> (!torch.vtensor<[],si32>, !torch.vtensor<[],si64>) {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %values, %indices = torch.aten.max.dim %arg0, %int0, %false : !torch.vtensor<[],si32>, !torch.int, !torch.bool -> !torch.vtensor<[],si32>, !torch.vtensor<[],si64>
  return %values, %indices : !torch.vtensor<[],si32>, !torch.vtensor<[],si64>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.min.dim$rank0(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[],f32>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[],si64>) {
// CHECK:         %[[IDX:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:         %[[IDX_TORCH:.*]] = torch_c.from_builtin_tensor %[[IDX]] : tensor<i64> -> !torch.vtensor<[],si64>
// CHECK:         return %[[ARG0]], %[[IDX_TORCH]] : !torch.vtensor<[],f32>, !torch.vtensor<[],si64>
func.func @torch.aten.min.dim$rank0(%arg0: !torch.vtensor<[],f32>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[],si64>) {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %values, %indices = torch.aten.min.dim %arg0, %int0, %false : !torch.vtensor<[],f32>, !torch.int, !torch.bool -> !torch.vtensor<[],f32>, !torch.vtensor<[],si64>
  return %values, %indices : !torch.vtensor<[],f32>, !torch.vtensor<[],si64>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.amax$rank0(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
// CHECK:         return %[[ARG0]] : !torch.vtensor<[],f32>
func.func @torch.aten.amax$rank0(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
  %int0 = torch.constant.int 0
  %true = torch.constant.bool true
  %dims = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %0 = torch.aten.amax %arg0, %dims, %true : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.amin$rank0(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
// CHECK:         return %[[ARG0]] : !torch.vtensor<[],f32>
func.func @torch.aten.amin$rank0(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
  %int0 = torch.constant.int 0
  %true = torch.constant.bool true
  %dims = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %0 = torch.aten.amin %arg0, %dims, %true : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

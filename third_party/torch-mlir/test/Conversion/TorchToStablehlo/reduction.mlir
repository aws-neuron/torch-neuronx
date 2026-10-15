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

// -----

// CHECK-LABEL:  func.func @torch.aten.max.dim$unsigned(
// CHECK:         stablehlo.compare  GE, %{{.*}}, %{{.*}},  UNSIGNED : (tensor<ui8>, tensor<ui8>) -> tensor<i1>
// CHECK:         stablehlo.compare  EQ, %{{.*}}, %{{.*}},  UNSIGNED : (tensor<ui8>, tensor<ui8>) -> tensor<i1>
func.func @torch.aten.max.dim$unsigned(%arg0: !torch.vtensor<[4,8],ui8>) -> (!torch.vtensor<[4],ui8>, !torch.vtensor<[4],si64>) {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %values, %indices = torch.aten.max.dim %arg0, %int1, %false : !torch.vtensor<[4,8],ui8>, !torch.int, !torch.bool -> !torch.vtensor<[4],ui8>, !torch.vtensor<[4],si64>
  return %values, %indices : !torch.vtensor<[4],ui8>, !torch.vtensor<[4],si64>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.min.dim$unsigned(
// CHECK:         stablehlo.compare  LE, %{{.*}}, %{{.*}},  UNSIGNED : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK:         stablehlo.compare  EQ, %{{.*}}, %{{.*}},  UNSIGNED : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
func.func @torch.aten.min.dim$unsigned(%arg0: !torch.vtensor<[4,8],ui32>) -> (!torch.vtensor<[4],ui32>, !torch.vtensor<[4],si64>) {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %values, %indices = torch.aten.min.dim %arg0, %int1, %false : !torch.vtensor<[4,8],ui32>, !torch.int, !torch.bool -> !torch.vtensor<[4],ui32>, !torch.vtensor<[4],si64>
  return %values, %indices : !torch.vtensor<[4],ui32>, !torch.vtensor<[4],si64>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.max.dim$signed(
// CHECK:         stablehlo.compare  GE, %{{.*}}, %{{.*}},  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK:         stablehlo.compare  EQ, %{{.*}}, %{{.*}},  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
func.func @torch.aten.max.dim$signed(%arg0: !torch.vtensor<[4,8],si32>) -> (!torch.vtensor<[4],si32>, !torch.vtensor<[4],si64>) {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %values, %indices = torch.aten.max.dim %arg0, %int1, %false : !torch.vtensor<[4,8],si32>, !torch.int, !torch.bool -> !torch.vtensor<[4],si32>, !torch.vtensor<[4],si64>
  return %values, %indices : !torch.vtensor<[4],si32>, !torch.vtensor<[4],si64>
}

// -----

// The identity of a max reduce is the LOWEST value of the element type. For ui8
// that value is 0, not the signed minimum (which stores as 128).
// CHECK-LABEL:  func.func @torch.aten.amax$unsigned_init(
// CHECK:         %[[INIT:.*]] = stablehlo.constant dense<0> : tensor<ui8>
// CHECK:         stablehlo.reduce(%{{.*}} init: %[[INIT]]) applies stablehlo.maximum
func.func @torch.aten.amax$unsigned_init(%arg0: !torch.vtensor<[4,8],ui8>) -> !torch.vtensor<[4],ui8> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %dims = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %0 = torch.aten.amax %arg0, %dims, %false : !torch.vtensor<[4,8],ui8>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4],ui8>
  return %0 : !torch.vtensor<[4],ui8>
}

// -----

// The identity of a min reduce is the HIGHEST value of the element type. For ui8
// that value is 255, not the signed maximum 127.
// CHECK-LABEL:  func.func @torch.aten.amin$unsigned_init(
// CHECK:         %[[INIT:.*]] = stablehlo.constant dense<255> : tensor<ui8>
// CHECK:         stablehlo.reduce(%{{.*}} init: %[[INIT]]) applies stablehlo.minimum
func.func @torch.aten.amin$unsigned_init(%arg0: !torch.vtensor<[4,8],ui8>) -> !torch.vtensor<[4],ui8> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %dims = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %0 = torch.aten.amin %arg0, %dims, %false : !torch.vtensor<[4,8],ui8>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4],ui8>
  return %0 : !torch.vtensor<[4],ui8>
}

// -----

// i1 holds 0 and 1, so the identity of a min reduce over bool is true.
// CHECK-LABEL:  func.func @torch.aten.amin$bool_init(
// CHECK:         %[[INIT:.*]] = stablehlo.constant dense<true> : tensor<i1>
// CHECK:         stablehlo.reduce(%{{.*}} init: %[[INIT]]) applies stablehlo.minimum
func.func @torch.aten.amin$bool_init(%arg0: !torch.vtensor<[4,8],i1>) -> !torch.vtensor<[4],i1> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %dims = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %0 = torch.aten.amin %arg0, %dims, %false : !torch.vtensor<[4,8],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4],i1>
  return %0 : !torch.vtensor<[4],i1>
}

// -----

// The identity of a max reduce over bool is false.
// CHECK-LABEL:  func.func @torch.aten.amax$bool_init(
// CHECK:         %[[INIT:.*]] = stablehlo.constant dense<false> : tensor<i1>
// CHECK:         stablehlo.reduce(%{{.*}} init: %[[INIT]]) applies stablehlo.maximum
func.func @torch.aten.amax$bool_init(%arg0: !torch.vtensor<[4,8],i1>) -> !torch.vtensor<[4],i1> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %dims = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %0 = torch.aten.amax %arg0, %dims, %false : !torch.vtensor<[4,8],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4],i1>
  return %0 : !torch.vtensor<[4],i1>
}

// -----

// A signed type keeps the signed extremes.
// CHECK-LABEL:  func.func @torch.aten.amax$signed_init(
// CHECK:         %[[INIT:.*]] = stablehlo.constant dense<-2147483648> : tensor<i32>
// CHECK:         stablehlo.reduce(%{{.*}} init: %[[INIT]]) applies stablehlo.maximum
func.func @torch.aten.amax$signed_init(%arg0: !torch.vtensor<[4,8],si32>) -> !torch.vtensor<[4],si32> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %dims = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %0 = torch.aten.amax %arg0, %dims, %false : !torch.vtensor<[4,8],si32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4],si32>
  return %0 : !torch.vtensor<[4],si32>
}

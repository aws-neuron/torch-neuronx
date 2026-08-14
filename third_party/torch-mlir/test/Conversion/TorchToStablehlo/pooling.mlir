// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL:   func.func @torch.aten.max_pool2d(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_5:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_6:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           %[[VAL_11:.*]] = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_10]]) <{padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 2, 1>, window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 1, 1>}> ({
// CHECK:           ^bb0(%[[VAL_12:.*]]: tensor<f32>, %[[VAL_13:.*]]: tensor<f32>):
// CHECK:             %[[VAL_14:.*]] = stablehlo.maximum %[[VAL_12]], %[[VAL_13]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_14]] : tensor<f32>
// CHECK:           }) : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_15:.*]] = torch_c.from_builtin_tensor %[[VAL_16:.*]] : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[VAL_15]] : !torch.vtensor<[?,?,?,?],f32>
// CHECK:         }
func.func @torch.aten.max_pool2d(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.aten.max_pool2d %arg0, %0, %1, %2, %3, %false : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  return %4 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.max_pool2d$padding(
// CHECK-SAME:                                             %[[VAL_0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_5:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_6:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           %[[VAL_9:.*]] = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_8]]) <{padding = dense<{{\[\[}}0, 0], [0, 0], [2, 2], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 2, 1>, window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 1, 1>}> ({
// CHECK:           ^bb0(%[[VAL_10:.*]]: tensor<f32>, %[[VAL_11:.*]]: tensor<f32>):
// CHECK:             %[[VAL_12:.*]] = stablehlo.maximum %[[VAL_10]], %[[VAL_11]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_12]] : tensor<f32>
// CHECK:           }) : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_13:.*]] = torch_c.from_builtin_tensor %[[VAL_14:.*]] : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[VAL_13]] : !torch.vtensor<[?,?,?,?],f32>
// CHECK:         }
func.func @torch.aten.max_pool2d$padding(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.max_pool2d %arg0, %0, %1, %2, %2, %false : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  return %3 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.max_pool2d$ceiloff(
// CHECK-SAME:                                             %[[VAL_0:.*]]: !torch.vtensor<[1,256,56,56],f32>) -> !torch.vtensor<[1,256,27,27],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,256,56,56],f32> -> tensor<1x256x56x56xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           %[[VAL_12:.*]] = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_11]]) <{padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
// CHECK:           ^bb0(%[[VAL_13:.*]]: tensor<f32>, %[[VAL_14:.*]]: tensor<f32>):
// CHECK:             %[[VAL_15:.*]] = stablehlo.maximum %[[VAL_13]], %[[VAL_14]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_15]] : tensor<f32>
// CHECK:           }) : (tensor<1x256x56x56xf32>, tensor<f32>) -> tensor<1x256x27x27xf32>
// CHECK:           %[[VAL_16:.*]] = torch_c.from_builtin_tensor %[[VAL_17:.*]] : tensor<1x256x27x27xf32> -> !torch.vtensor<[1,256,27,27],f32>
// CHECK:           return %[[VAL_16]] : !torch.vtensor<[1,256,27,27],f32>
// CHECK:         }
func.func @torch.aten.max_pool2d$ceiloff(%arg0: !torch.vtensor<[1,256,56,56],f32>) -> !torch.vtensor<[1,256,27,27],f32> {
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %0 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.aten.max_pool2d %arg0, %0, %1, %2, %3, %false : !torch.vtensor<[1,256,56,56],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,256,27,27],f32>
  return %4 : !torch.vtensor<[1,256,27,27],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.max_pool2d$ceilon(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !torch.vtensor<[1,256,56,56],f32>) -> !torch.vtensor<[1,256,28,28],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,256,56,56],f32> -> tensor<1x256x56x56xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           %[[VAL_12:.*]] = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_11]]) <{padding = dense<{{\[\[}}0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
// CHECK:           ^bb0(%[[VAL_13:.*]]: tensor<f32>, %[[VAL_14:.*]]: tensor<f32>):
// CHECK:             %[[VAL_15:.*]] = stablehlo.maximum %[[VAL_13]], %[[VAL_14]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_15]] : tensor<f32>
// CHECK:           }) : (tensor<1x256x56x56xf32>, tensor<f32>) -> tensor<1x256x28x28xf32>
// CHECK:           %[[VAL_16:.*]] = torch_c.from_builtin_tensor %[[VAL_17:.*]] : tensor<1x256x28x28xf32> -> !torch.vtensor<[1,256,28,28],f32>
// CHECK:           return %[[VAL_16]] : !torch.vtensor<[1,256,28,28],f32>
// CHECK:         }
func.func @torch.aten.max_pool2d$ceilon(%arg0: !torch.vtensor<[1,256,56,56],f32>) -> !torch.vtensor<[1,256,28,28],f32> {
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %true = torch.constant.bool true
  %int0 = torch.constant.int 0
  %0 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.aten.max_pool2d %arg0, %0, %1, %2, %3, %true : !torch.vtensor<[1,256,56,56],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,256,28,28],f32>
  return %4 : !torch.vtensor<[1,256,28,28],f32>
}


// -----

// CHECK-LABEL:   func.func @torch.aten.max_pool2d_with_indices(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: !torch.vtensor<[?,?,?],f32>) -> (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],si64>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_5]], %[[VAL_5]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_12:.*]] = tensor.dim %[[VAL_1]], %[[VAL_11]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_13:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_14:.*]] = tensor.dim %[[VAL_1]], %[[VAL_13]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_15:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_16:.*]] = tensor.dim %[[VAL_1]], %[[VAL_15]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_17:.*]] = tensor.from_elements %[[VAL_12]], %[[VAL_14]], %[[VAL_16]] : tensor<3xindex>
// CHECK:           %[[VAL_18:.*]] = arith.muli %[[VAL_16]], %[[VAL_14]] : index
// CHECK:           %[[VAL_19:.*]] = tensor.from_elements %[[VAL_12]], %[[VAL_18]] : tensor<2xindex>
// CHECK:           %[[VAL_20:.*]] = stablehlo.dynamic_iota %[[VAL_19]], dim = 1 : (tensor<2xindex>) -> tensor<?x?xi64>
// CHECK:           %[[VAL_21:.*]] = stablehlo.dynamic_reshape %[[VAL_20]], %[[VAL_17]] : (tensor<?x?xi64>, tensor<3xindex>) -> tensor<?x?x?xi64>
// CHECK:           %[[VAL_22:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           %[[VAL_23:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_24:.*]]:2 = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]]) <{padding = dense<0> : tensor<3x2xi64>, window_dimensions = array<i64: 1, 3, 3>, window_strides = array<i64: 1, 2, 2>}> ({
// CHECK:           ^bb0(%[[VAL_25:.*]]: tensor<f32>, %[[VAL_26:.*]]: tensor<i64>, %[[VAL_27:.*]]: tensor<f32>, %[[VAL_28:.*]]: tensor<i64>):
// CHECK:             %[[VAL_29:.*]] = stablehlo.compare  GE, %[[VAL_25]], %[[VAL_27]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             %[[VAL_30:.*]] = stablehlo.select %[[VAL_29]], %[[VAL_25]], %[[VAL_27]] : tensor<i1>, tensor<f32>
// CHECK:             %[[VAL_31:.*]] = stablehlo.compare  EQ, %[[VAL_25]], %[[VAL_27]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             %[[VAL_32:.*]] = stablehlo.minimum %[[VAL_26]], %[[VAL_28]] : tensor<i64>
// CHECK:             %[[VAL_33:.*]] = stablehlo.select %[[VAL_29]], %[[VAL_26]], %[[VAL_28]] : tensor<i1>, tensor<i64>
// CHECK:             %[[VAL_34:.*]] = stablehlo.select %[[VAL_31]], %[[VAL_32]], %[[VAL_33]] : tensor<i1>, tensor<i64>
// CHECK:             stablehlo.return %[[VAL_30]], %[[VAL_34]] : tensor<f32>, tensor<i64>
// CHECK:           }) : (tensor<?x?x?xf32>, tensor<?x?x?xi64>, tensor<f32>, tensor<i64>) -> (tensor<?x?x?xf32>, tensor<?x?x?xi64>)
// CHECK:           %[[VAL_35:.*]] = torch_c.from_builtin_tensor %[[VAL_36:.*]]#1 : tensor<?x?x?xi64> -> !torch.vtensor<[?,?,?],si64>
// CHECK:           %[[VAL_37:.*]] = torch_c.from_builtin_tensor %[[VAL_36]]#0 : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:           return %[[VAL_37]], %[[VAL_35]] : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],si64>
// CHECK:         }
func.func @torch.aten.max_pool2d_with_indices(%arg0: !torch.vtensor<[?,?,?],f32>) -> (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],si64>) {
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %result0, %result1 = torch.aten.max_pool2d_with_indices %arg0, %0, %1, %2, %3, %false : !torch.vtensor<[?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],si64>
  return %result0, %result1 : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.avg_pool2d(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_6:.*]] = torch.constant.none
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_11:.*]] = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_10]]) <{padding = dense<{{\[\[}}0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
// CHECK:           ^bb0(%[[VAL_12:.*]]: tensor<f32>, %[[VAL_13:.*]]: tensor<f32>):
// CHECK:             %[[VAL_14:.*]] = stablehlo.add %[[VAL_12]], %[[VAL_13]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_14]] : tensor<f32>
// CHECK:           }) : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_15:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_16:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_17:.*]] = tensor.dim %[[VAL_1]], %[[VAL_16]] : tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_18:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_19:.*]] = tensor.dim %[[VAL_1]], %[[VAL_18]] : tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_20:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_21:.*]] = tensor.dim %[[VAL_1]], %[[VAL_20]] : tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_22:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_23:.*]] = tensor.dim %[[VAL_1]], %[[VAL_22]] : tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_24:.*]] = tensor.from_elements %[[VAL_17]], %[[VAL_19]], %[[VAL_21]], %[[VAL_23]] : tensor<4xindex>
// CHECK:           %[[VAL_25:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_15]], %[[VAL_24]], dims = [] : (tensor<f32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_26:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_27:.*]] = "stablehlo.reduce_window"(%[[VAL_25]], %[[VAL_26]]) <{padding = dense<{{\[\[}}0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
// CHECK:           ^bb0(%[[VAL_28:.*]]: tensor<f32>, %[[VAL_29:.*]]: tensor<f32>):
// CHECK:             %[[VAL_30:.*]] = stablehlo.add %[[VAL_28]], %[[VAL_29]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_30]] : tensor<f32>
// CHECK:           }) : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_31:.*]] = stablehlo.divide %[[VAL_32:.*]], %[[VAL_33:.*]] : tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_34:.*]] = torch_c.from_builtin_tensor %[[VAL_31]] : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[VAL_34]] : !torch.vtensor<[?,?,?,?],f32>
// CHECK:         }
func.func @torch.aten.avg_pool2d(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.avg_pool2d %arg0, %0, %1, %2, %false, %false, %none : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?,?],f32>
  return %3 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.avg_pool2d$count_include_pad(
// CHECK-SAME:                                                       %[[VAL_0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_6:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_7:.*]] = torch.constant.none
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_12:.*]] = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_11]]) <{padding = dense<{{\[\[}}0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
// CHECK:           ^bb0(%[[VAL_13:.*]]: tensor<f32>, %[[VAL_14:.*]]: tensor<f32>):
// CHECK:             %[[VAL_15:.*]] = stablehlo.add %[[VAL_13]], %[[VAL_14]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_15]] : tensor<f32>
// CHECK:           }) : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_16:.*]] = stablehlo.constant dense<9> : tensor<i64>
// CHECK:           %[[VAL_17:.*]] = stablehlo.convert %[[VAL_16]] : (tensor<i64>) -> tensor<f32>
// CHECK:           %[[VAL_18:.*]] = chlo.broadcast_divide %[[VAL_19:.*]], %[[VAL_17]] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_20:.*]] = torch_c.from_builtin_tensor %[[VAL_18]] : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[VAL_20]] : !torch.vtensor<[?,?,?,?],f32>
// CHECK:         }
func.func @torch.aten.avg_pool2d$count_include_pad(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.avg_pool2d %arg0, %0, %1, %2, %false, %true, %none : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?,?],f32>
  return %3 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @aten.adaptive_avg_pool3d$basic(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !torch.vtensor<[1,1,8,8,8],f32>) -> !torch.vtensor<[1,1,4,4,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,8,8,8],f32> -> tensor<1x1x8x8x8xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_4:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_5:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_6:.*]] = torch.constant.none
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_2]], %[[VAL_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_2]], %[[VAL_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_11:.*]] = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_10]]) <{padding = dense<0> : tensor<5x2xi64>, window_dilations = array<i64: 1, 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 2, 2, 2>, window_strides = array<i64: 1, 1, 2, 2, 2>}> ({
// CHECK:           ^bb0(%[[VAL_12:.*]]: tensor<f32>, %[[VAL_13:.*]]: tensor<f32>):
// CHECK:             %[[VAL_14:.*]] = stablehlo.add %[[VAL_12]], %[[VAL_13]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_14]] : tensor<f32>
// CHECK:           }) : (tensor<1x1x8x8x8xf32>, tensor<f32>) -> tensor<1x1x4x4x4xf32>
// CHECK:           %[[VAL_15:.*]] = stablehlo.constant dense<8> : tensor<i64>
// CHECK:           %[[VAL_16:.*]] = stablehlo.convert %[[VAL_15]] : (tensor<i64>) -> tensor<f32>
// CHECK:           %[[VAL_17:.*]] = chlo.broadcast_divide %[[VAL_18:.*]], %[[VAL_16]] : (tensor<1x1x4x4x4xf32>, tensor<f32>) -> tensor<1x1x4x4x4xf32>
// CHECK:           %[[VAL_19:.*]] = torch_c.from_builtin_tensor %[[VAL_17]] : tensor<1x1x4x4x4xf32> -> !torch.vtensor<[1,1,4,4,4],f32>
// CHECK:           return %[[VAL_19]] : !torch.vtensor<[1,1,4,4,4],f32>
// CHECK:         }
func.func @aten.adaptive_avg_pool3d$basic(%arg0: !torch.vtensor<[1,1,8,8,8],f32>) -> !torch.vtensor<[1,1,4,4,4],f32> {
  %int2 = torch.constant.int 2
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int2, %int2, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0, %int0 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.avg_pool3d %arg0, %0, %1, %2, %false, %true, %none : !torch.vtensor<[1,1,8,8,8],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,1,4,4,4],f32>
  return %3 : !torch.vtensor<[1,1,4,4,4],f32>
}

// -----

// CHECK-LABEL:   func.func @aten.adaptive_avg_pool3d$non_square(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: !torch.vtensor<[1,1,8,12,16],f32>) -> !torch.vtensor<[1,1,4,6,8],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,8,12,16],f32> -> tensor<1x1x8x12x16xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_4:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_5:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_6:.*]] = torch.constant.none
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_2]], %[[VAL_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_2]], %[[VAL_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_11:.*]] = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_10]]) <{padding = dense<0> : tensor<5x2xi64>, window_dilations = array<i64: 1, 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 2, 2, 2>, window_strides = array<i64: 1, 1, 2, 2, 2>}> ({
// CHECK:           ^bb0(%[[VAL_12:.*]]: tensor<f32>, %[[VAL_13:.*]]: tensor<f32>):
// CHECK:             %[[VAL_14:.*]] = stablehlo.add %[[VAL_12]], %[[VAL_13]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_14]] : tensor<f32>
// CHECK:           }) : (tensor<1x1x8x12x16xf32>, tensor<f32>) -> tensor<1x1x4x6x8xf32>
// CHECK:           %[[VAL_15:.*]] = stablehlo.constant dense<8> : tensor<i64>
// CHECK:           %[[VAL_16:.*]] = stablehlo.convert %[[VAL_15]] : (tensor<i64>) -> tensor<f32>
// CHECK:           %[[VAL_17:.*]] = chlo.broadcast_divide %[[VAL_18:.*]], %[[VAL_16]] : (tensor<1x1x4x6x8xf32>, tensor<f32>) -> tensor<1x1x4x6x8xf32>
// CHECK:           %[[VAL_19:.*]] = torch_c.from_builtin_tensor %[[VAL_17]] : tensor<1x1x4x6x8xf32> -> !torch.vtensor<[1,1,4,6,8],f32>
// CHECK:           return %[[VAL_19]] : !torch.vtensor<[1,1,4,6,8],f32>
// CHECK:         }
func.func @aten.adaptive_avg_pool3d$non_square(%arg0: !torch.vtensor<[1,1,8,12,16],f32>) -> !torch.vtensor<[1,1,4,6,8],f32> {
  %int2 = torch.constant.int 2
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int2, %int2, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0, %int0 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.avg_pool3d %arg0, %0, %1, %2, %false, %true, %none : !torch.vtensor<[1,1,8,12,16],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,1,4,6,8],f32>
  return %3 : !torch.vtensor<[1,1,4,6,8],f32>
}


// -----

// CHECK-LABEL:   func.func @aten.avg_pool2d_backward$base_kernel(
// CHECK-SAME:                                                    %[[VAL_0:.*]]: !torch.vtensor<[1,3,32,32],f32>,
// CHECK-SAME:                                                    %[[VAL_1:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[1,3,32,32],f32> {
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 16
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 16
// CHECK:           %[[VAL_6:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_7:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_8:.*]] = torch.aten.expand %[[VAL_1]], %[[VAL_6]], %[[VAL_7]] : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,3,16,16],f32>
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_10:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_10]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_12:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[VAL_13:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_14:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_15:.*]] = torch.prim.ListConstruct %[[VAL_13]], %[[VAL_14]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_16:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_17:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_18:.*]] = torch.constant.none
// CHECK:           %[[VAL_19:.*]] = torch.aten.avg_pool2d_backward %[[VAL_8]], %[[VAL_0]], %[[VAL_11]], %[[VAL_12]], %[[VAL_15]], %[[VAL_16]], %[[VAL_17]], %[[VAL_18]] : !torch.vtensor<[1,3,16,16],f32>, !torch.vtensor<[1,3,32,32],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,3,32,32],f32>
// CHECK:           return %[[VAL_19]] : !torch.vtensor<[1,3,32,32],f32>
// CHECK:         }
func.func @aten.avg_pool2d_backward$base_kernel(%arg0: !torch.vtensor<[1,3,32,32],f32>, %arg1: !torch.vtensor<[],f32>) -> !torch.vtensor<[1,3,32,32],f32> {
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %int16 = torch.constant.int 16
  %int16_0 = torch.constant.int 16
  %0 = torch.prim.ListConstruct %int1, %int3, %int16, %int16_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %1 = torch.aten.expand %arg1, %0, %false : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,3,16,16],f32>
  %int2 = torch.constant.int 2
  %int2_1 = torch.constant.int 2
  %2 = torch.prim.ListConstruct %int2, %int2_1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %int0 = torch.constant.int 0
  %int0_2 = torch.constant.int 0
  %4 = torch.prim.ListConstruct %int0, %int0_2 : (!torch.int, !torch.int) -> !torch.list<int>
  %false_3 = torch.constant.bool false
  %true = torch.constant.bool true
  %none = torch.constant.none
  %5 = torch.aten.avg_pool2d_backward %1, %arg0, %2, %3, %4, %false_3, %true, %none : !torch.vtensor<[1,3,16,16],f32>, !torch.vtensor<[1,3,32,32],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,3,32,32],f32>
  return %5 : !torch.vtensor<[1,3,32,32],f32>
}


// -----

// CHECK-LABEL:   func.func @aten.avg_pool2d_backward$uneven_kernel(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: !torch.vtensor<[1,3,16,16],f32>,
// CHECK-SAME:                                                      %[[VAL_1:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[1,3,16,16],f32> {
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 5
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_6:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_7:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_8:.*]] = torch.aten.expand %[[VAL_1]], %[[VAL_6]], %[[VAL_7]] : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,3,5,3],f32>
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_10:.*]] = torch.constant.int 5
// CHECK:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_10]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_12:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[VAL_13:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_14:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_15:.*]] = torch.prim.ListConstruct %[[VAL_13]], %[[VAL_14]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_16:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_17:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_18:.*]] = torch.constant.none
// CHECK:           %[[VAL_19:.*]] = torch.aten.avg_pool2d_backward %[[VAL_8]], %[[VAL_0]], %[[VAL_11]], %[[VAL_12]], %[[VAL_15]], %[[VAL_16]], %[[VAL_17]], %[[VAL_18]] : !torch.vtensor<[1,3,5,3],f32>, !torch.vtensor<[1,3,16,16],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,3,16,16],f32>
// CHECK:           return %[[VAL_19]] : !torch.vtensor<[1,3,16,16],f32>
// CHECK:         }
func.func @aten.avg_pool2d_backward$uneven_kernel(%arg0: !torch.vtensor<[1,3,16,16],f32>, %arg1: !torch.vtensor<[],f32>) -> !torch.vtensor<[1,3,16,16],f32> {
    %int1 = torch.constant.int 1
    %int3 = torch.constant.int 3
    %int5 = torch.constant.int 5
    %int3_0 = torch.constant.int 3
    %0 = torch.prim.ListConstruct %int1, %int3, %int5, %int3_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.expand %arg1, %0, %false : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,3,5,3],f32>
    %int3_1 = torch.constant.int 3
    %int5_2 = torch.constant.int 5
    %2 = torch.prim.ListConstruct %int3_1, %int5_2 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %int0 = torch.constant.int 0
    %int0_3 = torch.constant.int 0
    %4 = torch.prim.ListConstruct %int0, %int0_3 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_4 = torch.constant.bool false
    %true = torch.constant.bool true
    %none = torch.constant.none
    %5 = torch.aten.avg_pool2d_backward %1, %arg0, %2, %3, %4, %false_4, %true, %none : !torch.vtensor<[1,3,5,3],f32>, !torch.vtensor<[1,3,16,16],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,3,16,16],f32>
    return %5 : !torch.vtensor<[1,3,16,16],f32>
}


// -----

// CHECK-LABEL:   func.func @aten.avg_pool2d_backward$uneven_stride(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: !torch.vtensor<[1,3,16,16],f32>,
// CHECK-SAME:                                                      %[[VAL_1:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[1,3,16,16],f32> {
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 8
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 5
// CHECK:           %[[VAL_6:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_7:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_8:.*]] = torch.aten.expand %[[VAL_1]], %[[VAL_6]], %[[VAL_7]] : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,3,8,5],f32>
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_10:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_10]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_12:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_13:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_14:.*]] = torch.prim.ListConstruct %[[VAL_12]], %[[VAL_13]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_15:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_16:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_17:.*]] = torch.prim.ListConstruct %[[VAL_15]], %[[VAL_16]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_18:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_19:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_20:.*]] = torch.constant.none
// CHECK:           %[[VAL_21:.*]] = torch.aten.avg_pool2d_backward %[[VAL_8]], %[[VAL_0]], %[[VAL_11]], %[[VAL_14]], %[[VAL_17]], %[[VAL_18]], %[[VAL_19]], %[[VAL_20]] : !torch.vtensor<[1,3,8,5],f32>, !torch.vtensor<[1,3,16,16],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,3,16,16],f32>
// CHECK:           return %[[VAL_21]] : !torch.vtensor<[1,3,16,16],f32>
// CHECK:         }
func.func @aten.avg_pool2d_backward$uneven_stride(%arg0: !torch.vtensor<[1,3,16,16],f32>, %arg1: !torch.vtensor<[],f32>) -> !torch.vtensor<[1,3,16,16],f32> {
    %int1 = torch.constant.int 1
    %int3 = torch.constant.int 3
    %int8 = torch.constant.int 8
    %int5 = torch.constant.int 5
    %0 = torch.prim.ListConstruct %int1, %int3, %int8, %int5 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.expand %arg1, %0, %false : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,3,8,5],f32>
    %int2 = torch.constant.int 2
    %int2_0 = torch.constant.int 2
    %2 = torch.prim.ListConstruct %int2, %int2_0 : (!torch.int, !torch.int) -> !torch.list<int>
    %int2_1 = torch.constant.int 2
    %int3_2 = torch.constant.int 3
    %3 = torch.prim.ListConstruct %int2_1, %int3_2 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0 = torch.constant.int 0
    %int0_3 = torch.constant.int 0
    %4 = torch.prim.ListConstruct %int0, %int0_3 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_4 = torch.constant.bool false
    %true = torch.constant.bool true
    %none = torch.constant.none
    %5 = torch.aten.avg_pool2d_backward %1, %arg0, %2, %3, %4, %false_4, %true, %none : !torch.vtensor<[1,3,8,5],f32>, !torch.vtensor<[1,3,16,16],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,3,16,16],f32>
    return %5 : !torch.vtensor<[1,3,16,16],f32>
}


// -----

// CHECK-LABEL:   func.func @aten.avg_pool2d_backward$uneven_padding(
// CHECK-SAME:                                                       %[[VAL_0:.*]]: !torch.vtensor<[1,3,16,16],f32>,
// CHECK-SAME:                                                       %[[VAL_1:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[1,3,16,16],f32> {
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 9
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 8
// CHECK:           %[[VAL_6:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_7:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_8:.*]] = torch.aten.expand %[[VAL_1]], %[[VAL_6]], %[[VAL_7]] : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,3,9,8],f32>
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_10:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_10]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_12:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[VAL_13:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_14:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_15:.*]] = torch.prim.ListConstruct %[[VAL_13]], %[[VAL_14]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_16:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_17:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_18:.*]] = torch.constant.none
// CHECK:           %[[VAL_19:.*]] = torch.aten.avg_pool2d_backward %[[VAL_8]], %[[VAL_0]], %[[VAL_11]], %[[VAL_12]], %[[VAL_15]], %[[VAL_16]], %[[VAL_17]], %[[VAL_18]] : !torch.vtensor<[1,3,9,8],f32>, !torch.vtensor<[1,3,16,16],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,3,16,16],f32>
// CHECK:           return %[[VAL_19]] : !torch.vtensor<[1,3,16,16],f32>
// CHECK:         }
func.func @aten.avg_pool2d_backward$uneven_padding(%arg0: !torch.vtensor<[1,3,16,16],f32>, %arg1: !torch.vtensor<[],f32>) -> !torch.vtensor<[1,3,16,16],f32> {
    %int1 = torch.constant.int 1
    %int3 = torch.constant.int 3
    %int9 = torch.constant.int 9
    %int8 = torch.constant.int 8
    %0 = torch.prim.ListConstruct %int1, %int3, %int9, %int8 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.expand %arg1, %0, %false : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,3,9,8],f32>
    %int2 = torch.constant.int 2
    %int2_0 = torch.constant.int 2
    %2 = torch.prim.ListConstruct %int2, %int2_0 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %int1_1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %4 = torch.prim.ListConstruct %int1_1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_2 = torch.constant.bool false
    %true = torch.constant.bool true
    %none = torch.constant.none
    %5 = torch.aten.avg_pool2d_backward %1, %arg0, %2, %3, %4, %false_2, %true, %none : !torch.vtensor<[1,3,9,8],f32>, !torch.vtensor<[1,3,16,16],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,3,16,16],f32>
    return %5 : !torch.vtensor<[1,3,16,16],f32>
}

// -----

// CHECK-LABEL:   func.func @aten._adaptive_avg_pool2d_backward$basic(
// CHECK-SAME:                                                        %[[VAL_0:.*]]: !torch.vtensor<[1,3,32,32],f32>,
// CHECK-SAME:                                                        %[[VAL_1:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[1,3,32,32],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 8
// CHECK:           %[[VAL_6:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_5]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_7:.*]] = torch_c.to_i64 %[[VAL_3]]
// CHECK:           %[[VAL_8:.*]] = arith.index_cast %[[VAL_7]] : i64 to index
// CHECK:           %[[VAL_9:.*]] = torch_c.to_i64 %[[VAL_4]]
// CHECK:           %[[VAL_10:.*]] = arith.index_cast %[[VAL_9]] : i64 to index
// CHECK:           %[[VAL_11:.*]] = torch_c.to_i64 %[[VAL_5]]
// CHECK:           %[[VAL_12:.*]] = arith.index_cast %[[VAL_11]] : i64 to index
// CHECK:           %[[VAL_13:.*]] = torch_c.to_i64 %[[VAL_5]]
// CHECK:           %[[VAL_14:.*]] = arith.index_cast %[[VAL_13]] : i64 to index
// CHECK:           %[[VAL_15:.*]] = tensor.from_elements %[[VAL_8]], %[[VAL_10]], %[[VAL_12]], %[[VAL_14]] : tensor<4xindex>
// CHECK:           %[[VAL_16:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_2]], %[[VAL_15]], dims = [] : (tensor<f32>, tensor<4xindex>) -> tensor<1x3x8x8xf32>
// CHECK:           %[[VAL_17:.*]] = stablehlo.constant dense<16> : tensor<i64>
// CHECK:           %[[VAL_18:.*]] = stablehlo.convert %[[VAL_17]] : (tensor<i64>) -> tensor<f32>
// CHECK:           %[[VAL_19:.*]] = chlo.broadcast_divide %[[VAL_16]], %[[VAL_18]] : (tensor<1x3x8x8xf32>, tensor<f32>) -> tensor<1x3x8x8xf32>
// CHECK:           %[[VAL_20:.*]] = stablehlo.reshape %[[VAL_19]] : (tensor<1x3x8x8xf32>) -> tensor<1x3x8x1x8x1xf32>
// CHECK:           %[[VAL_21:.*]] = stablehlo.broadcast_in_dim %[[VAL_20]], dims = [0, 1, 2, 3, 4, 5] : (tensor<1x3x8x1x8x1xf32>) -> tensor<1x3x8x4x8x4xf32>
// CHECK:           %[[VAL_22:.*]] = stablehlo.reshape %[[VAL_21]] : (tensor<1x3x8x4x8x4xf32>) -> tensor<1x3x32x32xf32>
// CHECK:           %[[VAL_23:.*]] = torch_c.from_builtin_tensor %[[VAL_22]] : tensor<1x3x32x32xf32> -> !torch.vtensor<[1,3,32,32],f32>
// CHECK:           return %[[VAL_23]] : !torch.vtensor<[1,3,32,32],f32>
// CHECK:         }
func.func @aten._adaptive_avg_pool2d_backward$basic(%arg0: !torch.vtensor<[1,3,32,32],f32>, %arg1: !torch.vtensor<[],f32>) -> !torch.vtensor<[1,3,32,32],f32> {
    %int1 = torch.constant.int 1
    %int3 = torch.constant.int 3
    %int8 = torch.constant.int 8
    %0 = torch.prim.ListConstruct %int1, %int3, %int8, %int8 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.broadcast_to %arg1, %0 : !torch.vtensor<[],f32>, !torch.list<int> -> !torch.vtensor<[1,3,8,8],f32>
    %2 = torch.aten._adaptive_avg_pool2d_backward %1, %arg0 : !torch.vtensor<[1,3,8,8],f32>, !torch.vtensor<[1,3,32,32],f32> -> !torch.vtensor<[1,3,32,32],f32>
    return %2 : !torch.vtensor<[1,3,32,32],f32>
}

// -----

// CHECK-LABEL:   func.func @aten.max_pool2d_with_indices_backward$basic(
// CHECK-SAME:                                                           %[[VAL_0:.*]]: !torch.vtensor<[1,1,4,4],f32>,
// CHECK-SAME:                                                           %[[VAL_1:.*]]: !torch.vtensor<[1,1,8,8],f32>,
// CHECK-SAME:                                                           %[[VAL_2:.*]]: !torch.vtensor<[1,1,4,4],si64>) -> !torch.vtensor<[1,1,8,8],f32> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,1,8,8],f32> -> tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,4,4],f32> -> tensor<1x1x4x4xf32>
// CHECK:           %[[VAL_5:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_8]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_8]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_7]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_12:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_13:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_14:.*]] = "stablehlo.select_and_scatter"(%[[VAL_3]], %[[VAL_4]], %[[VAL_13]]) <{padding = dense<0> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}> ({
// CHECK:           ^bb0(%[[VAL_15:.*]]: tensor<f32>, %[[VAL_16:.*]]: tensor<f32>):
// CHECK:             %[[VAL_17:.*]] = stablehlo.compare  GE, %[[VAL_15]], %[[VAL_16]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             stablehlo.return %[[VAL_17]] : tensor<i1>
// CHECK:           }, {
// CHECK:           ^bb0(%[[VAL_18:.*]]: tensor<f32>, %[[VAL_19:.*]]: tensor<f32>):
// CHECK:             %[[VAL_20:.*]] = stablehlo.add %[[VAL_18]], %[[VAL_19]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_20]] : tensor<f32>
// CHECK:           }) : (tensor<1x1x8x8xf32>, tensor<1x1x4x4xf32>, tensor<f32>) -> tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_21:.*]] = torch_c.from_builtin_tensor %[[VAL_22:.*]] : tensor<1x1x8x8xf32> -> !torch.vtensor<[1,1,8,8],f32>
// CHECK:           return %[[VAL_21]] : !torch.vtensor<[1,1,8,8],f32>
// CHECK:         }
func.func @aten.max_pool2d_with_indices_backward$basic(%arg0: !torch.vtensor<[1,1,4,4],f32>, %arg1: !torch.vtensor<[1,1,8,8],f32>, %arg2: !torch.vtensor<[1,1,4,4],si64>) -> !torch.vtensor<[1,1,8,8],f32> {
    %false = torch.constant.bool false
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.aten.max_pool2d_with_indices_backward %arg0, %arg1, %0, %1, %2, %3, %false, %arg2 : !torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,1,8,8],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.vtensor<[1,1,4,4],si64> -> !torch.vtensor<[1,1,8,8],f32>
    return %4 : !torch.vtensor<[1,1,8,8],f32>
  }


// -----

// CHECK-LABEL:   func.func @aten.max_pool2d_with_indices_backward$uneven_kernel(
// CHECK-SAME:                                                                   %[[VAL_0:.*]]: !torch.vtensor<[1,1,6,7],f32>,
// CHECK-SAME:                                                                   %[[VAL_1:.*]]: !torch.vtensor<[1,1,8,8],f32>,
// CHECK-SAME:                                                                   %[[VAL_2:.*]]: !torch.vtensor<[1,1,6,7],si64>) -> !torch.vtensor<[1,1,8,8],f32> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,1,8,8],f32> -> tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,6,7],f32> -> tensor<1x1x6x7xf32>
// CHECK:           %[[VAL_5:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_9]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_7]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_12:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_13:.*]] = torch.prim.ListConstruct %[[VAL_7]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_14:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_15:.*]] = "stablehlo.select_and_scatter"(%[[VAL_3]], %[[VAL_4]], %[[VAL_14]]) <{padding = dense<0> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 1, 3, 2>, window_strides = array<i64: 1, 1, 1, 1>}> ({
// CHECK:           ^bb0(%[[VAL_16:.*]]: tensor<f32>, %[[VAL_17:.*]]: tensor<f32>):
// CHECK:             %[[VAL_18:.*]] = stablehlo.compare  GE, %[[VAL_16]], %[[VAL_17]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             stablehlo.return %[[VAL_18]] : tensor<i1>
// CHECK:           }, {
// CHECK:           ^bb0(%[[VAL_19:.*]]: tensor<f32>, %[[VAL_20:.*]]: tensor<f32>):
// CHECK:             %[[VAL_21:.*]] = stablehlo.add %[[VAL_19]], %[[VAL_20]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_21]] : tensor<f32>
// CHECK:           }) : (tensor<1x1x8x8xf32>, tensor<1x1x6x7xf32>, tensor<f32>) -> tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_22:.*]] = torch_c.from_builtin_tensor %[[VAL_23:.*]] : tensor<1x1x8x8xf32> -> !torch.vtensor<[1,1,8,8],f32>
// CHECK:           return %[[VAL_22]] : !torch.vtensor<[1,1,8,8],f32>
// CHECK:         }
func.func @aten.max_pool2d_with_indices_backward$uneven_kernel(%arg0: !torch.vtensor<[1,1,6,7],f32>, %arg1: !torch.vtensor<[1,1,8,8],f32>, %arg2: !torch.vtensor<[1,1,6,7],si64>) -> !torch.vtensor<[1,1,8,8],f32> {
    %false = torch.constant.bool false
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %0 = torch.prim.ListConstruct %int3, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.aten.max_pool2d_with_indices_backward %arg0, %arg1, %0, %1, %2, %3, %false, %arg2 : !torch.vtensor<[1,1,6,7],f32>, !torch.vtensor<[1,1,8,8],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.vtensor<[1,1,6,7],si64> -> !torch.vtensor<[1,1,8,8],f32>
    return %4 : !torch.vtensor<[1,1,8,8],f32>
}


// -----


// CHECK-LABEL:   func.func @aten.max_pool2d_with_indices_backward$uneven_padding(
// CHECK-SAME:                                                                    %[[VAL_0:.*]]: !torch.vtensor<[1,1,15,17],f32>,
// CHECK-SAME:                                                                    %[[VAL_1:.*]]: !torch.vtensor<[1,1,16,16],f32>,
// CHECK-SAME:                                                                    %[[VAL_2:.*]]: !torch.vtensor<[1,1,15,17],si64>) -> !torch.vtensor<[1,1,16,16],f32> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,1,16,16],f32> -> tensor<1x1x16x16xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,15,17],f32> -> tensor<1x1x15x17xf32>
// CHECK:           %[[VAL_5:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_8]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_7]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_7]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_12:.*]] = torch.prim.ListConstruct %[[VAL_7]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_13:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_14:.*]] = "stablehlo.select_and_scatter"(%[[VAL_3]], %[[VAL_4]], %[[VAL_13]]) <{padding = dense<{{\[\[}}0, 0], [0, 0], [1, 1], [2, 2]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 1, 4, 4>, window_strides = array<i64: 1, 1, 1, 1>}> ({
// CHECK:           ^bb0(%[[VAL_15:.*]]: tensor<f32>, %[[VAL_16:.*]]: tensor<f32>):
// CHECK:             %[[VAL_17:.*]] = stablehlo.compare  GE, %[[VAL_15]], %[[VAL_16]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             stablehlo.return %[[VAL_17]] : tensor<i1>
// CHECK:           }, {
// CHECK:           ^bb0(%[[VAL_18:.*]]: tensor<f32>, %[[VAL_19:.*]]: tensor<f32>):
// CHECK:             %[[VAL_20:.*]] = stablehlo.add %[[VAL_18]], %[[VAL_19]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_20]] : tensor<f32>
// CHECK:           }) : (tensor<1x1x16x16xf32>, tensor<1x1x15x17xf32>, tensor<f32>) -> tensor<1x1x16x16xf32>
// CHECK:           %[[VAL_21:.*]] = torch_c.from_builtin_tensor %[[VAL_22:.*]] : tensor<1x1x16x16xf32> -> !torch.vtensor<[1,1,16,16],f32>
// CHECK:           return %[[VAL_21]] : !torch.vtensor<[1,1,16,16],f32>
// CHECK:         }
func.func @aten.max_pool2d_with_indices_backward$uneven_padding(%arg0: !torch.vtensor<[1,1,15,17],f32>, %arg1: !torch.vtensor<[1,1,16,16],f32>, %arg2: !torch.vtensor<[1,1,15,17],si64>) -> !torch.vtensor<[1,1,16,16],f32> {
    %false = torch.constant.bool false
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int4 = torch.constant.int 4
    %0 = torch.prim.ListConstruct %int4, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int1, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.aten.max_pool2d_with_indices_backward %arg0, %arg1, %0, %1, %2, %3, %false, %arg2 : !torch.vtensor<[1,1,15,17],f32>, !torch.vtensor<[1,1,16,16],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.vtensor<[1,1,15,17],si64> -> !torch.vtensor<[1,1,16,16],f32>
    return %4 : !torch.vtensor<[1,1,16,16],f32>
}


// -----


// CHECK-LABEL:   func.func @aten.max_pool2d_with_indices_backward$uneven_stride(
// CHECK-SAME:                                                                   %[[VAL_0:.*]]: !torch.vtensor<[1,1,4,3],f32>,
// CHECK-SAME:                                                                   %[[VAL_1:.*]]: !torch.vtensor<[1,1,8,8],f32>,
// CHECK-SAME:                                                                   %[[VAL_2:.*]]: !torch.vtensor<[1,1,4,3],si64>) -> !torch.vtensor<[1,1,8,8],f32> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,1,8,8],f32> -> tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,4,3],f32> -> tensor<1x1x4x3xf32>
// CHECK:           %[[VAL_5:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_9]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_8]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_12:.*]] = torch.prim.ListConstruct %[[VAL_7]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_13:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_14:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_15:.*]] = "stablehlo.select_and_scatter"(%[[VAL_3]], %[[VAL_4]], %[[VAL_14]]) <{padding = dense<0> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 3>}> ({
// CHECK:           ^bb0(%[[VAL_16:.*]]: tensor<f32>, %[[VAL_17:.*]]: tensor<f32>):
// CHECK:             %[[VAL_18:.*]] = stablehlo.compare  GE, %[[VAL_16]], %[[VAL_17]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             stablehlo.return %[[VAL_18]] : tensor<i1>
// CHECK:           }, {
// CHECK:           ^bb0(%[[VAL_19:.*]]: tensor<f32>, %[[VAL_20:.*]]: tensor<f32>):
// CHECK:             %[[VAL_21:.*]] = stablehlo.add %[[VAL_19]], %[[VAL_20]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_21]] : tensor<f32>
// CHECK:           }) : (tensor<1x1x8x8xf32>, tensor<1x1x4x3xf32>, tensor<f32>) -> tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_22:.*]] = torch_c.from_builtin_tensor %[[VAL_23:.*]] : tensor<1x1x8x8xf32> -> !torch.vtensor<[1,1,8,8],f32>
// CHECK:           return %[[VAL_22]] : !torch.vtensor<[1,1,8,8],f32>
// CHECK:         }
func.func @aten.max_pool2d_with_indices_backward$uneven_stride(%arg0: !torch.vtensor<[1,1,4,3],f32>, %arg1: !torch.vtensor<[1,1,8,8],f32>, %arg2: !torch.vtensor<[1,1,4,3],si64>) -> !torch.vtensor<[1,1,8,8],f32> {
    %false = torch.constant.bool false
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.aten.max_pool2d_with_indices_backward %arg0, %arg1, %0, %1, %2, %3, %false, %arg2 : !torch.vtensor<[1,1,4,3],f32>, !torch.vtensor<[1,1,8,8],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.vtensor<[1,1,4,3],si64> -> !torch.vtensor<[1,1,8,8],f32>
    return %4 : !torch.vtensor<[1,1,8,8],f32>
}


// -----


// CHECK-LABEL:   func.func @aten.max_pool2d_with_indices_backward$dynamic_batch(
// CHECK-SAME:                                                                   %[[VAL_0:.*]]: !torch.vtensor<[?,1,4,4],f32>,
// CHECK-SAME:                                                                   %[[VAL_1:.*]]: !torch.vtensor<[?,1,8,8],f32>,
// CHECK-SAME:                                                                   %[[VAL_2:.*]]: !torch.vtensor<[?,1,4,4],si64>) -> !torch.vtensor<[?,1,8,8],f32> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,1,8,8],f32> -> tensor<?x1x8x8xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,1,4,4],f32> -> tensor<?x1x4x4xf32>
// CHECK:           %[[VAL_5:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_8]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_8]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_7]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_12:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_13:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_14:.*]] = "stablehlo.select_and_scatter"(%[[VAL_3]], %[[VAL_4]], %[[VAL_13]]) <{padding = dense<0> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}> ({
// CHECK:           ^bb0(%[[VAL_15:.*]]: tensor<f32>, %[[VAL_16:.*]]: tensor<f32>):
// CHECK:             %[[VAL_17:.*]] = stablehlo.compare  GE, %[[VAL_15]], %[[VAL_16]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             stablehlo.return %[[VAL_17]] : tensor<i1>
// CHECK:           }, {
// CHECK:           ^bb0(%[[VAL_18:.*]]: tensor<f32>, %[[VAL_19:.*]]: tensor<f32>):
// CHECK:             %[[VAL_20:.*]] = stablehlo.add %[[VAL_18]], %[[VAL_19]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_20]] : tensor<f32>
// CHECK:           }) : (tensor<?x1x8x8xf32>, tensor<?x1x4x4xf32>, tensor<f32>) -> tensor<?x1x8x8xf32>
// CHECK:           %[[VAL_21:.*]] = torch_c.from_builtin_tensor %[[VAL_22:.*]] : tensor<?x1x8x8xf32> -> !torch.vtensor<[?,1,8,8],f32>
// CHECK:           return %[[VAL_21]] : !torch.vtensor<[?,1,8,8],f32>
// CHECK:         }
func.func @aten.max_pool2d_with_indices_backward$dynamic_batch(%grad_out: !torch.vtensor<[?,1,4,4],f32>, %self: !torch.vtensor<[?,1,8,8],f32>, %indices: !torch.vtensor<[?,1,4,4],si64>) -> !torch.vtensor<[?,1,8,8],f32> {
    %ceil_mode = torch.constant.bool false
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %kernel_size = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %stride = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %padding = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %dilation = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %result = torch.aten.max_pool2d_with_indices_backward %grad_out, %self, %kernel_size, %stride, %padding, %dilation, %ceil_mode, %indices : !torch.vtensor<[?,1,4,4],f32>, !torch.vtensor<[?,1,8,8],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.vtensor<[?,1,4,4],si64> -> !torch.vtensor<[?,1,8,8],f32>
    return %result : !torch.vtensor<[?,1,8,8],f32>
}


// -----


// Dynamic H and W (spatial dims) — ceilMode=false, so padding is static and
// select_and_scatter lowers cleanly without knowing H or W at compile time.
// CHECK-LABEL:   func.func @aten.max_pool2d_with_indices_backward$dynamic_hw(
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %{{.*}} : !torch.vtensor<[1,1,?,?],f32> -> tensor<1x1x?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %{{.*}} : !torch.vtensor<[1,1,?,?],f32> -> tensor<1x1x?x?xf32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = "stablehlo.select_and_scatter"(%[[VAL_3]], %[[VAL_4]], %[[VAL_5]]) <{padding = dense<0> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: tensor<f32>, %[[VAL_8:.*]]: tensor<f32>):
// CHECK:             %[[VAL_9:.*]] = stablehlo.compare  GE, %[[VAL_7]], %[[VAL_8]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             stablehlo.return %[[VAL_9]] : tensor<i1>
// CHECK:           }, {
// CHECK:           ^bb0(%[[VAL_10:.*]]: tensor<f32>, %[[VAL_11:.*]]: tensor<f32>):
// CHECK:             %[[VAL_12:.*]] = stablehlo.add %[[VAL_10]], %[[VAL_11]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_12]] : tensor<f32>
// CHECK:           }) : (tensor<1x1x?x?xf32>, tensor<1x1x?x?xf32>, tensor<f32>) -> tensor<1x1x?x?xf32>
// CHECK:           %[[VAL_13:.*]] = torch_c.from_builtin_tensor %[[VAL_14:.*]] : tensor<1x1x?x?xf32> -> !torch.vtensor<[1,1,?,?],f32>
// CHECK:           return %[[VAL_13]] : !torch.vtensor<[1,1,?,?],f32>
// CHECK:         }
func.func @aten.max_pool2d_with_indices_backward$dynamic_hw(%grad_out: !torch.vtensor<[1,1,?,?],f32>, %self: !torch.vtensor<[1,1,?,?],f32>, %indices: !torch.vtensor<[1,1,?,?],si64>) -> !torch.vtensor<[1,1,?,?],f32> {
    %ceil_mode = torch.constant.bool false
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %kernel_size = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %stride = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %padding = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %dilation = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %result = torch.aten.max_pool2d_with_indices_backward %grad_out, %self, %kernel_size, %stride, %padding, %dilation, %ceil_mode, %indices : !torch.vtensor<[1,1,?,?],f32>, !torch.vtensor<[1,1,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.vtensor<[1,1,?,?],si64> -> !torch.vtensor<[1,1,?,?],f32>
    return %result : !torch.vtensor<[1,1,?,?],f32>
}


// -----


// ceilMode=true with dynamic H and W: extra trailing padding cannot be
// computed without knowing the spatial sizes, so it is skipped and
// padding stays zero. The op still lowers to select_and_scatter.
// CHECK-LABEL:   func.func @aten.max_pool2d_with_indices_backward$dynamic_hw_ceil(
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %{{.*}} : !torch.vtensor<[1,1,?,?],f32> -> tensor<1x1x?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %{{.*}} : !torch.vtensor<[1,1,?,?],f32> -> tensor<1x1x?x?xf32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = "stablehlo.select_and_scatter"(%[[VAL_3]], %[[VAL_4]], %[[VAL_5]]) <{padding = dense<0> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}> ({
// CHECK:           ^bb0(%[[VAL_7:.*]]: tensor<f32>, %[[VAL_8:.*]]: tensor<f32>):
// CHECK:             %[[VAL_9:.*]] = stablehlo.compare  GE, %[[VAL_7]], %[[VAL_8]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             stablehlo.return %[[VAL_9]] : tensor<i1>
// CHECK:           }, {
// CHECK:           ^bb0(%[[VAL_10:.*]]: tensor<f32>, %[[VAL_11:.*]]: tensor<f32>):
// CHECK:             %[[VAL_12:.*]] = stablehlo.add %[[VAL_10]], %[[VAL_11]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_12]] : tensor<f32>
// CHECK:           }) : (tensor<1x1x?x?xf32>, tensor<1x1x?x?xf32>, tensor<f32>) -> tensor<1x1x?x?xf32>
// CHECK:           %[[VAL_13:.*]] = torch_c.from_builtin_tensor %[[VAL_14:.*]] : tensor<1x1x?x?xf32> -> !torch.vtensor<[1,1,?,?],f32>
// CHECK:           return %[[VAL_13]] : !torch.vtensor<[1,1,?,?],f32>
// CHECK:         }
func.func @aten.max_pool2d_with_indices_backward$dynamic_hw_ceil(%grad_out: !torch.vtensor<[1,1,?,?],f32>, %self: !torch.vtensor<[1,1,?,?],f32>, %indices: !torch.vtensor<[1,1,?,?],si64>) -> !torch.vtensor<[1,1,?,?],f32> {
    %ceil_mode = torch.constant.bool true
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %kernel_size = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %stride = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %padding = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %dilation = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %result = torch.aten.max_pool2d_with_indices_backward %grad_out, %self, %kernel_size, %stride, %padding, %dilation, %ceil_mode, %indices : !torch.vtensor<[1,1,?,?],f32>, !torch.vtensor<[1,1,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.vtensor<[1,1,?,?],si64> -> !torch.vtensor<[1,1,?,?],f32>
    return %result : !torch.vtensor<[1,1,?,?],f32>
}


// -----


// CHECK-LABEL:   func.func @aten.max_pool2d_with_indices_backward$dynamic_channel(
// CHECK-SAME:                                                                     %[[VAL_0:.*]]: !torch.vtensor<[1,?,4,4],f32>,
// CHECK-SAME:                                                                     %[[VAL_1:.*]]: !torch.vtensor<[1,?,8,8],f32>,
// CHECK-SAME:                                                                     %[[VAL_2:.*]]: !torch.vtensor<[1,?,4,4],si64>) -> !torch.vtensor<[1,?,8,8],f32> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,?,8,8],f32> -> tensor<1x?x8x8xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,?,4,4],f32> -> tensor<1x?x4x4xf32>
// CHECK:           %[[VAL_5:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_8]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_8]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_7]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_12:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_13:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_14:.*]] = "stablehlo.select_and_scatter"(%[[VAL_3]], %[[VAL_4]], %[[VAL_13]]) <{padding = dense<0> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}> ({
// CHECK:           ^bb0(%[[VAL_15:.*]]: tensor<f32>, %[[VAL_16:.*]]: tensor<f32>):
// CHECK:             %[[VAL_17:.*]] = stablehlo.compare  GE, %[[VAL_15]], %[[VAL_16]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             stablehlo.return %[[VAL_17]] : tensor<i1>
// CHECK:           }, {
// CHECK:           ^bb0(%[[VAL_18:.*]]: tensor<f32>, %[[VAL_19:.*]]: tensor<f32>):
// CHECK:             %[[VAL_20:.*]] = stablehlo.add %[[VAL_18]], %[[VAL_19]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_20]] : tensor<f32>
// CHECK:           }) : (tensor<1x?x8x8xf32>, tensor<1x?x4x4xf32>, tensor<f32>) -> tensor<1x?x8x8xf32>
// CHECK:           %[[VAL_21:.*]] = torch_c.from_builtin_tensor %[[VAL_22:.*]] : tensor<1x?x8x8xf32> -> !torch.vtensor<[1,?,8,8],f32>
// CHECK:           return %[[VAL_21]] : !torch.vtensor<[1,?,8,8],f32>
// CHECK:         }
func.func @aten.max_pool2d_with_indices_backward$dynamic_channel(%grad_out: !torch.vtensor<[1,?,4,4],f32>, %self: !torch.vtensor<[1,?,8,8],f32>, %indices: !torch.vtensor<[1,?,4,4],si64>) -> !torch.vtensor<[1,?,8,8],f32> {
    %ceil_mode = torch.constant.bool false
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %kernel_size = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %stride = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %padding = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %dilation = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %result = torch.aten.max_pool2d_with_indices_backward %grad_out, %self, %kernel_size, %stride, %padding, %dilation, %ceil_mode, %indices : !torch.vtensor<[1,?,4,4],f32>, !torch.vtensor<[1,?,8,8],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.vtensor<[1,?,4,4],si64> -> !torch.vtensor<[1,?,8,8],f32>
    return %result : !torch.vtensor<[1,?,8,8],f32>
}


// -----


// CHECK-LABEL:   func.func @aten.max_pool2d_with_indices$empty_stride(
// CHECK-SAME:                                                         %[[VAL_0:.*]]: !torch.vtensor<[1,1,8,8],f32>) -> (!torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,1,4,4],si64>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,8,8],f32> -> tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_6:.*]] = torch.prim.ListConstruct %[[VAL_5]], %[[VAL_5]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_11:.*]] = tensor.dim %[[VAL_1]], %[[VAL_10]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_13:.*]] = tensor.dim %[[VAL_1]], %[[VAL_12]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_14:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_15:.*]] = tensor.dim %[[VAL_1]], %[[VAL_14]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_16:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_17:.*]] = tensor.dim %[[VAL_1]], %[[VAL_16]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_18:.*]] = tensor.from_elements %[[VAL_11]], %[[VAL_13]], %[[VAL_15]], %[[VAL_17]] : tensor<4xindex>
// CHECK:           %[[VAL_19:.*]] = arith.muli %[[VAL_17]], %[[VAL_15]] : index
// CHECK:           %[[VAL_20:.*]] = tensor.from_elements %[[VAL_11]], %[[VAL_13]], %[[VAL_19]] : tensor<3xindex>
// CHECK:           %[[VAL_21:.*]] = stablehlo.dynamic_iota %[[VAL_20]], dim = 2 : (tensor<3xindex>) -> tensor<1x1x64xi64>
// CHECK:           %[[VAL_22:.*]] = stablehlo.dynamic_reshape %[[VAL_21]], %[[VAL_18]] : (tensor<1x1x64xi64>, tensor<4xindex>) -> tensor<1x1x8x8xi64>
// CHECK:           %[[VAL_23:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           %[[VAL_24:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_25:.*]]:2 = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]]) <{padding = dense<0> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}> ({
// CHECK:           ^bb0(%[[VAL_26:.*]]: tensor<f32>, %[[VAL_27:.*]]: tensor<i64>, %[[VAL_28:.*]]: tensor<f32>, %[[VAL_29:.*]]: tensor<i64>):
// CHECK:             %[[VAL_30:.*]] = stablehlo.compare  GE, %[[VAL_26]], %[[VAL_28]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             %[[VAL_31:.*]] = stablehlo.select %[[VAL_30]], %[[VAL_26]], %[[VAL_28]] : tensor<i1>, tensor<f32>
// CHECK:             %[[VAL_32:.*]] = stablehlo.compare  EQ, %[[VAL_26]], %[[VAL_28]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             %[[VAL_33:.*]] = stablehlo.minimum %[[VAL_27]], %[[VAL_29]] : tensor<i64>
// CHECK:             %[[VAL_34:.*]] = stablehlo.select %[[VAL_30]], %[[VAL_27]], %[[VAL_29]] : tensor<i1>, tensor<i64>
// CHECK:             %[[VAL_35:.*]] = stablehlo.select %[[VAL_32]], %[[VAL_33]], %[[VAL_34]] : tensor<i1>, tensor<i64>
// CHECK:             stablehlo.return %[[VAL_31]], %[[VAL_35]] : tensor<f32>, tensor<i64>
// CHECK:           }) : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xi64>, tensor<f32>, tensor<i64>) -> (tensor<1x1x4x4xf32>, tensor<1x1x4x4xi64>)
// CHECK:           %[[VAL_36:.*]] = torch_c.from_builtin_tensor %[[VAL_37:.*]]#1 : tensor<1x1x4x4xi64> -> !torch.vtensor<[1,1,4,4],si64>
// CHECK:           %[[VAL_38:.*]] = torch_c.from_builtin_tensor %[[VAL_37]]#0 : tensor<1x1x4x4xf32> -> !torch.vtensor<[1,1,4,4],f32>
// CHECK:           return %[[VAL_38]], %[[VAL_36]] : !torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,1,4,4],si64>
// CHECK:         }
func.func @aten.max_pool2d_with_indices$empty_stride(%arg0: !torch.vtensor<[1,1,8,8],f32>) -> (!torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,1,4,4],si64>) {
    %false = torch.constant.bool false
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %result0, %result1 = torch.aten.max_pool2d_with_indices %arg0, %0, %1, %2, %3, %false : !torch.vtensor<[1,1,8,8],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,1,4,4],si64>
    return %result0, %result1 : !torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,1,4,4],si64>
}

// -----

// CHECK-LABEL:   func.func @aten.im2col$basic(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !torch.vtensor<[1,1,28,28],f32>) -> !torch.vtensor<[1,9,676],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,28,28],f32> -> tensor<1x1x28x28xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_5]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_9]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_12:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_13:.*]] = torch.prim.ListConstruct %[[VAL_11]], %[[VAL_12]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_14:.*]] = stablehlo.iota dim = 0 : tensor<3x1x1x1xi32>
// CHECK:           %[[VAL_15:.*]] = stablehlo.iota dim = 1 : tensor<1x3x1x1xi32>
// CHECK:           %[[VAL_16:.*]] = stablehlo.iota dim = 2 : tensor<1x1x26x1xi32>
// CHECK:           %[[VAL_17:.*]] = stablehlo.iota dim = 3 : tensor<1x1x1x26xi32>
// CHECK:           %[[VAL_18:.*]] = stablehlo.broadcast_in_dim %[[VAL_14]], dims = [0, 1, 2, 3] : (tensor<3x1x1x1xi32>) -> tensor<3x3x26x26xi32>
// CHECK:           %[[VAL_19:.*]] = stablehlo.broadcast_in_dim %[[VAL_15]], dims = [0, 1, 2, 3] : (tensor<1x3x1x1xi32>) -> tensor<3x3x26x26xi32>
// CHECK:           %[[VAL_20:.*]] = stablehlo.broadcast_in_dim %[[VAL_16]], dims = [0, 1, 2, 3] : (tensor<1x1x26x1xi32>) -> tensor<3x3x26x26xi32>
// CHECK:           %[[VAL_21:.*]] = stablehlo.broadcast_in_dim %[[VAL_17]], dims = [0, 1, 2, 3] : (tensor<1x1x1x26xi32>) -> tensor<3x3x26x26xi32>
// CHECK:           %[[VAL_22:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK:           %[[VAL_23:.*]] = chlo.broadcast_multiply %[[VAL_18]], %[[VAL_22]] : (tensor<3x3x26x26xi32>, tensor<i32>) -> tensor<3x3x26x26xi32>
// CHECK:           %[[VAL_24:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK:           %[[VAL_25:.*]] = chlo.broadcast_multiply %[[VAL_20]], %[[VAL_24]] : (tensor<3x3x26x26xi32>, tensor<i32>) -> tensor<3x3x26x26xi32>
// CHECK:           %[[VAL_26:.*]] = chlo.broadcast_add %[[VAL_23]], %[[VAL_25]] : (tensor<3x3x26x26xi32>, tensor<3x3x26x26xi32>) -> tensor<3x3x26x26xi32>
// CHECK:           %[[VAL_27:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK:           %[[VAL_28:.*]] = chlo.broadcast_multiply %[[VAL_19]], %[[VAL_27]] : (tensor<3x3x26x26xi32>, tensor<i32>) -> tensor<3x3x26x26xi32>
// CHECK:           %[[VAL_29:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK:           %[[VAL_30:.*]] = chlo.broadcast_multiply %[[VAL_21]], %[[VAL_29]] : (tensor<3x3x26x26xi32>, tensor<i32>) -> tensor<3x3x26x26xi32>
// CHECK:           %[[VAL_31:.*]] = chlo.broadcast_add %[[VAL_28]], %[[VAL_30]] : (tensor<3x3x26x26xi32>, tensor<3x3x26x26xi32>) -> tensor<3x3x26x26xi32>
// CHECK:           %[[VAL_32:.*]] = stablehlo.reshape %[[VAL_26]] : (tensor<3x3x26x26xi32>) -> tensor<6084x1xi32>
// CHECK:           %[[VAL_33:.*]] = stablehlo.reshape %[[VAL_31]] : (tensor<3x3x26x26xi32>) -> tensor<6084x1xi32>
// CHECK:           %[[VAL_34:.*]] = stablehlo.concatenate %[[VAL_32]], %[[VAL_33]], dim = 1 : (tensor<6084x1xi32>, tensor<6084x1xi32>) -> tensor<6084x2xi32>
// CHECK:           %[[VAL_35:.*]] = "stablehlo.gather"(%[[VAL_1]], %[[VAL_34]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2, 3], start_index_map = [2, 3], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1, 1, 1>}> : (tensor<1x1x28x28xf32>, tensor<6084x2xi32>) -> tensor<1x1x6084xf32>
// CHECK:           %[[VAL_36:.*]] = stablehlo.reshape %[[VAL_35]] : (tensor<1x1x6084xf32>) -> tensor<1x9x676xf32>
// CHECK:           %[[VAL_37:.*]] = torch_c.from_builtin_tensor %[[VAL_36]] : tensor<1x9x676xf32> -> !torch.vtensor<[1,9,676],f32>
// CHECK:           return %[[VAL_37]] : !torch.vtensor<[1,9,676],f32>
// CHECK:         }
func.func @aten.im2col$basic(%arg0: !torch.vtensor<[1,1,28,28],f32>) -> !torch.vtensor<[1,9,676],f32> {
    %int3 = torch.constant.int 3
    %int3_0 = torch.constant.int 3
    %0 = torch.prim.ListConstruct %int3, %int3_0 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1 = torch.constant.int 1
    %int1_1 = torch.constant.int 1
    %1 = torch.prim.ListConstruct %int1, %int1_1 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0 = torch.constant.int 0
    %int0_2 = torch.constant.int 0
    %2 = torch.prim.ListConstruct %int0, %int0_2 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_3 = torch.constant.int 1
    %int1_4 = torch.constant.int 1
    %3 = torch.prim.ListConstruct %int1_3, %int1_4 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.aten.im2col %arg0, %0, %1, %2, %3 : !torch.vtensor<[1,1,28,28],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.vtensor<[1,9,676],f32>
    return %4 : !torch.vtensor<[1,9,676],f32>
}

// -----

// CHECK-LABEL:   func.func @aten.im2col$uneven_kernel(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !torch.vtensor<[1,3,32,64],f32>) -> !torch.vtensor<[1,24,594],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,3,32,64],f32> -> tensor<1x3x32x64xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_5]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_9]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_12:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_13:.*]] = torch.prim.ListConstruct %[[VAL_11]], %[[VAL_12]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_14:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_15:.*]] = stablehlo.pad %[[VAL_1]], %[[VAL_14]], low = [0, 0, 2, 2], high = [0, 0, 2, 2], interior = [0, 0, 0, 0] : (tensor<1x3x32x64xf32>, tensor<f32>) -> tensor<1x3x36x68xf32>
// CHECK:           %[[VAL_16:.*]] = stablehlo.iota dim = 0 : tensor<2x1x1x1xi32>
// CHECK:           %[[VAL_17:.*]] = stablehlo.iota dim = 1 : tensor<1x4x1x1xi32>
// CHECK:           %[[VAL_18:.*]] = stablehlo.iota dim = 2 : tensor<1x1x18x1xi32>
// CHECK:           %[[VAL_19:.*]] = stablehlo.iota dim = 3 : tensor<1x1x1x33xi32>
// CHECK:           %[[VAL_20:.*]] = stablehlo.broadcast_in_dim %[[VAL_16]], dims = [0, 1, 2, 3] : (tensor<2x1x1x1xi32>) -> tensor<2x4x18x33xi32>
// CHECK:           %[[VAL_21:.*]] = stablehlo.broadcast_in_dim %[[VAL_17]], dims = [0, 1, 2, 3] : (tensor<1x4x1x1xi32>) -> tensor<2x4x18x33xi32>
// CHECK:           %[[VAL_22:.*]] = stablehlo.broadcast_in_dim %[[VAL_18]], dims = [0, 1, 2, 3] : (tensor<1x1x18x1xi32>) -> tensor<2x4x18x33xi32>
// CHECK:           %[[VAL_23:.*]] = stablehlo.broadcast_in_dim %[[VAL_19]], dims = [0, 1, 2, 3] : (tensor<1x1x1x33xi32>) -> tensor<2x4x18x33xi32>
// CHECK:           %[[VAL_24:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK:           %[[VAL_25:.*]] = chlo.broadcast_multiply %[[VAL_20]], %[[VAL_24]] : (tensor<2x4x18x33xi32>, tensor<i32>) -> tensor<2x4x18x33xi32>
// CHECK:           %[[VAL_26:.*]] = stablehlo.constant dense<2> : tensor<i32>
// CHECK:           %[[VAL_27:.*]] = chlo.broadcast_multiply %[[VAL_22]], %[[VAL_26]] : (tensor<2x4x18x33xi32>, tensor<i32>) -> tensor<2x4x18x33xi32>
// CHECK:           %[[VAL_28:.*]] = chlo.broadcast_add %[[VAL_25]], %[[VAL_27]] : (tensor<2x4x18x33xi32>, tensor<2x4x18x33xi32>) -> tensor<2x4x18x33xi32>
// CHECK:           %[[VAL_29:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK:           %[[VAL_30:.*]] = chlo.broadcast_multiply %[[VAL_21]], %[[VAL_29]] : (tensor<2x4x18x33xi32>, tensor<i32>) -> tensor<2x4x18x33xi32>
// CHECK:           %[[VAL_31:.*]] = stablehlo.constant dense<2> : tensor<i32>
// CHECK:           %[[VAL_32:.*]] = chlo.broadcast_multiply %[[VAL_23]], %[[VAL_31]] : (tensor<2x4x18x33xi32>, tensor<i32>) -> tensor<2x4x18x33xi32>
// CHECK:           %[[VAL_33:.*]] = chlo.broadcast_add %[[VAL_30]], %[[VAL_32]] : (tensor<2x4x18x33xi32>, tensor<2x4x18x33xi32>) -> tensor<2x4x18x33xi32>
// CHECK:           %[[VAL_34:.*]] = stablehlo.reshape %[[VAL_28]] : (tensor<2x4x18x33xi32>) -> tensor<4752x1xi32>
// CHECK:           %[[VAL_35:.*]] = stablehlo.reshape %[[VAL_33]] : (tensor<2x4x18x33xi32>) -> tensor<4752x1xi32>
// CHECK:           %[[VAL_36:.*]] = stablehlo.concatenate %[[VAL_34]], %[[VAL_35]], dim = 1 : (tensor<4752x1xi32>, tensor<4752x1xi32>) -> tensor<4752x2xi32>
// CHECK:           %[[VAL_37:.*]] = "stablehlo.gather"(%[[VAL_15]], %[[VAL_36]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2, 3], start_index_map = [2, 3], index_vector_dim = 1>, slice_sizes = array<i64: 1, 3, 1, 1>}> : (tensor<1x3x36x68xf32>, tensor<4752x2xi32>) -> tensor<1x3x4752xf32>
// CHECK:           %[[VAL_38:.*]] = stablehlo.reshape %[[VAL_37]] : (tensor<1x3x4752xf32>) -> tensor<1x24x594xf32>
// CHECK:           %[[VAL_39:.*]] = torch_c.from_builtin_tensor %[[VAL_38]] : tensor<1x24x594xf32> -> !torch.vtensor<[1,24,594],f32>
// CHECK:           return %[[VAL_39]] : !torch.vtensor<[1,24,594],f32>
// CHECK:         }
func.func @aten.im2col$uneven_kernel(%arg0: !torch.vtensor<[1,3,32,64],f32>) -> !torch.vtensor<[1,24,594],f32> {
    %int2 = torch.constant.int 2
    %int4 = torch.constant.int 4
    %0 = torch.prim.ListConstruct %int2, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1 = torch.constant.int 1
    %int1_0 = torch.constant.int 1
    %1 = torch.prim.ListConstruct %int1, %int1_0 : (!torch.int, !torch.int) -> !torch.list<int>
    %int2_1 = torch.constant.int 2
    %int2_2 = torch.constant.int 2
    %2 = torch.prim.ListConstruct %int2_1, %int2_2 : (!torch.int, !torch.int) -> !torch.list<int>
    %int2_3 = torch.constant.int 2
    %int2_4 = torch.constant.int 2
    %3 = torch.prim.ListConstruct %int2_3, %int2_4 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.aten.im2col %arg0, %0, %1, %2, %3 : !torch.vtensor<[1,3,32,64],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.vtensor<[1,24,594],f32>
    return %4 : !torch.vtensor<[1,24,594],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.cumsum$static_last_dim(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],f32> -> tensor<3x4xf32>
// CHECK:           %[[TRIL:.*]] = stablehlo.constant dense<
// CHECK:           %[[DOT:.*]] = stablehlo.dot_general %[[INPUT]], %[[TRIL]], contracting_dims = [1] x [0] : (tensor<3x4xf32>, tensor<4x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[DOT]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.cumsum$static_last_dim(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
    %int1 = torch.constant.int 1
    %none = torch.constant.none
    %0 = torch.aten.cumsum %arg0, %int1, %none : !torch.vtensor<[3,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,4],f32>
    return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.cumsum$static_non_last_dim(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],f32> -> tensor<3x4xf32>
// CHECK:           %[[TRANSPOSED:.*]] = stablehlo.transpose %[[INPUT]], dims = [1, 0] : (tensor<3x4xf32>) -> tensor<4x3xf32>
// CHECK:           %[[TRIL:.*]] = stablehlo.constant dense<
// CHECK:           %[[DOT:.*]] = stablehlo.dot_general %[[TRANSPOSED]], %[[TRIL]], contracting_dims = [1] x [0] : (tensor<4x3xf32>, tensor<3x3xf32>) -> tensor<4x3xf32>
// CHECK:           %[[RESULT:.*]] = stablehlo.transpose %[[DOT]], dims = [1, 0] : (tensor<4x3xf32>) -> tensor<3x4xf32>
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.cumsum$static_non_last_dim(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
    %int0 = torch.constant.int 0
    %none = torch.constant.none
    %0 = torch.aten.cumsum %arg0, %int0, %none : !torch.vtensor<[3,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,4],f32>
    return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.cumsum$dynamic_last_dim(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[3,?],f32>) -> !torch.vtensor<[3,?],f32> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,?],f32> -> tensor<3x?xf32>
// CHECK:           %[[DIM:.*]] = tensor.dim %[[INPUT]], %{{.*}} : tensor<3x?xf32>
// CHECK:           %[[DIMI64:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:           %[[SHAPE:.*]] = tensor.from_elements %[[DIMI64]], %[[DIMI64]] : tensor<2xi64>
// CHECK:           %[[ROW_IOTA:.*]] = stablehlo.dynamic_iota %[[SHAPE]], dim = 0 : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:           %[[COL_IOTA:.*]] = stablehlo.dynamic_iota %[[SHAPE]], dim = 1 : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:           %[[MASK:.*]] = stablehlo.compare LE, %[[ROW_IOTA]], %[[COL_IOTA]], SIGNED : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi1>
// CHECK:           %[[TRIL:.*]] = stablehlo.convert %[[MASK]] : (tensor<?x?xi1>) -> tensor<?x?xf32>
// CHECK:           %[[DOT:.*]] = stablehlo.dot_general %[[INPUT]], %[[TRIL]], contracting_dims = [1] x [0] : (tensor<3x?xf32>, tensor<?x?xf32>) -> tensor<3x?xf32>
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[DOT]] : tensor<3x?xf32> -> !torch.vtensor<[3,?],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[3,?],f32>
// CHECK:         }
func.func @torch.aten.cumsum$dynamic_last_dim(%arg0: !torch.vtensor<[3,?],f32>) -> !torch.vtensor<[3,?],f32> {
    %int1 = torch.constant.int 1
    %none = torch.constant.none
    %0 = torch.aten.cumsum %arg0, %int1, %none : !torch.vtensor<[3,?],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,?],f32>
    return %0 : !torch.vtensor<[3,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.cumsum$dynamic_non_last_dim(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[?,4],f32>) -> !torch.vtensor<[?,4],f32> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,4],f32> -> tensor<?x4xf32>
// CHECK:           %[[TRANSPOSED:.*]] = stablehlo.transpose %[[INPUT]], dims = [1, 0] : (tensor<?x4xf32>) -> tensor<4x?xf32>
// CHECK:           %[[DIM:.*]] = tensor.dim %[[TRANSPOSED]], %{{.*}} : tensor<4x?xf32>
// CHECK:           %[[DIMI64:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:           %[[SHAPE:.*]] = tensor.from_elements %[[DIMI64]], %[[DIMI64]] : tensor<2xi64>
// CHECK:           %[[ROW_IOTA:.*]] = stablehlo.dynamic_iota %[[SHAPE]], dim = 0 : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:           %[[COL_IOTA:.*]] = stablehlo.dynamic_iota %[[SHAPE]], dim = 1 : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:           %[[MASK:.*]] = stablehlo.compare LE, %[[ROW_IOTA]], %[[COL_IOTA]], SIGNED : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi1>
// CHECK:           %[[TRIL:.*]] = stablehlo.convert %[[MASK]] : (tensor<?x?xi1>) -> tensor<?x?xf32>
// CHECK:           %[[DOT:.*]] = stablehlo.dot_general %[[TRANSPOSED]], %[[TRIL]], contracting_dims = [1] x [0] : (tensor<4x?xf32>, tensor<?x?xf32>) -> tensor<4x?xf32>
// CHECK:           %[[RESULT:.*]] = stablehlo.transpose %[[DOT]], dims = [1, 0] : (tensor<4x?xf32>) -> tensor<?x4xf32>
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<?x4xf32> -> !torch.vtensor<[?,4],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,4],f32>
// CHECK:         }
func.func @torch.aten.cumsum$dynamic_non_last_dim(%arg0: !torch.vtensor<[?,4],f32>) -> !torch.vtensor<[?,4],f32> {
    %int0 = torch.constant.int 0
    %none = torch.constant.none
    %0 = torch.aten.cumsum %arg0, %int0, %none : !torch.vtensor<[?,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[?,4],f32>
    return %0 : !torch.vtensor<[?,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.cumsum$static_int(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[5],si32>) -> !torch.vtensor<[5],si64> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[5],si32> -> tensor<5xi32>
// CHECK:           %[[PROMOTED:.*]] = stablehlo.convert %[[INPUT]] : (tensor<5xi32>) -> tensor<5xi64>
// CHECK:           %[[TRIL:.*]] = stablehlo.constant dense<
// CHECK:           %[[DOT:.*]] = stablehlo.dot_general %[[PROMOTED]], %[[TRIL]], contracting_dims = [0] x [0] : (tensor<5xi64>, tensor<5x5xi64>) -> tensor<5xi64>
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[DOT]] : tensor<5xi64> -> !torch.vtensor<[5],si64>
// CHECK:           return %[[OUT]] : !torch.vtensor<[5],si64>
// CHECK:         }
func.func @torch.aten.cumsum$static_int(%arg0: !torch.vtensor<[5],si32>) -> !torch.vtensor<[5],si64> {
    %int0 = torch.constant.int 0
    %none = torch.constant.none
    %0 = torch.aten.cumsum %arg0, %int0, %none : !torch.vtensor<[5],si32>, !torch.int, !torch.none -> !torch.vtensor<[5],si64>
    return %0 : !torch.vtensor<[5],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.cumsum$rank1_dynamic(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?],f32> -> tensor<?xf32>
// CHECK:           %[[DIM:.*]] = tensor.dim %[[INPUT]], %{{.*}} : tensor<?xf32>
// CHECK:           %[[DIMI64:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:           %[[SHAPE:.*]] = tensor.from_elements %[[DIMI64]], %[[DIMI64]] : tensor<2xi64>
// CHECK:           %[[ROW_IOTA:.*]] = stablehlo.dynamic_iota %[[SHAPE]], dim = 0 : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:           %[[COL_IOTA:.*]] = stablehlo.dynamic_iota %[[SHAPE]], dim = 1 : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:           %[[MASK:.*]] = stablehlo.compare LE, %[[ROW_IOTA]], %[[COL_IOTA]], SIGNED : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi1>
// CHECK:           %[[TRIL:.*]] = stablehlo.convert %[[MASK]] : (tensor<?x?xi1>) -> tensor<?x?xf32>
// CHECK:           %[[DOT:.*]] = stablehlo.dot_general %[[INPUT]], %[[TRIL]], contracting_dims = [0] x [0] : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?xf32>
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[DOT]] : tensor<?xf32> -> !torch.vtensor<[?],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?],f32>
// CHECK:         }
func.func @torch.aten.cumsum$rank1_dynamic(%arg0: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> {
    %int0 = torch.constant.int 0
    %none = torch.constant.none
    %0 = torch.aten.cumsum %arg0, %int0, %none : !torch.vtensor<[?],f32>, !torch.int, !torch.none -> !torch.vtensor<[?],f32>
    return %0 : !torch.vtensor<[?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.cumsum$rank3_middle_dim_dynamic(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[2,?,4],f32>) -> !torch.vtensor<[2,?,4],f32> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,?,4],f32> -> tensor<2x?x4xf32>
// CHECK:           %[[TRANSPOSED:.*]] = stablehlo.transpose %[[INPUT]], dims = [0, 2, 1] : (tensor<2x?x4xf32>) -> tensor<2x4x?xf32>
// CHECK:           %[[DIM:.*]] = tensor.dim %[[TRANSPOSED]], %{{.*}} : tensor<2x4x?xf32>
// CHECK:           %[[DIMI64:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:           %[[SHAPE:.*]] = tensor.from_elements %[[DIMI64]], %[[DIMI64]] : tensor<2xi64>
// CHECK:           %[[ROW_IOTA:.*]] = stablehlo.dynamic_iota %[[SHAPE]], dim = 0 : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:           %[[COL_IOTA:.*]] = stablehlo.dynamic_iota %[[SHAPE]], dim = 1 : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:           %[[MASK:.*]] = stablehlo.compare LE, %[[ROW_IOTA]], %[[COL_IOTA]], SIGNED : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi1>
// CHECK:           %[[TRIL:.*]] = stablehlo.convert %[[MASK]] : (tensor<?x?xi1>) -> tensor<?x?xf32>
// CHECK:           %[[DOT:.*]] = stablehlo.dot_general %[[TRANSPOSED]], %[[TRIL]], contracting_dims = [2] x [0] : (tensor<2x4x?xf32>, tensor<?x?xf32>) -> tensor<2x4x?xf32>
// CHECK:           %[[RESULT:.*]] = stablehlo.transpose %[[DOT]], dims = [0, 2, 1] : (tensor<2x4x?xf32>) -> tensor<2x?x4xf32>
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<2x?x4xf32> -> !torch.vtensor<[2,?,4],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[2,?,4],f32>
// CHECK:         }
func.func @torch.aten.cumsum$rank3_middle_dim_dynamic(%arg0: !torch.vtensor<[2,?,4],f32>) -> !torch.vtensor<[2,?,4],f32> {
    %int1 = torch.constant.int 1
    %none = torch.constant.none
    %0 = torch.aten.cumsum %arg0, %int1, %none : !torch.vtensor<[2,?,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[2,?,4],f32>
    return %0 : !torch.vtensor<[2,?,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.cumsum$dynamic_int(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[?],si32>) -> !torch.vtensor<[?],si64> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?],si32> -> tensor<?xi32>
// CHECK:           %[[PROMOTED:.*]] = stablehlo.convert %[[INPUT]] : (tensor<?xi32>) -> tensor<?xi64>
// CHECK:           %[[DIM:.*]] = tensor.dim %[[PROMOTED]], %{{.*}} : tensor<?xi64>
// CHECK:           %[[DIMI64:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:           %[[SHAPE:.*]] = tensor.from_elements %[[DIMI64]], %[[DIMI64]] : tensor<2xi64>
// CHECK:           %[[ROW_IOTA:.*]] = stablehlo.dynamic_iota %[[SHAPE]], dim = 0 : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:           %[[COL_IOTA:.*]] = stablehlo.dynamic_iota %[[SHAPE]], dim = 1 : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:           %[[MASK:.*]] = stablehlo.compare LE, %[[ROW_IOTA]], %[[COL_IOTA]], SIGNED : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi1>
// CHECK:           %[[TRIL:.*]] = stablehlo.convert %[[MASK]] : (tensor<?x?xi1>) -> tensor<?x?xi64>
// CHECK:           %[[DOT:.*]] = stablehlo.dot_general %[[PROMOTED]], %[[TRIL]], contracting_dims = [0] x [0] : (tensor<?xi64>, tensor<?x?xi64>) -> tensor<?xi64>
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[DOT]] : tensor<?xi64> -> !torch.vtensor<[?],si64>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?],si64>
// CHECK:         }
func.func @torch.aten.cumsum$dynamic_int(%arg0: !torch.vtensor<[?],si32>) -> !torch.vtensor<[?],si64> {
    %int0 = torch.constant.int 0
    %none = torch.constant.none
    %0 = torch.aten.cumsum %arg0, %int0, %none : !torch.vtensor<[?],si32>, !torch.int, !torch.none -> !torch.vtensor<[?],si64>
    return %0 : !torch.vtensor<[?],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.cumsum$dtype_promotion(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[4],f16>) -> !torch.vtensor<[4],f32> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4],f16> -> tensor<4xf16>
// CHECK:           %[[PROMOTED:.*]] = stablehlo.convert %[[INPUT]] : (tensor<4xf16>) -> tensor<4xf32>
// CHECK:           %[[TRIL:.*]] = stablehlo.constant dense<
// CHECK:           %[[DOT:.*]] = stablehlo.dot_general %[[PROMOTED]], %[[TRIL]], contracting_dims = [0] x [0] : (tensor<4xf32>, tensor<4x4xf32>) -> tensor<4xf32>
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[DOT]] : tensor<4xf32> -> !torch.vtensor<[4],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[4],f32>
// CHECK:         }
func.func @torch.aten.cumsum$dtype_promotion(%arg0: !torch.vtensor<[4],f16>) -> !torch.vtensor<[4],f32> {
    %int0 = torch.constant.int 0
    %int6 = torch.constant.int 6
    %0 = torch.aten.cumsum %arg0, %int0, %int6 : !torch.vtensor<[4],f16>, !torch.int, !torch.int -> !torch.vtensor<[4],f32>
    return %0 : !torch.vtensor<[4],f32>
}

// -----
// CHECK-LABEL:   func.func @aten.im2col$dynamic_spatial(
// CHECK-SAME:      %[[VAL_0:.*]]: !torch.vtensor<[1,3,?,?],f32>) -> !torch.vtensor<[1,27,?],f32> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,3,?,?],f32> -> tensor<1x3x?x?xf32>
// CHECK:           stablehlo.pad
// CHECK:           stablehlo.dynamic_iota
// CHECK:           stablehlo.dynamic_iota
// CHECK:           stablehlo.dynamic_broadcast_in_dim
// CHECK:           stablehlo.dynamic_broadcast_in_dim
// CHECK:           stablehlo.dynamic_broadcast_in_dim
// CHECK:           stablehlo.dynamic_broadcast_in_dim
// CHECK:           chlo.broadcast_multiply
// CHECK:           chlo.broadcast_multiply
// CHECK:           chlo.broadcast_add
// CHECK:           chlo.broadcast_multiply
// CHECK:           chlo.broadcast_multiply
// CHECK:           chlo.broadcast_add
// CHECK:           stablehlo.dynamic_reshape
// CHECK:           stablehlo.dynamic_reshape
// CHECK:           stablehlo.concatenate
// CHECK:           "stablehlo.dynamic_gather"
// CHECK:           stablehlo.dynamic_reshape
// CHECK:         }
func.func @aten.im2col$dynamic_spatial(%arg0: !torch.vtensor<[1,3,?,?],f32>) -> !torch.vtensor<[1,27,?],f32> {
    %int3 = torch.constant.int 3
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.aten.im2col %arg0, %0, %1, %2, %3 : !torch.vtensor<[1,3,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.vtensor<[1,27,?],f32>
    return %4 : !torch.vtensor<[1,27,?],f32>
}

// -----

// CHECK-LABEL:   func.func @aten.im2col$dynamic_spatial_with_padding(
// CHECK-SAME:      %[[VAL_0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:           stablehlo.pad
// CHECK:           stablehlo.dynamic_iota
// CHECK:           stablehlo.dynamic_iota
// CHECK:           stablehlo.dynamic_broadcast_in_dim
// CHECK:           "stablehlo.dynamic_gather"
// CHECK:           stablehlo.dynamic_reshape
// CHECK:         }
func.func @aten.im2col$dynamic_spatial_with_padding(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.aten.im2col %arg0, %0, %1, %2, %3 : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.vtensor<[?,?,?],f32>
    return %4 : !torch.vtensor<[?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @aten.im2col$dynamic_spatial_stride2(
// CHECK-SAME:      %[[VAL_0:.*]]: !torch.vtensor<[1,4,?,?],f32>) -> !torch.vtensor<[1,16,?],f32> {
// CHECK:           stablehlo.dynamic_iota
// CHECK:           stablehlo.dynamic_iota
// CHECK:           stablehlo.dynamic_broadcast_in_dim
// CHECK:           chlo.broadcast_multiply
// CHECK:           chlo.broadcast_add
// CHECK:           "stablehlo.dynamic_gather"
// CHECK:           stablehlo.dynamic_reshape
// CHECK:         }
func.func @aten.im2col$dynamic_spatial_stride2(%arg0: !torch.vtensor<[1,4,?,?],f32>) -> !torch.vtensor<[1,16,?],f32> {
    %int2 = torch.constant.int 2
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.aten.im2col %arg0, %0, %1, %2, %3 : !torch.vtensor<[1,4,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.vtensor<[1,16,?],f32>
    return %4 : !torch.vtensor<[1,16,?],f32>
}

// -----

// Verify MaxPool2dWithIndices with dynamic spatial dims lowers via ReduceWindow
// (the TorchToNeuron decomposition bails out, so it falls through to TorchToStablehlo).
// CHECK-LABEL:   func.func @torch.aten.max_pool2d_with_indices$dynamic_spatial(
// CHECK-SAME:        %[[ARG0:.*]]: !torch.vtensor<[1,1,?,?],f32>) -> (!torch.vtensor<[1,1,?,?],f32>, !torch.vtensor<[1,1,?,?],si64>) {
// CHECK:           stablehlo.reduce_window
// CHECK:         }
func.func @torch.aten.max_pool2d_with_indices$dynamic_spatial(%arg0: !torch.vtensor<[1,1,?,?],f32>) -> (!torch.vtensor<[1,1,?,?],f32>, !torch.vtensor<[1,1,?,?],si64>) {
    %int2 = torch.constant.int 2
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %result0, %result1 = torch.aten.max_pool2d_with_indices %arg0, %0, %1, %2, %3, %false : !torch.vtensor<[1,1,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,?,?],f32>, !torch.vtensor<[1,1,?,?],si64>
    return %result0, %result1 : !torch.vtensor<[1,1,?,?],f32>, !torch.vtensor<[1,1,?,?],si64>
}

// -----


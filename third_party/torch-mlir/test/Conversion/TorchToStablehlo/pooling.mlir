// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL:   func.func @torch.aten.max_pool2d(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %int2 = torch.constant.int 2
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %false = torch.constant.bool false
// CHECK:           %[[VAL_2:.*]] = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_3:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_6:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           %[[VAL_7:.*]] = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_6]]) <{padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 2, 1>, window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 1, 1>}> ({
// CHECK:           ^bb0(%[[VAL_8:.*]]: tensor<f32>, %[[VAL_9:.*]]: tensor<f32>):
// CHECK:             %[[VAL_10:.*]] = stablehlo.maximum %[[VAL_8]], %[[VAL_9]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_10]] : tensor<f32>
// CHECK:           }) : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_11:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[VAL_11]] : !torch.vtensor<[?,?,?,?],f32>
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
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %int2 = torch.constant.int 2
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %false = torch.constant.bool false
// CHECK:           %[[VAL_2:.*]] = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_3:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_5]])
// CHECK{LITERAL}:    <{padding = dense<[[0, 0], [0, 0], [2, 2], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 2, 1>, window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 1, 1>}> ({
// CHECK:           ^bb0(%[[VAL_8:.*]]: tensor<f32>, %[[VAL_9:.*]]: tensor<f32>):
// CHECK:             %[[VAL_10:.*]] = stablehlo.maximum %[[VAL_8]], %[[VAL_9]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_10]] : tensor<f32>
// CHECK:           }) : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[?,?,?,?],f32>
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
// CHECK-SAME:                                      %[[VAL_0:.*]]: !torch.vtensor<[1,256,56,56],f32>) -> !torch.vtensor<[1,256,27,27],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,256,56,56],f32> -> tensor<1x256x56x56xf32>
// CHECK:           %int3 = torch.constant.int 3
// CHECK:           %int2 = torch.constant.int 2
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %false = torch.constant.bool false
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %[[VAL_2:.*]] = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_3:.*]] = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_5]])
// CHECK{LITERAL}:    <{padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
// CHECK:           ^bb0(%[[VAL_8:.*]]: tensor<f32>, %[[VAL_9:.*]]: tensor<f32>):
// CHECK:             %[[VAL_10:.*]] = stablehlo.maximum %[[VAL_8]], %[[VAL_9]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_10]] : tensor<f32>
// CHECK:           }) : (tensor<1x256x56x56xf32>, tensor<f32>) -> tensor<1x256x27x27xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<1x256x27x27xf32> -> !torch.vtensor<[1,256,27,27],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[1,256,27,27],f32>
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
// CHECK-SAME:                                      %[[VAL_0:.*]]: !torch.vtensor<[1,256,56,56],f32>) -> !torch.vtensor<[1,256,28,28],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,256,56,56],f32> -> tensor<1x256x56x56xf32>
// CHECK:           %int3 = torch.constant.int 3
// CHECK:           %int2 = torch.constant.int 2
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %true = torch.constant.bool true
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %[[VAL_2:.*]] = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_3:.*]] = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_5]])
// CHECK{LITERAL}:    <{padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
// CHECK:           ^bb0(%[[VAL_8:.*]]: tensor<f32>, %[[VAL_9:.*]]: tensor<f32>):
// CHECK:             %[[VAL_10:.*]] = stablehlo.maximum %[[VAL_8]], %[[VAL_9]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_10]] : tensor<f32>
// CHECK:           }) : (tensor<1x256x56x56xf32>, tensor<f32>) -> tensor<1x256x28x28xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<1x256x28x28xf32> -> !torch.vtensor<[1,256,28,28],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[1,256,28,28],f32>
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

// CHECK-LABEL:  func.func @torch.aten.max_pool2d_with_indices(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?],f32>) -> (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],si64>) {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:         %[[INT3:.*]] = torch.constant.int 3
// CHECK:         %[[INT2:.*]] = torch.constant.int 2
// CHECK:         %[[FALSE:.*]] = torch.constant.bool false
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[T1:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[T2:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[T3:.*]] = torch.prim.ListConstruct %[[INT0]], %[[INT0]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[T4:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[T5:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?x?xf32>
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM_0:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?x?xf32>
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[DIM_1:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x?x?xf32>
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[DIM]], %[[DIM_0]], %[[DIM_1]] : tensor<3xindex>
// CHECK:         %[[T9:.*]] = arith.muli %[[DIM_1]], %[[DIM_0]] : index
// CHECK:         %[[FROM_ELEMENTS_2:.*]] = tensor.from_elements %[[DIM]], %[[T9]] : tensor<2xindex>
// CHECK:         %[[T10:.*]] = stablehlo.dynamic_iota %[[FROM_ELEMENTS_2]], dim = 1 : (tensor<2xindex>) -> tensor<?x?xi64>
// CHECK:         %[[T11:.*]] = stablehlo.dynamic_reshape %[[T10]], %[[FROM_ELEMENTS]] : (tensor<?x?xi64>, tensor<3xindex>) -> tensor<?x?x?xi64>
// CHECK:         %[[T12:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:         %[[T13:.*]]:2 = "stablehlo.reduce_window"(%[[T0]], %[[T11]], %[[T5]], %[[T12]]) <{padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3>, window_strides = array<i64: 1, 2, 2>}> ({
// CHECK:         ^bb0(%[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<i64>, %[[ARG3:.*]]: tensor<f32>, %[[ARG4:.*]]: tensor<i64>):
// CHECK:         %[[T16:.*]] = stablehlo.compare  GE, %[[ARG1]], %[[ARG3]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:         %[[T17:.*]] = stablehlo.select %[[T16]], %[[ARG1]], %[[ARG3]] : tensor<i1>, tensor<f32>
// CHECK:         %[[T18:.*]] = stablehlo.compare  EQ, %[[ARG1]], %[[ARG3]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:         %[[T19:.*]] = stablehlo.minimum %[[ARG2]], %[[ARG4]] : tensor<i64>
// CHECK:         %[[T20:.*]] = stablehlo.select %[[T16]], %[[ARG2]], %[[ARG4]] : tensor<i1>, tensor<i64>
// CHECK:         %[[T21:.*]] = stablehlo.select %[[T18]], %[[T19]], %[[T20]] : tensor<i1>, tensor<i64>
// CHECK:         stablehlo.return %[[T17]], %[[T21]] : tensor<f32>, tensor<i64>
// CHECK:         }) : (tensor<?x?x?xf32>, tensor<?x?x?xi64>, tensor<f32>, tensor<i64>) -> (tensor<?x?x?xf32>, tensor<?x?x?xi64>)
// CHECK-DAG:         %[[T14:.*]] = torch_c.from_builtin_tensor %[[T13]]#0 : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
// CHECK-DAG:         %[[T15:.*]] = torch_c.from_builtin_tensor %[[T13]]#1 : tensor<?x?x?xi64> -> !torch.vtensor<[?,?,?],si64>
// CHECK:         return %[[T14]], %[[T15]] : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],si64>
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
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %int3 = torch.constant.int 3
// CHECK:           %int2 = torch.constant.int 2
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %false = torch.constant.bool false
// CHECK:           %none = torch.constant.none
// CHECK:           %[[VAL_2:.*]] = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_3:.*]] = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_5]])
// CHECK{LITERAL}:    <{padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
// CHECK:           ^bb0(%[[IVAL_0:.*]]: tensor<f32>, %[[IVAL_1:.*]]: tensor<f32>):
// CHECK:             %[[IVAL_2:.*]] = stablehlo.add %[[IVAL_0]], %[[IVAL_1]] : tensor<f32>
// CHECK:             stablehlo.return %[[IVAL_2]] : tensor<f32>
// CHECK:           }) : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[IDX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = tensor.dim %[[VAL_1]], %[[IDX_0]] : tensor<?x?x?x?xf32>
// CHECK:           %[[IDX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_10:.*]] = tensor.dim %[[VAL_1]], %[[IDX_1]] : tensor<?x?x?x?xf32>
// CHECK:           %[[IDX_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_12:.*]] = tensor.dim %[[VAL_1]], %[[IDX_2]] : tensor<?x?x?x?xf32>
// CHECK:           %[[IDX_3:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_14:.*]] = tensor.dim %[[VAL_1]], %[[IDX_3]] : tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_16:.*]] = tensor.from_elements %[[VAL_8]], %[[VAL_10]], %[[VAL_12]], %[[VAL_14]] : tensor<4xindex>
// CHECK:           %[[VAL_17:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_7]], %[[VAL_16]], dims = [] : (tensor<f32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_18:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_19:.*]] = "stablehlo.reduce_window"(%[[VAL_17]], %[[VAL_18]])
// CHECK{LITERAL}:    <{padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
// CHECK:           ^bb0(%[[IVAL_3:.*]]: tensor<f32>, %[[IVAL_4:.*]]: tensor<f32>):
// CHECK:             %[[IVAL_5:.*]] = stablehlo.add %[[IVAL_3]], %[[IVAL_4]] : tensor<f32>
// CHECK:             stablehlo.return %[[IVAL_5]] : tensor<f32>
// CHECK:           }) : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_20:.*]] = stablehlo.divide %[[VAL_6]], %[[VAL_19]] : tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_21:.*]] = torch_c.from_builtin_tensor %[[VAL_20]] : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[VAL_21]] : !torch.vtensor<[?,?,?,?],f32>
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

// CHECK-LABEL:  func.func @torch.aten.avg_pool2d$count_include_pad(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[INT3:.*]] = torch.constant.int 3
// CHECK:           %[[INT2:.*]] = torch.constant.int 2
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[T1:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T2:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T3:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T4:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[T5:.*]] = "stablehlo.reduce_window"(%[[T0]], %[[T4]])
// CHECK{LITERAL}:    <{padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
// CHECK:           ^bb0(%[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<f32>):
// CHECK:             %[[T10:.*]] = stablehlo.add %[[ARG1]], %[[ARG2]] : tensor<f32>
// CHECK:             stablehlo.return %[[T10]] : tensor<f32>
// CHECK:           }) : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[T6:.*]] = stablehlo.constant dense<9> : tensor<i64>
// CHECK:           %[[T7:.*]] = stablehlo.convert %[[T6]] : (tensor<i64>) -> tensor<f32>
// CHECK:           %[[T8:.*]] = chlo.broadcast_divide %[[T5]], %[[T7]] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[T9:.*]] = torch_c.from_builtin_tensor %[[T8]] : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[T9]] : !torch.vtensor<[?,?,?,?],f32>
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
// CHECK-SAME:                    %[[VAL_0:.*]]: !torch.vtensor<[1,3,16,16],f32>,
// CHECK-SAME:                    %[[VAL_1:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[1,3,16,16],f32> {
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
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[1,1,4,4],si64> -> tensor<1x1x4x4xi64>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,1,8,8],f32> -> tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,4,4],f32> -> tensor<1x1x4x4xf32>
// CHECK:           %[[VAL_6:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_9]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_9]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_12:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_8]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_13:.*]] = torch.prim.ListConstruct %[[VAL_7]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_14:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_15:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_16:.*]] = tensor.dim %[[VAL_4]], %[[VAL_15]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_17:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_18:.*]] = tensor.dim %[[VAL_4]], %[[VAL_17]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_19:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_20:.*]] = tensor.dim %[[VAL_4]], %[[VAL_19]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_21:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_22:.*]] = tensor.dim %[[VAL_4]], %[[VAL_21]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_23:.*]] = tensor.from_elements %[[VAL_16]], %[[VAL_18]], %[[VAL_20]], %[[VAL_22]] : tensor<4xindex>
// CHECK:           %[[VAL_24:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_14]], %[[VAL_23]], dims = [] : (tensor<f32>, tensor<4xindex>) -> tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_25:.*]] = stablehlo.reshape %[[VAL_5]] : (tensor<1x1x4x4xf32>) -> tensor<1x1x16xf32>
// CHECK:           %[[VAL_26:.*]] = stablehlo.reshape %[[VAL_3]] : (tensor<1x1x4x4xi64>) -> tensor<1x1x16x1xi64>
// CHECK:           %[[VAL_27:.*]] = stablehlo.reshape %[[VAL_24]] : (tensor<1x1x8x8xf32>) -> tensor<1x1x64xf32>
// CHECK:           %[[VAL_28:.*]] = "stablehlo.scatter"(%[[VAL_27]], %[[VAL_26]], %[[VAL_25]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [2], input_batching_dims = [0, 1], scatter_indices_batching_dims = [0, 1], scatter_dims_to_operand_dims = [2], index_vector_dim = 3>, unique_indices = false}> ({
// CHECK:           ^bb0(%[[VAL_29:.*]]: tensor<f32>, %[[VAL_30:.*]]: tensor<f32>):
// CHECK:             %[[VAL_31:.*]] = stablehlo.add %[[VAL_29]], %[[VAL_30]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_31]] : tensor<f32>
// CHECK:           }) : (tensor<1x1x64xf32>, tensor<1x1x16x1xi64>, tensor<1x1x16xf32>) -> tensor<1x1x64xf32>
// CHECK:           %[[VAL_32:.*]] = stablehlo.reshape %[[VAL_33:.*]] : (tensor<1x1x64xf32>) -> tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_34:.*]] = torch_c.from_builtin_tensor %[[VAL_32]] : tensor<1x1x8x8xf32> -> !torch.vtensor<[1,1,8,8],f32>
// CHECK:           return %[[VAL_34]] : !torch.vtensor<[1,1,8,8],f32>
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
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[1,1,6,7],si64> -> tensor<1x1x6x7xi64>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,1,8,8],f32> -> tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,6,7],f32> -> tensor<1x1x6x7xf32>
// CHECK:           %[[VAL_6:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_10:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_10]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_12:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_8]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_13:.*]] = torch.prim.ListConstruct %[[VAL_7]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_14:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_8]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_15:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_16:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_17:.*]] = tensor.dim %[[VAL_4]], %[[VAL_16]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_18:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_19:.*]] = tensor.dim %[[VAL_4]], %[[VAL_18]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_20:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_21:.*]] = tensor.dim %[[VAL_4]], %[[VAL_20]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_22:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_23:.*]] = tensor.dim %[[VAL_4]], %[[VAL_22]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_24:.*]] = tensor.from_elements %[[VAL_17]], %[[VAL_19]], %[[VAL_21]], %[[VAL_23]] : tensor<4xindex>
// CHECK:           %[[VAL_25:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_15]], %[[VAL_24]], dims = [] : (tensor<f32>, tensor<4xindex>) -> tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_26:.*]] = stablehlo.reshape %[[VAL_5]] : (tensor<1x1x6x7xf32>) -> tensor<1x1x42xf32>
// CHECK:           %[[VAL_27:.*]] = stablehlo.reshape %[[VAL_3]] : (tensor<1x1x6x7xi64>) -> tensor<1x1x42x1xi64>
// CHECK:           %[[VAL_28:.*]] = stablehlo.reshape %[[VAL_25]] : (tensor<1x1x8x8xf32>) -> tensor<1x1x64xf32>
// CHECK:           %[[VAL_29:.*]] = "stablehlo.scatter"(%[[VAL_28]], %[[VAL_27]], %[[VAL_26]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [2], input_batching_dims = [0, 1], scatter_indices_batching_dims = [0, 1], scatter_dims_to_operand_dims = [2], index_vector_dim = 3>, unique_indices = false}> ({
// CHECK:           ^bb0(%[[VAL_30:.*]]: tensor<f32>, %[[VAL_31:.*]]: tensor<f32>):
// CHECK:             %[[VAL_32:.*]] = stablehlo.add %[[VAL_30]], %[[VAL_31]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_32]] : tensor<f32>
// CHECK:           }) : (tensor<1x1x64xf32>, tensor<1x1x42x1xi64>, tensor<1x1x42xf32>) -> tensor<1x1x64xf32>
// CHECK:           %[[VAL_33:.*]] = stablehlo.reshape %[[VAL_34:.*]] : (tensor<1x1x64xf32>) -> tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_35:.*]] = torch_c.from_builtin_tensor %[[VAL_33]] : tensor<1x1x8x8xf32> -> !torch.vtensor<[1,1,8,8],f32>
// CHECK:           return %[[VAL_35]] : !torch.vtensor<[1,1,8,8],f32>
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
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[1,1,15,17],si64> -> tensor<1x1x15x17xi64>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,1,16,16],f32> -> tensor<1x1x16x16xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,15,17],f32> -> tensor<1x1x15x17xf32>
// CHECK:           %[[VAL_6:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_9]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_8]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_12:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_13:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_8]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_14:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_15:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_16:.*]] = tensor.dim %[[VAL_4]], %[[VAL_15]] : tensor<1x1x16x16xf32>
// CHECK:           %[[VAL_17:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_18:.*]] = tensor.dim %[[VAL_4]], %[[VAL_17]] : tensor<1x1x16x16xf32>
// CHECK:           %[[VAL_19:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_20:.*]] = tensor.dim %[[VAL_4]], %[[VAL_19]] : tensor<1x1x16x16xf32>
// CHECK:           %[[VAL_21:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_22:.*]] = tensor.dim %[[VAL_4]], %[[VAL_21]] : tensor<1x1x16x16xf32>
// CHECK:           %[[VAL_23:.*]] = tensor.from_elements %[[VAL_16]], %[[VAL_18]], %[[VAL_20]], %[[VAL_22]] : tensor<4xindex>
// CHECK:           %[[VAL_24:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_14]], %[[VAL_23]], dims = [] : (tensor<f32>, tensor<4xindex>) -> tensor<1x1x16x16xf32>
// CHECK:           %[[VAL_25:.*]] = stablehlo.reshape %[[VAL_5]] : (tensor<1x1x15x17xf32>) -> tensor<1x1x255xf32>
// CHECK:           %[[VAL_26:.*]] = stablehlo.reshape %[[VAL_3]] : (tensor<1x1x15x17xi64>) -> tensor<1x1x255x1xi64>
// CHECK:           %[[VAL_27:.*]] = stablehlo.reshape %[[VAL_24]] : (tensor<1x1x16x16xf32>) -> tensor<1x1x256xf32>
// CHECK:           %[[VAL_28:.*]] = "stablehlo.scatter"(%[[VAL_27]], %[[VAL_26]], %[[VAL_25]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [2], input_batching_dims = [0, 1], scatter_indices_batching_dims = [0, 1], scatter_dims_to_operand_dims = [2], index_vector_dim = 3>, unique_indices = false}> ({
// CHECK:           ^bb0(%[[VAL_29:.*]]: tensor<f32>, %[[VAL_30:.*]]: tensor<f32>):
// CHECK:             %[[VAL_31:.*]] = stablehlo.add %[[VAL_29]], %[[VAL_30]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_31]] : tensor<f32>
// CHECK:           }) : (tensor<1x1x256xf32>, tensor<1x1x255x1xi64>, tensor<1x1x255xf32>) -> tensor<1x1x256xf32>
// CHECK:           %[[VAL_32:.*]] = stablehlo.reshape %[[VAL_33:.*]] : (tensor<1x1x256xf32>) -> tensor<1x1x16x16xf32>
// CHECK:           %[[VAL_34:.*]] = torch_c.from_builtin_tensor %[[VAL_32]] : tensor<1x1x16x16xf32> -> !torch.vtensor<[1,1,16,16],f32>
// CHECK:           return %[[VAL_34]] : !torch.vtensor<[1,1,16,16],f32>
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
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[1,1,4,3],si64> -> tensor<1x1x4x3xi64>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,1,8,8],f32> -> tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,4,3],f32> -> tensor<1x1x4x3xf32>
// CHECK:           %[[VAL_6:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_10:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_10]], %[[VAL_10]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_12:.*]] = torch.prim.ListConstruct %[[VAL_10]], %[[VAL_9]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_13:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_8]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_14:.*]] = torch.prim.ListConstruct %[[VAL_7]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_15:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_16:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_17:.*]] = tensor.dim %[[VAL_4]], %[[VAL_16]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_18:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_19:.*]] = tensor.dim %[[VAL_4]], %[[VAL_18]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_20:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_21:.*]] = tensor.dim %[[VAL_4]], %[[VAL_20]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_22:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_23:.*]] = tensor.dim %[[VAL_4]], %[[VAL_22]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_24:.*]] = tensor.from_elements %[[VAL_17]], %[[VAL_19]], %[[VAL_21]], %[[VAL_23]] : tensor<4xindex>
// CHECK:           %[[VAL_25:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_15]], %[[VAL_24]], dims = [] : (tensor<f32>, tensor<4xindex>) -> tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_26:.*]] = stablehlo.reshape %[[VAL_5]] : (tensor<1x1x4x3xf32>) -> tensor<1x1x12xf32>
// CHECK:           %[[VAL_27:.*]] = stablehlo.reshape %[[VAL_3]] : (tensor<1x1x4x3xi64>) -> tensor<1x1x12x1xi64>
// CHECK:           %[[VAL_28:.*]] = stablehlo.reshape %[[VAL_25]] : (tensor<1x1x8x8xf32>) -> tensor<1x1x64xf32>
// CHECK:           %[[VAL_29:.*]] = "stablehlo.scatter"(%[[VAL_28]], %[[VAL_27]], %[[VAL_26]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [2], input_batching_dims = [0, 1], scatter_indices_batching_dims = [0, 1], scatter_dims_to_operand_dims = [2], index_vector_dim = 3>, unique_indices = false}> ({
// CHECK:           ^bb0(%[[VAL_30:.*]]: tensor<f32>, %[[VAL_31:.*]]: tensor<f32>):
// CHECK:             %[[VAL_32:.*]] = stablehlo.add %[[VAL_30]], %[[VAL_31]] : tensor<f32>
// CHECK:             stablehlo.return %[[VAL_32]] : tensor<f32>
// CHECK:           }) : (tensor<1x1x64xf32>, tensor<1x1x12x1xi64>, tensor<1x1x12xf32>) -> tensor<1x1x64xf32>
// CHECK:           %[[VAL_33:.*]] = stablehlo.reshape %[[VAL_34:.*]] : (tensor<1x1x64xf32>) -> tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_35:.*]] = torch_c.from_builtin_tensor %[[VAL_33]] : tensor<1x1x8x8xf32> -> !torch.vtensor<[1,1,8,8],f32>
// CHECK:           return %[[VAL_35]] : !torch.vtensor<[1,1,8,8],f32>
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
// CHECK:           %[[VAL_10:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           %[[VAL_11:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_12:.*]] = tensor.dim %[[VAL_1]], %[[VAL_11]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_13:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_14:.*]] = tensor.dim %[[VAL_1]], %[[VAL_13]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_15:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_16:.*]] = tensor.dim %[[VAL_1]], %[[VAL_15]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_17:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_18:.*]] = tensor.dim %[[VAL_1]], %[[VAL_17]] : tensor<1x1x8x8xf32>
// CHECK:           %[[VAL_19:.*]] = tensor.from_elements %[[VAL_12]], %[[VAL_14]], %[[VAL_16]], %[[VAL_18]] : tensor<4xindex>
// CHECK:           %[[VAL_20:.*]] = arith.muli %[[VAL_18]], %[[VAL_16]] : index
// CHECK:           %[[VAL_21:.*]] = tensor.from_elements %[[VAL_12]], %[[VAL_14]], %[[VAL_20]] : tensor<3xindex>
// CHECK:           %[[VAL_22:.*]] = stablehlo.dynamic_iota %[[VAL_21]], dim = 2 : (tensor<3xindex>) -> tensor<1x1x64xi64>
// CHECK:           %[[VAL_23:.*]] = stablehlo.dynamic_reshape %[[VAL_22]], %[[VAL_19]] : (tensor<1x1x64xi64>, tensor<4xindex>) -> tensor<1x1x8x8xi64>
// CHECK:           %[[VAL_24:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_25:.*]]:2 = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_23]], %[[VAL_10]], %[[VAL_24]]) <{padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}> ({
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

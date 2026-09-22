// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s


// -----

// CHECK-LABEL:   func.func @torch.vtensor.literal$basic() -> !torch.vtensor<[],f32> {
// CHECK:           %[[VAL_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:           return %[[VAL_1]] : !torch.vtensor<[],f32>
func.func @torch.vtensor.literal$basic() -> !torch.vtensor<[],f32> {
  %0 = torch.vtensor.literal(dense<0.0> : tensor<f32>) : !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.vtensor.literal$signed() -> !torch.vtensor<[2],si64> {
// CHECK:           %[[VAL_0:.*]] = stablehlo.constant dense<1> : tensor<2xi64>
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<2xi64> -> !torch.vtensor<[2],si64>
// CHECK:           return %[[VAL_1]] : !torch.vtensor<[2],si64>
func.func @torch.vtensor.literal$signed() -> !torch.vtensor<[2],si64> {
  %0 = torch.vtensor.literal(dense<1> : tensor<2xsi64>) : !torch.vtensor<[2],si64>
  return %0 : !torch.vtensor<[2],si64>
}

// -----

// CHECK-LABEL:  func.func @torch.prim.NumToTensor.Scalar$basic(
// CHECK-SAME:         ) -> !torch.vtensor<[],si64> {
// CHECK:         %[[CST:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK:         %[[FROM:.*]] = torch_c.from_builtin_tensor %[[CST]] : tensor<i64> -> !torch.vtensor<[],si64>
// CHECK:         return %[[FROM]] : !torch.vtensor<[],si64>
func.func @torch.prim.NumToTensor.Scalar$basic() -> !torch.vtensor<[], si64> {
  %int1 = torch.constant.int 1
  %0 = torch.prim.NumToTensor.Scalar %int1 : !torch.int -> !torch.vtensor<[], si64>
  return %0 : !torch.vtensor<[], si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.contiguous(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           return %[[VAL_0]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.contiguous(%arg0: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.contiguous %arg0, %int0 : !torch.vtensor<[4,64],f32>, !torch.int -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.reciprocal(
// CHECK-SAME:                                             %[[VAL_0:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:           %[[VAL_2:.*]] = "chlo.constant_like"(%[[VAL_1]]) <{value = 1.000000e+00 : f32}> : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.divide %[[VAL_2]], %[[VAL_1]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[?,?,?],f32>
func.func @torch.aten.reciprocal(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %0 = torch.aten.reciprocal %arg0 : !torch.vtensor<[?,?,?],f32> -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}


// -----

// CHECK-LABEL:   func.func @torch.aten.reciprocal$integer(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: !torch.vtensor<[?,?,?],si64>) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?],si64> -> tensor<?x?x?xi64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.convert %[[VAL_1]] : (tensor<?x?x?xi64>) -> tensor<?x?x?xf32>
// CHECK:           %[[VAL_3:.*]] = "chlo.constant_like"(%[[VAL_2]]) <{value = 1.000000e+00 : f32}> : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.divide %[[VAL_3]], %[[VAL_2]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?,?],f32>
func.func @torch.aten.reciprocal$integer(%arg0: !torch.vtensor<[?,?,?],si64>) -> !torch.vtensor<[?,?,?],f32> {
  %0 = torch.aten.reciprocal %arg0 : !torch.vtensor<[?,?,?],si64> -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}


// -----

// CHECK-LABEL:   func.func @torch.aten.transpose$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = stablehlo.transpose %[[VAL_1]], dims = [1, 0] : (tensor<4x3xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[3,4],f32>
func.func @torch.aten.transpose$basic(%arg0: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[3,4],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.aten.transpose.int %arg0, %int0, %int1 : !torch.vtensor<[4,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}


// -----

// CHECK-LABEL:   func.func @torch.aten.broadcast_to$dynamic_implicit(
// CHECK-SAME:                                             %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[8,4,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %int-1 = torch.constant.int -1
// CHECK:           %int4 = torch.constant.int 4
// CHECK:           %int8 = torch.constant.int 8
// CHECK:           %[[VAL_2:.*]] = torch.prim.ListConstruct %int8, %int4, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_i64 %int8
// CHECK:           %[[VAL_4:.*]] = arith.index_cast %[[VAL_3:.*]] : i64 to index
// CHECK:           %[[VAL_5:.*]] = torch_c.to_i64 %int4
// CHECK:           %[[VAL_6:.*]] = arith.index_cast %[[VAL_5]] : i64 to index
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_8:.*]] = tensor.dim %[[VAL_1:.*]], %[[VAL_7]] : tensor<?x?xf32>
// CHECK:           %[[VAL_9:.*]] = tensor.from_elements %[[VAL_4]], %[[VAL_6]], %[[VAL_8]] : tensor<3xindex>
// CHECK:           %[[VAL_10:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_1]], %[[VAL_9]], dims = [1, 2] : (tensor<?x?xf32>, tensor<3xindex>) -> tensor<8x4x?xf32>
// CHECK:           %[[VAL_11:.*]] = torch_c.from_builtin_tensor %[[VAL_10]] : tensor<8x4x?xf32> -> !torch.vtensor<[8,4,?],f32>
// CHECK:           return %[[VAL_11]] : !torch.vtensor<[8,4,?],f32>
func.func @torch.aten.broadcast_to$dynamic_implicit(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[8,4,?],f32> {
  %int-1 = torch.constant.int -1
  %int4 = torch.constant.int 4
  %int8 = torch.constant.int 8
  %0 = torch.prim.ListConstruct %int8, %int4, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.broadcast_to %arg0, %0 : !torch.vtensor<[?,?],f32>, !torch.list<int> -> !torch.vtensor<[8,4,?],f32>
  return %1 : !torch.vtensor<[8,4,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.batch_norm$training(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !torch.vtensor<[?,3,?,?],f32>) -> !torch.vtensor<[?,3,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,3,?,?],f32> -> tensor<?x3x?x?xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK:           %true = torch.constant.bool true
// CHECK:           %float1.000000e-01 = torch.constant.float 1.000000e-01
// CHECK:           %float1.000000e-05 = torch.constant.float 1.000000e-05
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = tensor.dim %[[VAL_1]], %[[VAL_4]] : tensor<?x3x?x?xf32>
// CHECK:           %[[VAL_6:.*]] = tensor.from_elements %[[VAL_5]] : tensor<1xindex>
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]], %[[VAL_9:.*]] = "stablehlo.batch_norm_training"(%[[VAL_1]], %[[VAL_3]], %[[VAL_2]]) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<?x3x?x?xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<?x3x?x?xf32>, tensor<3xf32>, tensor<3xf32>)
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<?x3x?x?xf32> -> !torch.vtensor<[?,3,?,?],f32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[?,3,?,?],f32>
func.func @torch.aten.batch_norm$training(%arg0: !torch.vtensor<[?,3,?,?],f32>) -> !torch.vtensor<[?,3,?,?],f32> {
  %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %true = torch.constant.bool true
  %float1.000000e-01 = torch.constant.float 1.000000e-01
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %2 = torch.aten.batch_norm %arg0, %1, %0, %0, %1, %true, %float1.000000e-01, %float1.000000e-05, %true : !torch.vtensor<[?,3,?,?],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[?,3,?,?],f32>
  return %2 : !torch.vtensor<[?,3,?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.batch_norm$inference(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,3,?,?],f32>) -> !torch.vtensor<[?,3,?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,3,?,?],f32> -> tensor<?x3x?x?xf32>
// CHECK:         %[[T1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
// CHECK:         %[[T2:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK:         %[[TRUE:.*]] = torch.constant.bool true
// CHECK:         %[[FALSE:.*]] = torch.constant.bool false
// CHECK:         %[[FLOAT1:.*]].000000e-01 = torch.constant.float 1.000000e-01
// CHECK:         %[[FLOAT1:.*]].000000e-05 = torch.constant.float 1.000000e-05
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T3:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x3x?x?xf32>
// CHECK:         %[[T4:.*]] = tensor.from_elements %[[T3]] : tensor<1xindex>
// CHECK:         %[[T5:.*]] = tensor.cast %[[T0]] : tensor<?x3x?x?xf32> to tensor<?x3x?x?xf32>
// CHECK:         %[[T6:.*]] = "stablehlo.batch_norm_inference"(%[[T5]], %[[T2]], %[[T1]], %[[T1]], %[[T2]]) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<?x3x?x?xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<?x3x?x?xf32>
// CHECK:         %[[T7:.*]] = tensor.cast %[[T6]] : tensor<?x3x?x?xf32> to tensor<?x3x?x?xf32>
// CHECK:         %[[T8:.*]] = torch_c.from_builtin_tensor %[[T7]] : tensor<?x3x?x?xf32> -> !torch.vtensor<[?,3,?,?],f32>
// CHECK:         return %[[T8]] : !torch.vtensor<[?,3,?,?],f32>
func.func @torch.aten.batch_norm$inference(%arg0: !torch.vtensor<[?,3,?,?],f32>) -> !torch.vtensor<[?,3,?,?],f32> {
  %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %true = torch.constant.bool true
  %false = torch.constant.bool false
  %float1.000000e-01 = torch.constant.float 1.000000e-01
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %2 = torch.aten.batch_norm %arg0, %1, %0, %0, %1, %false, %float1.000000e-01, %float1.000000e-05, %true : !torch.vtensor<[?,3,?,?],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[?,3,?,?],f32>
  return %2 : !torch.vtensor<[?,3,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.batch_norm$no_bias_weight(
// CHECK-SAME:                                                    %[[VAL_0:.*]]: !torch.vtensor<[?,3,?,?],f32>) -> !torch.vtensor<[?,3,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,3,?,?],f32> -> tensor<?x3x?x?xf32>
// CHECK:           %none = torch.constant.none
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK:           %true = torch.constant.bool true
// CHECK:           %float1.000000e-01 = torch.constant.float 1.000000e-01
// CHECK:           %float1.000000e-05 = torch.constant.float 1.000000e-05
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = tensor.dim %[[VAL_1]], %[[VAL_4]] : tensor<?x3x?x?xf32>
// CHECK:           %[[VAL_6:.*]] = tensor.from_elements %[[VAL_5]] : tensor<1xindex>
// CHECK:           %[[VAL_7:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_8:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_7]], %[[VAL_6]], dims = [] : (tensor<f32>, tensor<1xindex>) -> tensor<3xf32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_9]], %[[VAL_6]], dims = [] : (tensor<f32>, tensor<1xindex>) -> tensor<3xf32>
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]], %[[VAL_13:.*]] = "stablehlo.batch_norm_training"(%[[VAL_1]], %[[VAL_8]], %[[VAL_10]]) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<?x3x?x?xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<?x3x?x?xf32>, tensor<3xf32>, tensor<3xf32>)
// CHECK:           %[[VAL_14:.*]] = torch_c.from_builtin_tensor %[[VAL_11]] : tensor<?x3x?x?xf32> -> !torch.vtensor<[?,3,?,?],f32>
// CHECK:           return %[[VAL_14]] : !torch.vtensor<[?,3,?,?],f32>
func.func @torch.aten.batch_norm$no_bias_weight(%arg0: !torch.vtensor<[?,3,?,?],f32>) -> !torch.vtensor<[?,3,?,?],f32> {
  %none = torch.constant.none
  %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %true = torch.constant.bool true
  %float1.000000e-01 = torch.constant.float 1.000000e-01
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %2 = torch.aten.batch_norm %arg0, %none, %none, %0, %1, %true, %float1.000000e-01, %float1.000000e-05, %true : !torch.vtensor<[?,3,?,?],f32>, !torch.none, !torch.none, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[?,3,?,?],f32>
  return %2 : !torch.vtensor<[?,3,?,?],f32>
}


// CHECK-LABEL:   func @torch.aten.native_layer_norm(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !torch.vtensor<[3,7,4,5],f32>) -> !torch.vtensor<[3,7,4,5],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,7,4,5],f32> -> tensor<3x7x4x5xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<4x5xf32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<4x5xf32>
// CHECK:           %int4 = torch.constant.int 4
// CHECK:           %int5 = torch.constant.int 5
// CHECK:           %float1.000000e-05 = torch.constant.float 1.000000e-05
// CHECK:           %true = torch.constant.bool true
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %int4, %int5 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = stablehlo.constant dense<[1, 21, 20]> : tensor<3xi64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.dynamic_reshape %[[VAL_1]], %[[VAL_5]] : (tensor<3x7x4x5xf32>, tensor<3xi64>) -> tensor<1x21x20xf32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<21xf32>
// CHECK:           %[[VAL_8:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<21xf32>
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]], %[[VAL_11:.*]] = "stablehlo.batch_norm_training"(%[[VAL_6]], %[[VAL_7]], %[[VAL_8]]) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x21x20xf32>, tensor<21xf32>, tensor<21xf32>) -> (tensor<1x21x20xf32>, tensor<21xf32>, tensor<21xf32>)
// CHECK:           %[[VAL_12:.*]] = stablehlo.constant dense<[3, 7, 4, 5]> : tensor<4xi64>
// CHECK:           %[[VAL_13:.*]] = stablehlo.dynamic_reshape %[[VAL_9]], %[[VAL_12]] : (tensor<1x21x20xf32>, tensor<4xi64>) -> tensor<3x7x4x5xf32>
// CHECK:           %[[VAL_14:.*]] = stablehlo.constant dense<[3, 7, 1, 1]> : tensor<4xi64>
// CHECK:           %[[VAL_15:.*]] = stablehlo.dynamic_reshape %[[VAL_10]], %[[VAL_14]] : (tensor<21xf32>, tensor<4xi64>) -> tensor<3x7x1x1xf32>
// CHECK:           %[[VAL_16:.*]] = stablehlo.constant dense<[3, 7, 1, 1]> : tensor<4xi64>
// CHECK:           %[[VAL_17:.*]] = stablehlo.dynamic_reshape %[[VAL_11]], %[[VAL_16]] : (tensor<21xf32>, tensor<4xi64>) -> tensor<3x7x1x1xf32>
// CHECK:           %[[VAL_18:.*]] = stablehlo.broadcast_in_dim %[[VAL_3]], dims = [2, 3] : (tensor<4x5xf32>) -> tensor<3x7x4x5xf32>
// CHECK:           %[[VAL_19:.*]] = stablehlo.broadcast_in_dim %[[VAL_2]], dims = [2, 3] : (tensor<4x5xf32>) -> tensor<3x7x4x5xf32>
// CHECK:           %[[VAL_20:.*]] = stablehlo.multiply %[[VAL_13]], %[[VAL_18]] : tensor<3x7x4x5xf32>
// CHECK:           %[[VAL_21:.*]] = stablehlo.add %[[VAL_20]], %[[VAL_19]] : tensor<3x7x4x5xf32>
// CHECK:           %[[VAL_22:.*]] = torch_c.from_builtin_tensor %[[VAL_21:.*]] : tensor<3x7x4x5xf32> -> !torch.vtensor<[3,7,4,5],f32>
// CHECK:           return %[[VAL_22]] : !torch.vtensor<[3,7,4,5],f32>
func.func @torch.aten.native_layer_norm(%arg0: !torch.vtensor<[3,7,4,5],f32>) -> !torch.vtensor<[3,7,4,5],f32> {
  %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<4x5xf32>) : !torch.vtensor<[4,5],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<4x5xf32>) : !torch.vtensor<[4,5],f32>
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %true = torch.constant.bool true
  %2 = torch.prim.ListConstruct %int4, %int5 : (!torch.int, !torch.int) -> !torch.list<int>
  %result0, %result1, %result2 = torch.aten.native_layer_norm %arg0, %2, %1, %0, %float1.000000e-05 : !torch.vtensor<[3,7,4,5],f32>, !torch.list<int>, !torch.vtensor<[4,5],f32>, !torch.vtensor<[4,5],f32>, !torch.float -> !torch.vtensor<[3,7,4,5],f32>, !torch.vtensor<[3,7,1,1],f32>, !torch.vtensor<[3,7,1,1],f32>
  return %result0 : !torch.vtensor<[3,7,4,5],f32>
}

// -----


// CHECK-LABEL:  func.func @torch.aten.cat$convert(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>, %[[ARG1:.*]]: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[T0:.*]] = torch.prim.ListConstruct %[[ARG0]], %[[ARG1]] : (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],si32>) -> !torch.list<vtensor>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:         %[[T2:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?,?],si32> -> tensor<?x?xi32>
// CHECK:         %[[T3:.*]] = stablehlo.convert %[[T2]] : (tensor<?x?xi32>) -> tensor<?x?xf32>
// CHECK:         %[[T4:.*]] = stablehlo.concatenate %[[T1]], %[[T3]], dim = 0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T5:.*]] = torch_c.from_builtin_tensor %[[T4]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[T5]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.cat$convert(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],si32>) -> !torch.list<vtensor>
  %1 = torch.aten.cat %0, %int0 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %1 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.cat(
// CHECK-SAME:                              %[[ARG_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                              %[[ARG_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %[[VAL_0:.*]] = torch.prim.ListConstruct %[[ARG_0]], %[[ARG_1]] : (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>) -> !torch.list<vtensor>
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[ARG_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.concatenate %[[VAL_1]], %[[VAL_2]], dim = 0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.cat(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>) -> !torch.list<vtensor>
  %1 = torch.aten.cat %0, %int0 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %1 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.runtime.assert(
// CHECK-SAME:                                    %[[ARG_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           return %[[ARG_0]] : !torch.vtensor<[?,?],f32>
func.func @torch.runtime.assert(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %true = torch.constant.bool true
  torch.runtime.assert %true, "this should not fail"
  return %arg0: !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.bitwise_left_shift.Tensor(
// CHECK-SAME:                              %[[ARG_0:.*]]: !torch.vtensor<[3,4],si32>,
// CHECK-SAME:                              %[[ARG_1:.*]]: !torch.vtensor<[3,1],si32>) -> !torch.vtensor<[3,4],si32> {
// CHECK-DAG:       %[[VAL_0:.*]] = torch_c.to_builtin_tensor %[[ARG_0:.*]] : !torch.vtensor<[3,4],si32> -> tensor<3x4xi32>
// CHECK-DAG:       %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[ARG_1:.*]] : !torch.vtensor<[3,1],si32> -> tensor<3x1xi32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.broadcast_in_dim %[[VAL_1:.*]], dims = [0, 1] : (tensor<3x1xi32>) -> tensor<3x4xi32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.shift_left %[[VAL_0:.*]], %[[VAL_2:.*]] : tensor<3x4xi32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3:.*]] : tensor<3x4xi32> -> !torch.vtensor<[3,4],si32>
// CHECK:           return %[[VAL_4:.*]] : !torch.vtensor<[3,4],si32>
func.func @torch.aten.bitwise_left_shift.Tensor(%arg0: !torch.vtensor<[3,4],si32>, %arg1: !torch.vtensor<[3,1],si32>) -> !torch.vtensor<[3,4],si32> {
  %0 = torch.aten.bitwise_left_shift.Tensor %arg0, %arg1 : !torch.vtensor<[3,4],si32>, !torch.vtensor<[3,1],si32> -> !torch.vtensor<[3,4],si32>
  return %0 : !torch.vtensor<[3,4],si32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.bitwise_left_shift.Tensor$scalar_lhs(
// CHECK-SAME:                              %[[ARG_0:.*]]: !torch.vtensor<[],si32>,
// CHECK-SAME:                              %[[ARG_1:.*]]: !torch.vtensor<[1],si32>) -> !torch.vtensor<[1],si32> {
// CHECK-DAG:       %[[VAL_0:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[],si32> -> tensor<i32>
// CHECK-DAG:       %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[ARG_1]] : !torch.vtensor<[1],si32> -> tensor<1xi32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.broadcast_in_dim %[[VAL_0]], dims = [] : (tensor<i32>) -> tensor<1xi32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.shift_left %[[VAL_2]], %[[VAL_1]] : tensor<1xi32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<1xi32> -> !torch.vtensor<[1],si32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[1],si32>
func.func @torch.aten.bitwise_left_shift.Tensor$scalar_lhs(%arg0: !torch.vtensor<[],si32>, %arg1: !torch.vtensor<[1],si32>) -> !torch.vtensor<[1],si32> {
  %0 = torch.aten.bitwise_left_shift.Tensor %arg0, %arg1 : !torch.vtensor<[],si32>, !torch.vtensor<[1],si32> -> !torch.vtensor<[1],si32>
  return %0 : !torch.vtensor<[1],si32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.bitwise_right_shift.Tensor(
// CHECK-SAME:                              %[[ARG_0:.*]]: !torch.vtensor<[3,4],si64>,
// CHECK-SAME:                              %[[ARG_1:.*]]: !torch.vtensor<[3,4],si64>) -> !torch.vtensor<[3,4],si64> {
// CHECK-DAG:       %[[VAL_0:.*]] = torch_c.to_builtin_tensor %[[ARG_0:.*]] : !torch.vtensor<[3,4],si64> -> tensor<3x4xi64>
// CHECK-DAG:       %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[ARG_1:.*]] : !torch.vtensor<[3,4],si64> -> tensor<3x4xi64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.shift_right_arithmetic %[[VAL_0:.*]], %[[VAL_1:.*]] : tensor<3x4xi64>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2:.*]] : tensor<3x4xi64> -> !torch.vtensor<[3,4],si64>
// CHECK:           return %[[VAL_3:.*]] : !torch.vtensor<[3,4],si64>
func.func @torch.aten.bitwise_right_shift.Tensor(%arg0: !torch.vtensor<[3,4],si64>, %arg1: !torch.vtensor<[3,4],si64>) -> !torch.vtensor<[3,4],si64> {
  %0 = torch.aten.bitwise_right_shift.Tensor %arg0, %arg1 : !torch.vtensor<[3,4],si64>, !torch.vtensor<[3,4],si64> -> !torch.vtensor<[3,4],si64>
  return %0 : !torch.vtensor<[3,4],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.__rshift__.Tensor(
// CHECK-SAME:                              %[[ARG_0:.*]]: !torch.vtensor<[3,4],si32>,
// CHECK-SAME:                              %[[ARG_1:.*]]: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32> {
// CHECK-DAG:       %[[VAL_0:.*]] = torch_c.to_builtin_tensor %[[ARG_0:.*]] : !torch.vtensor<[3,4],si32> -> tensor<3x4xi32>
// CHECK-DAG:       %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[ARG_1:.*]] : !torch.vtensor<[3,4],si32> -> tensor<3x4xi32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.shift_right_arithmetic %[[VAL_0:.*]], %[[VAL_1:.*]] : tensor<3x4xi32>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2:.*]] : tensor<3x4xi32> -> !torch.vtensor<[3,4],si32>
// CHECK:           return %[[VAL_3:.*]] : !torch.vtensor<[3,4],si32>
func.func @torch.aten.__rshift__.Tensor(%arg0: !torch.vtensor<[3,4],si32>, %arg1: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32> {
  %0 = torch.aten.__rshift__.Tensor %arg0, %arg1 : !torch.vtensor<[3,4],si32>, !torch.vtensor<[3,4],si32> -> !torch.vtensor<[3,4],si32>
  return %0 : !torch.vtensor<[3,4],si32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.__rshift__.Tensor$broadcast(
// CHECK-SAME:                              %[[ARG_0:.*]]: !torch.vtensor<[3,4],si64>,
// CHECK-SAME:                              %[[ARG_1:.*]]: !torch.vtensor<[3,1],si64>) -> !torch.vtensor<[3,4],si64> {
// CHECK-DAG:       %[[VAL_0:.*]] = torch_c.to_builtin_tensor %[[ARG_1:.*]] : !torch.vtensor<[3,1],si64> -> tensor<3x1xi64>
// CHECK-DAG:       %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[ARG_0:.*]] : !torch.vtensor<[3,4],si64> -> tensor<3x4xi64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.broadcast_in_dim %[[VAL_0:.*]], dims = [0, 1] : (tensor<3x1xi64>) -> tensor<3x4xi64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.shift_right_arithmetic %[[VAL_1:.*]], %[[VAL_2:.*]] : tensor<3x4xi64>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3:.*]] : tensor<3x4xi64> -> !torch.vtensor<[3,4],si64>
// CHECK:           return %[[VAL_4:.*]] : !torch.vtensor<[3,4],si64>
func.func @torch.aten.__rshift__.Tensor$broadcast(%arg0: !torch.vtensor<[3,4],si64>, %arg1: !torch.vtensor<[3,1],si64>) -> !torch.vtensor<[3,4],si64> {
  %0 = torch.aten.__rshift__.Tensor %arg0, %arg1 : !torch.vtensor<[3,4],si64>, !torch.vtensor<[3,1],si64> -> !torch.vtensor<[3,4],si64>
  return %0 : !torch.vtensor<[3,4],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.tril(
// CHECK-SAME:                   %[[ARG_0:.*]]: !torch.vtensor<[2,3,5],f32>,
// CHECK-SAME:                   %[[ARG_1:.*]]: !torch.int) -> !torch.vtensor<[2,3,5],f32>
// CHECK-DAG:       %[[VAL_0:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[2,3,5],f32> -> tensor<2x3x5xf32>
// CHECK-DAG:       %[[VAL_1:.*]] = torch_c.to_i64 %[[ARG_1]]
// CHECK:           %[[VAL_4:.*]] = tensor.from_elements %[[VAL_1]] : tensor<1xi64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.iota dim = 1 : tensor<3x5xi64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.iota dim = 0 : tensor<3x5xi64>
// CHECK:           %[[VAL_5:.*]] = chlo.broadcast_add %[[VAL_3]], %[[VAL_4]] {broadcast_dimensions = array<i64: 1>} : (tensor<3x5xi64>, tensor<1xi64>) -> tensor<3x5xi64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.compare  LE, %[[VAL_2]], %[[VAL_5]],  SIGNED : (tensor<3x5xi64>, tensor<3x5xi64>) -> tensor<3x5xi1>
// CHECK:           %[[VAL_7:.*]] = stablehlo.broadcast_in_dim %[[VAL_6]], dims = [1, 2] : (tensor<3x5xi1>) -> tensor<2x3x5xi1>
// CHECK:           %[[VAL_8:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<2x3x5xf32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.select %[[VAL_7]], %[[VAL_0]], %[[VAL_8]] : tensor<2x3x5xi1>, tensor<2x3x5xf32>
// CHECK:           %[[VAL_10:.*]] = torch_c.from_builtin_tensor %[[VAL_9]] : tensor<2x3x5xf32> -> !torch.vtensor<[2,3,5],f32>
// CHECK:           return %[[VAL_10:.*]] : !torch.vtensor<[2,3,5],f32>
func.func @torch.aten.tril(%arg0: !torch.vtensor<[2,3,5],f32>, %arg1: !torch.int) -> !torch.vtensor<[2,3,5],f32> {
  %0 = torch.aten.tril %arg0, %arg1:!torch.vtensor<[2,3,5],f32>, !torch.int -> !torch.vtensor<[2,3,5],f32>
  return %0 : !torch.vtensor<[2,3,5],f32>
}

// -----

// Test: tril with dynamic batch dim (matrix dims static)
// CHECK-LABEL: func @torch.aten.tril_dynamic_batch
// CHECK-SAME:    %[[INPUT:.*]]: !torch.vtensor<[?,3,5],f32>, %[[DIAG:.*]]: !torch.int) -> !torch.vtensor<[?,3,5],f32>
// CHECK:         stablehlo.dynamic_iota
// CHECK:         stablehlo.dynamic_iota
// CHECK:         chlo.broadcast_add
// CHECK:         stablehlo.compare
// CHECK:         stablehlo.dynamic_broadcast_in_dim
// CHECK:         stablehlo.dynamic_broadcast_in_dim
// CHECK:         stablehlo.select
func.func @torch.aten.tril_dynamic_batch(%arg0: !torch.vtensor<[?,3,5],f32>, %arg1: !torch.int) -> !torch.vtensor<[?,3,5],f32> {
  %0 = torch.aten.tril %arg0, %arg1 : !torch.vtensor<[?,3,5],f32>, !torch.int -> !torch.vtensor<[?,3,5],f32>
  return %0 : !torch.vtensor<[?,3,5],f32>
}

// -----

// Test: tril with fully dynamic shape (matrix dims dynamic)
// CHECK-LABEL: func @torch.aten.tril_fully_dynamic
// CHECK-SAME:    %[[INPUT:.*]]: !torch.vtensor<[?,?,?],f32>, %[[DIAG:.*]]: !torch.int) -> !torch.vtensor<[?,?,?],f32>
// CHECK:         stablehlo.get_dimension_size
// CHECK:         stablehlo.get_dimension_size
// CHECK:         stablehlo.dynamic_iota
// CHECK:         stablehlo.dynamic_iota
// CHECK:         chlo.broadcast_add
// CHECK:         stablehlo.compare
// CHECK:         stablehlo.dynamic_broadcast_in_dim
// CHECK:         stablehlo.dynamic_broadcast_in_dim
// CHECK:         stablehlo.select
func.func @torch.aten.tril_fully_dynamic(%arg0: !torch.vtensor<[?,?,?],f32>, %arg1: !torch.int) -> !torch.vtensor<[?,?,?],f32> {
  %0 = torch.aten.tril %arg0, %arg1 : !torch.vtensor<[?,?,?],f32>, !torch.int -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// -----

// Test: tril 2D dynamic
// CHECK-LABEL: func @torch.aten.tril_2d_dynamic
// CHECK-SAME:    %[[INPUT:.*]]: !torch.vtensor<[?,?],f32>, %[[DIAG:.*]]: !torch.int) -> !torch.vtensor<[?,?],f32>
// CHECK:         stablehlo.dynamic_iota
// CHECK:         stablehlo.dynamic_iota
// CHECK:         stablehlo.compare
// CHECK:         stablehlo.select
func.func @torch.aten.tril_2d_dynamic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.int) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.tril %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.bitwise_or.Scalar(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !torch.vtensor<[4,5,6],si32>) -> !torch.vtensor<[4,5,6],si32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,5,6],si32> -> tensor<4x5x6xi32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 5
// CHECK:           %[[VAL_3:.*]] = arith.constant 5 : i64
// CHECK:           %[[VAL_4:.*]] = tensor.from_elements %[[VAL_3]] : tensor<1xi64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.convert %[[VAL_4]] : (tensor<1xi64>) -> tensor<1xi32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.reshape %[[VAL_5]] : (tensor<1xi32>) -> tensor<i32>
// CHECK:           %[[VAL_7:.*]] = chlo.broadcast_or %[[VAL_1]], %[[VAL_6]] : (tensor<4x5x6xi32>, tensor<i32>) -> tensor<4x5x6xi32>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<4x5x6xi32> -> !torch.vtensor<[4,5,6],si32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[4,5,6],si32>
// CHECK:         }
func.func @torch.aten.bitwise_or.Scalar(%arg0: !torch.vtensor<[4,5,6],si32>) -> !torch.vtensor<[4,5,6],si32> {
  %int5 = torch.constant.int 5
  %0 = torch.aten.bitwise_or.Scalar %arg0, %int5 : !torch.vtensor<[4,5,6],si32>, !torch.int -> !torch.vtensor<[4,5,6],si32>
  return %0 : !torch.vtensor<[4,5,6],si32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.bitwise_xor.Scalar(
// CHECK-SAME:                                             %[[VAL_0:.*]]: !torch.vtensor<[4,5,6],si32>) -> !torch.vtensor<[4,5,6],si32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,5,6],si32> -> tensor<4x5x6xi32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 8
// CHECK:           %[[VAL_3:.*]] = arith.constant 8 : i64
// CHECK:           %[[VAL_4:.*]] = tensor.from_elements %[[VAL_3]] : tensor<1xi64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.convert %[[VAL_4]] : (tensor<1xi64>) -> tensor<1xi32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.reshape %[[VAL_5]] : (tensor<1xi32>) -> tensor<i32>
// CHECK:           %[[VAL_7:.*]] = chlo.broadcast_xor %[[VAL_1]], %[[VAL_6]] : (tensor<4x5x6xi32>, tensor<i32>) -> tensor<4x5x6xi32>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<4x5x6xi32> -> !torch.vtensor<[4,5,6],si32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[4,5,6],si32>
// CHECK:         }
func.func @torch.aten.bitwise_xor.Scalar(%arg0: !torch.vtensor<[4,5,6],si32>) -> !torch.vtensor<[4,5,6],si32> {
  %int8 = torch.constant.int 8
  %0 = torch.aten.bitwise_xor.Scalar %arg0, %int8 : !torch.vtensor<[4,5,6],si32>, !torch.int -> !torch.vtensor<[4,5,6],si32>
  return %0 : !torch.vtensor<[4,5,6],si32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.fmod.Scalar(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !torch.vtensor<[4,3,2],f32>) -> !torch.vtensor<[4,3,2],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,3,2],f32> -> tensor<4x3x2xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 17
// CHECK:           %[[VAL_3:.*]] = arith.constant 17 : i64
// CHECK:           %[[VAL_4:.*]] = tensor.from_elements %[[VAL_3]] : tensor<1xi64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.convert %[[VAL_4]] : (tensor<1xi64>) -> tensor<1xf32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.reshape %[[VAL_5]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.broadcast_in_dim %[[VAL_6]], dims = [] : (tensor<f32>) -> tensor<4x3x2xf32>
// CHECK:           %[[VAL_8:.*]] = stablehlo.divide %[[VAL_1]], %[[VAL_7]] : tensor<4x3x2xf32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.sign %[[VAL_8]] : tensor<4x3x2xf32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.abs %[[VAL_8]] : tensor<4x3x2xf32>
// CHECK:           %[[VAL_11:.*]] = stablehlo.floor %[[VAL_10]] : tensor<4x3x2xf32>
// CHECK:           %[[VAL_12:.*]] = stablehlo.multiply %[[VAL_9]], %[[VAL_11]] : tensor<4x3x2xf32>
// CHECK:           %[[VAL_13:.*]] = stablehlo.multiply %[[VAL_12]], %[[VAL_7]] : tensor<4x3x2xf32>
// CHECK:           %[[VAL_14:.*]] = stablehlo.subtract %[[VAL_1]], %[[VAL_13]] : tensor<4x3x2xf32>
// CHECK:           %[[VAL_15:.*]] = torch_c.from_builtin_tensor %[[VAL_14]] : tensor<4x3x2xf32> -> !torch.vtensor<[4,3,2],f32>
// CHECK:           return %[[VAL_15]] : !torch.vtensor<[4,3,2],f32>
// CHECK:         }
func.func @torch.aten.fmod.Scalar(%arg0: !torch.vtensor<[4,3,2],f32>) -> !torch.vtensor<[4,3,2],f32> {
  %int17 = torch.constant.int 17
  %0 = torch.aten.fmod.Scalar %arg0, %int17 : !torch.vtensor<[4,3,2],f32>, !torch.int -> !torch.vtensor<[4,3,2],f32>
  return %0 : !torch.vtensor<[4,3,2],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.fmod.Scalar$dynamic(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !torch.vtensor<[?,3,2],f32>) -> !torch.vtensor<[?,3,2],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,3,2],f32> -> tensor<?x3x2xf32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 5 : i64
// CHECK:           %[[VAL_3:.*]] = tensor.from_elements %[[VAL_2]] : tensor<1xi64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.convert %[[VAL_3]] : (tensor<1xi64>) -> tensor<1xf32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.reshape %[[VAL_4]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = tensor.dim %[[VAL_1]], %[[VAL_6]] : tensor<?x3x2xf32>
// CHECK:           %[[VAL_8:.*]] = arith.index_cast %[[VAL_7]] : index to i64
// CHECK:           %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_10:.*]] = tensor.dim %[[VAL_1]], %[[VAL_9]] : tensor<?x3x2xf32>
// CHECK:           %[[VAL_11:.*]] = arith.index_cast %[[VAL_10]] : index to i64
// CHECK:           %[[VAL_12:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_13:.*]] = tensor.dim %[[VAL_1]], %[[VAL_12]] : tensor<?x3x2xf32>
// CHECK:           %[[VAL_14:.*]] = arith.index_cast %[[VAL_13]] : index to i64
// CHECK:           %[[VAL_15:.*]] = tensor.from_elements %[[VAL_8]], %[[VAL_11]], %[[VAL_14]] : tensor<3xi64>
// CHECK:           %[[VAL_16:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_5]], %[[VAL_15]], dims = [] : (tensor<f32>, tensor<3xi64>) -> tensor<?x3x2xf32>
// CHECK:           %[[VAL_17:.*]] = stablehlo.divide %[[VAL_1]], %[[VAL_16]] : tensor<?x3x2xf32>
// CHECK:           %[[VAL_18:.*]] = stablehlo.sign %[[VAL_17]] : tensor<?x3x2xf32>
// CHECK:           %[[VAL_19:.*]] = stablehlo.abs %[[VAL_17]] : tensor<?x3x2xf32>
// CHECK:           %[[VAL_20:.*]] = stablehlo.floor %[[VAL_19]] : tensor<?x3x2xf32>
// CHECK:           %[[VAL_21:.*]] = stablehlo.multiply %[[VAL_18]], %[[VAL_20]] : tensor<?x3x2xf32>
// CHECK:           %[[VAL_22:.*]] = stablehlo.multiply %[[VAL_21]], %[[VAL_16]] : tensor<?x3x2xf32>
// CHECK:           %[[VAL_23:.*]] = stablehlo.subtract %[[VAL_1]], %[[VAL_22]] : tensor<?x3x2xf32>
// CHECK:           %[[VAL_24:.*]] = torch_c.from_builtin_tensor %[[VAL_23]] : tensor<?x3x2xf32> -> !torch.vtensor<[?,3,2],f32>
// CHECK:           return %[[VAL_24]] : !torch.vtensor<[?,3,2],f32>
// CHECK:         }
func.func @torch.aten.fmod.Scalar$dynamic(%arg0: !torch.vtensor<[?,3,2],f32>) -> !torch.vtensor<[?,3,2],f32> {
  %int5 = torch.constant.int 5
  %0 = torch.aten.fmod.Scalar %arg0, %int5 : !torch.vtensor<[?,3,2],f32>, !torch.int -> !torch.vtensor<[?,3,2],f32>
  return %0 : !torch.vtensor<[?,3,2],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.div.Tensor_mode$none_rounding(
// CHECK-SAME:                                                         %[[VAL_0:.*]]: !torch.vtensor<[4,4],f32>,
// CHECK-SAME:                                                         %[[VAL_1:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[4,4],f32> {
// CHECK-DAG:       %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK-DAG:       %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,4],f32> -> tensor<4x4xf32>
// CHECK:           %[[VAL_4:.*]] = chlo.broadcast_divide %[[VAL_3]], %[[VAL_2]] : (tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<4x4xf32> -> !torch.vtensor<[4,4],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[4,4],f32>
// CHECK:         }
func.func @torch.aten.div.Tensor_mode$none_rounding(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.vtensor<[],f32>) -> !torch.vtensor<[4,4],f32> {
  %none = torch.constant.none
  %0 = torch.aten.div.Tensor_mode %arg0, %arg1, %none : !torch.vtensor<[4,4],f32>, !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[4,4],f32>
  return %0 : !torch.vtensor<[4,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.div.Tensor_mode$trunc_rounding(
// CHECK-SAME:                                                          %[[VAL_0:.*]]: !torch.vtensor<[4,4],f32>,
// CHECK-SAME:                                                          %[[VAL_1:.*]]: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
// CHECK-DAG:       %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,4],f32> -> tensor<4x4xf32>
// CHECK-DAG:       %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,4],f32> -> tensor<4x4xf32>
// CHECK:           %[[VAL_4:.*]] = chlo.broadcast_divide %[[VAL_2]], %[[VAL_3]] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.sign %[[VAL_4]] : tensor<4x4xf32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.abs %[[VAL_4]] : tensor<4x4xf32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.floor %[[VAL_6]] : tensor<4x4xf32>
// CHECK:           %[[VAL_8:.*]] = stablehlo.multiply %[[VAL_5]], %[[VAL_7]] : tensor<4x4xf32>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<4x4xf32> -> !torch.vtensor<[4,4],f32>
// CHECK:           return %[[VAL_9]] : !torch.vtensor<[4,4],f32>
// CHECK:         }
func.func @torch.aten.div.Tensor_mode$trunc_rounding(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
  %str = torch.constant.str "trunc"
  %0 = torch.aten.div.Tensor_mode %arg0, %arg1, %str : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f32>, !torch.str -> !torch.vtensor<[4,4],f32>
  return %0 : !torch.vtensor<[4,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.div.Tensor_mode$floor_rounding(
// CHECK-SAME:                                                          %[[VAL_0:.*]]: !torch.vtensor<[4,4],f32>,
// CHECK-SAME:                                                          %[[VAL_1:.*]]: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
// CHECK-DAG:       %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,4],f32> -> tensor<4x4xf32>
// CHECK-DAG:       %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,4],f32> -> tensor<4x4xf32>
// CHECK:           %[[MOD:.*]] = chlo.broadcast_remainder %[[VAL_2]], %[[VAL_3]] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK:           %[[NUM:.*]] = chlo.broadcast_subtract %[[VAL_2]], %[[MOD]] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK:           %[[DIV:.*]] = chlo.broadcast_divide %[[NUM]], %[[VAL_3]] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK:           %[[ZERO:.*]] = "chlo.constant_like"(%[[DIV]]) <{value = 0.000000e+00 : f32}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK:           %[[ONE:.*]] = "chlo.constant_like"(%[[DIV]]) <{value = 1.000000e+00 : f32}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK:           %[[C23:.*]] = "chlo.constant_like"(%[[DIV]]) <{value = 0x4B800000 : f32}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK:           %[[AX:.*]] = stablehlo.abs %[[VAL_2]] : tensor<4x4xf32>
// CHECK:           %[[RX:.*]] = chlo.broadcast_compare %[[AX]], %[[C23]] {comparison_direction = #chlo<comparison_direction LT>} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK:           %[[AY:.*]] = stablehlo.abs %[[VAL_3]] : tensor<4x4xf32>
// CHECK:           %[[RY:.*]] = chlo.broadcast_compare %[[AY]], %[[C23]] {comparison_direction = #chlo<comparison_direction LT>} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK:           %[[AQ:.*]] = stablehlo.abs %[[DIV]] : tensor<4x4xf32>
// CHECK:           %[[RQ:.*]] = chlo.broadcast_compare %[[AQ]], %[[C23]] {comparison_direction = #chlo<comparison_direction LT>} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK:           %[[AND1:.*]] = stablehlo.and %[[RX]], %[[RY]] : tensor<4x4xi1>
// CHECK:           %[[SAFE:.*]] = stablehlo.and %[[AND1]], %[[RQ]] : tensor<4x4xi1>
// CHECK:           %[[C22:.*]] = "chlo.constant_like"(%[[DIV]]) <{value = 0x4B000000 : f32}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK:           %[[DS:.*]] = stablehlo.select %[[SAFE]], %[[DIV]], %[[ZERO]] : tensor<4x4xi1>, tensor<4x4xf32>
// CHECK:           %[[RND:.*]] = stablehlo.round_nearest_even %[[DS]] : tensor<4x4xf32>
// CHECK:           %[[ADS:.*]] = stablehlo.abs %[[DS]] : tensor<4x4xf32>
// CHECK:           %[[BIG:.*]] = chlo.broadcast_compare %[[ADS]], %[[C22]] {comparison_direction = #chlo<comparison_direction GE>} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK:           %[[G1:.*]] = stablehlo.select %[[BIG]], %[[DS]], %[[RND]] : tensor<4x4xi1>, tensor<4x4xf32>
// CHECK:           %[[ISNAN:.*]] = chlo.broadcast_compare %[[DS]], %[[DS]] {comparison_direction = #chlo<comparison_direction NE>} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK:           %[[Q:.*]] = stablehlo.select %[[ISNAN]], %[[DS]], %[[G1]] : tensor<4x4xi1>, tensor<4x4xf32>
// CHECK:           %[[PROD:.*]] = chlo.broadcast_multiply %[[MOD]], %[[VAL_3]] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK:           %[[ADJ:.*]] = chlo.broadcast_compare %[[PROD]], %[[ZERO]] {comparison_direction = #chlo<comparison_direction LT>} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK:           %[[DN:.*]] = stablehlo.subtract %[[Q]], %[[ONE]] : tensor<4x4xf32>
// CHECK:           %[[EXACT:.*]] = stablehlo.select %[[ADJ]], %[[DN]], %[[Q]] : tensor<4x4xi1>, tensor<4x4xf32>
// CHECK:           %[[NQ:.*]] = chlo.broadcast_divide %[[VAL_2]], %[[VAL_3]] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK:           %[[FL:.*]] = stablehlo.floor %[[NQ]] : tensor<4x4xf32>
// CHECK:           %[[ANQ:.*]] = stablehlo.abs %[[NQ]] : tensor<4x4xf32>
// CHECK:           %[[NBIG:.*]] = chlo.broadcast_compare %[[ANQ]], %[[C22]] {comparison_direction = #chlo<comparison_direction GE>} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK:           %[[NG1:.*]] = stablehlo.select %[[NBIG]], %[[NQ]], %[[FL]] : tensor<4x4xi1>, tensor<4x4xf32>
// CHECK:           %[[NNAN:.*]] = chlo.broadcast_compare %[[NQ]], %[[NQ]] {comparison_direction = #chlo<comparison_direction NE>} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK:           %[[NAIVE:.*]] = stablehlo.select %[[NNAN]], %[[NQ]], %[[NG1]] : tensor<4x4xi1>, tensor<4x4xf32>
// CHECK:           %[[OUT:.*]] = stablehlo.select %[[SAFE]], %[[EXACT]], %[[NAIVE]] : tensor<4x4xi1>, tensor<4x4xf32>
// CHECK:           %[[RES:.*]] = torch_c.from_builtin_tensor %[[OUT]] : tensor<4x4xf32> -> !torch.vtensor<[4,4],f32>
// CHECK:           return %[[RES]] : !torch.vtensor<[4,4],f32>
// CHECK:         }
func.func @torch.aten.div.Tensor_mode$floor_rounding(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
  %str = torch.constant.str "floor"
  %0 = torch.aten.div.Tensor_mode %arg0, %arg1, %str : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f32>, !torch.str -> !torch.vtensor<[4,4],f32>
  return %0 : !torch.vtensor<[4,4],f32>
}

// -----

// CHECK-LABEL:   func.func @aten.fill.tensor(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[2,2],f32>,
// CHECK-SAME:                                %[[VAL_1:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[2,2],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,2],f32> -> tensor<2x2xf32>
// CHECK:           %[[VAL_4:.*]] = shape.shape_of %[[VAL_3]] : tensor<2x2xf32> -> tensor<2xindex>
// CHECK:           %[[VAL_5:.*]] = stablehlo.dynamic_broadcast_in_dim %[[VAL_2]], %[[VAL_4]], dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<2x2xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<2x2xf32> -> !torch.vtensor<[2,2],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[2,2],f32>
// CHECK:         }
func.func @aten.fill.tensor(%arg0: !torch.vtensor<[2,2],f32>, %arg1: !torch.vtensor<[],f32>) -> !torch.vtensor<[2,2],f32> {
  %0 = torch.aten.fill.Tensor %arg0, %arg1 : !torch.vtensor<[2,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[2,2],f32>
  return %0 : !torch.vtensor<[2,2],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten._conj$complex(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[2,3],complex<f32>>) -> !torch.vtensor<[2,3],complex<f32>> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,3],complex<f32>> -> tensor<2x3xcomplex<f32>>
// CHECK:           %[[VAL_2:.*]] = stablehlo.real %[[VAL_1]] : (tensor<2x3xcomplex<f32>>) -> tensor<2x3xf32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.imag %[[VAL_1]] : (tensor<2x3xcomplex<f32>>) -> tensor<2x3xf32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.negate %[[VAL_3]] : tensor<2x3xf32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.complex %[[VAL_2]], %[[VAL_4]] : tensor<2x3xcomplex<f32>>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<2x3xcomplex<f32>> -> !torch.vtensor<[2,3],complex<f32>>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[2,3],complex<f32>>
func.func @torch.aten._conj$complex(%arg0: !torch.vtensor<[2,3],complex<f32>>) -> !torch.vtensor<[2,3],complex<f32>> {
  %0 = torch.aten._conj %arg0 : !torch.vtensor<[2,3],complex<f32>> -> !torch.vtensor<[2,3],complex<f32>>
  return %0 : !torch.vtensor<[2,3],complex<f32>>
}

// -----

// CHECK-LABEL:   func.func @torch.aten._conj$real(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> {
// CHECK:           return %[[VAL_0]] : !torch.vtensor<[2,3],f32>
func.func @torch.aten._conj$real(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten._conj %arg0 : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}


// -----

// CHECK-LABEL:   func.func @torch.aten.sum.dim_IntList$0d(
// CHECK-SAME:                                             %[[ARG0:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
// CHECK:           return %[[ARG0]] : !torch.vtensor<[],f32>
func.func @torch.aten.sum.dim_IntList$0d(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
  %int0 = torch.constant.int 0
  %true = torch.constant.bool true
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %1 = torch.aten.sum.dim_IntList %arg0, %0, %true, %none : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[],f32>
  return %1 : !torch.vtensor<[],f32>
}

// -----

// Test: torch.aten.ones with dynamic shape (SymInt in shape list)
// CHECK-LABEL: func @torch.aten.ones_dynamic
// CHECK-SAME:    %[[DIM:.*]]: !torch.int) -> !torch.vtensor<[?,4],f32>
// CHECK:         %[[I64:.*]] = torch_c.to_i64 %[[DIM]]
// CHECK:         tensor.from_elements %[[I64]]
// CHECK:         stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:         stablehlo.dynamic_broadcast_in_dim
func.func @torch.aten.ones_dynamic(%arg0: !torch.int) -> !torch.vtensor<[?,4],f32> {
  %int4 = torch.constant.int 4
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %arg0, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.ones %0, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,4],f32>
  return %1 : !torch.vtensor<[?,4],f32>
}

// -----

// Test: torch.aten.zeros with dynamic shape
// CHECK-LABEL: func @torch.aten.zeros_dynamic
// CHECK-SAME:    %[[DIM:.*]]: !torch.int) -> !torch.vtensor<[?,?],f32>
// CHECK:         stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         stablehlo.dynamic_broadcast_in_dim
func.func @torch.aten.zeros_dynamic(%arg0: !torch.int) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.prim.ListConstruct %arg0, %arg0 : (!torch.int, !torch.int) -> !torch.list<int>
  %none = torch.constant.none
  %1 = torch.aten.zeros %0, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?],f32>
  return %1 : !torch.vtensor<[?,?],f32>
}

// -----

// Test: torch.aten.ones with fully dynamic 2D shape (exercises concatenate path)
// CHECK-LABEL:   func.func @torch.aten.ones_dynamic_2d(
// CHECK-SAME:        %[[ARG0:.*]]: !torch.int, %[[ARG1:.*]]: !torch.int) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[D0:.*]] = torch_c.to_i64 %[[ARG0]]
// CHECK:           %[[FE0:.*]] = tensor.from_elements %[[D0]] : tensor<1xi64>
// CHECK:           %[[D1:.*]] = torch_c.to_i64 %[[ARG1]]
// CHECK:           %[[FE1:.*]] = tensor.from_elements %[[D1]] : tensor<1xi64>
// CHECK:           %[[SHAPE:.*]] = stablehlo.concatenate %[[FE0]], %[[FE1]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           %[[CST:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[RESULT:.*]] = stablehlo.dynamic_broadcast_in_dim %[[CST]], %[[SHAPE]], dims = [] : (tensor<f32>, tensor<2xi64>) -> tensor<?x?xf32>
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.ones_dynamic_2d(%arg0: !torch.int, %arg1: !torch.int) -> !torch.vtensor<[?,?],f32> {
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.ones %0, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?],f32>
  return %1 : !torch.vtensor<[?,?],f32>
}

// -----

// Test: torch.aten.zeros with 3D dynamic integer type (exercises concatenate + integer constant)
// CHECK-LABEL:   func.func @torch.aten.zeros_dynamic_3d(
// CHECK-SAME:        %[[ARG0:.*]]: !torch.int, %[[ARG1:.*]]: !torch.int, %[[ARG2:.*]]: !torch.int) -> !torch.vtensor<[?,?,?],si64> {
// CHECK:           %[[D0:.*]] = torch_c.to_i64 %[[ARG0]]
// CHECK:           %[[FE0:.*]] = tensor.from_elements %[[D0]] : tensor<1xi64>
// CHECK:           %[[D1:.*]] = torch_c.to_i64 %[[ARG1]]
// CHECK:           %[[FE1:.*]] = tensor.from_elements %[[D1]] : tensor<1xi64>
// CHECK:           %[[D2:.*]] = torch_c.to_i64 %[[ARG2]]
// CHECK:           %[[FE2:.*]] = tensor.from_elements %[[D2]] : tensor<1xi64>
// CHECK:           %[[SHAPE:.*]] = stablehlo.concatenate %[[FE0]], %[[FE1]], %[[FE2]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[RESULT:.*]] = stablehlo.dynamic_broadcast_in_dim %[[CST]], %[[SHAPE]], dims = [] : (tensor<i64>, tensor<3xi64>) -> tensor<?x?x?xi64>
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<?x?x?xi64> -> !torch.vtensor<[?,?,?],si64>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,?,?],si64>
// CHECK:         }
func.func @torch.aten.zeros_dynamic_3d(%arg0: !torch.int, %arg1: !torch.int, %arg2: !torch.int) -> !torch.vtensor<[?,?,?],si64> {
  %none = torch.constant.none
  %int4 = torch.constant.int 4
  %0 = torch.prim.ListConstruct %arg0, %arg1, %arg2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.zeros %0, %int4, %none, %none, %none : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?,?],si64>
  return %1 : !torch.vtensor<[?,?,?],si64>
}

// -----

// Test: native_layer_norm with single dynamic batch dim
// CHECK-LABEL: func @torch.aten.native_layer_norm_dynamic_batch
// CHECK-SAME:    %[[INPUT:.*]]: !torch.vtensor<[?,7,4,5],f32>) -> !torch.vtensor<[?,7,4,5],f32>
// CHECK:         %[[T:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[?,7,4,5],f32> -> tensor<?x7x4x5xf32>
// CHECK:         %[[C1:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK:         %[[D0:.*]] = stablehlo.get_dimension_size %[[T]], dim = 0
// CHECK:         %[[F0:.*]] = stablehlo.multiply %[[C1]], %[[D0]] : tensor<i32>
// CHECK:         %[[D1:.*]] = stablehlo.get_dimension_size %[[T]], dim = 1
// CHECK:         %[[FEAT:.*]] = stablehlo.multiply %[[F0]], %[[D1]] : tensor<i32>
// CHECK:         %[[ONE:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK:         %[[EMB:.*]] = stablehlo.constant dense<20> : tensor<i32>
// CHECK:         %[[ONE_R:.*]] = stablehlo.reshape %[[ONE]] : (tensor<i32>) -> tensor<1xi32>
// CHECK:         %[[FEAT_R:.*]] = stablehlo.reshape %[[FEAT]] : (tensor<i32>) -> tensor<1xi32>
// CHECK:         %[[EMB_R:.*]] = stablehlo.reshape %[[EMB]] : (tensor<i32>) -> tensor<1xi32>
// CHECK:         %[[FLAT_SHAPE:.*]] = stablehlo.concatenate %[[ONE_R]], %[[FEAT_R]], %[[EMB_R]], dim = 0
// CHECK:         %[[RESHAPED:.*]] = stablehlo.dynamic_reshape %[[T]], %[[FLAT_SHAPE]] : (tensor<?x7x4x5xf32>, tensor<3xi32>) -> tensor<1x?x20xf32>
// CHECK:         %[[SCALE_S:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:         %[[OFFSET_S:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[FEAT_SHAPE:.*]] = stablehlo.reshape %[[FEAT]] : (tensor<i32>) -> tensor<1xi32>
// CHECK:         %[[SCALE:.*]] = stablehlo.dynamic_broadcast_in_dim %[[SCALE_S]], %[[FEAT_SHAPE]], dims = []
// CHECK:         %[[OFFSET:.*]] = stablehlo.dynamic_broadcast_in_dim %[[OFFSET_S]], %[[FEAT_SHAPE]], dims = []
// CHECK:         %[[BN_OUT:.*]], %[[BN_MEAN:.*]], %[[BN_VAR:.*]] = "stablehlo.batch_norm_training"(%[[RESHAPED]], %[[SCALE]], %[[OFFSET]]) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}>
// CHECK:         %[[OUT:.*]] = stablehlo.dynamic_reshape %[[BN_OUT]], {{.*}} : (tensor<1x?x20xf32>, {{.*}}) -> tensor<?x7x4x5xf32>
// CHECK:         %[[MEAN:.*]] = stablehlo.dynamic_reshape %[[BN_MEAN]], {{.*}} : (tensor<?xf32>, {{.*}}) -> tensor<?x7x1x1xf32>
// CHECK:         %[[VAR:.*]] = stablehlo.dynamic_reshape %[[BN_VAR]], {{.*}} : (tensor<?xf32>, {{.*}}) -> tensor<?x7x1x1xf32>
// CHECK:         stablehlo.dynamic_broadcast_in_dim
// CHECK:         stablehlo.dynamic_broadcast_in_dim
// CHECK:         stablehlo.multiply
// CHECK:         stablehlo.add
func.func @torch.aten.native_layer_norm_dynamic_batch(%arg0: !torch.vtensor<[?,7,4,5],f32>) -> !torch.vtensor<[?,7,4,5],f32> {
  %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<4x5xf32>) : !torch.vtensor<[4,5],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<4x5xf32>) : !torch.vtensor<[4,5],f32>
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %true = torch.constant.bool true
  %2 = torch.prim.ListConstruct %int4, %int5 : (!torch.int, !torch.int) -> !torch.list<int>
  %result0, %result1, %result2 = torch.aten.native_layer_norm %arg0, %2, %1, %0, %float1.000000e-05 : !torch.vtensor<[?,7,4,5],f32>, !torch.list<int>, !torch.vtensor<[4,5],f32>, !torch.vtensor<[4,5],f32>, !torch.float -> !torch.vtensor<[?,7,4,5],f32>, !torch.vtensor<[?,7,1,1],f32>, !torch.vtensor<[?,7,1,1],f32>
  return %result0 : !torch.vtensor<[?,7,4,5],f32>
}

// -----

// Test: native_layer_norm with multiple dynamic batch dims
// CHECK-LABEL: func @torch.aten.native_layer_norm_dynamic_multi_batch
// CHECK-SAME:    %[[INPUT:.*]]: !torch.vtensor<[?,?,64],f32>) -> !torch.vtensor<[?,?,64],f32>
// CHECK:         %[[T:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[?,?,64],f32> -> tensor<?x?x64xf32>
// CHECK:         %[[C1:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK:         %[[D0:.*]] = stablehlo.get_dimension_size %[[T]], dim = 0
// CHECK:         %[[F0:.*]] = stablehlo.multiply %[[C1]], %[[D0]] : tensor<i32>
// CHECK:         %[[D1:.*]] = stablehlo.get_dimension_size %[[T]], dim = 1
// CHECK:         %[[FEAT:.*]] = stablehlo.multiply %[[F0]], %[[D1]] : tensor<i32>
// CHECK:         stablehlo.dynamic_reshape {{.*}} -> tensor<1x?x64xf32>
// CHECK:         "stablehlo.batch_norm_training"
// CHECK:         stablehlo.dynamic_reshape {{.*}} -> tensor<?x?x64xf32>
// CHECK:         stablehlo.dynamic_reshape {{.*}} -> tensor<?x?x1xf32>
func.func @torch.aten.native_layer_norm_dynamic_multi_batch(%arg0: !torch.vtensor<[?,?,64],f32>) -> !torch.vtensor<[?,?,64],f32> {
  %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<64xf32>) : !torch.vtensor<[64],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<64xf32>) : !torch.vtensor<[64],f32>
  %int64 = torch.constant.int 64
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %true = torch.constant.bool true
  %2 = torch.prim.ListConstruct %int64 : (!torch.int) -> !torch.list<int>
  %result0, %result1, %result2 = torch.aten.native_layer_norm %arg0, %2, %1, %0, %float1.000000e-05 : !torch.vtensor<[?,?,64],f32>, !torch.list<int>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.float -> !torch.vtensor<[?,?,64],f32>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,1],f32>
  return %result0 : !torch.vtensor<[?,?,64],f32>
}

// -----

// Negative test: native_layer_norm with dynamic normalized dim should fail
func.func @torch.aten.native_layer_norm_dynamic_norm_dim(%arg0: !torch.vtensor<[3,?,?],f32>) -> !torch.vtensor<[3,?,?],f32> {
  %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<4x5xf32>) : !torch.vtensor<[4,5],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<4x5xf32>) : !torch.vtensor<[4,5],f32>
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %true = torch.constant.bool true
  %2 = torch.prim.ListConstruct %int4, %int5 : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @+1 {{failed to legalize operation 'torch.aten.native_layer_norm' that was explicitly marked illegal}}
  %result0, %result1, %result2 = torch.aten.native_layer_norm %arg0, %2, %1, %0, %float1.000000e-05 : !torch.vtensor<[3,?,?],f32>, !torch.list<int>, !torch.vtensor<[4,5],f32>, !torch.vtensor<[4,5],f32>, !torch.float -> !torch.vtensor<[3,?,?],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>
  return %result0 : !torch.vtensor<[3,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.constant_pad_nd$static(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,6],f32> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],f32> -> tensor<3x4xf32>
// CHECK:           %[[FLOAT:.*]] = torch.constant.float 0.000000e+00
// CHECK:           %[[F64_VAL:.*]] = torch_c.to_f64 %[[FLOAT]]
// CHECK:           %[[FE:.*]] = tensor.from_elements %[[F64_VAL]] : tensor<1xf64>
// CHECK:           %[[CVT:.*]] = stablehlo.convert %[[FE]] : (tensor<1xf64>) -> tensor<1xf32>
// CHECK:           %[[PAD_VAL:.*]] = stablehlo.reshape %[[CVT]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:           %[[RESULT:.*]] = stablehlo.pad %[[INPUT]], %[[PAD_VAL]], low = [0, 1], high = [0, 1], interior = [0, 0] : (tensor<3x4xf32>, tensor<f32>) -> tensor<3x6xf32>
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<3x6xf32> -> !torch.vtensor<[3,6],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[3,6],f32>
// CHECK:         }
func.func @torch.aten.constant_pad_nd$static(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,6],f32> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %float0 = torch.constant.float 0.0
    %pad = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %0 = torch.aten.constant_pad_nd %arg0, %pad, %float0 : !torch.vtensor<[3,4],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[3,6],f32>
    return %0 : !torch.vtensor<[3,6],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.constant_pad_nd$dynamic_pad(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                               %[[LOW:.*]]: !torch.int,
// CHECK-SAME:                               %[[HIGH:.*]]: !torch.int) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[FLOAT:.*]] = torch.constant.float 0.000000e+00
// CHECK:           %[[F64_VAL:.*]] = torch_c.to_f64 %[[FLOAT]]
// CHECK:           %[[FE:.*]] = tensor.from_elements %[[F64_VAL]] : tensor<1xf64>
// CHECK:           %[[CVT:.*]] = stablehlo.convert %[[FE]] : (tensor<1xf64>) -> tensor<1xf32>
// CHECK:           %[[PAD_VAL:.*]] = stablehlo.reshape %[[CVT]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:           %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK:           %[[LOW_I64:.*]] = torch_c.to_i64 %[[LOW]]
// CHECK:           %[[HIGH_I64:.*]] = torch_c.to_i64 %[[HIGH]]
// CHECK:           %[[PAD_LOW:.*]] = tensor.from_elements %[[ZERO]], %[[LOW_I64]] : tensor<2xi64>
// CHECK:           %[[PAD_HIGH:.*]] = tensor.from_elements %[[ZERO]], %[[HIGH_I64]] : tensor<2xi64>
// CHECK:           %[[PAD_INTERIOR:.*]] = tensor.from_elements %[[ZERO]], %[[ZERO]] : tensor<2xi64>
// CHECK:           %[[RESULT:.*]] = stablehlo.dynamic_pad %[[INPUT]], %[[PAD_VAL]], %[[PAD_LOW]], %[[PAD_HIGH]], %[[PAD_INTERIOR]] : (tensor<?x?xf32>, tensor<f32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x?xf32>
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.constant_pad_nd$dynamic_pad(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.vtensor<[?,?],f32> {
    %float0 = torch.constant.float 0.0
    %pad = torch.prim.ListConstruct %arg1, %arg2 : (!torch.int, !torch.int) -> !torch.list<int>
    %0 = torch.aten.constant_pad_nd %arg0, %pad, %float0 : !torch.vtensor<[?,?],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[?,?],f32>
    return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.constant_pad_nd$mixed_static_dynamic_pad(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[?,?,?],f32>,
// CHECK-SAME:                               %[[DYN_HIGH:.*]]: !torch.int) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:           %[[INT2:.*]] = torch.constant.int 2
// CHECK:           %[[FLOAT:.*]] = torch.constant.float 0.000000e+00
// CHECK:           %[[F64_VAL:.*]] = torch_c.to_f64 %[[FLOAT]]
// CHECK:           %[[FE:.*]] = tensor.from_elements %[[F64_VAL]] : tensor<1xf64>
// CHECK:           %[[CVT:.*]] = stablehlo.convert %[[FE]] : (tensor<1xf64>) -> tensor<1xf32>
// CHECK:           %[[PAD_VAL:.*]] = stablehlo.reshape %[[CVT]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:           %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK:           %[[CONST2:.*]] = torch_c.to_i64 %[[INT2]]
// CHECK:           %[[HIGH_I64:.*]] = torch_c.to_i64 %[[DYN_HIGH]]
// CHECK:           %[[PAD_LOW:.*]] = tensor.from_elements %[[ZERO]], %[[ZERO]], %[[CONST2]] : tensor<3xi64>
// CHECK:           %[[PAD_HIGH:.*]] = tensor.from_elements %[[ZERO]], %[[ZERO]], %[[HIGH_I64]] : tensor<3xi64>
// CHECK:           %[[PAD_INTERIOR:.*]] = tensor.from_elements %[[ZERO]], %[[ZERO]], %[[ZERO]] : tensor<3xi64>
// CHECK:           %[[RESULT:.*]] = stablehlo.dynamic_pad %[[INPUT]], %[[PAD_VAL]], %[[PAD_LOW]], %[[PAD_HIGH]], %[[PAD_INTERIOR]] : (tensor<?x?x?xf32>, tensor<f32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x?x?xf32>
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,?,?],f32>
// CHECK:         }
func.func @torch.aten.constant_pad_nd$mixed_static_dynamic_pad(%arg0: !torch.vtensor<[?,?,?],f32>, %arg1: !torch.int) -> !torch.vtensor<[?,?,?],f32> {
    %int2 = torch.constant.int 2
    %float0 = torch.constant.float 0.0
    %pad = torch.prim.ListConstruct %int2, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
    %0 = torch.aten.constant_pad_nd %arg0, %pad, %float0 : !torch.vtensor<[?,?,?],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[?,?,?],f32>
    return %0 : !torch.vtensor<[?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.constant_pad_nd$dynamic_pad_multi_dim(
// CHECK-SAME:      %[[VAL_0:[a-zA-Z0-9_]+]]: !torch.vtensor<[?,?,?],f32>,
// CHECK-SAME:      %[[P0:[a-zA-Z0-9_]+]]: !torch.int,
// CHECK-SAME:      %[[P1:[a-zA-Z0-9_]+]]: !torch.int,
// CHECK-SAME:      %[[P2:[a-zA-Z0-9_]+]]: !torch.int,
// CHECK-SAME:      %[[P3:[a-zA-Z0-9_]+]]: !torch.int) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:           %[[FLOAT:.*]] = torch.constant.float 0.000000e+00
// CHECK:           %[[F64_VAL:.*]] = torch_c.to_f64 %[[FLOAT]]
// CHECK:           %[[FE:.*]] = tensor.from_elements %[[F64_VAL]] : tensor<1xf64>
// CHECK:           %[[CVT:.*]] = stablehlo.convert %[[FE]] : (tensor<1xf64>) -> tensor<1xf32>
// CHECK:           %[[PAD_VAL:.*]] = stablehlo.reshape %[[CVT]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:           %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK:           %[[P0_I64:.*]] = torch_c.to_i64 %[[P0]]
// CHECK:           %[[P1_I64:.*]] = torch_c.to_i64 %[[P1]]
// CHECK:           %[[P2_I64:.*]] = torch_c.to_i64 %[[P2]]
// CHECK:           %[[P3_I64:.*]] = torch_c.to_i64 %[[P3]]
// CHECK:           %[[PAD_LOW:.*]] = tensor.from_elements %[[ZERO]], %[[P2_I64]], %[[P0_I64]] : tensor<3xi64>
// CHECK:           %[[PAD_HIGH:.*]] = tensor.from_elements %[[ZERO]], %[[P3_I64]], %[[P1_I64]] : tensor<3xi64>
// CHECK:           %[[PAD_INTERIOR:.*]] = tensor.from_elements %[[ZERO]], %[[ZERO]], %[[ZERO]] : tensor<3xi64>
// CHECK:           %[[RESULT:.*]] = stablehlo.dynamic_pad %[[INPUT]], %[[PAD_VAL]], %[[PAD_LOW]], %[[PAD_HIGH]], %[[PAD_INTERIOR]] : (tensor<?x?x?xf32>, tensor<f32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x?x?xf32>
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,?,?],f32>
// CHECK:         }
func.func @torch.aten.constant_pad_nd$dynamic_pad_multi_dim(%arg0: !torch.vtensor<[?,?,?],f32>, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.int, %arg4: !torch.int) -> !torch.vtensor<[?,?,?],f32> {
    %float0 = torch.constant.float 0.0
    %pad = torch.prim.ListConstruct %arg1, %arg2, %arg3, %arg4 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %0 = torch.aten.constant_pad_nd %arg0, %pad, %float0 : !torch.vtensor<[?,?,?],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[?,?,?],f32>
    return %0 : !torch.vtensor<[?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.constant_pad_nd$dynamic_pad_nonzero_fill(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[?,?],si32>,
// CHECK-SAME:                               %[[LOW:.*]]: !torch.int,
// CHECK-SAME:                               %[[HIGH:.*]]: !torch.int) -> !torch.vtensor<[?,?],si32> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],si32> -> tensor<?x?xi32>
// CHECK:           %[[INT_NEG1:.*]] = torch.constant.int -1
// CHECK:           %[[C_NEG1:.*]] = arith.constant -1 : i64
// CHECK:           %[[FE:.*]] = tensor.from_elements %[[C_NEG1]] : tensor<1xi64>
// CHECK:           %[[CVT:.*]] = stablehlo.convert %[[FE]] : (tensor<1xi64>) -> tensor<1xi32>
// CHECK:           %[[PAD_VAL:.*]] = stablehlo.reshape %[[CVT]] : (tensor<1xi32>) -> tensor<i32>
// CHECK:           %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK:           %[[LOW_I64:.*]] = torch_c.to_i64 %[[LOW]]
// CHECK:           %[[HIGH_I64:.*]] = torch_c.to_i64 %[[HIGH]]
// CHECK:           %[[PAD_LOW:.*]] = tensor.from_elements %[[ZERO]], %[[LOW_I64]] : tensor<2xi64>
// CHECK:           %[[PAD_HIGH:.*]] = tensor.from_elements %[[ZERO]], %[[HIGH_I64]] : tensor<2xi64>
// CHECK:           %[[PAD_INTERIOR:.*]] = tensor.from_elements %[[ZERO]], %[[ZERO]] : tensor<2xi64>
// CHECK:           %[[RESULT:.*]] = stablehlo.dynamic_pad %[[INPUT]], %[[PAD_VAL]], %[[PAD_LOW]], %[[PAD_HIGH]], %[[PAD_INTERIOR]] : (tensor<?x?xi32>, tensor<i32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x?xi32>
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<?x?xi32> -> !torch.vtensor<[?,?],si32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,?],si32>
// CHECK:         }
func.func @torch.aten.constant_pad_nd$dynamic_pad_nonzero_fill(%arg0: !torch.vtensor<[?,?],si32>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.vtensor<[?,?],si32> {
    %int_neg1 = torch.constant.int -1
    %pad = torch.prim.ListConstruct %arg1, %arg2 : (!torch.int, !torch.int) -> !torch.list<int>
    %0 = torch.aten.constant_pad_nd %arg0, %pad, %int_neg1 : !torch.vtensor<[?,?],si32>, !torch.list<int>, !torch.int -> !torch.vtensor<[?,?],si32>
    return %0 : !torch.vtensor<[?,?],si32>
}

// -----

// -----

// Test: reflection_pad1d with dynamic batch dim (padded dim is static)
// CHECK-LABEL: func @torch.aten.reflection_pad1d_dynamic_batch
// CHECK-SAME:    %[[INPUT:.*]]: !torch.vtensor<[?,8],f32>) -> !torch.vtensor<[?,12],f32>
// CHECK:         %[[T:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[?,8],f32> -> tensor<?x8xf32>
// CHECK:         stablehlo.get_dimension_size
// CHECK:         stablehlo.real_dynamic_slice %[[T]]
// CHECK:         stablehlo.reverse
// CHECK:         stablehlo.real_dynamic_slice %[[T]]
// CHECK:         stablehlo.reverse
// CHECK:         stablehlo.concatenate
func.func @torch.aten.reflection_pad1d_dynamic_batch(%arg0: !torch.vtensor<[?,8],f32>) -> !torch.vtensor<[?,12],f32> {
  %int2 = torch.constant.int 2
  %int2_0 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int2, %int2_0 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.reflection_pad1d %arg0, %0 : !torch.vtensor<[?,8],f32>, !torch.list<int> -> !torch.vtensor<[?,12],f32>
  return %1 : !torch.vtensor<[?,12],f32>
}

// -----

// Test: reflection_pad1d with dynamic batch and channel dims
// CHECK-LABEL: func @torch.aten.reflection_pad1d_dynamic_batch_channel
// CHECK-SAME:    %[[INPUT:.*]]: !torch.vtensor<[?,?,10],f32>) -> !torch.vtensor<[?,?,16],f32>
// CHECK:         %[[T:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[?,?,10],f32> -> tensor<?x?x10xf32>
// CHECK:         stablehlo.get_dimension_size
// CHECK:         stablehlo.real_dynamic_slice %[[T]]
// CHECK:         stablehlo.reverse
// CHECK:         stablehlo.real_dynamic_slice %[[T]]
// CHECK:         stablehlo.reverse
// CHECK:         stablehlo.concatenate
func.func @torch.aten.reflection_pad1d_dynamic_batch_channel(%arg0: !torch.vtensor<[?,?,10],f32>) -> !torch.vtensor<[?,?,16],f32> {
  %int3 = torch.constant.int 3
  %int3_0 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int3, %int3_0 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.reflection_pad1d %arg0, %0 : !torch.vtensor<[?,?,10],f32>, !torch.list<int> -> !torch.vtensor<[?,?,16],f32>
  return %1 : !torch.vtensor<[?,?,16],f32>
}

// -----

// Test: reflection_pad1d with fully dynamic shape (including padded dim)
// CHECK-LABEL: func @torch.aten.reflection_pad1d_dynamic_padded_dim
// CHECK-SAME:    %[[INPUT:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32>
// CHECK:         %[[T:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:         stablehlo.get_dimension_size
// CHECK:         stablehlo.real_dynamic_slice %[[T]]
// CHECK:         stablehlo.reverse
// CHECK:         stablehlo.subtract
// CHECK:         stablehlo.subtract
// CHECK:         stablehlo.real_dynamic_slice %[[T]]
// CHECK:         stablehlo.reverse
// CHECK:         stablehlo.concatenate
func.func @torch.aten.reflection_pad1d_dynamic_padded_dim(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int1, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.reflection_pad1d %arg0, %0 : !torch.vtensor<[?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?],f32>
  return %1 : !torch.vtensor<[?,?],f32>
}

// -----

// -----

// CHECK-LABEL:   func.func @torch.aten.clone$same_type(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           return %[[VAL_0]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.clone$same_type(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
    %int0 = torch.constant.int 0
    %0 = torch.aten.clone %arg0, %int0 : !torch.vtensor<[3,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>
    return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.clone$shape_refinement(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[1,?,?],f32> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:           %[[CAST:.*]] = tensor.cast %[[INPUT]] : tensor<?x?x?xf32> to tensor<1x?x?xf32>
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[CAST]] : tensor<1x?x?xf32> -> !torch.vtensor<[1,?,?],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[1,?,?],f32>
// CHECK:         }
func.func @torch.aten.clone$shape_refinement(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[1,?,?],f32> {
    %int0 = torch.constant.int 0
    %0 = torch.aten.clone %arg0, %int0 : !torch.vtensor<[?,?,?],f32>, !torch.int -> !torch.vtensor<[1,?,?],f32>
    return %0 : !torch.vtensor<[1,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.clone$dynamic_identity(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           return %[[VAL_0]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.clone$dynamic_identity(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
    %none = torch.constant.none
    %0 = torch.aten.clone %arg0, %none : !torch.vtensor<[?,?],f32>, !torch.none -> !torch.vtensor<[?,?],f32>
    return %0 : !torch.vtensor<[?,?],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.runtime.assert$dynamic(
// CHECK-SAME:                                    %[[ARG_0:.*]]: !torch.vtensor<[?,?],f32>, %[[ARG_1:.*]]: !torch.bool) -> !torch.vtensor<[?,?],f32> {
// CHECK-NOT:       torch.runtime.assert
// CHECK:           return %[[ARG_0]] : !torch.vtensor<[?,?],f32>
func.func @torch.runtime.assert$dynamic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.bool) -> !torch.vtensor<[?,?],f32> {
  torch.runtime.assert %arg1, "dynamic shape guard"
  return %arg0: !torch.vtensor<[?,?],f32>
}

// -----


// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten._native_batch_norm_legit_functional(
// CHECK-SAME:                                                              %arg0: !torch.vtensor<[1,3,8,8],f32>,
// CHECK-SAME:                                                              %arg1: !torch.vtensor<[],si64>,
// CHECK-SAME:                                                              %arg2: !torch.vtensor<[3],f32>,
// CHECK-SAME:                                                              %arg3: !torch.vtensor<[3],f32>,
// CHECK-SAME:                                                              %arg4: !torch.vtensor<[3],f32>,
// CHECK-SAME:                                                              %arg5: !torch.vtensor<[3],f32>) -> (!torch.vtensor<[],si64>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[1,3,8,8],f32>) {
// CHECK:           %[[VAL_6:.*]] = torch_c.to_builtin_tensor %arg3 : !torch.vtensor<[3],f32> -> tensor<3xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[3],f32> -> tensor<3xf32>
// CHECK:           %[[VAL_8:.*]] = torch_c.to_builtin_tensor %arg5 : !torch.vtensor<[3],f32> -> tensor<3xf32>
// CHECK:           %[[VAL_9:.*]] = torch_c.to_builtin_tensor %arg4 : !torch.vtensor<[3],f32> -> tensor<3xf32>
// CHECK:           %[[VAL_10:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[1,3,8,8],f32> -> tensor<1x3x8x8xf32>
// CHECK:           %[[VAL_11:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[],si64> -> tensor<i64>
// CHECK:           %[[VAL_12:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_13:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_14:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_15:.*]] = tensor.from_elements %[[VAL_13]] : tensor<1xi64>
// CHECK:           %[[VAL_16:.*]] = stablehlo.convert %[[VAL_15]] : tensor<1xi64>
// CHECK:           %[[VAL_17:.*]] = stablehlo.reshape %[[VAL_16]] : (tensor<1xi64>) -> tensor<i64>
// CHECK:           %[[VAL_18:.*]] = chlo.broadcast_add %[[VAL_11]], %[[VAL_17]] : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:           %[[VAL_19:.*]] = torch_c.from_builtin_tensor %[[VAL_18]] : tensor<i64> -> !torch.vtensor<[],si64>
// CHECK:           %[[VAL_20:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_21:.*]] = torch.constant.float 0.000000e+00
// CHECK:           %[[VAL_22:.*]] = torch.constant.float 1.000000e-05
// CHECK:           %[[VAL_23:.*]], %[[VAL_24:.*]], %[[VAL_25:.*]] = "stablehlo.batch_norm_training"(%[[VAL_10]], %[[VAL_9]], %[[VAL_8]]) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x3x8x8xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<1x3x8x8xf32>, tensor<3xf32>, tensor<3xf32>)
// CHECK:           %[[VAL_26:.*]] = torch_c.from_builtin_tensor %[[VAL_23]] : tensor<1x3x8x8xf32> -> !torch.vtensor<[1,3,8,8],f32>
// CHECK:           %[[VAL_27:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_28:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_29:.*]] = stablehlo.subtract %[[VAL_28]], %[[VAL_27]] : tensor<f32>
// CHECK:           %[[VAL_30:.*]] = chlo.broadcast_multiply %[[VAL_29]], %[[VAL_7]] : (tensor<f32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK:           %[[VAL_31:.*]] = chlo.broadcast_multiply %[[VAL_27]], %[[VAL_24]] : (tensor<f32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK:           %[[VAL_32:.*]] = stablehlo.add %[[VAL_30]], %[[VAL_31]] : tensor<3xf32>
// CHECK:           %[[VAL_33:.*]] = torch_c.from_builtin_tensor %[[VAL_32]] : tensor<3xf32> -> !torch.vtensor<[3],f32>
// CHECK:           %[[VAL_34:.*]] = chlo.broadcast_multiply %[[VAL_29]], %[[VAL_6]] : (tensor<f32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK:           %[[VAL_35:.*]] = chlo.broadcast_multiply %[[VAL_27]], %[[VAL_25]] : (tensor<f32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK:           %[[VAL_36:.*]] = stablehlo.add %[[VAL_34]], %[[VAL_35]] : tensor<3xf32>
// CHECK:           %[[VAL_37:.*]] = torch_c.from_builtin_tensor %[[VAL_36]] : tensor<3xf32> -> !torch.vtensor<[3],f32>
// CHECK:           return %[[VAL_19]], %[[VAL_33]], %[[VAL_37]], %[[VAL_26]] : !torch.vtensor<[],si64>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[1,3,8,8],f32>
// CHECK:         }
func.func @torch.aten._native_batch_norm_legit_functional(%arg0: !torch.vtensor<[1,3,8,8],f32>, %arg1: !torch.vtensor<[],si64>, %arg2: !torch.vtensor<[3],f32>, %arg3: !torch.vtensor<[3],f32>, %arg4: !torch.vtensor<[3],f32>, %arg5: !torch.vtensor<[3],f32>) -> (!torch.vtensor<[],si64>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[1,3,8,8],f32>) {
    %int1 = torch.constant.int 1
    %int1_0 = torch.constant.int 1
    %0 = torch.aten.add.Scalar %arg1, %int1, %int1_0 : !torch.vtensor<[],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
    %true = torch.constant.bool true
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %float1.000000e-05 = torch.constant.float 1.000000e-05
    %result0, %result1, %result2, %running_mean_out, %running_var_out = torch.aten._native_batch_norm_legit_functional %arg0, %arg4, %arg5, %arg2, %arg3, %true, %float0.000000e00, %float1.000000e-05 : !torch.vtensor<[1,3,8,8],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.bool, !torch.float, !torch.float -> !torch.vtensor<[1,3,8,8],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>
    return %0, %running_mean_out, %running_var_out, %result0 : !torch.vtensor<[],si64>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[1,3,8,8],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten._native_batch_norm_legit_no_training(
// CHECK-SAME:                                                               %arg0: !torch.vtensor<[1,3,8,8],f32>,
// CHECK-SAME:                                                               %arg1: !torch.vtensor<[3],f32>,
// CHECK-SAME:                                                               %arg2: !torch.vtensor<[3],f32>,
// CHECK-SAME:                                                               %arg3: !torch.vtensor<[3],f32>,
// CHECK-SAME:                                                               %arg4: !torch.vtensor<[3],f32>) -> !torch.vtensor<[1,3,8,8],f32> {
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[3],f32> -> tensor<3xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[3],f32> -> tensor<3xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.to_builtin_tensor %arg4 : !torch.vtensor<[3],f32> -> tensor<3xf32>
// CHECK:           %[[VAL_8:.*]] = torch_c.to_builtin_tensor %arg3 : !torch.vtensor<[3],f32> -> tensor<3xf32>
// CHECK:           %[[VAL_9:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[1,3,8,8],f32> -> tensor<1x3x8x8xf32>
// CHECK:           %[[VAL_10:.*]] = torch.constant.float 1.000000e-01
// CHECK:           %[[VAL_11:.*]] = torch.constant.float 1.000000e-05
// CHECK:           %[[VAL_12:.*]] = "stablehlo.batch_norm_inference"(%[[VAL_9]], %[[VAL_8]], %[[VAL_7]], %[[VAL_6]], %[[VAL_5]]) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x3x8x8xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<1x3x8x8xf32>
// CHECK:           %[[VAL_13:.*]] = torch_c.from_builtin_tensor %[[VAL_12]] : tensor<1x3x8x8xf32> -> !torch.vtensor<[1,3,8,8],f32>
// CHECK:           %[[VAL_14:.*]] = stablehlo.constant dense<> : tensor<0xf32>
// CHECK:           %[[VAL_15:.*]] = stablehlo.constant dense<> : tensor<0xf32>
// CHECK:           return %[[VAL_13]] : !torch.vtensor<[1,3,8,8],f32>
// CHECK:         }
func.func @torch.aten._native_batch_norm_legit_no_training(%arg0: !torch.vtensor<[1,3,8,8],f32>, %arg1: !torch.vtensor<[3],f32>, %arg2: !torch.vtensor<[3],f32>, %arg3: !torch.vtensor<[3],f32>, %arg4: !torch.vtensor<[3],f32>) -> !torch.vtensor<[1,3,8,8],f32> {
    %float1.000000e-01 = torch.constant.float 1.000000e-01
    %float1.000000e-05 = torch.constant.float 1.000000e-05
    %result0, %result1, %result2 = torch.aten._native_batch_norm_legit_no_training %arg0, %arg3, %arg4, %arg1, %arg2, %float1.000000e-01, %float1.000000e-05 : !torch.vtensor<[1,3,8,8],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,3,8,8],f32>, !torch.vtensor<[0],f32>, !torch.vtensor<[0],f32>
    return %result0 : !torch.vtensor<[1,3,8,8],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten._native_batch_norm_legit.no_stats(
// CHECK-SAME:                                                            %[[VAL_0:.*]]: !torch.vtensor<[1,3,8,8],f32>,
// CHECK-SAME:                                                            %[[VAL_1:.*]]: !torch.vtensor<[3],f32>,
// CHECK-SAME:                                                            %[[VAL_2:.*]]: !torch.vtensor<[3],f32>) -> !torch.vtensor<[1,3,8,8],f32> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[3],f32> -> tensor<3xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[3],f32> -> tensor<3xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,3,8,8],f32> -> tensor<1x3x8x8xf32>
// CHECK:           %[[VAL_6:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_7:.*]] = torch.constant.float 1.000000e-01
// CHECK:           %[[VAL_8:.*]] = torch.constant.float 1.000000e-05
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]], %[[VAL_11:.*]] = "stablehlo.batch_norm_training"(%[[VAL_5]], %[[VAL_4]], %[[VAL_3]]) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x3x8x8xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<1x3x8x8xf32>, tensor<3xf32>, tensor<3xf32>)
// CHECK:           %[[VAL_12:.*]] = torch_c.from_builtin_tensor %[[VAL_9]] : tensor<1x3x8x8xf32> -> !torch.vtensor<[1,3,8,8],f32>
// CHECK:           return %[[VAL_12]] : !torch.vtensor<[1,3,8,8],f32>
// CHECK:         }
func.func @torch.aten._native_batch_norm_legit.no_stats(%arg0: !torch.vtensor<[1,3,8,8],f32>, %arg1: !torch.vtensor<[3],f32>, %arg2: !torch.vtensor<[3],f32>) -> !torch.vtensor<[1,3,8,8],f32> {
    %true = torch.constant.bool true
    %float1.000000e-01 = torch.constant.float 1.000000e-01
    %float1.000000e-05 = torch.constant.float 1.000000e-05
    %result0, %result1, %result2 = torch.aten._native_batch_norm_legit.no_stats %arg0, %arg1, %arg2, %true, %float1.000000e-01, %float1.000000e-05 : !torch.vtensor<[1,3,8,8],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.bool, !torch.float, !torch.float -> !torch.vtensor<[1,3,8,8],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>
    return %result0 : !torch.vtensor<[1,3,8,8],f32>
}

// RUN: torch-mlir-opt -pass-pipeline='builtin.module(torch-raw-to-neuron-backend-pipeline)' -split-input-file %s | FileCheck %s

module {
  func.func @forward(%arg0: !torch.vtensor<[32,96,64],bf16>, %arg1: !torch.vtensor<[32,64,96],bf16>, %arg2: !torch.vtensor<[32,64,96],bf16>) -> !torch.vtensor<[32,96,64],bf16> {
    // CHECK: %[[VAL_0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[32,96,64],bf16> -> tensor<32x96x64xbf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %none_1 = torch.constant.none
    %false = torch.constant.bool false
    %none_2 = torch.constant.none

    %0 = torch.aten.zeros_like %arg0, %none, %none_0, %none_1, %false, %none_2 : !torch.vtensor<[32,96,64],bf16>, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[32,96,64],bf16>
    // CHECK: %[[VAL_2:.*]] = stablehlo.custom_call @AwsNeuronCustomNativeKernel(%[[INP_0:.*]], %[[INP_1:.*]], %[[INP_2:.*]], %[[VAL_3:.*]]) {backend_config = "eyJrZXJJ9"} : (tensor<32x64x96xbf16>, tensor<32x64x96xbf16>, tensor<32x96x64xbf16>, tensor<32x96x64xbf16>) -> tensor<32x96x64xbf16>
    %2 = torch.operator "stablehlo.custom_call"(%arg2, %arg1, %arg0, %0) {backend_config = "eyJrZXJJ9", call_target_name = "AwsNeuronCustomNativeKernel"} : (!torch.vtensor<[32,64,96],bf16>, !torch.vtensor<[32,64,96],bf16>, !torch.vtensor<[32,96,64],bf16>, !torch.vtensor<[32,96,64],bf16>) ->  !torch.vtensor<[32,96,64],bf16>
    // CHECK: %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<32x96x64xbf16> -> !torch.vtensor<[32,96,64],bf16>
    return %2 : !torch.vtensor<[32,96,64],bf16>
  }
}

// -----

// CHECK-LABEL:   func.func @forward_with_side_effect(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[1,128],si32>) -> !torch.vtensor<[1,128],si32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,128],si32> -> tensor<1x128xi32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.custom_call @AwsNeuronCustomNativeKernel(%[[VAL_1]]) {backend_config = "eyJrZXJJ9", has_side_effect = true} : (tensor<1x128xi32>) -> tensor<1x128xi32>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<1x128xi32> -> !torch.vtensor<[1,128],si32>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[1,128],si32>
// CHECK:         }
module {
  func.func @forward_with_side_effect(%arg0: !torch.vtensor<[1,128],si32>) -> !torch.vtensor<[1,128],si32> {
    %0 = torch.operator "stablehlo.custom_call"(%arg0) {backend_config = "eyJrZXJJ9", call_target_name = "AwsNeuronCustomNativeKernel", has_side_effect = true} : (!torch.vtensor<[1,128],si32>) -> !torch.vtensor<[1,128],si32>
    return %0 : !torch.vtensor<[1,128],si32>
  }
}

// -----


// CHECK-LABEL:   func.func @forward_without_side_effect(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !torch.vtensor<[1,128],si32>) -> !torch.vtensor<[1,128],si32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,128],si32> -> tensor<1x128xi32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.custom_call @AwsNeuronCustomNativeKernel(%[[VAL_1]]) {backend_config = "eyJrZXJJ9"} : (tensor<1x128xi32>) -> tensor<1x128xi32>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<1x128xi32> -> !torch.vtensor<[1,128],si32>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[1,128],si32>
// CHECK:         }
module {
  func.func @forward_without_side_effect(%arg0: !torch.vtensor<[1,128],si32>) -> !torch.vtensor<[1,128],si32> {
    %0 = torch.operator "stablehlo.custom_call"(%arg0) {backend_config = "eyJrZXJJ9", call_target_name = "AwsNeuronCustomNativeKernel"} : (!torch.vtensor<[1,128],si32>) -> !torch.vtensor<[1,128],si32>
    return %0 : !torch.vtensor<[1,128],si32>
  }
}

// RUN: WORLD_SIZE=8 torch-mlir-opt -pass-pipeline='builtin.module(torch-raw-to-neuron-backend-pipeline)' %s | FileCheck %s
// XFAIL: *

module {
  // CHECK-LABEL: func.func @test_all_gather_coalesced_default
  func.func @test_all_gather_coalesced_default(%arg0: !torch.vtensor<[2,4],f32>, %arg1: !torch.vtensor<[1,2],f32>) -> (!torch.vtensor<[16,4],f32>, !torch.vtensor<[8,2],f32>) {
    %int8 = torch.constant.int 8
    %group = torch.constant.str "default"

    // CHECK: %[[INPUT0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[2,4],f32> -> tensor<2x4xf32>
    // CHECK: %[[INPUT1:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,2],f32> -> tensor<1x2xf32>
    // CHECK: %[[RESULT:.*]]:2 = "stablehlo.all_gather"(%[[INPUT0]], %[[INPUT1]]) <{all_gather_dim = 0 : i64, channel_handle = {{.*}}, replica_groups = dense<{{\[}}[0, 1, 2, 3, 4, 5, 6, 7]{{\]}}> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<2x4xf32>, tensor<1x2xf32>) -> (tensor<16x4xf32>, tensor<8x2xf32>)
    // CHECK: %[[TUPLE:.*]] = stablehlo.tuple %[[RESULT]]#0, %[[RESULT]]#1 : tuple<tensor<16x4xf32>, tensor<8x2xf32>>
    // CHECK: %[[ELEM0:.*]] = stablehlo.get_tuple_element %[[TUPLE]][0] : (tuple<tensor<16x4xf32>, tensor<8x2xf32>>) -> tensor<16x4xf32>
    // CHECK: %[[OUTPUT0:.*]] = torch_c.from_builtin_tensor %[[ELEM0]] : tensor<16x4xf32> -> !torch.vtensor<[16,4],f32>
    // CHECK: %[[ELEM1:.*]] = stablehlo.get_tuple_element %[[TUPLE]][1] : (tuple<tensor<16x4xf32>, tensor<8x2xf32>>) -> tensor<8x2xf32>
    // CHECK: %[[OUTPUT1:.*]] = torch_c.from_builtin_tensor %[[ELEM1]] : tensor<8x2xf32> -> !torch.vtensor<[8,2],f32>
    %list = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,4],f32>, !torch.vtensor<[1,2],f32>) -> !torch.list<vtensor>
    %result = torch.operator "torch._c10d_functional.all_gather_into_tensor_coalesced"(%list, %int8, %group) : (!torch.list<vtensor>, !torch.int, !torch.str) -> !torch.list<vtensor>
    %out0, %out1 = torch.prim.ListUnpack %result : !torch.list<vtensor> -> !torch.vtensor<[16,4],f32>, !torch.vtensor<[8,2],f32>

    // CHECK: return %[[OUTPUT0]], %[[OUTPUT1]]
    return %out0, %out1 : !torch.vtensor<[16,4],f32>, !torch.vtensor<[8,2],f32>
  }

  // CHECK-LABEL: func.func @test_all_gather_coalesced_partial
  func.func @test_all_gather_coalesced_partial(%arg0: !torch.vtensor<[2,4],f32>, %arg1: !torch.vtensor<[1,2],f32>) -> (!torch.vtensor<[8,4],f32>, !torch.vtensor<[4,2],f32>) {
    %int4 = torch.constant.int 4
    %group = torch.constant.str "[[0, 2, 4, 6]]"

    // CHECK: %[[INPUT0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[2,4],f32> -> tensor<2x4xf32>
    // CHECK: %[[INPUT1:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,2],f32> -> tensor<1x2xf32>
    // CHECK: %[[RESULT:.*]]:2 = "stablehlo.all_gather"(%[[INPUT0]], %[[INPUT1]]) <{all_gather_dim = 0 : i64, channel_handle = {{.*}}, replica_groups = dense<{{\[}}[0, 2, 4, 6]{{\]}}> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<2x4xf32>, tensor<1x2xf32>) -> (tensor<8x4xf32>, tensor<4x2xf32>)
    // CHECK: %[[TUPLE:.*]] = stablehlo.tuple %[[RESULT]]#0, %[[RESULT]]#1 : tuple<tensor<8x4xf32>, tensor<4x2xf32>>
    // CHECK: %[[ELEM0:.*]] = stablehlo.get_tuple_element %[[TUPLE]][0] : (tuple<tensor<8x4xf32>, tensor<4x2xf32>>) -> tensor<8x4xf32>
    // CHECK: %[[OUTPUT0:.*]] = torch_c.from_builtin_tensor %[[ELEM0]] : tensor<8x4xf32> -> !torch.vtensor<[8,4],f32>
    // CHECK: %[[ELEM1:.*]] = stablehlo.get_tuple_element %[[TUPLE]][1] : (tuple<tensor<8x4xf32>, tensor<4x2xf32>>) -> tensor<4x2xf32>
    // CHECK: %[[OUTPUT1:.*]] = torch_c.from_builtin_tensor %[[ELEM1]] : tensor<4x2xf32> -> !torch.vtensor<[4,2],f32>
    %list = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,4],f32>, !torch.vtensor<[1,2],f32>) -> !torch.list<vtensor>
    %result = torch.operator "torch._c10d_functional.all_gather_into_tensor_coalesced"(%list, %int4, %group) : (!torch.list<vtensor>, !torch.int, !torch.str) -> !torch.list<vtensor>
    %out0, %out1 = torch.prim.ListUnpack %result : !torch.list<vtensor> -> !torch.vtensor<[8,4],f32>, !torch.vtensor<[4,2],f32>

    // CHECK: return %[[OUTPUT0]], %[[OUTPUT1]]
    return %out0, %out1 : !torch.vtensor<[8,4],f32>, !torch.vtensor<[4,2],f32>
  }
}

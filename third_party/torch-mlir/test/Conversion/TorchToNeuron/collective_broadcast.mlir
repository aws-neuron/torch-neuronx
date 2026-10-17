// RUN: WORLD_SIZE=8 torch-mlir-opt -pass-pipeline='builtin.module(torch-raw-to-neuron-backend-pipeline)' %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @test_broadcast_global_partial_id
  func.func @test_broadcast_global_partial_id(%arg0: !torch.vtensor<[8],f32>) -> !torch.vtensor<[8],f32> {
    %int0 = torch.constant.int 0
    %group = torch.constant.str "[[0, 2, 4, 6]]"

    // CHECK: %[[INPUT:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[8],f32> -> tensor<8xf32>
    // CHECK: %[[RESULT:.*]] = "stablehlo.collective_broadcast"(%[[INPUT]]) <{channel_handle = {{.*}}, replica_groups = dense<{{\[}}[0, 2, 4, 6]{{\]}}> : tensor<1x4xi64>}> : (tensor<8xf32>) -> tensor<8xf32>
    // CHECK: %[[OUTPUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<8xf32> -> !torch.vtensor<[8],f32>
    %0 = torch.operator "torch._c10d_functional.broadcast"(%arg0, %int0, %group) : (!torch.vtensor<[8],f32>, !torch.int, !torch.str) -> !torch.vtensor<[8],f32>

    // CHECK: return %[[OUTPUT]]
    return %0 : !torch.vtensor<[8],f32>
  }

  // CHECK-LABEL: func.func @test_broadcast_default
  func.func @test_broadcast_default(%arg0: !torch.vtensor<[8],f32>) -> !torch.vtensor<[8],f32> {
    %int0 = torch.constant.int 0
    %group = torch.constant.str "default"

    // CHECK: %[[INPUT:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[8],f32> -> tensor<8xf32>
    // CHECK: %[[RESULT:.*]] = "stablehlo.collective_broadcast"(%[[INPUT]]) <{channel_handle = {{.*}}, replica_groups = dense<{{\[}}[0, 1, 2, 3, 4, 5, 6, 7]{{\]}}> : tensor<1x8xi64>}> : (tensor<8xf32>) -> tensor<8xf32>
    // CHECK: %[[OUTPUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<8xf32> -> !torch.vtensor<[8],f32>
    %0 = torch.operator "torch._c10d_functional.broadcast"(%arg0, %int0, %group) : (!torch.vtensor<[8],f32>, !torch.int, !torch.str) -> !torch.vtensor<[8],f32>

    // CHECK: return %[[OUTPUT]]
    return %0 : !torch.vtensor<[8],f32>
  }

  // CHECK-LABEL: func.func @test_broadcast_non_default
  func.func @test_broadcast_non_default(%arg0: !torch.vtensor<[8],f32>) -> !torch.vtensor<[8],f32> {
    %int0 = torch.constant.int 0
    %group = torch.constant.str "1"

    // CHECK: %[[INPUT:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[8],f32> -> tensor<8xf32>
    // CHECK: %[[RESULT:.*]] = "stablehlo.collective_broadcast"(%[[INPUT]]) <{channel_handle = {{.*}}, replica_groups = dense<{{\[}}[0, 1, 2, 3, 4, 5, 6, 7]{{\]}}> : tensor<1x8xi64>}> : (tensor<8xf32>) -> tensor<8xf32>
    // CHECK: %[[OUTPUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<8xf32> -> !torch.vtensor<[8],f32>
    %0 = torch.operator "torch._c10d_functional.broadcast"(%arg0, %int0, %group) : (!torch.vtensor<[8],f32>, !torch.int, !torch.str) -> !torch.vtensor<[8],f32>

    // CHECK: return %[[OUTPUT]]
    return %0 : !torch.vtensor<[8],f32>
  }
}

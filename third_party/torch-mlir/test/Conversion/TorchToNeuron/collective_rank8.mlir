// RUN: WORLD_SIZE=8 torch-mlir-opt -pass-pipeline='builtin.module(torch-raw-to-neuron-backend-pipeline)' %s | FileCheck %s
// XFAIL: *

module {
  // CHECK-LABEL: func.func @test_reduce_scatter_sum_global_partial_id
  func.func @test_reduce_scatter_sum_global_partial_id(%arg0: !torch.vtensor<[8,4,2],f32>) -> !torch.vtensor<[4,4,2],f32> {
    %str = torch.constant.str "sum"
    %int2 = torch.constant.int 2
    %group = torch.constant.str "[[0, 2, 4, 6]]"

    // CHECK: %[[INPUT:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[8,4,2],f32> -> tensor<8x4x2xf32>
    // CHECK: %[[RESULT:.*]] = "stablehlo.reduce_scatter"(%[[INPUT]]) <{channel_handle = {{.*}}, replica_groups = dense<{{\[}}[0, 2, 4, 6]{{\]}}> : tensor<1x4xi64>, scatter_dimension = 0 : i64, use_global_device_ids}> ({
    // CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
    // CHECK:   %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[ARG1]] : tensor<f32>
    // CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
    // CHECK: }) : (tensor<8x4x2xf32>) -> tensor<4x4x2xf32>
    // CHECK: %[[OUTPUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<4x4x2xf32> -> !torch.vtensor<[4,4,2],f32>
    %0 = torch.operator "torch._c10d_functional.reduce_scatter_tensor"(%arg0, %str, %int2, %group) : (!torch.vtensor<[8,4,2],f32>, !torch.str, !torch.int, !torch.str) -> !torch.vtensor<[4,4,2],f32>

    // CHECK: return %[[OUTPUT]]
    return %0 : !torch.vtensor<[4,4,2],f32>
  }

  // CHECK-LABEL: func.func @test_reduce_scatter_sum_default
  func.func @test_reduce_scatter_sum_default(%arg0: !torch.vtensor<[8,4,2],f32>) -> !torch.vtensor<[4,4,2],f32> {
    %str = torch.constant.str "sum"
    %int2 = torch.constant.int 2
    %group = torch.constant.str "default"

    // CHECK: %[[INPUT:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[8,4,2],f32> -> tensor<8x4x2xf32>
    // CHECK: %[[RESULT:.*]] = "stablehlo.reduce_scatter"(%[[INPUT]]) <{channel_handle = {{.*}}, replica_groups = dense<{{\[}}[0, 1, 2, 3, 4, 5, 6, 7]{{\]}}> : tensor<1x8xi64>, scatter_dimension = 0 : i64, use_global_device_ids}> ({
    // CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
    // CHECK:   %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[ARG1]] : tensor<f32>
    // CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
    // CHECK: }) : (tensor<8x4x2xf32>) -> tensor<4x4x2xf32>
    // CHECK: %[[OUTPUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<4x4x2xf32> -> !torch.vtensor<[4,4,2],f32>
    %0 = torch.operator "torch._c10d_functional.reduce_scatter_tensor"(%arg0, %str, %int2, %group) : (!torch.vtensor<[8,4,2],f32>, !torch.str, !torch.int, !torch.str) -> !torch.vtensor<[4,4,2],f32>

    // CHECK: return %[[OUTPUT]]
    return %0 : !torch.vtensor<[4,4,2],f32>
  }

  // CHECK-LABEL: func.func @test_reduce_scatter_sum_non_default
  func.func @test_reduce_scatter_sum_non_default(%arg0: !torch.vtensor<[8,4,2],f32>) -> !torch.vtensor<[4,4,2],f32> {
    %str = torch.constant.str "sum"
    %int2 = torch.constant.int 2
    %group = torch.constant.str "1"

    // CHECK: %[[INPUT:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[8,4,2],f32> -> tensor<8x4x2xf32>
    // CHECK: %[[RESULT:.*]] = "stablehlo.reduce_scatter"(%[[INPUT]]) <{channel_handle = {{.*}}, replica_groups = dense<{{\[}}[0, 1, 2, 3, 4, 5, 6, 7]{{\]}}> : tensor<1x8xi64>, scatter_dimension = 0 : i64, use_global_device_ids}> ({
    // CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
    // CHECK:   %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[ARG1]] : tensor<f32>
    // CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
    // CHECK: }) : (tensor<8x4x2xf32>) -> tensor<4x4x2xf32>
    // CHECK: %[[OUTPUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<4x4x2xf32> -> !torch.vtensor<[4,4,2],f32>
    %0 = torch.operator "torch._c10d_functional.reduce_scatter_tensor"(%arg0, %str, %int2, %group) : (!torch.vtensor<[8,4,2],f32>, !torch.str, !torch.int, !torch.str) -> !torch.vtensor<[4,4,2],f32>

    // CHECK: return %[[OUTPUT]]
    return %0 : !torch.vtensor<[4,4,2],f32>
  }
}

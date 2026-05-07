// RUN: WORLD_SIZE=2 torch-mlir-opt -pass-pipeline='builtin.module(torch-raw-to-neuron-backend-pipeline)' %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @test_reduce_scatter_sum
  func.func @test_reduce_scatter_sum(%arg0: !torch.vtensor<[8,4,2],f32>) -> !torch.vtensor<[4,4,2],f32> {
    %str = torch.constant.str "sum"
    %int2 = torch.constant.int 2
    %group = torch.constant.str "default"

    // CHECK: %[[INPUT:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[8,4,2],f32> -> tensor<8x4x2xf32>
    // CHECK: %[[RESULT:.*]] = "stablehlo.reduce_scatter"(%[[INPUT]]) <{channel_handle = {{.*}}, replica_groups = dense<{{\[}}[0, 1]{{\]}}> : tensor<1x2xi64>, scatter_dimension = 0 : i64, use_global_device_ids}> ({
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
    %group = torch.constant.str "0"

    // CHECK: %[[INPUT:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[8,4,2],f32> -> tensor<8x4x2xf32>
    // CHECK: %[[RESULT:.*]] = "stablehlo.reduce_scatter"(%[[INPUT]]) <{channel_handle = {{.*}}, replica_groups = dense<{{\[}}[0, 1]{{\]}}> : tensor<1x2xi64>, scatter_dimension = 0 : i64, use_global_device_ids}> ({
    // CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
    // CHECK:   %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[ARG1]] : tensor<f32>
    // CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
    // CHECK: }) : (tensor<8x4x2xf32>) -> tensor<4x4x2xf32>
    // CHECK: %[[OUTPUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<4x4x2xf32> -> !torch.vtensor<[4,4,2],f32>
    %0 = torch.operator "torch._c10d_functional.reduce_scatter_tensor"(%arg0, %str, %int2, %group) : (!torch.vtensor<[8,4,2],f32>, !torch.str, !torch.int, !torch.str) -> !torch.vtensor<[4,4,2],f32>

    // CHECK: return %[[OUTPUT]]
    return %0 : !torch.vtensor<[4,4,2],f32>
  }

  // CHECK-LABEL: func.func @test_reduce_scatter_avg
  func.func @test_reduce_scatter_avg(%arg0: !torch.vtensor<[8,4,2],f32>) -> !torch.vtensor<[4,4,2],f32> {
    %str = torch.constant.str "avg"
    %int2 = torch.constant.int 2
    %group = torch.constant.str "default"

    // CHECK: %[[INPUT:.*]] = torch_c.to_builtin_tensor %arg0
    // CHECK: %[[REDUCE_RESULT:.*]] = "stablehlo.reduce_scatter"(%[[INPUT]])
    // CHECK: %[[DIVISOR:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<4x4x2xf32>
    // CHECK: %[[AVG_RESULT:.*]] = stablehlo.divide %[[REDUCE_RESULT]], %[[DIVISOR]] : tensor<4x4x2xf32>
    // CHECK: %[[OUTPUT:.*]] = torch_c.from_builtin_tensor %[[AVG_RESULT]]
    %0 = torch.operator "torch._c10d_functional.reduce_scatter_tensor"(%arg0, %str, %int2, %group) : (!torch.vtensor<[8,4,2],f32>, !torch.str, !torch.int, !torch.str) -> !torch.vtensor<[4,4,2],f32>

    return %0 : !torch.vtensor<[4,4,2],f32>
  }

  // CHECK-LABEL: func.func @test_wait_tensor_identity
  func.func @test_wait_tensor_identity(%arg0: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
    // CHECK-NOT: torch.operator
    // CHECK: return %arg0
    %0 = torch.operator "torch._c10d_functional.wait_tensor"(%arg0) : (!torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32>

    return %0 : !torch.vtensor<[4,4],f32>
  }

  // CHECK-LABEL: func.func @test_all_gather_basic
  func.func @test_all_gather_basic(%arg0: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[4,4],f32> {
    %int2 = torch.constant.int 2
    %group = torch.constant.str "default"

    // CHECK: %[[INPUT:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[2,4],f32> -> tensor<2x4xf32>
    // CHECK: %[[RESULT:.*]] = "stablehlo.all_gather"(%[[INPUT]]) <{all_gather_dim = 0 : i64, channel_handle = {{.*}}, replica_groups = dense<{{\[}}[0, 1]{{\]}}> : tensor<1x2xi64>, use_global_device_ids}> : (tensor<2x4xf32>) -> tensor<4x4xf32>
    // CHECK: %[[OUTPUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<4x4xf32> -> !torch.vtensor<[4,4],f32>
    %0 = torch.operator "torch._c10d_functional.all_gather_into_tensor"(%arg0, %int2, %group) : (!torch.vtensor<[2,4],f32>, !torch.int, !torch.str) -> !torch.vtensor<[4,4],f32>

    // CHECK: return %[[OUTPUT]]
    return %0 : !torch.vtensor<[4,4],f32>
  }

  // CHECK-LABEL: func.func @test_all_reduce_sum
  func.func @test_all_reduce_sum(%arg0: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
    %str = torch.constant.str "sum"
    %group = torch.constant.str "default"

    // CHECK: %[[INPUT:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[4,4],f32> -> tensor<4x4xf32>
    // CHECK: %[[RESULT:.*]] = "stablehlo.all_reduce"(%[[INPUT]]) <{channel_handle = {{.*}}, replica_groups = dense<{{\[}}[0, 1]{{\]}}> : tensor<1x2xi64>, use_global_device_ids}> ({
    // CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
    // CHECK:   %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[ARG1]] : tensor<f32>
    // CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
    // CHECK: }) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    // CHECK: %[[OUTPUT:.*]] = torch_c.from_builtin_tensor %[[RESULT]] : tensor<4x4xf32> -> !torch.vtensor<[4,4],f32>
    %0 = torch.operator "torch._c10d_functional.all_reduce"(%arg0, %str, %group) : (!torch.vtensor<[4,4],f32>, !torch.str, !torch.str) -> !torch.vtensor<[4,4],f32>

    // CHECK: return %[[OUTPUT]]
    return %0 : !torch.vtensor<[4,4],f32>
  }

  // CHECK-LABEL: func.func @test_all_reduce_avg
  func.func @test_all_reduce_avg(%arg0: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
    %str = torch.constant.str "avg"
    %group = torch.constant.str "default"

    // CHECK: %[[INPUT:.*]] = torch_c.to_builtin_tensor %arg0
    // CHECK: %[[REDUCE_RESULT:.*]] = "stablehlo.all_reduce"(%[[INPUT]])
    // CHECK: %[[DIVISOR:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<4x4xf32>
    // CHECK: %[[AVG_RESULT:.*]] = stablehlo.divide %[[REDUCE_RESULT]], %[[DIVISOR]] : tensor<4x4xf32>
    // CHECK: %[[OUTPUT:.*]] = torch_c.from_builtin_tensor %[[AVG_RESULT]]
    %0 = torch.operator "torch._c10d_functional.all_reduce"(%arg0, %str, %group) : (!torch.vtensor<[4,4],f32>, !torch.str, !torch.str) -> !torch.vtensor<[4,4],f32>

    return %0 : !torch.vtensor<[4,4],f32>
  }

  // CHECK-LABEL: func.func @test_all_reduce_coalesced
  func.func @test_all_reduce_coalesced(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.vtensor<[2,2],f32>) -> (!torch.vtensor<[4,4],f32>, !torch.vtensor<[2,2],f32>) {
    %str = torch.constant.str "sum"
    %group = torch.constant.str "default"

    // CHECK: %[[INPUT0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[4,4],f32> -> tensor<4x4xf32>
    // CHECK: %[[INPUT1:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[2,2],f32> -> tensor<2x2xf32>
    // CHECK: %[[RESULT:.*]]:2 = "stablehlo.all_reduce"(%[[INPUT0]], %[[INPUT1]]) <{channel_handle = {{.*}}, replica_groups = dense<{{\[}}[0, 1]{{\]}}> : tensor<1x2xi64>, use_global_device_ids}> ({
    // CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
    // CHECK:   %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[ARG1]] : tensor<f32>
    // CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
    // CHECK: }) : (tensor<4x4xf32>, tensor<2x2xf32>) -> (tensor<4x4xf32>, tensor<2x2xf32>)
    // CHECK: %[[TUPLE:.*]] = stablehlo.tuple %[[RESULT]]#0, %[[RESULT]]#1 : tuple<tensor<4x4xf32>, tensor<2x2xf32>>
    // CHECK: %[[ELEM0:.*]] = stablehlo.get_tuple_element %[[TUPLE]][0] : (tuple<tensor<4x4xf32>, tensor<2x2xf32>>) -> tensor<4x4xf32>
    // CHECK: %[[OUTPUT0:.*]] = torch_c.from_builtin_tensor %[[ELEM0]] : tensor<4x4xf32> -> !torch.vtensor<[4,4],f32>
    // CHECK: %[[ELEM1:.*]] = stablehlo.get_tuple_element %[[TUPLE]][1] : (tuple<tensor<4x4xf32>, tensor<2x2xf32>>) -> tensor<2x2xf32>
    // CHECK: %[[OUTPUT1:.*]] = torch_c.from_builtin_tensor %[[ELEM1]] : tensor<2x2xf32> -> !torch.vtensor<[2,2],f32>
    %list = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[4,4],f32>, !torch.vtensor<[2,2],f32>) -> !torch.list<vtensor>
    %result = torch.operator "torch._c10d_functional.all_reduce_coalesced"(%list, %str, %group) : (!torch.list<vtensor>, !torch.str, !torch.str) -> !torch.list<vtensor>
    %out0, %out1 = torch.prim.ListUnpack %result : !torch.list<vtensor> -> !torch.vtensor<[4,4],f32>, !torch.vtensor<[2,2],f32>

    // CHECK: return %[[OUTPUT0]], %[[OUTPUT1]]
    return %out0, %out1 : !torch.vtensor<[4,4],f32>, !torch.vtensor<[2,2],f32>
  }
}

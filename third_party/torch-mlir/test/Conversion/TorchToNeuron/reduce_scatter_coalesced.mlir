// RUN: WORLD_SIZE=2 torch-mlir-opt -pass-pipeline='builtin.module(torch-raw-to-neuron-backend-pipeline)' %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @test_reduce_scatter_coalesced_sum
  func.func @test_reduce_scatter_coalesced_sum(%arg0: !torch.vtensor<[8,4],f32>, %arg1: !torch.vtensor<[6,2],f32>) -> (!torch.vtensor<[4,4],f32>, !torch.vtensor<[3,2],f32>) {
    %str = torch.constant.str "sum"
    %int2 = torch.constant.int 2
    %group = torch.constant.str "default"

    // CHECK: %[[INPUT0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[8,4],f32> -> tensor<8x4xf32>
    // CHECK: %[[INPUT1:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[6,2],f32> -> tensor<6x2xf32>
    // CHECK: %[[RESULT0:.*]] = "stablehlo.reduce_scatter"(%[[INPUT0]]) <{channel_handle = {{.*}}, replica_groups = dense<{{\[}}[0, 1]{{\]}}> : tensor<1x2xi64>, scatter_dimension = 0 : i64, use_global_device_ids}> ({
    // CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
    // CHECK:   %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[ARG1]] : tensor<f32>
    // CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
    // CHECK: }) : (tensor<8x4xf32>) -> tensor<4x4xf32>
    // CHECK: %[[RESULT1:.*]] = "stablehlo.reduce_scatter"(%[[INPUT1]]) <{channel_handle = {{.*}}, replica_groups = dense<{{\[}}[0, 1]{{\]}}> : tensor<1x2xi64>, scatter_dimension = 0 : i64, use_global_device_ids}> ({
    // CHECK: ^bb0(%[[ARG2:.*]]: tensor<f32>, %[[ARG3:.*]]: tensor<f32>):
    // CHECK:   %[[ADD2:.*]] = stablehlo.add %[[ARG2]], %[[ARG3]] : tensor<f32>
    // CHECK:   stablehlo.return %[[ADD2]] : tensor<f32>
    // CHECK: }) : (tensor<6x2xf32>) -> tensor<3x2xf32>
    // CHECK: %[[TUPLE:.*]] = stablehlo.tuple %[[RESULT0]], %[[RESULT1]] : tuple<tensor<4x4xf32>, tensor<3x2xf32>>
    // CHECK: %[[ELEM0:.*]] = stablehlo.get_tuple_element %[[TUPLE]][0] : (tuple<tensor<4x4xf32>, tensor<3x2xf32>>) -> tensor<4x4xf32>
    // CHECK: %[[OUTPUT0:.*]] = torch_c.from_builtin_tensor %[[ELEM0]] : tensor<4x4xf32> -> !torch.vtensor<[4,4],f32>
    // CHECK: %[[ELEM1:.*]] = stablehlo.get_tuple_element %[[TUPLE]][1] : (tuple<tensor<4x4xf32>, tensor<3x2xf32>>) -> tensor<3x2xf32>
    // CHECK: %[[OUTPUT1:.*]] = torch_c.from_builtin_tensor %[[ELEM1]] : tensor<3x2xf32> -> !torch.vtensor<[3,2],f32>
    %list = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[8,4],f32>, !torch.vtensor<[6,2],f32>) -> !torch.list<vtensor>
    %result = torch.operator "torch._c10d_functional.reduce_scatter_tensor_coalesced"(%list, %str, %int2, %group) : (!torch.list<vtensor>, !torch.str, !torch.int, !torch.str) -> !torch.list<vtensor>
    %out0, %out1 = torch.prim.ListUnpack %result : !torch.list<vtensor> -> !torch.vtensor<[4,4],f32>, !torch.vtensor<[3,2],f32>

    // CHECK: return %[[OUTPUT0]], %[[OUTPUT1]]
    return %out0, %out1 : !torch.vtensor<[4,4],f32>, !torch.vtensor<[3,2],f32>
  }

  // CHECK-LABEL: func.func @test_reduce_scatter_coalesced_avg
  func.func @test_reduce_scatter_coalesced_avg(%arg0: !torch.vtensor<[8,4],f32>, %arg1: !torch.vtensor<[6,2],f32>) -> (!torch.vtensor<[4,4],f32>, !torch.vtensor<[3,2],f32>) {
    %str = torch.constant.str "avg"
    %int2 = torch.constant.int 2
    %group = torch.constant.str "default"

    // CHECK: %[[INPUT0:.*]] = torch_c.to_builtin_tensor %arg0
    // CHECK: %[[INPUT1:.*]] = torch_c.to_builtin_tensor %arg1
    // CHECK: %[[REDUCE0:.*]] = "stablehlo.reduce_scatter"(%[[INPUT0]])
    // CHECK: %[[DIVISOR0:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<4x4xf32>
    // CHECK: %[[AVG0:.*]] = stablehlo.divide %[[REDUCE0]], %[[DIVISOR0]] : tensor<4x4xf32>
    // CHECK: %[[REDUCE1:.*]] = "stablehlo.reduce_scatter"(%[[INPUT1]])
    // CHECK: %[[DIVISOR1:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<3x2xf32>
    // CHECK: %[[AVG1:.*]] = stablehlo.divide %[[REDUCE1]], %[[DIVISOR1]] : tensor<3x2xf32>
    // CHECK: %[[TUPLE:.*]] = stablehlo.tuple %[[AVG0]], %[[AVG1]]
    %list = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[8,4],f32>, !torch.vtensor<[6,2],f32>) -> !torch.list<vtensor>
    %result = torch.operator "torch._c10d_functional.reduce_scatter_tensor_coalesced"(%list, %str, %int2, %group) : (!torch.list<vtensor>, !torch.str, !torch.int, !torch.str) -> !torch.list<vtensor>
    %out0, %out1 = torch.prim.ListUnpack %result : !torch.list<vtensor> -> !torch.vtensor<[4,4],f32>, !torch.vtensor<[3,2],f32>

    return %out0, %out1 : !torch.vtensor<[4,4],f32>, !torch.vtensor<[3,2],f32>
  }

  // CHECK-LABEL: func.func @test_reduce_scatter_coalesced_prod
  func.func @test_reduce_scatter_coalesced_prod(%arg0: !torch.vtensor<[4,2],f32>, %arg1: !torch.vtensor<[8,3],f32>) -> (!torch.vtensor<[2,2],f32>, !torch.vtensor<[4,3],f32>) {
    %str = torch.constant.str "prod"
    %int2 = torch.constant.int 2
    %group = torch.constant.str "default"

    // CHECK: %[[RESULT0:.*]] = "stablehlo.reduce_scatter"({{.*}}) <{channel_handle = {{.*}}, replica_groups = dense<{{\[}}[0, 1]{{\]}}> : tensor<1x2xi64>, scatter_dimension = 0 : i64, use_global_device_ids}> ({
    // CHECK: ^bb0(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
    // CHECK:   %[[MUL:.*]] = stablehlo.multiply %[[ARG0]], %[[ARG1]] : tensor<f32>
    // CHECK:   stablehlo.return %[[MUL]] : tensor<f32>
    // CHECK: }) : (tensor<4x2xf32>) -> tensor<2x2xf32>
    %list = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[4,2],f32>, !torch.vtensor<[8,3],f32>) -> !torch.list<vtensor>
    %result = torch.operator "torch._c10d_functional.reduce_scatter_tensor_coalesced"(%list, %str, %int2, %group) : (!torch.list<vtensor>, !torch.str, !torch.int, !torch.str) -> !torch.list<vtensor>
    %out0, %out1 = torch.prim.ListUnpack %result : !torch.list<vtensor> -> !torch.vtensor<[2,2],f32>, !torch.vtensor<[4,3],f32>

    return %out0, %out1 : !torch.vtensor<[2,2],f32>, !torch.vtensor<[4,3],f32>
  }

  // CHECK-LABEL: func.func @test_reduce_scatter_coalesced_single
  func.func @test_reduce_scatter_coalesced_single(%arg0: !torch.vtensor<[10,5],f32>) -> !torch.vtensor<[5,5],f32> {
    %str = torch.constant.str "sum"
    %int2 = torch.constant.int 2
    %group = torch.constant.str "default"

    // CHECK: %[[INPUT:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[10,5],f32> -> tensor<10x5xf32>
    // CHECK: %[[RESULT:.*]] = "stablehlo.reduce_scatter"(%[[INPUT]]) <{channel_handle = {{.*}}, replica_groups = dense<{{\[}}[0, 1]{{\]}}> : tensor<1x2xi64>, scatter_dimension = 0 : i64, use_global_device_ids}>
    // CHECK: %[[TUPLE:.*]] = stablehlo.tuple %[[RESULT]] : tuple<tensor<5x5xf32>>
    // CHECK: %[[ELEM:.*]] = stablehlo.get_tuple_element %[[TUPLE]][0]
    // CHECK: %[[OUTPUT:.*]] = torch_c.from_builtin_tensor %[[ELEM]] : tensor<5x5xf32> -> !torch.vtensor<[5,5],f32>
    %list = torch.prim.ListConstruct %arg0 : (!torch.vtensor<[10,5],f32>) -> !torch.list<vtensor>
    %result = torch.operator "torch._c10d_functional.reduce_scatter_tensor_coalesced"(%list, %str, %int2, %group) : (!torch.list<vtensor>, !torch.str, !torch.int, !torch.str) -> !torch.list<vtensor>
    %out0 = torch.prim.ListUnpack %result : !torch.list<vtensor> -> !torch.vtensor<[5,5],f32>

    // CHECK: return %[[OUTPUT]]
    return %out0 : !torch.vtensor<[5,5],f32>
  }
}

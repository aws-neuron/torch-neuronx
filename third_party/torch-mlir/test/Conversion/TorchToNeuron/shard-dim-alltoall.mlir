// RUN: torch-mlir-opt %s -convert-torch-to-neuron -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @shard_dim_alltoall_gather0_shard1
// CHECK: %[[T0:.*]] = torch_c.to_builtin_tensor %arg0
// CHECK: %[[T1:.*]] = "stablehlo.all_to_all"(%[[T0]]) <{channel_handle = {{.*}}, concat_dimension = 0 : i64, replica_groups = dense<{{\[\[}}0, 1{{\]\]}}> : tensor<1x2xi64>, split_count = 2 : i64, split_dimension = 1 : i64}> : (tensor<4x8xf32>) -> tensor<8x4xf32>
// CHECK: %[[T2:.*]] = torch_c.from_builtin_tensor %[[T1]]
func.func @shard_dim_alltoall_gather0_shard1(%arg0: !torch.vtensor<[4,8],f32>) -> !torch.vtensor<[8,4],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %str = torch.constant.str "[[0,1]]"
  %0 = torch.operator "torch._dtensor.shard_dim_alltoall"(%arg0, %int0, %int1, %str) : (!torch.vtensor<[4,8],f32>, !torch.int, !torch.int, !torch.str) -> !torch.vtensor<[8,4],f32>
  return %0 : !torch.vtensor<[8,4],f32>
}

// -----

// CHECK-LABEL: func.func @shard_dim_alltoall_gather1_shard0
// CHECK: %[[T0:.*]] = torch_c.to_builtin_tensor %arg0
// CHECK: %[[T1:.*]] = "stablehlo.all_to_all"(%[[T0]]) <{channel_handle = {{.*}}, concat_dimension = 1 : i64, replica_groups = dense<{{\[\[}}0, 1{{\]\]}}> : tensor<1x2xi64>, split_count = 2 : i64, split_dimension = 0 : i64}> : (tensor<8x4xf32>) -> tensor<4x8xf32>
// CHECK: %[[T2:.*]] = torch_c.from_builtin_tensor %[[T1]]
func.func @shard_dim_alltoall_gather1_shard0(%arg0: !torch.vtensor<[8,4],f32>) -> !torch.vtensor<[4,8],f32> {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %str = torch.constant.str "[[0,1]]"
  %0 = torch.operator "torch._dtensor.shard_dim_alltoall"(%arg0, %int1, %int0, %str) : (!torch.vtensor<[8,4],f32>, !torch.int, !torch.int, !torch.str) -> !torch.vtensor<[4,8],f32>
  return %0 : !torch.vtensor<[4,8],f32>
}

// -----

// CHECK-LABEL: func.func @shard_dim_alltoall_3d_gather0_shard2
// CHECK: %[[T0:.*]] = torch_c.to_builtin_tensor %arg0
// CHECK: %[[T1:.*]] = "stablehlo.all_to_all"(%[[T0]]) <{channel_handle = {{.*}}, concat_dimension = 0 : i64, replica_groups = dense<{{\[\[}}0, 1, 2, 3{{\]\]}}> : tensor<1x4xi64>, split_count = 4 : i64, split_dimension = 2 : i64}> : (tensor<2x4x16xf32>) -> tensor<8x4x4xf32>
// CHECK: %[[T2:.*]] = torch_c.from_builtin_tensor %[[T1]]
func.func @shard_dim_alltoall_3d_gather0_shard2(%arg0: !torch.vtensor<[2,4,16],f32>) -> !torch.vtensor<[8,4,4],f32> {
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %str = torch.constant.str "[[0,1,2,3]]"
  %0 = torch.operator "torch._dtensor.shard_dim_alltoall"(%arg0, %int0, %int2, %str) : (!torch.vtensor<[2,4,16],f32>, !torch.int, !torch.int, !torch.str) -> !torch.vtensor<[8,4,4],f32>
  return %0 : !torch.vtensor<[8,4,4],f32>
}

// -----

// CHECK-LABEL: func.func @shard_dim_alltoall_3d_gather1_shard2
// CHECK: %[[T0:.*]] = torch_c.to_builtin_tensor %arg0
// CHECK: %[[T1:.*]] = "stablehlo.all_to_all"(%[[T0]]) <{channel_handle = {{.*}}, concat_dimension = 1 : i64, replica_groups = dense<{{\[\[}}0, 1, 2{{\]\]}}> : tensor<1x3xi64>, split_count = 3 : i64, split_dimension = 2 : i64}> : (tensor<4x3x6xf32>) -> tensor<4x9x2xf32>
// CHECK: %[[T2:.*]] = torch_c.from_builtin_tensor %[[T1]]
func.func @shard_dim_alltoall_3d_gather1_shard2(%arg0: !torch.vtensor<[4,3,6],f32>) -> !torch.vtensor<[4,9,2],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %str = torch.constant.str "[[0,1,2]]"
  %0 = torch.operator "torch._dtensor.shard_dim_alltoall"(%arg0, %int1, %int2, %str) : (!torch.vtensor<[4,3,6],f32>, !torch.int, !torch.int, !torch.str) -> !torch.vtensor<[4,9,2],f32>
  return %0 : !torch.vtensor<[4,9,2],f32>
}

// -----

// CHECK-LABEL: func.func @shard_dim_alltoall_same_dim
// CHECK: %[[T0:.*]] = torch_c.to_builtin_tensor %arg0
// CHECK: %[[T1:.*]] = "stablehlo.all_to_all"(%[[T0]]) <{channel_handle = {{.*}}, concat_dimension = 0 : i64, replica_groups = dense<{{\[\[}}0, 1{{\]\]}}> : tensor<1x2xi64>, split_count = 2 : i64, split_dimension = 0 : i64}> : (tensor<8x4xf32>) -> tensor<8x4xf32>
// CHECK: %[[T2:.*]] = torch_c.from_builtin_tensor %[[T1]]
func.func @shard_dim_alltoall_same_dim(%arg0: !torch.vtensor<[8,4],f32>) -> !torch.vtensor<[8,4],f32> {
  %int0 = torch.constant.int 0
  %str = torch.constant.str "[[0,1]]"
  %0 = torch.operator "torch._dtensor.shard_dim_alltoall"(%arg0, %int0, %int0, %str) : (!torch.vtensor<[8,4],f32>, !torch.int, !torch.int, !torch.str) -> !torch.vtensor<[8,4],f32>
  return %0 : !torch.vtensor<[8,4],f32>
}

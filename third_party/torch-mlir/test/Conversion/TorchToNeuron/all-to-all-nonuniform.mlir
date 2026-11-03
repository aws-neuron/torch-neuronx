// RUN: torch-mlir-opt %s -convert-torch-to-neuron -split-input-file | FileCheck %s

// Rank 1 of send = [[1, 3], [2, 2]]: output_split_sizes = [3, 2], input_split_sizes = [2, 2].
// Each chunk is padded to 4 rows, exchanged with an even all_to_all, then trimmed.

// CHECK-LABEL: func.func @all_to_all_nonuniform_2d
func.func @all_to_all_nonuniform_2d(%arg0: !torch.vtensor<[4,8],f32>) -> !torch.vtensor<[5,8],f32> {
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %output_splits = torch.prim.ListConstruct %int3, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %input_splits = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %str = torch.constant.str "[[0, 1]]"

  // CHECK: %[[T0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[4,8],f32> -> tensor<4x8xf32>
  // CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[S0:.*]] = stablehlo.slice %[[T0]] [0:2, 0:8] : (tensor<4x8xf32>) -> tensor<2x8xf32>
  // CHECK: %[[P0:.*]] = stablehlo.pad %[[S0]], %[[ZERO]], low = [0, 0], high = [2, 0], interior = [0, 0] : (tensor<2x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  // CHECK: %[[S1:.*]] = stablehlo.slice %[[T0]] [2:4, 0:8] : (tensor<4x8xf32>) -> tensor<2x8xf32>
  // CHECK: %[[P1:.*]] = stablehlo.pad %[[S1]], %[[ZERO]], low = [0, 0], high = [2, 0], interior = [0, 0] : (tensor<2x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  // CHECK: %[[PACKED:.*]] = stablehlo.concatenate %[[P0]], %[[P1]], dim = 0 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<8x8xf32>
  // CHECK: %[[A2A:.*]] = "stablehlo.all_to_all"(%[[PACKED]]) <{channel_handle = {{.*}}, concat_dimension = 0 : i64, replica_groups = dense<{{\[\[}}0, 1]]> : tensor<1x2xi64>, split_count = 2 : i64, split_dimension = 0 : i64}> : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK: %[[R0:.*]] = stablehlo.slice %[[A2A]] [0:3, 0:8] : (tensor<8x8xf32>) -> tensor<3x8xf32>
  // CHECK: %[[R1:.*]] = stablehlo.slice %[[A2A]] [4:6, 0:8] : (tensor<8x8xf32>) -> tensor<2x8xf32>
  // CHECK: %[[OUT:.*]] = stablehlo.concatenate %[[R0]], %[[R1]], dim = 0 : (tensor<3x8xf32>, tensor<2x8xf32>) -> tensor<5x8xf32>
  // CHECK: torch_c.from_builtin_tensor %[[OUT]] : tensor<5x8xf32> -> !torch.vtensor<[5,8],f32>
  %result = torch.operator "torch._c10d_functional.all_to_all_single"(%arg0, %output_splits, %input_splits, %str) : (!torch.vtensor<[4,8],f32>, !torch.list<int>, !torch.list<int>, !torch.str) -> !torch.vtensor<[5,8],f32>

  return %result : !torch.vtensor<[5,8],f32>
}

// -----

// Rank 0 of send = [[1, 3], [2, 2]]: output_split_sizes = [1, 2], input_split_sizes = [1, 3].

// CHECK-LABEL: func.func @all_to_all_nonuniform_1d
func.func @all_to_all_nonuniform_1d(%arg0: !torch.vtensor<[4],f32>) -> !torch.vtensor<[3],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %output_splits = torch.prim.ListConstruct %int1, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %input_splits = torch.prim.ListConstruct %int1, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %str = torch.constant.str "[[0, 1]]"

  // CHECK: %[[T0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[4],f32> -> tensor<4xf32>
  // CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[S0:.*]] = stablehlo.slice %[[T0]] [0:1] : (tensor<4xf32>) -> tensor<1xf32>
  // CHECK: %[[P0:.*]] = stablehlo.pad %[[S0]], %[[ZERO]], low = [0], high = [3], interior = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: %[[S1:.*]] = stablehlo.slice %[[T0]] [1:4] : (tensor<4xf32>) -> tensor<3xf32>
  // CHECK: %[[P1:.*]] = stablehlo.pad %[[S1]], %[[ZERO]], low = [0], high = [1], interior = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: %[[PACKED:.*]] = stablehlo.concatenate %[[P0]], %[[P1]], dim = 0 : (tensor<4xf32>, tensor<4xf32>) -> tensor<8xf32>
  // CHECK: %[[A2A:.*]] = "stablehlo.all_to_all"(%[[PACKED]]) {{.*}} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK: %[[R0:.*]] = stablehlo.slice %[[A2A]] [0:1] : (tensor<8xf32>) -> tensor<1xf32>
  // CHECK: %[[R1:.*]] = stablehlo.slice %[[A2A]] [4:6] : (tensor<8xf32>) -> tensor<2xf32>
  // CHECK: %[[OUT:.*]] = stablehlo.concatenate %[[R0]], %[[R1]], dim = 0 : (tensor<1xf32>, tensor<2xf32>) -> tensor<3xf32>
  %result = torch.operator "torch._c10d_functional.all_to_all_single"(%arg0, %output_splits, %input_splits, %str) : (!torch.vtensor<[4],f32>, !torch.list<int>, !torch.list<int>, !torch.str) -> !torch.vtensor<[3],f32>

  return %result : !torch.vtensor<[3],f32>
}

// -----

// Rank 1 of send = [[0, 4], [4, 0]]: a zero-size chunk becomes a zero block of 4 rows.

// CHECK-LABEL: func.func @all_to_all_nonuniform_zero_chunk
func.func @all_to_all_nonuniform_zero_chunk(%arg0: !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[4,2],f32> {
  %int0 = torch.constant.int 0
  %int4 = torch.constant.int 4
  %output_splits = torch.prim.ListConstruct %int4, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %input_splits = torch.prim.ListConstruct %int4, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %str = torch.constant.str "[[0, 1]]"

  // CHECK: %[[T0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[4,2],f32> -> tensor<4x2xf32>
  // CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[EMPTY:.*]] = stablehlo.broadcast_in_dim %[[ZERO]], dims = [] : (tensor<f32>) -> tensor<4x2xf32>
  // CHECK: %[[PACKED:.*]] = stablehlo.concatenate %[[T0]], %[[EMPTY]], dim = 0 : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<8x2xf32>
  // CHECK: %[[A2A:.*]] = "stablehlo.all_to_all"(%[[PACKED]]) {{.*}} : (tensor<8x2xf32>) -> tensor<8x2xf32>
  // CHECK: %[[R0:.*]] = stablehlo.slice %[[A2A]] [0:4, 0:2] : (tensor<8x2xf32>) -> tensor<4x2xf32>
  // CHECK: %[[OUT:.*]] = stablehlo.concatenate %[[R0]], dim = 0 : (tensor<4x2xf32>) -> tensor<4x2xf32>
  %result = torch.operator "torch._c10d_functional.all_to_all_single"(%arg0, %output_splits, %input_splits, %str) : (!torch.vtensor<[4,2],f32>, !torch.list<int>, !torch.list<int>, !torch.str) -> !torch.vtensor<[4,2],f32>

  return %result : !torch.vtensor<[4,2],f32>
}

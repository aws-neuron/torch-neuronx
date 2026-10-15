// RUN: torch-mlir-opt %s -convert-torch-to-neuron -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @all_to_all_single
func.func @all_to_all_single(%arg0: !torch.vtensor<[8],f32>) -> !torch.vtensor<[8],f32> {
  %none = torch.constant.none
  %str = torch.constant.str "default"
  // CHECK-NOT: torch.operator
  // CHECK: %[[T0:.*]] = torch_c.to_builtin_tensor %arg0
  // CHECK: %[[T1:.*]] = "stablehlo.all_to_all"(%[[T0]]) <{channel_handle = {{.*}}, concat_dimension = 0 : i64, replica_groups = dense<0> : tensor<1x1xi64>, split_count = 1 : i64, split_dimension = 0 : i64}> : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK: %[[T2:.*]] = torch_c.from_builtin_tensor %[[T1]]
  %0 = torch.operator "torch._c10d_functional.all_to_all_single"(%arg0, %none, %none, %str) : (!torch.vtensor<[8],f32>, !torch.none, !torch.none, !torch.str) -> !torch.vtensor<[8],f32>
  return %0 : !torch.vtensor<[8],f32>
}

// -----

// CHECK-LABEL: func.func @all_to_all_single_with_lists
func.func @all_to_all_single_with_lists(%arg0: !torch.vtensor<[8],f32>) -> !torch.vtensor<[8],f32> {
  %int2 = torch.constant.int 2
  %list = torch.prim.ListConstruct %int2, %int2, %int2, %int2 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %str = torch.constant.str "default"
  // CHECK-NOT: torch.operator
  // CHECK: %{{.*}} = torch_c.to_builtin_tensor
  // CHECK: %{{.*}} = "stablehlo.all_to_all"(%{{.*}}) <{channel_handle = {{.*}}, concat_dimension = 0 : i64, replica_groups = dense<0> : tensor<1x1xi64>, split_count = 1 : i64, split_dimension = 0 : i64}> : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK: %{{.*}} = torch_c.from_builtin_tensor
  %0 = torch.operator "torch._c10d_functional.all_to_all_single"(%arg0, %list, %list, %str) : (!torch.vtensor<[8],f32>, !torch.list<int>, !torch.list<int>, !torch.str) -> !torch.vtensor<[8],f32>
  return %0 : !torch.vtensor<[8],f32>
}

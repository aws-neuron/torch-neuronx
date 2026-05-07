// RUN: env WORLD_SIZE=2 RANK=1 torch-mlir-opt %s -convert-torch-to-neuron -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @all_to_all_nonuniform
func.func @all_to_all_nonuniform(%arg0: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
  %int0 = torch.constant.int 0
  %int4 = torch.constant.int 4
  %input_splits = torch.prim.ListConstruct %int0, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
  %output_splits = torch.prim.ListConstruct %int0, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
  %str = torch.constant.str "[[0, 1]]"

  // CHECK: %[[T0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[4],f32> -> tensor<4xf32>
  // CHECK: %[[GATHER:.*]] = "stablehlo.all_gather"(%[[T0]]) <{all_gather_dim = 0 : i64, channel_handle = {{.*}}, replica_groups = dense<{{\[\[}}0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> : (tensor<4xf32>) -> tensor<8xf32>
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %[[GATHER]] [4:8] : (tensor<8xf32>) -> tensor<4xf32>
  // CHECK: %[[CONCAT:.*]] = stablehlo.concatenate %[[SLICE]], dim = 0 : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %[[T1:.*]] = torch_c.from_builtin_tensor %[[CONCAT]] : tensor<4xf32> -> !torch.vtensor<[4],f32>
  %result = torch.operator "torch._c10d_functional.all_to_all_single"(%arg0, %input_splits, %output_splits, %str) : (!torch.vtensor<[4],f32>, !torch.list<int>, !torch.list<int>, !torch.str) -> !torch.vtensor<[4],f32>

  return %result : !torch.vtensor<[4],f32>
}

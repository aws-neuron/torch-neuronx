// RUN: not torch-mlir-opt -pass-pipeline='builtin.module(torch-raw-to-neuron-backend-pipeline)' %s 2>&1 | FileCheck %s

// CHECK: max_pool2d: padding must be >= 0
func.func @max_pool2d_padding_negative(%arg0: !torch.vtensor<[1,1,4,4],f32>) -> (!torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,1,4,4],si64>) {
  %int2 = torch.constant.int 2
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %neg1 = torch.constant.int -1
  %false = torch.constant.bool false
  %kernel = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %stride = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %neg1, %neg1 : (!torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %values, %indices = torch.aten.max_pool2d_with_indices %arg0, %kernel, %stride, %padding, %dilation, %false : !torch.vtensor<[1,1,4,4],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,1,4,4],si64>
  return %values, %indices : !torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,1,4,4],si64>
}

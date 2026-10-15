// RUN: not torch-mlir-opt -pass-pipeline='builtin.module(torch-raw-to-neuron-backend-pipeline)' %s 2>&1 | FileCheck %s

// Padding (5) equals the effective kernel size (dilation=2, kernelSize=3,
// kernelSpan = 2*(3-1)+1 = 5). The corner window is entirely padding with
// no real input overlap. padding=4 would be valid (1 real element overlap).

// CHECK: max_pool2d: padding must be less than the effective kernel size
func.func @max_pool2d_padding_exceeds_kernel(%arg0: !torch.vtensor<[1,1,3,3],f32>) -> (!torch.vtensor<[1,1,3,3],f32>, !torch.vtensor<[1,1,3,3],si64>) {
  %int3 = torch.constant.int 3
  %int5 = torch.constant.int 5
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %kernel = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %stride = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int5, %int5 : (!torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %values, %indices = torch.aten.max_pool2d_with_indices %arg0, %kernel, %stride, %padding, %dilation, %false : !torch.vtensor<[1,1,3,3],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,3,3],f32>, !torch.vtensor<[1,1,3,3],si64>
  return %values, %indices : !torch.vtensor<[1,1,3,3],f32>, !torch.vtensor<[1,1,3,3],si64>
}

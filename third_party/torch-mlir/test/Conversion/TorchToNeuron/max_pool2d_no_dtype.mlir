// RUN: not torch-mlir-opt -pass-pipeline='builtin.module(torch-raw-to-neuron-backend-pipeline)' %s 2>&1 | FileCheck %s

// Verify that the pre-gate rejects dtype-less input tensors with a clear error.

// CHECK: max_pool2d: input tensor has no dtype.
func.func @max_pool2d_no_dtype(%arg0: !torch.vtensor<[1,1,4,4],unk>) -> (!torch.vtensor<[1,1,2,2],unk>, !torch.vtensor<[1,1,2,2],si64>) {
  %int2 = torch.constant.int 2
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %kernel = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %stride = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %values, %indices = torch.aten.max_pool2d_with_indices %arg0, %kernel, %stride, %padding, %dilation, %false : !torch.vtensor<[1,1,4,4],unk>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,2,2],unk>, !torch.vtensor<[1,1,2,2],si64>
  return %values, %indices : !torch.vtensor<[1,1,2,2],unk>, !torch.vtensor<[1,1,2,2],si64>
}

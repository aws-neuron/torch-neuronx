// RUN: not torch-mlir-opt -pass-pipeline='builtin.module(torch-raw-to-neuron-backend-pipeline)' %s 2>&1 | FileCheck %s

// Kernel (3x3) larger than input (2x2) with stride=2: strideBudget=-1.
// Floor division must use signed floor (round toward -inf) not C++ truncation
// (round toward zero) to correctly compute outputSize=0 and reject.
// Without signedFloorDivision, C++ truncation gives -1/2=0, outputSize=1
// (wrong), and the output-too-small guard does not fire.

// CHECK: max_pool2d: output size is too small
func.func @max_pool2d_floor_division_negative(%arg0: !torch.vtensor<[1,1,2,2],f32>) -> (!torch.vtensor<[1,1,1,1],f32>, !torch.vtensor<[1,1,1,1],si64>) {
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %kernel = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %stride = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %values, %indices = torch.aten.max_pool2d_with_indices %arg0, %kernel, %stride, %padding, %dilation, %false : !torch.vtensor<[1,1,2,2],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,1,1],f32>, !torch.vtensor<[1,1,1,1],si64>
  return %values, %indices : !torch.vtensor<[1,1,1,1],f32>, !torch.vtensor<[1,1,1,1],si64>
}

// RUN: torch-mlir-opt -pass-pipeline='builtin.module(torch-raw-to-neuron-backend-pipeline)' %s | FileCheck %s

// Padding (4) is one less than the effective kernel size (dilation=2,
// kernelSize=3, kernelSpan = 2*(3-1)+1 = 5). The corner window overlaps
// exactly 1 real input element. This is the maximum valid padding under
// the correctness limit (padding < kernelSpan), and exceeds
// PyTorch's stricter trace-time limit (padding <= kernelSpan / 2 = 2).
// Verifies hand-written MLIR can use the full correctness range.

// CHECK-LABEL: func.func @max_pool2d_padding_at_correctness_limit
// CHECK-NOT: max_pool2d_with_indices
func.func @max_pool2d_padding_at_correctness_limit(%arg0: !torch.vtensor<[1,1,3,3],f32>) -> (!torch.vtensor<[1,1,7,7],f32>, !torch.vtensor<[1,1,7,7],si64>) {
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %kernel = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %stride = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int4, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %values, %indices = torch.aten.max_pool2d_with_indices %arg0, %kernel, %stride, %padding, %dilation, %false : !torch.vtensor<[1,1,3,3],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,7,7],f32>, !torch.vtensor<[1,1,7,7],si64>
  return %values, %indices : !torch.vtensor<[1,1,7,7],f32>, !torch.vtensor<[1,1,7,7],si64>
}

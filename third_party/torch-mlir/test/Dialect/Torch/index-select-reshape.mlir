// RUN: torch-mlir-opt <%s --pass-pipeline='builtin.module(torch-backend-to-stablehlo-backend-pipeline)' | FileCheck %s
// CHECK-LABEL: @index_select_scalar_index
func.func @index_select_scalar_index(%input: !torch.vtensor<[2],si32>, %index: !torch.vtensor<[],si32>) -> !torch.vtensor<[1],si32> {
  %int0 = torch.constant.int 0
  // CHECK: stablehlo.reshape
  // CHECK-SAME: (tensor<i32>) -> tensor<1xi32>
  // CHECK: stablehlo.gather
  // CHECK-SAME: -> tensor<1xi32>
  %result = torch.aten.index_select %input, %int0, %index : !torch.vtensor<[2],si32>, !torch.int, !torch.vtensor<[],si32> -> !torch.vtensor<[1],si32>
  return %result : !torch.vtensor<[1],si32>
}

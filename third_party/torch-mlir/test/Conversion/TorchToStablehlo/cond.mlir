// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s

// torch.prim.If lowers to stablehlo.if: the !torch.bool condition is converted
// to i1 and wrapped into a 0-d tensor<i1> (via tensor.from_elements), and the
// then/else regions are inlined into the true/false branches.
// CHECK-LABEL:  func.func @torch.prim.if$basic(
// CHECK-SAME:     %[[PRED:.*]]: !torch.bool,
// CHECK-SAME:     %[[A:.*]]: !torch.vtensor<[4],f32>,
// CHECK-SAME:     %[[B:.*]]: !torch.vtensor<[4],f32>
func.func @torch.prim.if$basic(%pred: !torch.bool, %a: !torch.vtensor<[4],f32>, %b: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
  // CHECK-DAG:  %[[I1:.*]] = torch_c.to_i1 %[[PRED]]
  // CHECK-DAG:  %[[TA:.*]] = torch_c.to_builtin_tensor %[[A]] : !torch.vtensor<[4],f32> -> tensor<4xf32>
  // CHECK-DAG:  %[[TB:.*]] = torch_c.to_builtin_tensor %[[B]] : !torch.vtensor<[4],f32> -> tensor<4xf32>
  // CHECK:      %[[PREDT:.*]] = tensor.from_elements %[[I1]] : tensor<i1>
  // CHECK:      %[[IF:.*]] = "stablehlo.if"(%[[PREDT]]) ({
  // CHECK:        stablehlo.return %[[TA]] : tensor<4xf32>
  // CHECK:      }, {
  // CHECK:        stablehlo.return %[[TB]] : tensor<4xf32>
  // CHECK:      }) : (tensor<i1>) -> tensor<4xf32>
  %0 = torch.prim.If %pred -> (!torch.vtensor<[4],f32>) {
    torch.prim.If.yield %a : !torch.vtensor<[4],f32>
  } else {
    torch.prim.If.yield %b : !torch.vtensor<[4],f32>
  }
  // CHECK:      %[[OUT:.*]] = torch_c.from_builtin_tensor %[[IF]] : tensor<4xf32> -> !torch.vtensor<[4],f32>
  // CHECK:      return %[[OUT]] : !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// -----

// A branch body with a real computation converts inside the region: the
// aten op becomes its stablehlo equivalent and torch.prim.If.yield becomes
// stablehlo.return of the converted value.
// CHECK-LABEL:  func.func @torch.prim.if$compute(
func.func @torch.prim.if$compute(%pred: !torch.bool, %x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
  // CHECK:      "stablehlo.if"
  // CHECK:        chlo.broadcast_add
  // CHECK:        stablehlo.return %{{.*}} : tensor<4xf32>
  // CHECK:      }, {
  // CHECK:        chlo.broadcast_subtract
  // CHECK:        stablehlo.return %{{.*}} : tensor<4xf32>
  %float1 = torch.constant.float 1.000000e+00
  %int1 = torch.constant.int 1
  %0 = torch.prim.If %pred -> (!torch.vtensor<[4],f32>) {
    %1 = torch.aten.add.Scalar %x, %float1, %int1 : !torch.vtensor<[4],f32>, !torch.float, !torch.int -> !torch.vtensor<[4],f32>
    torch.prim.If.yield %1 : !torch.vtensor<[4],f32>
  } else {
    %2 = torch.aten.sub.Scalar %x, %float1, %int1 : !torch.vtensor<[4],f32>, !torch.float, !torch.int -> !torch.vtensor<[4],f32>
    torch.prim.If.yield %2 : !torch.vtensor<[4],f32>
  }
  return %0 : !torch.vtensor<[4],f32>
}

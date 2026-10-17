// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s

// Special-function lowerings to CHLO. The non-broadcast CHLO ops are finished to
// StableHLO by the upstream chlo-legalize-to-stablehlo pass. Binary ops emit the
// non-broadcast CHLO form directly (the broadcast variants are not handled by the
// Neuron CHLO custom-lowering pass).

// CHECK-LABEL:  func.func @torch.aten.erfc$basic(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:         %[[T1:.*]] = chlo.erfc %[[T0]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         %[[T2:.*]] = torch_c.from_builtin_tensor %[[T1]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[T2]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.erfc$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.erfc %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.erfinv$basic(
// CHECK:         %[[T1:.*]] = chlo.erf_inv %{{.*}} : tensor<?x?xf32> -> tensor<?x?xf32>
func.func @torch.aten.erfinv$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.erfinv %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.lgamma$basic(
// CHECK:         %[[T1:.*]] = chlo.lgamma %{{.*}} : tensor<?x?xf32> -> tensor<?x?xf32>
func.func @torch.aten.lgamma$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.lgamma %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.digamma$basic(
// CHECK:         %[[T1:.*]] = chlo.digamma %{{.*}} : tensor<?x?xf32> -> tensor<?x?xf32>
func.func @torch.aten.digamma$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.digamma %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.special_i1e$basic(
// CHECK:         %[[T1:.*]] = chlo.bessel_i1e %{{.*}} : tensor<?x?xf32> -> tensor<?x?xf32>
func.func @torch.aten.special_i1e$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.special_i1e %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.nextafter$basic(
// CHECK-DAG:     %[[T0:.*]] = torch_c.to_builtin_tensor %{{.*}} : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK-DAG:     %[[T1:.*]] = torch_c.to_builtin_tensor %{{.*}} : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:         %[[T2:.*]] = chlo.next_after %{{.*}}, %{{.*}} : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
func.func @torch.aten.nextafter$basic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.nextafter %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.special_zeta$basic(
// CHECK:         %[[T2:.*]] = chlo.zeta %{{.*}}, %{{.*}} : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
func.func @torch.aten.special_zeta$basic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.special_zeta %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

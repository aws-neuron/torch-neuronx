// RUN: torch-mlir-opt %s | FileCheck %s

// Verify that f8E8M0FNU (OCP MX E8M0 scale dtype) is accepted as a valid
// tensor element type and round-trips through the torch dialect.

// CHECK-LABEL: func.func @f8E8M0FNU_roundtrip
func.func @f8E8M0FNU_roundtrip(%arg0: !torch.vtensor<[4,8],f8E8M0FNU>) -> !torch.vtensor<[4,8],f8E8M0FNU> {
  // CHECK: return %arg0 : !torch.vtensor<[4,8],f8E8M0FNU>
  return %arg0 : !torch.vtensor<[4,8],f8E8M0FNU>
}

// CHECK-LABEL: func.func @f8E8M0FNU_unknown_size
func.func @f8E8M0FNU_unknown_size(%arg0: !torch.vtensor<[?,?],f8E8M0FNU>) -> !torch.vtensor<[?,?],f8E8M0FNU> {
  // CHECK: return %arg0 : !torch.vtensor<[?,?],f8E8M0FNU>
  return %arg0 : !torch.vtensor<[?,?],f8E8M0FNU>
}

// CHECK-LABEL: func.func @f8E8M0FNU_1d
func.func @f8E8M0FNU_1d(%arg0: !torch.vtensor<[32],f8E8M0FNU>) -> !torch.vtensor<[32],f8E8M0FNU> {
  // CHECK: return %arg0 : !torch.vtensor<[32],f8E8M0FNU>
  return %arg0 : !torch.vtensor<[32],f8E8M0FNU>
}

// RUN: torch-mlir-opt %s -convert-torch-to-neuron -split-input-file | FileCheck %s

// Dynamic spatial dimensions are left untouched by TorchToNeuron so they fall
// through to the TorchToStablehlo decomposition.
// CHECK-LABEL:   func.func @max_pool2d_with_indices$dynamic_spatial(
// CHECK-SAME:        %[[ARG0:[a-zA-Z0-9_]+]]: !torch.vtensor<[1,1,?,?],f32>) -> (!torch.vtensor<[1,1,?,?],f32>, !torch.vtensor<[1,1,?,?],si64>) {
// CHECK:           %[[INT2:.*]] = torch.constant.int 2
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[KS:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[ST:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[PD:.*]] = torch.prim.ListConstruct %[[INT0]], %[[INT0]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[DL:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[R0:.*]], %[[R1:.*]] = torch.aten.max_pool2d_with_indices %[[ARG0]], %[[KS]], %[[ST]], %[[PD]], %[[DL]], %[[FALSE]] : !torch.vtensor<[1,1,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,?,?],f32>, !torch.vtensor<[1,1,?,?],si64>
// CHECK:           return %[[R0]], %[[R1]] : !torch.vtensor<[1,1,?,?],f32>, !torch.vtensor<[1,1,?,?],si64>
func.func @max_pool2d_with_indices$dynamic_spatial(%arg0: !torch.vtensor<[1,1,?,?],f32>) -> (!torch.vtensor<[1,1,?,?],f32>, !torch.vtensor<[1,1,?,?],si64>) {
  %int2 = torch.constant.int 2
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %result0, %result1 = torch.aten.max_pool2d_with_indices %arg0, %0, %1, %2, %3, %false : !torch.vtensor<[1,1,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,?,?],f32>, !torch.vtensor<[1,1,?,?],si64>
  return %result0, %result1 : !torch.vtensor<[1,1,?,?],f32>, !torch.vtensor<[1,1,?,?],si64>
}

// -----

// Static shapes are decomposed by TorchToNeuron via the aten.im2col lowering
// rather than being handed off to TorchToStablehlo.
// CHECK-LABEL:   func.func @max_pool2d_with_indices$static(
// CHECK-SAME:        %[[ARG0:[a-zA-Z0-9_]+]]: !torch.vtensor<[1,1,8,8],f32>) -> (!torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,1,4,4],si64>) {
// CHECK-NOT:       torch.aten.max_pool2d_with_indices
// CHECK:           %[[IM2COL:.*]] = torch.aten.im2col %[[ARG0]], {{.*}} : !torch.vtensor<[1,1,8,8],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.vtensor<[1,4,16],f32>
// CHECK:           %[[COLS:.*]] = torch.aten.reshape %[[IM2COL]], {{.*}} -> !torch.vtensor<[1,1,4,16],f32>
// CHECK:           %[[VALUES:.*]], %[[INDICES:.*]] = torch.aten.max.dim %[[COLS]], {{.*}} -> !torch.vtensor<[1,1,16],f32>, !torch.vtensor<[1,1,16],si64>
// CHECK:           torch.aten.max.dim %[[COLS]], {{.*}} -> !torch.vtensor<[1,1,16],f32>, !torch.vtensor<[1,1,16],si64>
// CHECK:           torch.aten.floor_divide.Scalar
// CHECK:           torch.aten.remainder.Scalar
// CHECK:           %[[VALOUT:.*]] = torch.aten.reshape %[[VALUES]], {{.*}} -> !torch.vtensor<[1,1,4,4],f32>
// CHECK:           return %[[VALOUT]], %{{.*}} : !torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,1,4,4],si64>
func.func @max_pool2d_with_indices$static(%arg0: !torch.vtensor<[1,1,8,8],f32>) -> (!torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,1,4,4],si64>) {
  %int2 = torch.constant.int 2
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %result0, %result1 = torch.aten.max_pool2d_with_indices %arg0, %0, %1, %2, %3, %false : !torch.vtensor<[1,1,8,8],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,1,4,4],si64>
  return %result0, %result1 : !torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,1,4,4],si64>
}

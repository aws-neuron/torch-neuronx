// RUN: torch-mlir-opt -torch-adjust-calling-conventions -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL:   func.func @basic(
// CHECK-SAME:                %[[ARG:.*]]: !torch.vtensor<[2,3,?],f32>) -> !torch.tensor {
// CHECK:           %[[ERASED:.*]] = torch.tensor_static_info_cast %[[ARG]] : !torch.vtensor<[2,3,?],f32> to !torch.vtensor
// CHECK:           %[[NONVAL_TENSOR:.*]] = torch.copy.to_tensor %[[ERASED]] : !torch.tensor
// CHECK:           return %[[NONVAL_TENSOR]] : !torch.tensor
func.func @basic(%arg0: !torch.tensor {torch.type_bound = !torch.vtensor<[2,3,?],f32>}) -> !torch.tensor {
  return %arg0 : !torch.tensor
}

// -----

// CHECK-LABEL:   func.func @no_type_bound(
// CHECK-SAME:                        %[[ARG:.*]]: !torch.tensor) -> !torch.tensor {
// CHECK:           return %[[ARG]] : !torch.tensor
func.func @no_type_bound(%arg0: !torch.tensor) -> !torch.tensor {
  return %arg0 : !torch.tensor
}

// -----

// CHECK-LABEL:   func.func @call(
// CHECK-SAME:               %[[ARG:.*]]: !torch.vtensor<[2,3,?],f32>) -> !torch.tensor {
// CHECK:           %[[ARG_ERASED:.*]] = torch.tensor_static_info_cast %[[ARG]] : !torch.vtensor<[2,3,?],f32> to !torch.vtensor
// CHECK:           %[[ARG_NONVAL:.*]] = torch.copy.to_tensor %[[ARG_ERASED]] : !torch.tensor
// CHECK:           %[[INFO_ADDED:.*]] = torch.tensor_static_info_cast %[[ARG_NONVAL]] : !torch.tensor to !torch.tensor<[2,3,?],f32>
// CHECK:           %[[CALL_ARG:.*]] = torch.copy.to_vtensor %[[INFO_ADDED]] : !torch.vtensor<[2,3,?],f32>
// CHECK:           %[[CALL_RES:.*]] = call @call(%[[CALL_ARG]]) : (!torch.vtensor<[2,3,?],f32>) -> !torch.tensor
// CHECK:           return %[[ARG_NONVAL]] : !torch.tensor
func.func @call(%arg0: !torch.tensor {torch.type_bound = !torch.vtensor<[2,3,?],f32>}) -> !torch.tensor {
  %0 = call @call(%arg0) : (!torch.tensor) -> !torch.tensor
  return %arg0 : !torch.tensor
}

// -----

// CHECK-LABEL:   func.func @none_return() {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           return
func.func @none_return() -> !torch.none {
  %1 = torch.constant.none
  return %1 : !torch.none
}

// CHECK-LABEL:   func.func @none_call_return() {
// CHECK:           call @none_return() : () -> ()
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           "test.use"(%[[NONE]]) : (!torch.none) -> ()
// CHECK:           return
func.func @none_call_return() {
  %0 = call @none_return() : () -> !torch.none
  "test.use"(%0) : (!torch.none) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @tuple_return(
// CHECK:                     %[[ARG0:.*]]: !torch.vtensor<[?],f32>,
// CHECK:                     %[[ARG1:.*]]: !torch.vtensor<[?],f32>) -> (!torch.tensor, !torch.tensor) {
// CHECK:         %[[ARG1_ERASED:.*]] = torch.tensor_static_info_cast %[[ARG1]] : !torch.vtensor<[?],f32> to !torch.vtensor
// CHECK:         %[[ARG1_NONVAL:.*]] = torch.copy.to_tensor %[[ARG1_ERASED]] : !torch.tensor
// CHECK:         %[[ARG0_ERASED:.*]] = torch.tensor_static_info_cast %[[ARG0]] : !torch.vtensor<[?],f32> to !torch.vtensor
// CHECK:         %[[ARG0_NONVAL:.*]] = torch.copy.to_tensor %[[ARG0_ERASED]] : !torch.tensor
// CHECK:         %[[TUPLE:.*]] = torch.prim.TupleConstruct %[[ARG0_NONVAL]], %[[ARG1_NONVAL]] : !torch.tensor, !torch.tensor -> !torch.tuple<tensor, tensor>
// CHECK:         %[[CST0:.*]] = torch.constant.int 0
// CHECK:         %[[RET0:.*]] = torch.prim.TupleIndex %[[TUPLE]], %[[CST0]] : !torch.tuple<tensor, tensor>, !torch.int -> !torch.tensor
// CHECK:         %[[CST1:.*]] = torch.constant.int 1
// CHECK:         %[[RET1:.*]] = torch.prim.TupleIndex %[[TUPLE]], %[[CST1]] : !torch.tuple<tensor, tensor>, !torch.int -> !torch.tensor
// CHECK:         return %[[RET0]], %[[RET1]] : !torch.tensor, !torch.tensor
func.func @tuple_return(%arg0: !torch.tensor {torch.type_bound = !torch.vtensor<[?],f32>},
                   %arg1: !torch.tensor {torch.type_bound = !torch.vtensor<[?],f32>}) -> !torch.tuple<tensor, tensor> {
  %1 = torch.prim.TupleConstruct %arg0, %arg1 : !torch.tensor, !torch.tensor -> !torch.tuple<tensor, tensor>
  return %1 : !torch.tuple<tensor, tensor>
}

// CHECK-LABEL: func.func @call_tuple_return(
// CHECK:                          %[[ARG0:.*]]: !torch.vtensor<[?],f32>,
// CHECK:                          %[[ARG1:.*]]: !torch.vtensor<[?],f32>) -> (!torch.tensor, !torch.tensor) {
// CHECK:         %[[ARG1_ERASED:.*]] = torch.tensor_static_info_cast %[[ARG1]] : !torch.vtensor<[?],f32> to !torch.vtensor
// CHECK:         %[[ARG1_NONVAL:.*]] = torch.copy.to_tensor %[[ARG1_ERASED]] : !torch.tensor
// CHECK:         %[[ARG0_ERASED:.*]] = torch.tensor_static_info_cast %[[ARG0]] : !torch.vtensor<[?],f32> to !torch.vtensor
// CHECK:         %[[ARG0_NONVAL:.*]] = torch.copy.to_tensor %[[ARG0_ERASED]] : !torch.tensor
// CHECK:         %[[ARG0_NONVAL_SHAPED:.*]] = torch.tensor_static_info_cast %[[ARG0_NONVAL]] : !torch.tensor to !torch.tensor<[?],f32>
// CHECK:         %[[ARG0_VAL_SHAPED:.*]] = torch.copy.to_vtensor %[[ARG0_NONVAL_SHAPED]] : !torch.vtensor<[?],f32>
// CHECK:         %[[ARG1_NONVAL_SHAPED:.*]] = torch.tensor_static_info_cast %[[ARG1_NONVAL]] : !torch.tensor to !torch.tensor<[?],f32>
// CHECK:         %[[ARG1_VAL_SHAPED:.*]] = torch.copy.to_vtensor %[[ARG1_NONVAL_SHAPED]] : !torch.vtensor<[?],f32>
// CHECK:         %[[RETS:.*]]:2 = call @tuple_return(%[[ARG0_VAL_SHAPED]], %[[ARG1_VAL_SHAPED]]) : (!torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>) -> (!torch.tensor, !torch.tensor)
// CHECK:         %[[TUPLE:.*]] = torch.prim.TupleConstruct %[[RETS]]#0, %[[RETS]]#1 : !torch.tensor, !torch.tensor -> !torch.tuple<tensor, tensor>
// CHECK:         %[[CST0:.*]] = torch.constant.int 0
// CHECK:         %[[RET0:.*]] = torch.prim.TupleIndex %[[TUPLE]], %[[CST0]] : !torch.tuple<tensor, tensor>, !torch.int -> !torch.tensor
// CHECK:         %[[CST1:.*]] = torch.constant.int 1
// CHECK:         %[[RET1:.*]] = torch.prim.TupleIndex %[[TUPLE]], %[[CST1]] : !torch.tuple<tensor, tensor>, !torch.int -> !torch.tensor
// CHECK:         return %[[RET0]], %[[RET1]] : !torch.tensor, !torch.tensor
func.func @call_tuple_return(%arg0: !torch.tensor {torch.type_bound = !torch.vtensor<[?],f32>},
                        %arg1: !torch.tensor {torch.type_bound = !torch.vtensor<[?],f32>}) -> !torch.tuple<tensor, tensor> {
  %0 = call @tuple_return(%arg0, %arg1) : (!torch.tensor, !torch.tensor) -> !torch.tuple<tensor, tensor>
  return %0 : !torch.tuple<tensor, tensor>
}

// -----

// CHECK-LABEL: func.func @main(
// CHECK-SAME:                %[[ARG0:.*]]: !torch.vtensor<[4,8],f32>,
// CHECK-SAME:                %[[ARG1:.*]]: !torch.vtensor<[4,8],f32>) -> (!torch.vtensor<[8,8],f32>, !torch.vtensor<[8],f32>) {
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[LIST0:.*]] = torch.prim.ListConstruct %[[INT0]] : (!torch.int) -> !torch.list<int>
// CHECK:         %[[TRUE:.*]] = torch.constant.bool true
// CHECK:         %[[NONE:.*]] = torch.constant.none
// CHECK:         %[[SUM:.*]] = torch.aten.sum.dim_IntList %[[ARG1]], %[[LIST0]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[4,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,8],f32>
// CHECK:         %[[INT8:.*]] = torch.constant.int 8
// CHECK:         %[[LIST8:.*]] = torch.prim.ListConstruct %[[INT8]] : (!torch.int) -> !torch.list<int>
// CHECK:         %[[VIEW:.*]] = torch.aten.view %[[SUM]], %[[LIST8]] : !torch.vtensor<[1,8],f32>, !torch.list<int> -> !torch.vtensor<[8],f32>
// CHECK:         %[[T:.*]] = torch.aten.t %[[ARG0]] : !torch.vtensor<[4,8],f32> -> !torch.vtensor<[8,4],f32>
// CHECK:         %[[MM:.*]] = torch.aten.mm %[[T]], %[[ARG1]] : !torch.vtensor<[8,4],f32>, !torch.vtensor<[4,8],f32> -> !torch.vtensor<[8,8],f32>
// CHECK:         %[[NONE_0:.*]] = torch.constant.none
// CHECK:         return %[[MM]], %[[VIEW]] : !torch.vtensor<[8,8],f32>, !torch.vtensor<[8],f32>
func.func @main(%arg0: !torch.vtensor<[4,8],f32>, %arg1: !torch.vtensor<[4,8],f32>) -> (!torch.vtensor<[8,8],f32>, !torch.none, !torch.vtensor<[8],f32>) {
  %int0 = torch.constant.int 0
  %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %true = torch.constant.bool true
  %none = torch.constant.none
  %1 = torch.aten.sum.dim_IntList %arg1, %0, %true, %none : !torch.vtensor<[4,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,8],f32>
  %int8 = torch.constant.int 8
  %2 = torch.prim.ListConstruct %int8 : (!torch.int) -> !torch.list<int>
  %3 = torch.aten.view %1, %2 : !torch.vtensor<[1,8],f32>, !torch.list<int> -> !torch.vtensor<[8],f32>
  %4 = torch.aten.t %arg0 : !torch.vtensor<[4,8],f32> -> !torch.vtensor<[8,4],f32>
  %5 = torch.aten.mm %4, %arg1 : !torch.vtensor<[8,4],f32>, !torch.vtensor<[4,8],f32> -> !torch.vtensor<[8,8],f32>
  %none_0 = torch.constant.none
  return %5, %none_0, %3 : !torch.vtensor<[8,8],f32>, !torch.none, !torch.vtensor<[8],f32>
}

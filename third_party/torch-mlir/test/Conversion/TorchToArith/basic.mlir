// RUN: torch-mlir-opt <%s -convert-torch-to-arith -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @add_int(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: !torch.int, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.int) -> !torch.int {
// CHECK:         %[[B:.*]] = torch_c.to_i64 %[[ARG1]]
// CHECK:         %[[A:.*]] = torch_c.to_i64 %[[ARG0]]
// CHECK:         %[[RES:.*]] = arith.addi %[[A]], %[[B]] : i64
// CHECK:         %[[OUT:.*]] = torch_c.from_i64 %[[RES]]
// CHECK:         return %[[OUT]] : !torch.int
func.func @add_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  %0 = torch.aten.add.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.int
  return %0 : !torch.int
}

// -----

// CHECK-LABEL: func.func @mul_int(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: !torch.int, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.int) -> !torch.int {
// CHECK:         %[[B:.*]] = torch_c.to_i64 %[[ARG1]]
// CHECK:         %[[A:.*]] = torch_c.to_i64 %[[ARG0]]
// CHECK:         %[[RES:.*]] = arith.muli %[[A]], %[[B]] : i64
// CHECK:         %[[OUT:.*]] = torch_c.from_i64 %[[RES]]
// CHECK:         return %[[OUT]] : !torch.int
func.func @mul_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  %0 = torch.aten.mul.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.int
  return %0 : !torch.int
}

// -----

// CHECK-LABEL: func.func @sub_int(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: !torch.int, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.int) -> !torch.int {
// CHECK:         %[[B:.*]] = torch_c.to_i64 %[[ARG1]]
// CHECK:         %[[A:.*]] = torch_c.to_i64 %[[ARG0]]
// CHECK:         %[[RES:.*]] = arith.subi %[[A]], %[[B]] : i64
// CHECK:         %[[OUT:.*]] = torch_c.from_i64 %[[RES]]
// CHECK:         return %[[OUT]] : !torch.int
func.func @sub_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  %0 = torch.aten.sub.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.int
  return %0 : !torch.int
}

// -----

// CHECK-LABEL: func.func @neg_int(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: !torch.int) -> !torch.int {
// CHECK:         %[[A:.*]] = torch_c.to_i64 %[[ARG0]]
// CHECK:         %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK:         %[[RES:.*]] = arith.subi %[[ZERO]], %[[A]] : i64
// CHECK:         %[[OUT:.*]] = torch_c.from_i64 %[[RES]]
// CHECK:         return %[[OUT]] : !torch.int
func.func @neg_int(%arg0: !torch.int) -> !torch.int {
  %0 = torch.aten.neg.int %arg0 : !torch.int -> !torch.int
  return %0 : !torch.int
}

// -----

// CHECK-LABEL: func.func @constant_int() -> !torch.int {
// CHECK:         %[[C:.*]] = arith.constant 42 : i64
// CHECK:         %[[OUT:.*]] = torch_c.from_i64 %[[C]]
// CHECK:         return %[[OUT]] : !torch.int
func.func @constant_int() -> !torch.int {
  %0 = torch.constant.int 42
  return %0 : !torch.int
}

// -----

// CHECK-LABEL: func.func @constant_float() -> !torch.float {
// CHECK:         %[[C:.*]] = arith.constant 3.140000e+00 : f64
// CHECK:         %[[OUT:.*]] = torch_c.from_f64 %[[C]]
// CHECK:         return %[[OUT]] : !torch.float
func.func @constant_float() -> !torch.float {
  %0 = torch.constant.float 3.14
  return %0 : !torch.float
}

// -----

// CHECK-LABEL: func.func @constant_bool() -> !torch.bool {
// CHECK:         %[[C:.*]] = arith.constant true
// CHECK:         %[[OUT:.*]] = torch_c.from_i1 %[[C]]
// CHECK:         return %[[OUT]] : !torch.bool
func.func @constant_bool() -> !torch.bool {
  %0 = torch.constant.bool true
  return %0 : !torch.bool
}

// -----

// CHECK-LABEL: func.func @eq_int(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: !torch.int, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.int) -> !torch.bool {
// CHECK:         %[[B:.*]] = torch_c.to_i64 %[[ARG1]]
// CHECK:         %[[A:.*]] = torch_c.to_i64 %[[ARG0]]
// CHECK:         %[[CMP:.*]] = arith.cmpi eq, %[[A]], %[[B]] : i64
// CHECK:         %[[OUT:.*]] = torch_c.from_i1 %[[CMP]]
// CHECK:         return %[[OUT]] : !torch.bool
func.func @eq_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.bool {
  %0 = torch.aten.eq.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// -----

// CHECK-LABEL: func.func @lt_int(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: !torch.int, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.int) -> !torch.bool {
// CHECK:         %[[B:.*]] = torch_c.to_i64 %[[ARG1]]
// CHECK:         %[[A:.*]] = torch_c.to_i64 %[[ARG0]]
// CHECK:         %[[CMP:.*]] = arith.cmpi slt, %[[A]], %[[B]] : i64
// CHECK:         %[[OUT:.*]] = torch_c.from_i1 %[[CMP]]
// CHECK:         return %[[OUT]] : !torch.bool
func.func @lt_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.bool {
  %0 = torch.aten.lt.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// -----

// CHECK-LABEL: func.func @div_int(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: !torch.int, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.int) -> !torch.float {
// CHECK:         %[[B:.*]] = torch_c.to_i64 %[[ARG1]]
// CHECK:         %[[A:.*]] = torch_c.to_i64 %[[ARG0]]
// CHECK:         %[[AF:.*]] = arith.sitofp %[[A]] : i64 to f64
// CHECK:         %[[BF:.*]] = arith.sitofp %[[B]] : i64 to f64
// CHECK:         %[[RES:.*]] = arith.divf %[[AF]], %[[BF]] : f64
// CHECK:         %[[OUT:.*]] = torch_c.from_f64 %[[RES]]
// CHECK:         return %[[OUT]] : !torch.float
func.func @div_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.float {
  %0 = torch.aten.div.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.float
  return %0 : !torch.float
}

// -----

// CHECK-LABEL: func.func @floordiv_int(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: !torch.int, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.int) -> !torch.int {
// CHECK:         %[[B:.*]] = torch_c.to_i64 %[[ARG1]]
// CHECK:         %[[A:.*]] = torch_c.to_i64 %[[ARG0]]
// CHECK:         %[[RES:.*]] = arith.floordivsi %[[A]], %[[B]] : i64
// CHECK:         %[[OUT:.*]] = torch_c.from_i64 %[[RES]]
// CHECK:         return %[[OUT]] : !torch.int
func.func @floordiv_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  %0 = torch.aten.floordiv.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.int
  return %0 : !torch.int
}

// -----

// CHECK-LABEL: func.func @remainder_int(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: !torch.int, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.int) -> !torch.int {
// CHECK:         %[[B:.*]] = torch_c.to_i64 %[[ARG1]]
// CHECK:         %[[A:.*]] = torch_c.to_i64 %[[ARG0]]
// CHECK:         %[[RES:.*]] = arith.remsi %[[A]], %[[B]] : i64
// CHECK:         %[[OUT:.*]] = torch_c.from_i64 %[[RES]]
// CHECK:         return %[[OUT]] : !torch.int
func.func @remainder_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  %0 = torch.aten.remainder.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.int
  return %0 : !torch.int
}

// -----

// CHECK-LABEL: func.func @add_float_int(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: !torch.float, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.int) -> !torch.float {
// CHECK:         %[[B:.*]] = torch_c.to_i64 %[[ARG1]]
// CHECK:         %[[A:.*]] = torch_c.to_f64 %[[ARG0]]
// CHECK:         %[[BF:.*]] = arith.sitofp %[[B]] : i64 to f64
// CHECK:         %[[RES:.*]] = arith.addf %[[A]], %[[BF]] : f64
// CHECK:         %[[OUT:.*]] = torch_c.from_f64 %[[RES]]
// CHECK:         return %[[OUT]] : !torch.float
func.func @add_float_int(%arg0: !torch.float, %arg1: !torch.int) -> !torch.float {
  %0 = torch.aten.add.float_int %arg0, %arg1 : !torch.float, !torch.int -> !torch.float
  return %0 : !torch.float
}

// -----

// CHECK-LABEL: func.func @sub_float(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: !torch.float, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.float) -> !torch.float {
// CHECK:         %[[B:.*]] = torch_c.to_f64 %[[ARG1]]
// CHECK:         %[[A:.*]] = torch_c.to_f64 %[[ARG0]]
// CHECK:         %[[RES:.*]] = arith.subf %[[A]], %[[B]] : f64
// CHECK:         %[[OUT:.*]] = torch_c.from_f64 %[[RES]]
// CHECK:         return %[[OUT]] : !torch.float
func.func @sub_float(%arg0: !torch.float, %arg1: !torch.float) -> !torch.float {
  %0 = torch.aten.sub.float %arg0, %arg1 : !torch.float, !torch.float -> !torch.float
  return %0 : !torch.float
}

// -----

// CHECK-LABEL: func.func @mul_float(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: !torch.float, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.float) -> !torch.float {
// CHECK:         %[[B:.*]] = torch_c.to_f64 %[[ARG1]]
// CHECK:         %[[A:.*]] = torch_c.to_f64 %[[ARG0]]
// CHECK:         %[[RES:.*]] = arith.mulf %[[A]], %[[B]] : f64
// CHECK:         %[[OUT:.*]] = torch_c.from_f64 %[[RES]]
// CHECK:         return %[[OUT]] : !torch.float
func.func @mul_float(%arg0: !torch.float, %arg1: !torch.float) -> !torch.float {
  %0 = torch.aten.mul.float %arg0, %arg1 : !torch.float, !torch.float -> !torch.float
  return %0 : !torch.float
}

// -----

// CHECK-LABEL: func.func @div_float(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: !torch.float, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.float) -> !torch.float {
// CHECK:         %[[B:.*]] = torch_c.to_f64 %[[ARG1]]
// CHECK:         %[[A:.*]] = torch_c.to_f64 %[[ARG0]]
// CHECK:         %[[RES:.*]] = arith.divf %[[A]], %[[B]] : f64
// CHECK:         %[[OUT:.*]] = torch_c.from_f64 %[[RES]]
// CHECK:         return %[[OUT]] : !torch.float
func.func @div_float(%arg0: !torch.float, %arg1: !torch.float) -> !torch.float {
  %0 = torch.aten.div.float %arg0, %arg1 : !torch.float, !torch.float -> !torch.float
  return %0 : !torch.float
}

// -----

// CHECK-LABEL: func.func @ge_int(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: !torch.int, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.int) -> !torch.bool {
// CHECK:         %[[B:.*]] = torch_c.to_i64 %[[ARG1]]
// CHECK:         %[[A:.*]] = torch_c.to_i64 %[[ARG0]]
// CHECK:         %[[CMP:.*]] = arith.cmpi sge, %[[A]], %[[B]] : i64
// CHECK:         %[[OUT:.*]] = torch_c.from_i1 %[[CMP]]
// CHECK:         return %[[OUT]] : !torch.bool
func.func @ge_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.bool {
  %0 = torch.aten.ge.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// -----

// CHECK-LABEL: func.func @ne_int(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: !torch.int, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.int) -> !torch.bool {
// CHECK:         %[[B:.*]] = torch_c.to_i64 %[[ARG1]]
// CHECK:         %[[A:.*]] = torch_c.to_i64 %[[ARG0]]
// CHECK:         %[[CMP:.*]] = arith.cmpi ne, %[[A]], %[[B]] : i64
// CHECK:         %[[OUT:.*]] = torch_c.from_i1 %[[CMP]]
// CHECK:         return %[[OUT]] : !torch.bool
func.func @ne_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.bool {
  %0 = torch.aten.ne.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// -----

// CHECK-LABEL: func.func @bool_int(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: !torch.int) -> !torch.bool {
// CHECK:         %[[A:.*]] = torch_c.to_i64 %[[ARG0]]
// CHECK:         %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK:         %[[TRUE:.*]] = arith.constant true
// CHECK:         %[[FALSE:.*]] = arith.constant false
// CHECK:         %[[CMP:.*]] = arith.cmpi ne, %[[A]], %[[ZERO]] : i64
// CHECK:         %[[SEL:.*]] = arith.select %[[CMP]], %[[TRUE]], %[[FALSE]] : i1
// CHECK:         %[[OUT:.*]] = torch_c.from_i1 %[[SEL]]
// CHECK:         return %[[OUT]] : !torch.bool
func.func @bool_int(%arg0: !torch.int) -> !torch.bool {
  %0 = torch.aten.Bool.int %arg0 : !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// -----

// CHECK-LABEL: func.func @int_bool(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: !torch.bool) -> !torch.int {
// CHECK:         %[[A:.*]] = torch_c.to_i1 %[[ARG0]]
// CHECK:         %[[RES:.*]] = arith.extui %[[A]] : i1 to i64
// CHECK:         %[[OUT:.*]] = torch_c.from_i64 %[[RES]]
// CHECK:         return %[[OUT]] : !torch.int
func.func @int_bool(%arg0: !torch.bool) -> !torch.int {
  %0 = torch.aten.Int.bool %arg0 : !torch.bool -> !torch.int
  return %0 : !torch.int
}

// -----

// CHECK-LABEL:   func.func @pow_int(
// CHECK-SAME:        %[[ARG0:.*]]: !torch.int, %[[ARG1:.*]]: !torch.int) -> !torch.int {
// CHECK:           %[[B:.*]] = torch_c.to_i64 %[[ARG1]]
// CHECK:           %[[A:.*]] = torch_c.to_i64 %[[ARG0]]
// CHECK:           %[[RES:.*]] = math.ipowi %[[A]], %[[B]] : i64
// CHECK:           %[[OUT:.*]] = torch_c.from_i64 %[[RES]]
// CHECK:           return %[[OUT]] : !torch.int
func.func @pow_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  %0 = torch.aten.pow.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.int
  return %0 : !torch.int
}

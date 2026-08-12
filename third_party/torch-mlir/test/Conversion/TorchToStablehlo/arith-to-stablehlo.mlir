// RUN: torch-mlir-opt -convert-torch-to-neuron-custom-lowering -split-input-file %s | FileCheck %s

module {
  func.func @test_extf(%arg0: tensor<2x2xf16>) -> tensor<2x2xf32> {
    %0 = arith.extf %arg0 : tensor<2x2xf16> to tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
// CHECK-LABEL:   func.func @test_extf(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<2x2xf16>) -> tensor<2x2xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.convert %[[VAL_0]] : (tensor<2x2xf16>) -> tensor<2x2xf32>
// CHECK:           return %[[VAL_1]] : tensor<2x2xf32>

// -----

module {
  func.func @test_extui(%arg0: tensor<2x2xi8>) -> tensor<2x2xi32> {
    %0 = arith.extui %arg0 : tensor<2x2xi8> to tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}
// CHECK-LABEL:   func.func @test_extui(
// CHECK-SAME:                          %[[VAL_0:.*]]: tensor<2x2xi8>) -> tensor<2x2xi32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.convert %[[VAL_0]] : (tensor<2x2xi8>) -> tensor<2x2xi32>
// CHECK:           return %[[VAL_1]] : tensor<2x2xi32>

// -----

module {
  func.func @test_maxui(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = arith.maxui %arg0, %arg1 : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}
// CHECK-LABEL:   func.func @test_maxui(
// CHECK-SAME:                          %[[VAL_0:.*]]: tensor<2x2xi32>,
// CHECK-SAME:                          %[[VAL_1:.*]]: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.maximum %[[VAL_0]], %[[VAL_1]] : tensor<2x2xi32>
// CHECK:           return %[[VAL_2]] : tensor<2x2xi32>

// -----

module {
  func.func @test_minui(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = arith.minui %arg0, %arg1 : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}
// CHECK-LABEL:   func.func @test_minui(
// CHECK-SAME:                          %[[VAL_0:.*]]: tensor<2x2xi32>,
// CHECK-SAME:                          %[[VAL_1:.*]]: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.minimum %[[VAL_0]], %[[VAL_1]] : tensor<2x2xi32>
// CHECK:           return %[[VAL_2]] : tensor<2x2xi32>

// -----

module {
  func.func @test_addf(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
// CHECK-LABEL:   func.func @test_addf(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<2x2xf32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.add %[[VAL_0]], %[[VAL_1]] : tensor<2x2xf32>
// CHECK:           return %[[VAL_2]] : tensor<2x2xf32>

// -----

module {
  func.func @test_andi(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = arith.andi %arg0, %arg1 : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}
// CHECK-LABEL:   func.func @test_andi(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<2x2xi32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.and %[[VAL_0]], %[[VAL_1]] : tensor<2x2xi32>
// CHECK:           return %[[VAL_2]] : tensor<2x2xi32>

// -----

module {
  func.func @test_bitcast(%arg0: tensor<2x2xf32>) -> tensor<2x2xi32> {
    %0 = arith.bitcast %arg0 : tensor<2x2xf32> to tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}
// CHECK-LABEL:   func.func @test_bitcast(
// CHECK-SAME:                            %[[VAL_0:.*]]: tensor<2x2xf32>) -> tensor<2x2xi32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.convert %[[VAL_0]] : (tensor<2x2xf32>) -> tensor<2x2xi32>
// CHECK:           return %[[VAL_1]] : tensor<2x2xi32>

// -----

module {
  func.func @test_cmpf(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
    %0 = arith.cmpf olt, %arg0, %arg1 : tensor<2x2xf32>
    return %0 : tensor<2x2xi1>
  }
}
// CHECK-LABEL:   func.func @test_cmpf(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<2x2xf32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: tensor<2x2xf32>) -> tensor<2x2xi1> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.compare  LT, %[[VAL_0]], %[[VAL_1]],  FLOAT : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
// CHECK:           return %[[VAL_2]] : tensor<2x2xi1>

// -----

module {
  func.func @test_divf(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = arith.divf %arg0, %arg1 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
// CHECK-LABEL:   func.func @test_divf(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<2x2xf32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.divide %[[VAL_0]], %[[VAL_1]] : tensor<2x2xf32>
// CHECK:           return %[[VAL_2]] : tensor<2x2xf32>

// -----

module {
  func.func @test_fptosi(%arg0: tensor<2x2xf32>) -> tensor<2x2xi32> {
    %0 = arith.fptosi %arg0 : tensor<2x2xf32> to tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}
// CHECK-LABEL:   func.func @test_fptosi(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<2x2xf32>) -> tensor<2x2xi32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.convert %[[VAL_0]] : (tensor<2x2xf32>) -> tensor<2x2xi32>
// CHECK:           return %[[VAL_1]] : tensor<2x2xi32>

// -----

module {
  func.func @test_fptoui(%arg0: tensor<2x2xf32>) -> tensor<2x2xi32> {
    %0 = arith.fptoui %arg0 : tensor<2x2xf32> to tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}
// CHECK-LABEL:   func.func @test_fptoui(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<2x2xf32>) -> tensor<2x2xi32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.convert %[[VAL_0]] : (tensor<2x2xf32>) -> tensor<2x2xi32>
// CHECK:           return %[[VAL_1]] : tensor<2x2xi32>

// -----

module {
  func.func @test_index_castui(%arg0: tensor<2x2xi32>) -> tensor<2x2xi64> {
    %0 = tensor.empty() : tensor<2x2xindex>
    %1 = arith.index_castui %0 : tensor<2x2xindex> to tensor<2x2xi64>
    return %1 : tensor<2x2xi64>
  }
}
// CHECK-LABEL:   func.func @test_index_castui(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<2x2xi32>) -> tensor<2x2xi64> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<0> : tensor<2x2xi64>
// CHECK:           return %[[VAL_1]] : tensor<2x2xi64>

// -----

module {
  func.func @test_maximumf(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = arith.maximumf %arg0, %arg1 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
// CHECK-LABEL:   func.func @test_maximumf(
// CHECK-SAME:                             %[[VAL_0:.*]]: tensor<2x2xf32>,
// CHECK-SAME:                             %[[VAL_1:.*]]: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.maximum %[[VAL_0]], %[[VAL_1]] : tensor<2x2xf32>
// CHECK:           return %[[VAL_2]] : tensor<2x2xf32>

// -----

module {
  func.func @test_minimumf(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = arith.minimumf %arg0, %arg1 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
// CHECK-LABEL:   func.func @test_minimumf(
// CHECK-SAME:                             %[[VAL_0:.*]]: tensor<2x2xf32>,
// CHECK-SAME:                             %[[VAL_1:.*]]: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.minimum %[[VAL_0]], %[[VAL_1]] : tensor<2x2xf32>
// CHECK:           return %[[VAL_2]] : tensor<2x2xf32>

// -----

module {
  func.func @test_mulf(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = arith.mulf %arg0, %arg1 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
// CHECK-LABEL:   func.func @test_mulf(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<2x2xf32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.multiply %[[VAL_0]], %[[VAL_1]] : tensor<2x2xf32>
// CHECK:           return %[[VAL_2]] : tensor<2x2xf32>

// -----

module {
  func.func @test_combined(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
    %0 = arith.addf %arg0, %arg1 : tensor<2x2xf32>
    %1 = arith.mulf %0, %arg1 : tensor<2x2xf32>
    %2 = arith.divf %1, %arg0 : tensor<2x2xf32>
    %3 = arith.maximumf %2, %arg0 : tensor<2x2xf32>
    %4 = arith.minimumf %3, %arg1 : tensor<2x2xf32>
    %5 = arith.cmpf olt, %4, %arg0 : tensor<2x2xf32>
    return %5 : tensor<2x2xi1>
  }
}
// CHECK-LABEL:   func.func @test_combined(
// CHECK-SAME:                             %[[VAL_0:.*]]: tensor<2x2xf32>, %[[VAL_1:.*]]: tensor<2x2xf32>) -> tensor<2x2xi1> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.add %[[VAL_0]], %[[VAL_1]] : tensor<2x2xf32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.multiply %[[VAL_2]], %[[VAL_1]] : tensor<2x2xf32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.divide %[[VAL_3]], %[[VAL_0]] : tensor<2x2xf32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.maximum %[[VAL_4]], %[[VAL_0]] : tensor<2x2xf32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.minimum %[[VAL_5]], %[[VAL_1]] : tensor<2x2xf32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.compare  LT, %[[VAL_6]], %[[VAL_0]],  FLOAT : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
// CHECK:           return %[[VAL_7]] : tensor<2x2xi1>

// -----

module {
  func.func @test_combined_comprehensive(
    %arg0: tensor<4x4xf32>,
    %arg1: tensor<4x4xf32>,
    %arg2: tensor<4x4xf16>,
    %arg3: tensor<4x4xi8>,
    %arg4: tensor<4x4xi32>
  ) -> (tensor<4x4xf32>, tensor<4x4xi1>, tensor<4x4xi32>) {

    %0 = arith.addf %arg0, %arg1 : tensor<4x4xf32>
    %1 = arith.mulf %0, %arg1 : tensor<4x4xf32>
    %2 = arith.divf %1, %arg0 : tensor<4x4xf32>
    %3 = arith.maximumf %2, %arg0 : tensor<4x4xf32>
    %4 = arith.minimumf %3, %arg1 : tensor<4x4xf32>
    %5 = arith.extf %arg2 : tensor<4x4xf16> to tensor<4x4xf32>
    %6 = arith.addf %4, %5 : tensor<4x4xf32>
    %7 = arith.cmpf olt, %6, %arg0 : tensor<4x4xf32>
    %8 = arith.cmpf oge, %6, %arg1 : tensor<4x4xf32>
    %9 = arith.cmpf oeq, %3, %4 : tensor<4x4xf32>
    %10 = arith.fptosi %6 : tensor<4x4xf32> to tensor<4x4xi32>
    %11 = arith.fptoui %3 : tensor<4x4xf32> to tensor<4x4xi32>
    %12 = arith.extui %arg3 : tensor<4x4xi8> to tensor<4x4xi32>
    %13 = arith.maxui %10, %11 : tensor<4x4xi32>
    %14 = arith.minui %13, %12 : tensor<4x4xi32>
    %15 = arith.andi %14, %arg4 : tensor<4x4xi32>
    %16 = arith.bitcast %6 : tensor<4x4xf32> to tensor<4x4xi32>
    %17 = arith.andi %15, %16 : tensor<4x4xi32>
    %19 = arith.bitcast %17 : tensor<4x4xi32> to tensor<4x4xf32>
    %20 = arith.mulf %19, %6 : tensor<4x4xf32>
    %21 = arith.cmpf une, %20, %6 : tensor<4x4xf32>

    return %20, %21, %17 : tensor<4x4xf32>, tensor<4x4xi1>, tensor<4x4xi32>
  }
}
// CHECK-LABEL:   func.func @test_combined_comprehensive(
// CHECK-SAME:                                           %[[VAL_0:.*]]: tensor<4x4xf32>, %[[VAL_1:.*]]: tensor<4x4xf32>, %[[VAL_2:.*]]: tensor<4x4xf16>, %[[VAL_3:.*]]: tensor<4x4xi8>, %[[VAL_4:.*]]: tensor<4x4xi32>) -> (tensor<4x4xf32>, tensor<4x4xi1>, tensor<4x4xi32>) {
// CHECK:           %[[VAL_5:.*]] = stablehlo.add %[[VAL_0]], %[[VAL_1]] : tensor<4x4xf32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.multiply %[[VAL_5]], %[[VAL_1]] : tensor<4x4xf32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.divide %[[VAL_6]], %[[VAL_0]] : tensor<4x4xf32>
// CHECK:           %[[VAL_8:.*]] = stablehlo.maximum %[[VAL_7]], %[[VAL_0]] : tensor<4x4xf32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.minimum %[[VAL_8]], %[[VAL_1]] : tensor<4x4xf32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.convert %[[VAL_2]] : (tensor<4x4xf16>) -> tensor<4x4xf32>
// CHECK:           %[[VAL_11:.*]] = stablehlo.add %[[VAL_9]], %[[VAL_10]] : tensor<4x4xf32>
// CHECK:           %[[VAL_12:.*]] = stablehlo.compare  LT, %[[VAL_11]], %[[VAL_0]],  FLOAT : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK:           %[[VAL_13:.*]] = stablehlo.compare  GE, %[[VAL_11]], %[[VAL_1]],  FLOAT : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK:           %[[VAL_14:.*]] = stablehlo.compare  EQ, %[[VAL_8]], %[[VAL_9]],  FLOAT : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK:           %[[VAL_15:.*]] = stablehlo.convert %[[VAL_11]] : (tensor<4x4xf32>) -> tensor<4x4xi32>
// CHECK:           %[[VAL_16:.*]] = stablehlo.convert %[[VAL_8]] : (tensor<4x4xf32>) -> tensor<4x4xi32>
// CHECK:           %[[VAL_17:.*]] = stablehlo.convert %[[VAL_3]] : (tensor<4x4xi8>) -> tensor<4x4xi32>
// CHECK:           %[[VAL_18:.*]] = stablehlo.maximum %[[VAL_15]], %[[VAL_16]] : tensor<4x4xi32>
// CHECK:           %[[VAL_19:.*]] = stablehlo.minimum %[[VAL_18]], %[[VAL_17]] : tensor<4x4xi32>
// CHECK:           %[[VAL_20:.*]] = stablehlo.and %[[VAL_19]], %[[VAL_4]] : tensor<4x4xi32>
// CHECK:           %[[VAL_21:.*]] = stablehlo.convert %[[VAL_11]] : (tensor<4x4xf32>) -> tensor<4x4xi32>
// CHECK:           %[[VAL_22:.*]] = stablehlo.and %[[VAL_20]], %[[VAL_21]] : tensor<4x4xi32>
// CHECK:           %[[VAL_23:.*]] = stablehlo.convert %[[VAL_22]] : (tensor<4x4xi32>) -> tensor<4x4xf32>
// CHECK:           %[[VAL_24:.*]] = stablehlo.multiply %[[VAL_23]], %[[VAL_11]] : tensor<4x4xf32>
// CHECK:           %[[VAL_25:.*]] = stablehlo.compare  NE, %[[VAL_24]], %[[VAL_11]],  FLOAT : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK:           return %[[VAL_24]], %[[VAL_25]], %[[VAL_22]] : tensor<4x4xf32>, tensor<4x4xi1>, tensor<4x4xi32>

// -----

module {
  func.func @test_data_flow(
    %arg0: tensor<2x2xf16>,
    %arg1: tensor<2x2xi8>
  ) -> tensor<2x2xf32> {

    %0 = arith.extf %arg0 : tensor<2x2xf16> to tensor<2x2xf32>
    %1 = arith.extui %arg1 : tensor<2x2xi8> to tensor<2x2xi32>
    %2 = arith.bitcast %1 : tensor<2x2xi32> to tensor<2x2xf32>
    %3 = arith.addf %0, %2 : tensor<2x2xf32>
    %4 = arith.mulf %3, %0 : tensor<2x2xf32>
    %5 = arith.fptosi %4 : tensor<2x2xf32> to tensor<2x2xi32>
    %6 = arith.fptoui %4 : tensor<2x2xf32> to tensor<2x2xi32>
    %7 = arith.maxui %5, %6 : tensor<2x2xi32>
    %8 = arith.minui %5, %6 : tensor<2x2xi32>
    %9 = arith.andi %7, %8 : tensor<2x2xi32>
    %10 = arith.bitcast %9 : tensor<2x2xi32> to tensor<2x2xf32>
    %11 = arith.maximumf %10, %4 : tensor<2x2xf32>
    %12 = arith.minimumf %11, %3 : tensor<2x2xf32>
    %13 = arith.divf %12, %0 : tensor<2x2xf32>

    return %13 : tensor<2x2xf32>
  }
}
// CHECK-LABEL:   func.func @test_data_flow(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<2x2xf16>, %[[VAL_1:.*]]: tensor<2x2xi8>) -> tensor<2x2xf32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.convert %[[VAL_0]] : (tensor<2x2xf16>) -> tensor<2x2xf32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.convert %[[VAL_1]] : (tensor<2x2xi8>) -> tensor<2x2xi32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.convert %[[VAL_3]] : (tensor<2x2xi32>) -> tensor<2x2xf32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.add %[[VAL_2]], %[[VAL_4]] : tensor<2x2xf32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.multiply %[[VAL_5]], %[[VAL_2]] : tensor<2x2xf32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.convert %[[VAL_6]] : (tensor<2x2xf32>) -> tensor<2x2xi32>
// CHECK:           %[[VAL_8:.*]] = stablehlo.convert %[[VAL_6]] : (tensor<2x2xf32>) -> tensor<2x2xi32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.maximum %[[VAL_7]], %[[VAL_8]] : tensor<2x2xi32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.minimum %[[VAL_7]], %[[VAL_8]] : tensor<2x2xi32>
// CHECK:           %[[VAL_11:.*]] = stablehlo.and %[[VAL_9]], %[[VAL_10]] : tensor<2x2xi32>
// CHECK:           %[[VAL_12:.*]] = stablehlo.convert %[[VAL_11]] : (tensor<2x2xi32>) -> tensor<2x2xf32>
// CHECK:           %[[VAL_13:.*]] = stablehlo.maximum %[[VAL_12]], %[[VAL_6]] : tensor<2x2xf32>
// CHECK:           %[[VAL_14:.*]] = stablehlo.minimum %[[VAL_13]], %[[VAL_5]] : tensor<2x2xf32>
// CHECK:           %[[VAL_15:.*]] = stablehlo.divide %[[VAL_14]], %[[VAL_2]] : tensor<2x2xf32>
// CHECK:           return %[[VAL_15]] : tensor<2x2xf32>

// -----

// CHECK-LABEL:   func.func @arith_muli_integer(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: tensor<i32>) -> tensor<i32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.multiply %[[VAL_0]], %[[VAL_1]] : tensor<i32>
// CHECK:           return %[[VAL_2]] : tensor<i32>
func.func @arith_muli_integer(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = arith.muli %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL:   func.func @arith_negf_float(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.negate %[[VAL_0]] : tensor<f32>
// CHECK:           return %[[VAL_1]] : tensor<f32>
func.func @arith_negf_float(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = arith.negf %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL:   func.func @arith_ori_boolean(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<i1>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: tensor<i1>) -> tensor<i1> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.or %[[VAL_0]], %[[VAL_1]] : tensor<i1>
// CHECK:           return %[[VAL_2]] : tensor<i1>
func.func @arith_ori_boolean(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
  %0 = arith.ori %arg0, %arg1 : tensor<i1>
  return %0 : tensor<i1>
}

// -----

// CHECK-LABEL:   func.func @arith_ori_i64(
// CHECK-SAME:                             %[[VAL_0:.*]]: tensor<i64>,
// CHECK-SAME:                             %[[VAL_1:.*]]: tensor<i64>) -> tensor<i64> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.or %[[VAL_0]], %[[VAL_1]] : tensor<i64>
// CHECK:           return %[[VAL_2]] : tensor<i64>
func.func @arith_ori_i64(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
  %0 = arith.ori %arg0, %arg1 : tensor<i64>
  return %0 : tensor<i64>
}

// -----

// CHECK-LABEL:   func.func @arith_remf_float(
// CHECK-SAME:                                %[[VAL_0:.*]]: tensor<f32>,
// CHECK-SAME:                                %[[VAL_1:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.remainder %[[VAL_0]], %[[VAL_1]] : tensor<f32>
// CHECK:           return %[[VAL_2]] : tensor<f32>
func.func @arith_remf_float(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = arith.remf %arg0, %arg1 : tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL:   func.func @arith_remsi_signed_integer(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: tensor<i32>) -> tensor<i32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.remainder %[[VAL_0]], %[[VAL_1]] : tensor<i32>
// CHECK:           return %[[VAL_2]] : tensor<i32>
func.func @arith_remsi_signed_integer(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = arith.remsi %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL:   func.func @arith_remui_unsigned_integer(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:                                            %[[VAL_1:.*]]: tensor<i32>) -> tensor<i32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.remainder %[[VAL_0]], %[[VAL_1]] : tensor<i32>
// CHECK:           return %[[VAL_2]] : tensor<i32>
func.func @arith_remui_unsigned_integer(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = arith.remui %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL:   func.func @arith_shli_shift_left(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: tensor<i32>) -> tensor<i32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.shift_left %[[VAL_0]], %[[VAL_1]] : tensor<i32>
// CHECK:           return %[[VAL_2]] : tensor<i32>
func.func @arith_shli_shift_left(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = arith.shli %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL:   func.func @arith_shrsi_shift_right_signed(
// CHECK-SAME:                                              %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: tensor<i32>) -> tensor<i32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.shift_right_arithmetic %[[VAL_0]], %[[VAL_1]] : tensor<i32>
// CHECK:           return %[[VAL_2]] : tensor<i32>
func.func @arith_shrsi_shift_right_signed(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = arith.shrsi %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL:   func.func @arith_shrui_shift_right_unsigned(
// CHECK-SAME:                                                %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:                                                %[[VAL_1:.*]]: tensor<i32>) -> tensor<i32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.shift_right_logical %[[VAL_0]], %[[VAL_1]] : tensor<i32>
// CHECK:           return %[[VAL_2]] : tensor<i32>
func.func @arith_shrui_shift_right_unsigned(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = arith.shrui %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL:   func.func @arith_sitofp_signed_int_to_float(
// CHECK-SAME:                                                %[[VAL_0:.*]]: tensor<i32>) -> tensor<f32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.convert %[[VAL_0]] : (tensor<i32>) -> tensor<f32>
// CHECK:           return %[[VAL_1]] : tensor<f32>
func.func @arith_sitofp_signed_int_to_float(%arg0: tensor<i32>) -> tensor<f32> {
  %0 = arith.sitofp %arg0 : tensor<i32> to tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL:   func.func @arith_subf_float(
// CHECK-SAME:                                %[[VAL_0:.*]]: tensor<f32>,
// CHECK-SAME:                                %[[VAL_1:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.subtract %[[VAL_0]], %[[VAL_1]] : tensor<f32>
// CHECK:           return %[[VAL_2]] : tensor<f32>
func.func @arith_subf_float(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = arith.subf %arg0, %arg1 : tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL:   func.func @arith_subi_integer(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: tensor<i32>) -> tensor<i32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.subtract %[[VAL_0]], %[[VAL_1]] : tensor<i32>
// CHECK:           return %[[VAL_2]] : tensor<i32>
func.func @arith_subi_integer(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = arith.subi %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL:   func.func @arith_uitofp_unsigned_int_to_float(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: tensor<i32>) -> tensor<f32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.convert %[[VAL_0]] : (tensor<i32>) -> tensor<f32>
// CHECK:           return %[[VAL_1]] : tensor<f32>
func.func @arith_uitofp_unsigned_int_to_float(%arg0: tensor<i32>) -> tensor<f32> {
  %0 = arith.uitofp %arg0 : tensor<i32> to tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL:   func.func @arith_xori_boolean(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<i1>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: tensor<i1>) -> tensor<i1> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.xor %[[VAL_0]], %[[VAL_1]] : tensor<i1>
// CHECK:           return %[[VAL_2]] : tensor<i1>
func.func @arith_xori_boolean(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
  %0 = arith.xori %arg0, %arg1 : tensor<i1>
  return %0 : tensor<i1>
}

// -----

// CHECK-LABEL:   func.func @arith_xori_i64(
// CHECK-SAME:                             %[[VAL_0:.*]]: tensor<i64>,
// CHECK-SAME:                             %[[VAL_1:.*]]: tensor<i64>) -> tensor<i64> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.xor %[[VAL_0]], %[[VAL_1]] : tensor<i64>
// CHECK:           return %[[VAL_2]] : tensor<i64>
func.func @arith_xori_i64(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
  %0 = arith.xori %arg0, %arg1 : tensor<i64>
  return %0 : tensor<i64>
}

// -----

// CHECK-LABEL:   func.func @arith_divsi_basic(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: tensor<i32>) -> tensor<i32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.divide %[[VAL_0]], %[[VAL_1]] : tensor<i32>
// CHECK:           return %[[VAL_2]] : tensor<i32>
func.func @arith_divsi_basic(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = arith.divsi %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL:   func.func @arith_divui_basic(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: tensor<i32>) -> tensor<i32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.bitcast_convert %[[VAL_0]] : (tensor<i32>) -> tensor<ui32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.bitcast_convert %[[VAL_1]] : (tensor<i32>) -> tensor<ui32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.divide %[[VAL_2]], %[[VAL_3]] : tensor<ui32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.bitcast_convert %[[VAL_4]] : (tensor<ui32>) -> tensor<i32>
// CHECK:           return %[[VAL_5]] : tensor<i32>
func.func @arith_divui_basic(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = arith.divui %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL:   func.func @arith_div_chained_operations(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<i32>, %[[VAL_1:.*]]: tensor<i32>, %[[VAL_2:.*]]: tensor<i32>) -> tensor<i32> {
// CHECK:           %[[VAL_3:.*]] = stablehlo.add %[[VAL_0]], %[[VAL_1]] : tensor<i32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.divide %[[VAL_3]], %[[VAL_2]] : tensor<i32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.multiply %[[VAL_4]], %[[VAL_0]] : tensor<i32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.bitcast_convert %[[VAL_5]] : (tensor<i32>) -> tensor<ui32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.bitcast_convert %[[VAL_1]] : (tensor<i32>) -> tensor<ui32>
// CHECK:           %[[VAL_8:.*]] = stablehlo.divide %[[VAL_6]], %[[VAL_7]] : tensor<ui32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.bitcast_convert %[[VAL_8]] : (tensor<ui32>) -> tensor<i32>
// CHECK:           return %[[VAL_9]] : tensor<i32>
func.func @arith_div_chained_operations(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
  %0 = arith.addi %arg0, %arg1 : tensor<i32>
  %1 = arith.divsi %0, %arg2 : tensor<i32>
  %2 = arith.muli %1, %arg0 : tensor<i32>
  %3 = arith.divui %2, %arg1 : tensor<i32>
  return %3 : tensor<i32>
}

// -----

// CHECK-LABEL:   func.func @arith_mixed_div_and_rem(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tensor<i64>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: tensor<i64>) -> (tensor<i64>, tensor<i64>) {
// CHECK:           %[[VAL_2:.*]] = stablehlo.divide %[[VAL_0]], %[[VAL_1]] : tensor<i64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.remainder %[[VAL_0]], %[[VAL_1]] : tensor<i64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.subtract %[[VAL_2]], %[[VAL_3]] : tensor<i64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.bitcast_convert %[[VAL_4]] : (tensor<i64>) -> tensor<ui64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.bitcast_convert %[[VAL_1]] : (tensor<i64>) -> tensor<ui64>
// CHECK:           %[[VAL_7:.*]] = stablehlo.divide %[[VAL_5]], %[[VAL_6]] : tensor<ui64>
// CHECK:           %[[VAL_8:.*]] = stablehlo.bitcast_convert %[[VAL_7]] : (tensor<ui64>) -> tensor<i64>
// CHECK:           return %[[VAL_8]], %[[VAL_3]] : tensor<i64>, tensor<i64>
func.func @arith_mixed_div_and_rem(%arg0: tensor<i64>, %arg1: tensor<i64>) -> (tensor<i64>, tensor<i64>) {
  %0 = arith.divsi %arg0, %arg1 : tensor<i64>
  %1 = arith.remsi %arg0, %arg1 : tensor<i64>
  %2 = arith.subi %0, %1 : tensor<i64>
  %3 = arith.divui %2, %arg1 : tensor<i64>
  return %3, %1 : tensor<i64>, tensor<i64>
}

// -----

// CHECK-LABEL:   func.func @arith_ceildivsi_i32(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: tensor<i32>) -> tensor<i32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.convert %[[VAL_0]] : (tensor<i32>) -> tensor<f64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.convert %[[VAL_1]] : (tensor<i32>) -> tensor<f64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.divide %[[VAL_2]], %[[VAL_3]] : tensor<f64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.ceil %[[VAL_4]] : tensor<f64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.convert %[[VAL_5]] : (tensor<f64>) -> tensor<i32>
// CHECK:           return %[[VAL_6]] : tensor<i32>
func.func @arith_ceildivsi_i32(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = arith.ceildivsi %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL:   func.func @arith_ceildivsi_i64(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<i64>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: tensor<i64>) -> tensor<i64> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.convert %[[VAL_0]] : (tensor<i64>) -> tensor<f64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.convert %[[VAL_1]] : (tensor<i64>) -> tensor<f64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.divide %[[VAL_2]], %[[VAL_3]] : tensor<f64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.ceil %[[VAL_4]] : tensor<f64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.convert %[[VAL_5]] : (tensor<f64>) -> tensor<i64>
// CHECK:           return %[[VAL_6]] : tensor<i64>
func.func @arith_ceildivsi_i64(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
  %0 = arith.ceildivsi %arg0, %arg1 : tensor<i64>
  return %0 : tensor<i64>
}

// -----

// CHECK-LABEL:   func.func @arith_floordivsi_i32(
// CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: tensor<i32>) -> tensor<i32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.convert %[[VAL_0]] : (tensor<i32>) -> tensor<f64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.convert %[[VAL_1]] : (tensor<i32>) -> tensor<f64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.divide %[[VAL_2]], %[[VAL_3]] : tensor<f64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.floor %[[VAL_4]] : tensor<f64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.convert %[[VAL_5]] : (tensor<f64>) -> tensor<i32>
// CHECK:           return %[[VAL_6]] : tensor<i32>
func.func @arith_floordivsi_i32(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = arith.floordivsi %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL:   func.func @arith_floordivsi_i64(
// CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<i64>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: tensor<i64>) -> tensor<i64> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.convert %[[VAL_0]] : (tensor<i64>) -> tensor<f64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.convert %[[VAL_1]] : (tensor<i64>) -> tensor<f64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.divide %[[VAL_2]], %[[VAL_3]] : tensor<f64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.floor %[[VAL_4]] : tensor<f64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.convert %[[VAL_5]] : (tensor<f64>) -> tensor<i64>
// CHECK:           return %[[VAL_6]] : tensor<i64>
func.func @arith_floordivsi_i64(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
  %0 = arith.floordivsi %arg0, %arg1 : tensor<i64>
  return %0 : tensor<i64>
}

// -----

// CHECK-LABEL:   func.func @test_trunci_i64_to_i32(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<2x2xi64>) -> tensor<2x2xi32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.convert %[[VAL_0]] : (tensor<2x2xi64>) -> tensor<2x2xi32>
// CHECK:           return %[[VAL_1]] : tensor<2x2xi32>
module {
  func.func @test_trunci_i64_to_i32(%arg0: tensor<2x2xi64>) -> tensor<2x2xi32> {
    %0 = arith.trunci %arg0 : tensor<2x2xi64> to tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}

// -----

// CHECK-LABEL:   func.func @test_truncf_f64_to_f32(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<2x2xf64>) -> tensor<2x2xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.convert %[[VAL_0]] : (tensor<2x2xf64>) -> tensor<2x2xf32>
// CHECK:           return %[[VAL_1]] : tensor<2x2xf32>
module {
  func.func @test_truncf_f64_to_f32(%arg0: tensor<2x2xf64>) -> tensor<2x2xf32> {
    %0 = arith.truncf %arg0 : tensor<2x2xf64> to tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}


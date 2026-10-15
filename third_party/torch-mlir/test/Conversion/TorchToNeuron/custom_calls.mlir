// RUN: torch-mlir-opt -pass-pipeline='builtin.module(torch-raw-to-neuron-backend-pipeline)' -split-input-file %s | FileCheck %s

module {
  func.func @test_valid_topk(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[3],f32> {
    %int24 = torch.constant.int 24
    %0 = torch.prim.ListConstruct %int24 : (!torch.int) -> !torch.list<int>
    %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[2,3,4],f32>, !torch.list<int> -> !torch.vtensor<[24],f32>
    // CHECK: %[[VAL_0:.*]] = torch_c.to_builtin_tensor %1 : !torch.vtensor<[24],f32> -> tensor<24xf32>
    %int3 = torch.constant.int 3
    %int-1 = torch.constant.int -1
    %true = torch.constant.bool true
    %true_0 = torch.constant.bool true
    // CHECK: %[[VAL_1:.*]] = stablehlo.custom_call @AwsNeuronTopK(%2) {backend_config = "3"} : (tensor<24xf32>) -> (tensor<3xf32>, tensor<3xi64>)
    %values, %indices = torch.aten.topk %1, %int3, %int-1, %true, %true_0 : !torch.vtensor<[24],f32>, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[3],f32>, !torch.vtensor<[3],si64>
    return %values : !torch.vtensor<[3],f32>
  }

  func.func @test_valid_topk_3d(%arg0: !torch.vtensor<[4,5,6],f32>) -> !torch.vtensor<[4,5,3],f32> {
    // CHECK: %[[VAL_0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[4,5,6],f32> -> tensor<4x5x6xf32>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %true = torch.constant.bool true
    %true_0 = torch.constant.bool true
    // CHECK: %[[VAL_1:.*]] = stablehlo.custom_call @AwsNeuronTopK(%0) {backend_config = "3"} : (tensor<4x5x6xf32>) -> (tensor<4x5x3xf32>, tensor<4x5x3xi64>)
    %values, %indices = torch.aten.topk %arg0, %int3, %int2, %true, %true_0 : !torch.vtensor<[4,5,6],f32>, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[4,5,3],f32>, !torch.vtensor<[4,5,3],si64>
    return %values : !torch.vtensor<[4,5,3],f32>
  }

  // CHECK-LABEL:   func.func @test_neuron_silu(
  // CHECK-SAME:                    %[[VAL_0:.*]]: !torch.vtensor<[4,5],f32>) -> !torch.vtensor<[4,5],f32> {
  // CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,5],f32> -> tensor<4x5xf32>
  // CHECK:           %[[VAL_2:.*]] = stablehlo.custom_call @AwsNeuronSilu(%[[VAL_1]]) : (tensor<4x5xf32>) -> tensor<4x5xf32>
  // CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<4x5xf32> -> !torch.vtensor<[4,5],f32>
  // CHECK:           return %[[VAL_3]] : !torch.vtensor<[4,5],f32>
  // CHECK:         }
  func.func @test_neuron_silu(%arg0: !torch.vtensor<[4,5],f32>) -> !torch.vtensor<[4,5],f32> {
    %0 = torch.aten.silu %arg0 : !torch.vtensor<[4,5],f32> -> !torch.vtensor<[4,5],f32>
    return %0 : !torch.vtensor<[4,5],f32>
  }

  // CHECK-LABEL:   func.func @test_neuron_erf(
  // CHECK-SAME:                    %[[VAL_0:.*]]: !torch.vtensor<[4,5],f32>) -> !torch.vtensor<[4,5],f32> {
  // CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,5],f32> -> tensor<4x5xf32>
  // CHECK:           %[[VAL_2:.*]] = stablehlo.custom_call @AwsNeuronErf(%[[VAL_1]]) : (tensor<4x5xf32>) -> tensor<4x5xf32>
  // CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<4x5xf32> -> !torch.vtensor<[4,5],f32>
  // CHECK:           return %[[VAL_3]] : !torch.vtensor<[4,5],f32>
  // CHECK:         }
  func.func @test_neuron_erf(%arg0: !torch.vtensor<[4,5],f32>) -> !torch.vtensor<[4,5],f32> {
    %0 = torch.aten.erf %arg0 : !torch.vtensor<[4,5],f32> -> !torch.vtensor<[4,5],f32>
    return %0 : !torch.vtensor<[4,5],f32>
  }

  // bf16 is the dtype that matters here: it is what the hand-rolled GELU in the
  // vision encoders feeds to erf.
  // CHECK-LABEL:   func.func @test_neuron_erf_bf16(
  // CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %{{.*}} : !torch.vtensor<[4,5],bf16> -> tensor<4x5xbf16>
  // CHECK:           %[[VAL_2:.*]] = stablehlo.custom_call @AwsNeuronErf(%[[VAL_1]]) : (tensor<4x5xbf16>) -> tensor<4x5xbf16>
  // CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<4x5xbf16> -> !torch.vtensor<[4,5],bf16>
  // CHECK:           return %[[VAL_3]] : !torch.vtensor<[4,5],bf16>
  func.func @test_neuron_erf_bf16(%arg0: !torch.vtensor<[4,5],bf16>) -> !torch.vtensor<[4,5],bf16> {
    %0 = torch.aten.erf %arg0 : !torch.vtensor<[4,5],bf16> -> !torch.vtensor<[4,5],bf16>
    return %0 : !torch.vtensor<[4,5],bf16>
  }
}

// -----

// CHECK-LABEL:   func.func @test_aten.topk$largest_false(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !torch.vtensor<[10,20,30],f32>) -> (!torch.vtensor<[10,20,6],f32>, !torch.vtensor<[10,20,6],si64>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[10,20,30],f32> -> tensor<10x20x30xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 6
// CHECK:           %[[VAL_3:.*]] = torch.constant.int -1
// CHECK:           %[[VAL_4:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_5:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_6:.*]] = stablehlo.negate %[[VAL_1]] : tensor<10x20x30xf32>
// CHECK:           %[[VAL_7:.*]]:2 = stablehlo.custom_call @AwsNeuronTopK(%[[VAL_6]]) {backend_config = "6"} : (tensor<10x20x30xf32>) -> (tensor<10x20x6xf32>, tensor<10x20x6xi64>)
// The index output is clamped to [0, n) with max(idx, 0) so a degenerate -1
// sentinel (all-NaN input) can never reach a downstream gather/scatter.
// CHECK-DAG:       %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<10x20x6xi64>
// CHECK-DAG:       %[[CLAMP:.*]] = stablehlo.maximum %[[VAL_7]]#1, %[[ZERO]] : tensor<10x20x6xi64>
// CHECK-DAG:       %[[NEGVAL:.*]] = stablehlo.negate %[[VAL_7]]#0 : tensor<10x20x6xf32>
// CHECK:           %[[IDXOUT:.*]] = torch_c.from_builtin_tensor %[[CLAMP]] : tensor<10x20x6xi64> -> !torch.vtensor<[10,20,6],si64>
// CHECK:           return
func.func @test_aten.topk$largest_false(%arg0: !torch.vtensor<[10,20,30],f32>) -> (!torch.vtensor<[10,20,6],f32>, !torch.vtensor<[10,20,6],si64>) {
    %int6 = torch.constant.int 6
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %values, %indices = torch.aten.topk %arg0, %int6, %int-1, %false, %true : !torch.vtensor<[10,20,30],f32>, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[10,20,6],f32>, !torch.vtensor<[10,20,6],si64>
    return %values, %indices : !torch.vtensor<[10,20,6],f32>, !torch.vtensor<[10,20,6],si64>
}

// -----

// CHECK-LABEL:   func.func @test_aten.topk$other_dim(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[10,20,30],f32>) -> (!torch.vtensor<[10,6,30],f32>, !torch.vtensor<[10,6,30],si64>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[10,20,30],f32> -> tensor<10x20x30xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 6
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_5:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_6:.*]] = stablehlo.transpose %[[VAL_1]], dims = [0, 2, 1] : (tensor<10x20x30xf32>) -> tensor<10x30x20xf32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.negate %[[VAL_6]] : tensor<10x30x20xf32>
// CHECK:           %[[VAL_8:.*]]:2 = stablehlo.custom_call @AwsNeuronTopK(%[[VAL_7]]) {backend_config = "6"} : (tensor<10x30x20xf32>) -> (tensor<10x30x6xf32>, tensor<10x30x6xi64>)
// The index output is clamped to [0, n) with max(idx, 0) before the inverse
// transpose, so a degenerate -1 sentinel can never reach a downstream scatter.
// CHECK-DAG:       %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<10x30x6xi64>
// CHECK-DAG:       %[[CLAMP:.*]] = stablehlo.maximum %[[VAL_8]]#1, %[[ZERO]] : tensor<10x30x6xi64>
// CHECK:           %[[IDXT:.*]] = stablehlo.transpose %[[CLAMP]], dims = [0, 2, 1] : (tensor<10x30x6xi64>) -> tensor<10x6x30xi64>
// CHECK:           %[[IDXOUT:.*]] = torch_c.from_builtin_tensor %[[IDXT]] : tensor<10x6x30xi64> -> !torch.vtensor<[10,6,30],si64>
// CHECK:           return
func.func @test_aten.topk$other_dim(%arg0: !torch.vtensor<[10,20,30],f32>) -> (!torch.vtensor<[10,6,30],f32>, !torch.vtensor<[10,6,30],si64>) {
    %int6 = torch.constant.int 6
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %values, %indices = torch.aten.topk %arg0, %int6, %int1, %false, %true : !torch.vtensor<[10,20,30],f32>, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[10,6,30],f32>, !torch.vtensor<[10,6,30],si64>
    return %values, %indices : !torch.vtensor<[10,6,30],f32>, !torch.vtensor<[10,6,30],si64>
}

// -----

// Regression: MoE-router top-k (largest, dim=-1). When the router logits are
// all-NaN, @AwsNeuronTopK returns index -1 for every slot; that -1 must be
// clamped to a valid position so the downstream routing-weight-grad scatter
// never indexes column -1 (device NRT 1006 out-of-bounds).
// CHECK-LABEL:   func.func @test_aten.topk$index_clamp(
// CHECK:           %[[IN:.*]] = torch_c.to_builtin_tensor %{{.*}} : !torch.vtensor<[96,512],f32> -> tensor<96x512xf32>
// CHECK:           %[[TOPK:.*]]:2 = stablehlo.custom_call @AwsNeuronTopK(%[[IN]]) {backend_config = "10"} : (tensor<96x512xf32>) -> (tensor<96x10xf32>, tensor<96x10xi64>)
// CHECK-DAG:       %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<96x10xi64>
// CHECK-DAG:       %[[CLAMP:.*]] = stablehlo.maximum %[[TOPK]]#1, %[[ZERO]] : tensor<96x10xi64>
// CHECK:           %[[IDXOUT:.*]] = torch_c.from_builtin_tensor %[[CLAMP]] : tensor<96x10xi64> -> !torch.vtensor<[96,10],si64>
// CHECK:           return
func.func @test_aten.topk$index_clamp(%arg0: !torch.vtensor<[96,512],f32>) -> (!torch.vtensor<[96,10],f32>, !torch.vtensor<[96,10],si64>) {
    %int10 = torch.constant.int 10
    %int-1 = torch.constant.int -1
    %true = torch.constant.bool true
    %true_0 = torch.constant.bool true
    %values, %indices = torch.aten.topk %arg0, %int10, %int-1, %true, %true_0 : !torch.vtensor<[96,512],f32>, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[96,10],f32>, !torch.vtensor<[96,10],si64>
    return %values, %indices : !torch.vtensor<[96,10],f32>, !torch.vtensor<[96,10],si64>
}

// -----

// CHECK-LABEL:   func.func @test_aten.sort$basic(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[10,20],f32>) -> (!torch.vtensor<[10,20],f32>, !torch.vtensor<[10,20],si64>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[10,20],f32> -> tensor<10x20xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int -1
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_4:.*]] = stablehlo.negate %[[VAL_1]] : tensor<10x20xf32>
// CHECK:           %[[VAL_5:.*]]:2 = stablehlo.custom_call @AwsNeuronTopK(%[[VAL_4]]) {backend_config = "20"} : (tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xi64>)
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]]#1 : tensor<10x20xi64> -> !torch.vtensor<[10,20],si64>
// CHECK:           %[[VAL_7:.*]] = stablehlo.negate %[[VAL_5]]#0 : tensor<10x20xf32>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<10x20xf32> -> !torch.vtensor<[10,20],f32>
// CHECK:           return %[[VAL_8]], %[[VAL_6]] : !torch.vtensor<[10,20],f32>, !torch.vtensor<[10,20],si64>
// CHECK:         }
func.func @test_aten.sort$basic(%arg0: !torch.vtensor<[10,20],f32>) -> (!torch.vtensor<[10,20],f32>, !torch.vtensor<[10,20],si64>) {
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %values, %indices = torch.aten.sort %arg0, %int-1, %false : !torch.vtensor<[10,20],f32>, !torch.int, !torch.bool -> !torch.vtensor<[10,20],f32>, !torch.vtensor<[10,20],si64>
    return %values, %indices : !torch.vtensor<[10,20],f32>, !torch.vtensor<[10,20],si64>
}

// -----

// CHECK-LABEL:   func.func @test_aten.sort$descending_true(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !torch.vtensor<[10,20],f32>) -> (!torch.vtensor<[10,20],f32>, !torch.vtensor<[10,20],si64>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[10,20],f32> -> tensor<10x20xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int -1
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_4:.*]]:2 = stablehlo.custom_call @AwsNeuronTopK(%[[VAL_1]]) {backend_config = "20"} : (tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xi64>)
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]]#1 : tensor<10x20xi64> -> !torch.vtensor<[10,20],si64>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_4]]#0 : tensor<10x20xf32> -> !torch.vtensor<[10,20],f32>
// CHECK:           return %[[VAL_6]], %[[VAL_5]] : !torch.vtensor<[10,20],f32>, !torch.vtensor<[10,20],si64>
// CHECK:         }
func.func @test_aten.sort$descending_true(%arg0: !torch.vtensor<[10,20],f32>) -> (!torch.vtensor<[10,20],f32>, !torch.vtensor<[10,20],si64>) {
    %int-1 = torch.constant.int -1
    %true = torch.constant.bool true
    %values, %indices = torch.aten.sort %arg0, %int-1, %true : !torch.vtensor<[10,20],f32>, !torch.int, !torch.bool -> !torch.vtensor<[10,20],f32>, !torch.vtensor<[10,20],si64>
    return %values, %indices : !torch.vtensor<[10,20],f32>, !torch.vtensor<[10,20],si64>
}

// -----

// CHECK-LABEL:   func.func @test_aten.sort$other_dim(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[10,20,30],f32>) -> (!torch.vtensor<[10,20,30],f32>, !torch.vtensor<[10,20,30],si64>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[10,20,30],f32> -> tensor<10x20x30xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_4:.*]] = stablehlo.transpose %[[VAL_1]], dims = [0, 2, 1] : (tensor<10x20x30xf32>) -> tensor<10x30x20xf32>
// CHECK:           %[[VAL_5:.*]]:2 = stablehlo.custom_call @AwsNeuronTopK(%[[VAL_4]]) {backend_config = "20"} : (tensor<10x30x20xf32>) -> (tensor<10x30x20xf32>, tensor<10x30x20xi64>)
// CHECK:           %[[VAL_6:.*]] = stablehlo.transpose %[[VAL_5]]#0, dims = [0, 2, 1] : (tensor<10x30x20xf32>) -> tensor<10x20x30xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<10x20x30xf32> -> !torch.vtensor<[10,20,30],f32>
// CHECK:           %[[VAL_8:.*]] = stablehlo.transpose %[[VAL_5]]#1, dims = [0, 2, 1] : (tensor<10x30x20xi64>) -> tensor<10x20x30xi64>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<10x20x30xi64> -> !torch.vtensor<[10,20,30],si64>
// CHECK:           return %[[VAL_7]], %[[VAL_9]] : !torch.vtensor<[10,20,30],f32>, !torch.vtensor<[10,20,30],si64>
// CHECK:         }
func.func @test_aten.sort$other_dim(%arg0: !torch.vtensor<[10,20,30],f32>) -> (!torch.vtensor<[10,20,30],f32>, !torch.vtensor<[10,20,30],si64>) {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %values, %indices = torch.aten.sort %arg0, %int1, %true : !torch.vtensor<[10,20,30],f32>, !torch.int, !torch.bool -> !torch.vtensor<[10,20,30],f32>, !torch.vtensor<[10,20,30],si64>
    return %values, %indices : !torch.vtensor<[10,20,30],f32>, !torch.vtensor<[10,20,30],si64>
}

// -----

// CHECK-LABEL:   func.func @test_aten.sort$scalar_asc(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !torch.vtensor<[],f32>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[],si64>) {
// CHECK:           %[[VAL_1:.*]] = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : !torch.vtensor<[],f32>, !torch.vtensor<[],si64>
// CHECK:         }
func.func @test_aten.sort$scalar_asc(%arg0: !torch.vtensor<[],f32>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[],si64>) {
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %values, %indices = torch.aten.sort %arg0, %int-1, %false : !torch.vtensor<[],f32>, !torch.int, !torch.bool -> !torch.vtensor<[],f32>, !torch.vtensor<[],si64>
    return %values, %indices : !torch.vtensor<[],f32>, !torch.vtensor<[],si64>
}

// -----

// CHECK-LABEL:   func.func @test_aten.sort$scalar_desc(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[],f32>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[],si64>) {
// CHECK:           %[[VAL_1:.*]] = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : !torch.vtensor<[],f32>, !torch.vtensor<[],si64>
// CHECK:         }
func.func @test_aten.sort$scalar_desc(%arg0: !torch.vtensor<[],f32>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[],si64>) {
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %values, %indices = torch.aten.sort %arg0, %int0, %true : !torch.vtensor<[],f32>, !torch.int, !torch.bool -> !torch.vtensor<[],f32>, !torch.vtensor<[],si64>
    return %values, %indices : !torch.vtensor<[],f32>, !torch.vtensor<[],si64>
}

// -----

// CHECK-LABEL:   func.func @test_quantize_mx_e5m2_last_dim(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !torch.vtensor<[890,128],bf16>) -> (!torch.vtensor<[890,128],f8E5M2>, !torch.vtensor<[890,4],f8E8M0FNU>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[890,128],bf16> -> tensor<890x128xbf16>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int -1
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 32
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 23
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_6:.*]]:2 = stablehlo.custom_call @QuantizeMX(%[[VAL_1]]) {backend_config = "{\22dtype\22: \22float8_e5m2\22, \22dim\22: -1, \22block_size\22: 32, \22scale_method\22: \22EMAX\22}"} : (tensor<890x128xbf16>) -> (tensor<890x128xf8E5M2>, tensor<890x4xf8E8M0FNU>)
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]]#1 : tensor<890x4xf8E8M0FNU> -> !torch.vtensor<[890,4],f8E8M0FNU>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_6]]#0 : tensor<890x128xf8E5M2> -> !torch.vtensor<[890,128],f8E5M2>
// CHECK:           return %[[VAL_8]], %[[VAL_7]] : !torch.vtensor<[890,128],f8E5M2>, !torch.vtensor<[890,4],f8E8M0FNU>
// CHECK:         }
func.func @test_quantize_mx_e5m2_last_dim(%arg0: !torch.vtensor<[890,128],bf16>) -> (!torch.vtensor<[890,128],f8E5M2>, !torch.vtensor<[890,4],f8E8M0FNU>) {
    %int-1 = torch.constant.int -1
    %int32 = torch.constant.int 32
    %int23 = torch.constant.int 23
    %int0 = torch.constant.int 0
    %output, %scale = torch.aten.quantize_mx %arg0, %int-1, %int32, %int23, %int0 : !torch.vtensor<[890,128],bf16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[890,128],f8E5M2>, !torch.vtensor<[890,4],f8E8M0FNU>
    return %output, %scale : !torch.vtensor<[890,128],f8E5M2>, !torch.vtensor<[890,4],f8E8M0FNU>
}

// -----

// Test ui8 scale-output convention (legacy PyTorch): the custom call still
// returns f8E8M0FNU and a bitcast_convert injects a ui8 view as a compat shim.
// CHECK-LABEL:   func.func @test_quantize_mx_e5m2_scale_ui8(
// CHECK-SAME:                                                %[[VAL_0:.*]]: !torch.vtensor<[890,128],bf16>) -> (!torch.vtensor<[890,128],f8E5M2>, !torch.vtensor<[890,4],ui8>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[890,128],bf16> -> tensor<890x128xbf16>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int -1
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 32
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 23
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_6:.*]]:2 = stablehlo.custom_call @QuantizeMX(%[[VAL_1]]) {backend_config = "{\22dtype\22: \22float8_e5m2\22, \22dim\22: -1, \22block_size\22: 32, \22scale_method\22: \22EMAX\22}"} : (tensor<890x128xbf16>) -> (tensor<890x128xf8E5M2>, tensor<890x4xf8E8M0FNU>)
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]]#0 : tensor<890x128xf8E5M2> -> !torch.vtensor<[890,128],f8E5M2>
// CHECK:           %[[VAL_8:.*]] = stablehlo.bitcast_convert %[[VAL_6]]#1 : (tensor<890x4xf8E8M0FNU>) -> tensor<890x4xui8>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<890x4xui8> -> !torch.vtensor<[890,4],ui8>
// CHECK:           return %[[VAL_7]], %[[VAL_9]] : !torch.vtensor<[890,128],f8E5M2>, !torch.vtensor<[890,4],ui8>
// CHECK:         }
func.func @test_quantize_mx_e5m2_scale_ui8(%arg0: !torch.vtensor<[890,128],bf16>) -> (!torch.vtensor<[890,128],f8E5M2>, !torch.vtensor<[890,4],ui8>) {
    %int-1 = torch.constant.int -1
    %int32 = torch.constant.int 32
    %int23 = torch.constant.int 23
    %int0 = torch.constant.int 0
    %output, %scale = torch.aten.quantize_mx %arg0, %int-1, %int32, %int23, %int0 : !torch.vtensor<[890,128],bf16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[890,128],f8E5M2>, !torch.vtensor<[890,4],ui8>
    return %output, %scale : !torch.vtensor<[890,128],f8E5M2>, !torch.vtensor<[890,4],ui8>
}

// -----

// CHECK-LABEL:   func.func @test_quantize_mx_e4m3fn(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[64,256],f16>) -> (!torch.vtensor<[64,256],f8E4M3FN>, !torch.vtensor<[64,8],f8E8M0FNU>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[64,256],f16> -> tensor<64x256xf16>
// CHECK:           %[[VAL_2:.*]]:2 = stablehlo.custom_call @QuantizeMX(%[[VAL_1]]) {backend_config = "{\22dtype\22: \22float8_e4m3fn\22, \22dim\22: -1, \22block_size\22: 32, \22scale_method\22: \22EMAX\22}"} : (tensor<64x256xf16>) -> (tensor<64x256xf8E4M3FN>, tensor<64x8xf8E8M0FNU>)
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]]#1 : tensor<64x8xf8E8M0FNU> -> !torch.vtensor<[64,8],f8E8M0FNU>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_2]]#0 : tensor<64x256xf8E4M3FN> -> !torch.vtensor<[64,256],f8E4M3FN>
// CHECK:           return %[[VAL_4]], %[[VAL_3]] : !torch.vtensor<[64,256],f8E4M3FN>, !torch.vtensor<[64,8],f8E8M0FNU>
// CHECK:         }
func.func @test_quantize_mx_e4m3fn(%arg0: !torch.vtensor<[64,256],f16>) -> (!torch.vtensor<[64,256],f8E4M3FN>, !torch.vtensor<[64,8],f8E8M0FNU>) {
    %int-1 = torch.constant.int -1
    %int32 = torch.constant.int 32
    %int24 = torch.constant.int 24
    %int0 = torch.constant.int 0
    %output, %scale = torch.aten.quantize_mx %arg0, %int-1, %int32, %int24, %int0 : !torch.vtensor<[64,256],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[64,256],f8E4M3FN>, !torch.vtensor<[64,8],f8E8M0FNU>
    return %output, %scale : !torch.vtensor<[64,256],f8E4M3FN>, !torch.vtensor<[64,8],f8E8M0FNU>
}

// -----

// CHECK-LABEL:   func.func @test_quantize_mx_second_to_last_dim(
// CHECK-SAME:                                                    %[[VAL_0:.*]]: !torch.vtensor<[128,64],bf16>) -> (!torch.vtensor<[128,64],f8E5M2>, !torch.vtensor<[4,64],f8E8M0FNU>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[128,64],bf16> -> tensor<128x64xbf16>
// CHECK:           %[[VAL_2:.*]]:2 = stablehlo.custom_call @QuantizeMX(%[[VAL_1]]) {backend_config = "{\22dtype\22: \22float8_e5m2\22, \22dim\22: -2, \22block_size\22: 32, \22scale_method\22: \22EMAX\22}"} : (tensor<128x64xbf16>) -> (tensor<128x64xf8E5M2>, tensor<4x64xf8E8M0FNU>)
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]]#1 : tensor<4x64xf8E8M0FNU> -> !torch.vtensor<[4,64],f8E8M0FNU>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_2]]#0 : tensor<128x64xf8E5M2> -> !torch.vtensor<[128,64],f8E5M2>
// CHECK:           return %[[VAL_4]], %[[VAL_3]] : !torch.vtensor<[128,64],f8E5M2>, !torch.vtensor<[4,64],f8E8M0FNU>
// CHECK:         }
func.func @test_quantize_mx_second_to_last_dim(%arg0: !torch.vtensor<[128,64],bf16>) -> (!torch.vtensor<[128,64],f8E5M2>, !torch.vtensor<[4,64],f8E8M0FNU>) {
    %int-2 = torch.constant.int -2
    %int32 = torch.constant.int 32
    %int23 = torch.constant.int 23
    %int0 = torch.constant.int 0
    %output, %scale = torch.aten.quantize_mx %arg0, %int-2, %int32, %int23, %int0 : !torch.vtensor<[128,64],bf16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[128,64],f8E5M2>, !torch.vtensor<[4,64],f8E8M0FNU>
    return %output, %scale : !torch.vtensor<[128,64],f8E5M2>, !torch.vtensor<[4,64],f8E8M0FNU>
}

// -----

// CHECK-LABEL:   func.func @test_quantize_mx_3d(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[4,512,1024],bf16>) -> (!torch.vtensor<[4,512,1024],f8E5M2>, !torch.vtensor<[4,512,32],f8E8M0FNU>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,512,1024],bf16> -> tensor<4x512x1024xbf16>
// CHECK:           %[[VAL_2:.*]]:2 = stablehlo.custom_call @QuantizeMX(%[[VAL_1]]) {backend_config = "{\22dtype\22: \22float8_e5m2\22, \22dim\22: -1, \22block_size\22: 32, \22scale_method\22: \22EMAX\22}"} : (tensor<4x512x1024xbf16>) -> (tensor<4x512x1024xf8E5M2>, tensor<4x512x32xf8E8M0FNU>)
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]]#1 : tensor<4x512x32xf8E8M0FNU> -> !torch.vtensor<[4,512,32],f8E8M0FNU>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_2]]#0 : tensor<4x512x1024xf8E5M2> -> !torch.vtensor<[4,512,1024],f8E5M2>
// CHECK:           return %[[VAL_4]], %[[VAL_3]] : !torch.vtensor<[4,512,1024],f8E5M2>, !torch.vtensor<[4,512,32],f8E8M0FNU>
// CHECK:         }
func.func @test_quantize_mx_3d(%arg0: !torch.vtensor<[4,512,1024],bf16>) -> (!torch.vtensor<[4,512,1024],f8E5M2>, !torch.vtensor<[4,512,32],f8E8M0FNU>) {
    %int-1 = torch.constant.int -1
    %int32 = torch.constant.int 32
    %int23 = torch.constant.int 23
    %int0 = torch.constant.int 0
    %output, %scale = torch.aten.quantize_mx %arg0, %int-1, %int32, %int23, %int0 : !torch.vtensor<[4,512,1024],bf16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,512,1024],f8E5M2>, !torch.vtensor<[4,512,32],f8E8M0FNU>
    return %output, %scale : !torch.vtensor<[4,512,1024],f8E5M2>, !torch.vtensor<[4,512,32],f8E8M0FNU>
}

// -----

// CHECK-LABEL:   func.func @test_quantize_mx_4d_second_last(
// CHECK-SAME:                                                %[[VAL_0:.*]]: !torch.vtensor<[2,8,1024,512],bf16>) -> (!torch.vtensor<[2,8,1024,512],f8E5M2>, !torch.vtensor<[2,8,32,512],f8E8M0FNU>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,8,1024,512],bf16> -> tensor<2x8x1024x512xbf16>
// CHECK:           %[[VAL_2:.*]]:2 = stablehlo.custom_call @QuantizeMX(%[[VAL_1]]) {backend_config = "{\22dtype\22: \22float8_e5m2\22, \22dim\22: -2, \22block_size\22: 32, \22scale_method\22: \22EMAX\22}"} : (tensor<2x8x1024x512xbf16>) -> (tensor<2x8x1024x512xf8E5M2>, tensor<2x8x32x512xf8E8M0FNU>)
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]]#1 : tensor<2x8x32x512xf8E8M0FNU> -> !torch.vtensor<[2,8,32,512],f8E8M0FNU>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_2]]#0 : tensor<2x8x1024x512xf8E5M2> -> !torch.vtensor<[2,8,1024,512],f8E5M2>
// CHECK:           return %[[VAL_4]], %[[VAL_3]] : !torch.vtensor<[2,8,1024,512],f8E5M2>, !torch.vtensor<[2,8,32,512],f8E8M0FNU>
// CHECK:         }
func.func @test_quantize_mx_4d_second_last(%arg0: !torch.vtensor<[2,8,1024,512],bf16>) -> (!torch.vtensor<[2,8,1024,512],f8E5M2>, !torch.vtensor<[2,8,32,512],f8E8M0FNU>) {
    %int-2 = torch.constant.int -2
    %int32 = torch.constant.int 32
    %int23 = torch.constant.int 23
    %int0 = torch.constant.int 0
    %output, %scale = torch.aten.quantize_mx %arg0, %int-2, %int32, %int23, %int0 : !torch.vtensor<[2,8,1024,512],bf16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,8,1024,512],f8E5M2>, !torch.vtensor<[2,8,32,512],f8E8M0FNU>
    return %output, %scale : !torch.vtensor<[2,8,1024,512],f8E5M2>, !torch.vtensor<[2,8,32,512],f8E8M0FNU>
}

// -----

// CHECK-LABEL:   func.func @test_neuron_rand(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[],si64>) -> !torch.vtensor<[10,20],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[],si64> -> tensor<i64>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 10
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 20
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = torch.constant.none
// CHECK:           %[[VAL_6:.*]] = torch.constant.device "neuron"
// CHECK:           %[[VAL_7:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_8:.*]] = stablehlo.convert %[[VAL_1]] : (tensor<i64>) -> tensor<ui64>
// CHECK:           %[[VAL_9:.*]] = stablehlo.reshape %[[VAL_8]] : (tensor<ui64>) -> tensor<1xui64>
// CHECK:           %[[VAL_10:.*]] = stablehlo.constant dense<0> : tensor<ui64>
// CHECK:           %[[VAL_11:.*]] = stablehlo.reshape %[[VAL_10]] : (tensor<ui64>) -> tensor<1xui64>
// CHECK:           %[[VAL_12:.*]] = stablehlo.concatenate %[[VAL_9]], %[[VAL_11]], dim = 0 : (tensor<1xui64>, tensor<1xui64>) -> tensor<2xui64>
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = stablehlo.rng_bit_generator %[[VAL_12]], algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<10x20xui32>)
// CHECK:           %[[VAL_15:.*]] = stablehlo.constant dense<9> : tensor<10x20xui32>
// CHECK:           %[[VAL_16:.*]] = stablehlo.shift_right_logical %[[VAL_14]], %[[VAL_15]] : tensor<10x20xui32>
// CHECK:           %[[VAL_17:.*]] = stablehlo.convert %[[VAL_16]] : (tensor<10x20xui32>) -> tensor<10x20xf32>
// CHECK:           %[[VAL_18:.*]] = stablehlo.constant dense<1.1920929E-7> : tensor<10x20xf32>
// CHECK:           %[[VAL_19:.*]] = stablehlo.multiply %[[VAL_17]], %[[VAL_18]] : tensor<10x20xf32>
// CHECK:           %[[VAL_20:.*]] = torch_c.from_builtin_tensor %[[VAL_19]] : tensor<10x20xf32> -> !torch.vtensor<[10,20],f32>
// CHECK:           return %[[VAL_20]] : !torch.vtensor<[10,20],f32>
// CHECK:         }
func.func @test_neuron_rand(%arg0: !torch.vtensor<[],si64>) -> !torch.vtensor<[10,20],f32> {
    %int10 = torch.constant.int 10
    %int20 = torch.constant.int 20
    %0 = torch.prim.ListConstruct %int10, %int20 : (!torch.int, !torch.int) -> !torch.list<int>
    %none = torch.constant.none
    %neuron = torch.constant.device "neuron"
    %false = torch.constant.bool false
    %1 = torch.operator "torch.neuron.rand"(%0, %arg0, %none, %neuron, %false) : (!torch.list<int>, !torch.vtensor<[],si64>, !torch.none, !torch.Device, !torch.bool) -> !torch.vtensor<[10,20],f32>
    return %1 : !torch.vtensor<[10,20],f32>
}

// -----

// CHECK-LABEL:   func.func @test_neuron_rand_bernoulli(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[],si64>) -> !torch.vtensor<[10,20],i1> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[],si64> -> tensor<i64>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 10
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 20
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = torch.constant.none
// CHECK:           %[[VAL_6:.*]] = torch.constant.device "neuron"
// CHECK:           %[[VAL_7:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_8:.*]] = stablehlo.convert %[[VAL_1]] : (tensor<i64>) -> tensor<ui64>
// CHECK:           %[[VAL_9:.*]] = stablehlo.reshape %[[VAL_8]] : (tensor<ui64>) -> tensor<1xui64>
// CHECK:           %[[VAL_10:.*]] = stablehlo.constant dense<0> : tensor<ui64>
// CHECK:           %[[VAL_11:.*]] = stablehlo.reshape %[[VAL_10]] : (tensor<ui64>) -> tensor<1xui64>
// CHECK:           %[[VAL_12:.*]] = stablehlo.concatenate %[[VAL_9]], %[[VAL_11]], dim = 0 : (tensor<1xui64>, tensor<1xui64>) -> tensor<2xui64>
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = stablehlo.rng_bit_generator %[[VAL_12]], algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<10x20xui32>)
// CHECK:           %[[VAL_15:.*]] = stablehlo.constant dense<9> : tensor<10x20xui32>
// CHECK:           %[[VAL_16:.*]] = stablehlo.shift_right_logical %[[VAL_14]], %[[VAL_15]] : tensor<10x20xui32>
// CHECK:           %[[VAL_17:.*]] = stablehlo.convert %[[VAL_16]] : (tensor<10x20xui32>) -> tensor<10x20xf32>
// CHECK:           %[[VAL_18:.*]] = stablehlo.constant dense<1.1920929E-7> : tensor<10x20xf32>
// CHECK:           %[[VAL_19:.*]] = stablehlo.multiply %[[VAL_17]], %[[VAL_18]] : tensor<10x20xf32>
// CHECK:           %[[VAL_20:.*]] = torch_c.from_builtin_tensor %[[VAL_19]] : tensor<10x20xf32> -> !torch.vtensor<[10,20],f32>
// CHECK:           %[[VAL_21:.*]] = torch.constant.float 4.000000e-01
// CHECK:           %[[VAL_22:.*]] = torch.aten.gt.Scalar %[[VAL_20]], %[[VAL_21]] : !torch.vtensor<[10,20],f32>, !torch.float -> !torch.vtensor<[10,20],i1>
// CHECK:           return %[[VAL_22]] : !torch.vtensor<[10,20],i1>
// CHECK:         }
func.func @test_neuron_rand_bernoulli(%arg0: !torch.vtensor<[],si64>) -> !torch.vtensor<[10,20],i1> {
    %int10 = torch.constant.int 10
    %int20 = torch.constant.int 20
    %0 = torch.prim.ListConstruct %int10, %int20 : (!torch.int, !torch.int) -> !torch.list<int>
    %none = torch.constant.none
    %neuron = torch.constant.device "neuron"
    %false = torch.constant.bool false
    %1 = torch.operator "torch.neuron.rand"(%0, %arg0, %none, %neuron, %false) : (!torch.list<int>, !torch.vtensor<[],si64>, !torch.none, !torch.Device, !torch.bool) -> !torch.vtensor<[10,20],f32>
    %float4.000000e-01 = torch.constant.float 4.000000e-01
    %2 = torch.aten.gt.Scalar %1, %float4.000000e-01 : !torch.vtensor<[10,20],f32>, !torch.float -> !torch.vtensor<[10,20],i1>
    return %2 : !torch.vtensor<[10,20],i1>
}

// -----

// CHECK-LABEL:   func.func @test_neuron_rand_with_inputs(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !torch.vtensor<[2,3,4],f32>,
// CHECK-SAME:                                            %[[VAL_1:.*]]: !torch.vtensor<[],si64>) -> !torch.vtensor<[2,3,4],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[],si64> -> tensor<i64>
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_6:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_7:.*]] = torch.constant.none
// CHECK:           %[[VAL_8:.*]] = torch.constant.device "neuron"
// CHECK:           %[[VAL_9:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_10:.*]] = stablehlo.convert %[[VAL_2]] : (tensor<i64>) -> tensor<ui64>
// CHECK:           %[[VAL_11:.*]] = stablehlo.reshape %[[VAL_10]] : (tensor<ui64>) -> tensor<1xui64>
// CHECK:           %[[VAL_12:.*]] = stablehlo.constant dense<0> : tensor<ui64>
// CHECK:           %[[VAL_13:.*]] = stablehlo.reshape %[[VAL_12]] : (tensor<ui64>) -> tensor<1xui64>
// CHECK:           %[[VAL_14:.*]] = stablehlo.concatenate %[[VAL_11]], %[[VAL_13]], dim = 0 : (tensor<1xui64>, tensor<1xui64>) -> tensor<2xui64>
// CHECK:           %[[VAL_15:.*]], %[[VAL_16:.*]] = stablehlo.rng_bit_generator %[[VAL_14]], algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2x3x4xui32>)
// CHECK:           %[[VAL_17:.*]] = stablehlo.constant dense<9> : tensor<2x3x4xui32>
// CHECK:           %[[VAL_18:.*]] = stablehlo.shift_right_logical %[[VAL_16]], %[[VAL_17]] : tensor<2x3x4xui32>
// CHECK:           %[[VAL_19:.*]] = stablehlo.convert %[[VAL_18]] : (tensor<2x3x4xui32>) -> tensor<2x3x4xf32>
// CHECK:           %[[VAL_20:.*]] = stablehlo.constant dense<1.1920929E-7> : tensor<2x3x4xf32>
// CHECK:           %[[VAL_21:.*]] = stablehlo.multiply %[[VAL_19]], %[[VAL_20]] : tensor<2x3x4xf32>
// CHECK:           %[[VAL_22:.*]] = torch_c.from_builtin_tensor %[[VAL_21]] : tensor<2x3x4xf32> -> !torch.vtensor<[2,3,4],f32>
// CHECK:           %[[VAL_23:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_24:.*]] = torch.aten.add.Tensor %[[VAL_0]], %[[VAL_22]], %[[VAL_23]] : !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>, !torch.int -> !torch.vtensor<[2,3,4],f32>
// CHECK:           return %[[VAL_24]] : !torch.vtensor<[2,3,4],f32>
// CHECK:         }
func.func @test_neuron_rand_with_inputs(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[],si64>) -> !torch.vtensor<[2,3,4],f32> {
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %int4 = torch.constant.int 4
    %0 = torch.prim.ListConstruct %int2, %int3, %int4 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %none = torch.constant.none
    %neuron = torch.constant.device "neuron"
    %false = torch.constant.bool false
    %1 = torch.operator "torch.neuron.rand"(%0, %arg1, %none, %neuron, %false) : (!torch.list<int>, !torch.vtensor<[],si64>, !torch.none, !torch.Device, !torch.bool) -> !torch.vtensor<[2,3,4],f32>
    %int1 = torch.constant.int 1
    %2 = torch.aten.add.Tensor %arg0, %1, %int1 : !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>, !torch.int -> !torch.vtensor<[2,3,4],f32>
    return %2 : !torch.vtensor<[2,3,4],f32>
}

// -----

// Test fused_quantize_mx_scaled_mm: basic 2D case with e5m2, out_dtype=bf16
// lhs: [64, 128] bf16 (M=64, K=128), rhs: [256, 128] bf16 (N=256, K=128)
// Both quantize along last dim (K=128): FP8 [..., K], scale K/32=4
// result: [64, 256] bf16 (M × N)
// CHECK-LABEL:   func.func @test_fused_quantize_mx_scaled_mm_basic(
// CHECK-SAME:                                                       %[[LHS:.*]]: !torch.vtensor<[64,128],bf16>,
// CHECK-SAME:                                                       %[[RHS:.*]]: !torch.vtensor<[256,128],bf16>)
// CHECK-DAG:       %[[LHS_T:.*]] = torch_c.to_builtin_tensor %[[LHS]] : !torch.vtensor<[64,128],bf16> -> tensor<64x128xbf16>
// CHECK-DAG:       %[[RHS_T:.*]] = torch_c.to_builtin_tensor %[[RHS]] : !torch.vtensor<[256,128],bf16> -> tensor<256x128xbf16>
// CHECK:           %[[LHS_Q:.*]]:2 = stablehlo.custom_call @QuantizeMX(%[[LHS_T]])
// CHECK-SAME:        {backend_config = "{\22dtype\22: \22float8_e5m2\22, \22dim\22: -1, \22block_size\22: 32, \22scale_method\22: \22EMAX\22}"}
// CHECK-SAME:        : (tensor<64x128xbf16>) -> (tensor<64x128xf8E5M2>, tensor<64x4xf8E8M0FNU>)
// CHECK:           %[[RHS_Q:.*]]:2 = stablehlo.custom_call @QuantizeMX(%[[RHS_T]])
// CHECK-SAME:        {backend_config = "{\22dtype\22: \22float8_e5m2\22, \22dim\22: -1, \22block_size\22: 32, \22scale_method\22: \22EMAX\22}"}
// CHECK-SAME:        : (tensor<256x128xbf16>) -> (tensor<256x128xf8E5M2>, tensor<256x4xf8E8M0FNU>)
// CHECK:           %[[DOT:.*]] = stablehlo.custom_call @__op$block_scaled_dot(%[[LHS_Q]]#0, %[[RHS_Q]]#0, %[[LHS_Q]]#1, %[[RHS_Q]]#1)
// CHECK-SAME:        {backend_config = "{\22scaled_dot_backend_config\22: {\22lhs_batch_dimensions\22: [], \22rhs_batch_dimensions\22: [], \22lhs_contracting_dimensions\22: [1], \22rhs_contracting_dimensions\22: [1], \22dequantize_type\22: \22BF16\22, \22element_dtype\22: \22float8_e5m2\22}}"}
// CHECK-SAME:        : (tensor<64x128xf8E5M2>, tensor<256x128xf8E5M2>, tensor<64x4xf8E8M0FNU>, tensor<256x4xf8E8M0FNU>) -> tensor<64x256xbf16>
// CHECK:           %[[OUT:.*]] = torch_c.from_builtin_tensor %[[DOT]] : tensor<64x256xbf16> -> !torch.vtensor<[64,256],bf16>
// CHECK:           return %[[OUT]] : !torch.vtensor<[64,256],bf16>
func.func @test_fused_quantize_mx_scaled_mm_basic(
    %arg0: !torch.vtensor<[64,128],bf16>,
    %arg1: !torch.vtensor<[256,128],bf16>) -> !torch.vtensor<[64,256],bf16> {
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %int23 = torch.constant.int 23
    %0 = torch.operator "torch.neuron.fused_quantize_mx_scaled_mm"(
        %arg0, %arg1, %none, %none, %int15, %int23)
        : (!torch.vtensor<[64,128],bf16>, !torch.vtensor<[256,128],bf16>,
           !torch.none, !torch.none, !torch.int, !torch.int)
        -> !torch.vtensor<[64,256],bf16>
    return %0 : !torch.vtensor<[64,256],bf16>
}

// -----

// Test fused_quantize_mx_scaled_mm: e4m3fn fp8_dtype with f16 inputs
// lhs: [32, 64] f16 (M=32, K=64), rhs: [128, 64] f16 (N=128, K=64)
// K=64: FP8 shape K, scale K/32=2
// result: [32, 128] bf16
// CHECK-LABEL:   func.func @test_fused_quantize_mx_scaled_mm_e4m3fn(
// CHECK:           %[[LHS_Q:.*]]:2 = stablehlo.custom_call @QuantizeMX
// CHECK-SAME:        {backend_config = "{\22dtype\22: \22float8_e4m3fn\22, \22dim\22: -1, \22block_size\22: 32, \22scale_method\22: \22EMAX\22}"}
// CHECK-SAME:        : (tensor<32x64xf16>) -> (tensor<32x64xf8E4M3FN>, tensor<32x2xf8E8M0FNU>)
// CHECK:           %[[RHS_Q:.*]]:2 = stablehlo.custom_call @QuantizeMX
// CHECK-SAME:        {backend_config = "{\22dtype\22: \22float8_e4m3fn\22, \22dim\22: -1, \22block_size\22: 32, \22scale_method\22: \22EMAX\22}"}
// CHECK-SAME:        : (tensor<128x64xf16>) -> (tensor<128x64xf8E4M3FN>, tensor<128x2xf8E8M0FNU>)
// CHECK:           %[[DOT:.*]] = stablehlo.custom_call @__op$block_scaled_dot(%[[LHS_Q]]#0, %[[RHS_Q]]#0, %[[LHS_Q]]#1, %[[RHS_Q]]#1)
// CHECK-SAME:        \22element_dtype\22: \22float8_e4m3fn\22
// CHECK-SAME:        : (tensor<32x64xf8E4M3FN>, tensor<128x64xf8E4M3FN>, tensor<32x2xf8E8M0FNU>, tensor<128x2xf8E8M0FNU>) -> tensor<32x128xbf16>
func.func @test_fused_quantize_mx_scaled_mm_e4m3fn(
    %arg0: !torch.vtensor<[32,64],f16>,
    %arg1: !torch.vtensor<[128,64],f16>) -> !torch.vtensor<[32,128],bf16> {
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %int24 = torch.constant.int 24
    %0 = torch.operator "torch.neuron.fused_quantize_mx_scaled_mm"(
        %arg0, %arg1, %none, %none, %int15, %int24)
        : (!torch.vtensor<[32,64],f16>, !torch.vtensor<[128,64],f16>,
           !torch.none, !torch.none, !torch.int, !torch.int)
        -> !torch.vtensor<[32,128],bf16>
    return %0 : !torch.vtensor<[32,128],bf16>
}

// -----

// Test fused_quantize_mx_scaled_mm: 3D batched case
// lhs: [4, 512, 1024] bf16, rhs: [4, 2048, 1024] bf16
// K=1024: FP8 shape K, scale K/32=32
// result: [4, 512, 2048] bf16
// batch_dims=[0], contracting_dims=[2]
// CHECK-LABEL:   func.func @test_fused_quantize_mx_scaled_mm_3d(
// CHECK:           %[[LHS_Q:.*]]:2 = stablehlo.custom_call @QuantizeMX
// CHECK-SAME:        : (tensor<4x512x1024xbf16>) -> (tensor<4x512x1024xf8E5M2>, tensor<4x512x32xf8E8M0FNU>)
// CHECK:           %[[RHS_Q:.*]]:2 = stablehlo.custom_call @QuantizeMX
// CHECK-SAME:        : (tensor<4x2048x1024xbf16>) -> (tensor<4x2048x1024xf8E5M2>, tensor<4x2048x32xf8E8M0FNU>)
// CHECK:           %[[DOT:.*]] = stablehlo.custom_call @__op$block_scaled_dot(%[[LHS_Q]]#0, %[[RHS_Q]]#0, %[[LHS_Q]]#1, %[[RHS_Q]]#1)
// CHECK-SAME:        {backend_config = "{\22scaled_dot_backend_config\22: {\22lhs_batch_dimensions\22: [0], \22rhs_batch_dimensions\22: [0], \22lhs_contracting_dimensions\22: [2], \22rhs_contracting_dimensions\22: [2], \22dequantize_type\22: \22BF16\22, \22element_dtype\22: \22float8_e5m2\22}}"}
// CHECK-SAME:        : (tensor<4x512x1024xf8E5M2>, tensor<4x2048x1024xf8E5M2>, tensor<4x512x32xf8E8M0FNU>, tensor<4x2048x32xf8E8M0FNU>) -> tensor<4x512x2048xbf16>
func.func @test_fused_quantize_mx_scaled_mm_3d(
    %arg0: !torch.vtensor<[4,512,1024],bf16>,
    %arg1: !torch.vtensor<[4,2048,1024],bf16>) -> !torch.vtensor<[4,512,2048],bf16> {
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %int23 = torch.constant.int 23
    %0 = torch.operator "torch.neuron.fused_quantize_mx_scaled_mm"(
        %arg0, %arg1, %none, %none, %int15, %int23)
        : (!torch.vtensor<[4,512,1024],bf16>, !torch.vtensor<[4,2048,1024],bf16>,
           !torch.none, !torch.none, !torch.int, !torch.int)
        -> !torch.vtensor<[4,512,2048],bf16>
    return %0 : !torch.vtensor<[4,512,2048],bf16>
}

// -----

// Test fused_quantize_mx_scaled_mm: f32 output dtype
// lhs: [64, 128] bf16, rhs: [256, 128] bf16
// out_dtype=6 (Float) → dequantize_type="F32", result type f32
// CHECK-LABEL:   func.func @test_fused_quantize_mx_scaled_mm_f32_output(
// CHECK:           %[[DOT:.*]] = stablehlo.custom_call @__op$block_scaled_dot
// CHECK-SAME:        \22dequantize_type\22: \22F32\22
// CHECK-SAME:        -> tensor<64x256xf32>
func.func @test_fused_quantize_mx_scaled_mm_f32_output(
    %arg0: !torch.vtensor<[64,128],bf16>,
    %arg1: !torch.vtensor<[256,128],bf16>) -> !torch.vtensor<[64,256],f32> {
    %none = torch.constant.none
    %int6 = torch.constant.int 6
    %int23 = torch.constant.int 23
    %0 = torch.operator "torch.neuron.fused_quantize_mx_scaled_mm"(
        %arg0, %arg1, %none, %none, %int6, %int23)
        : (!torch.vtensor<[64,128],bf16>, !torch.vtensor<[256,128],bf16>,
           !torch.none, !torch.none, !torch.int, !torch.int)
        -> !torch.vtensor<[64,256],f32>
    return %0 : !torch.vtensor<[64,256],f32>
}

// -----

// Test _scaled_mm: basic 2D case, FP8 inputs, out_dtype=bf16, fp8_dtype=e5m2
// lhs: [64, 128] f8E5M2 (M=64, K=128), rhs: [256, 128] f8E5M2 (N=256, K=128, transposed)
// lhs_scale: [64, 4] f8E8M0FNU (K/32 = 4), rhs_scale: [256, 4] f8E8M0FNU
// result: [64, 256] bf16
// CHECK-LABEL:   func.func @test_scaled_mm_basic(
// CHECK-SAME:                                    %[[LHS:.*]]: !torch.vtensor<[64,128],f8E5M2>,
// CHECK-SAME:                                    %[[SA:.*]]: !torch.vtensor<[64,4],f8E8M0FNU>,
// CHECK-SAME:                                    %[[RHS:.*]]: !torch.vtensor<[256,128],f8E5M2>,
// CHECK-SAME:                                    %[[SB:.*]]: !torch.vtensor<[256,4],f8E8M0FNU>)
// CHECK-DAG:       %[[LHS_T:.*]] = torch_c.to_builtin_tensor %[[LHS]] : !torch.vtensor<[64,128],f8E5M2> -> tensor<64x128xf8E5M2>
// CHECK-DAG:       %[[RHS_T:.*]] = torch_c.to_builtin_tensor %[[RHS]] : !torch.vtensor<[256,128],f8E5M2> -> tensor<256x128xf8E5M2>
// CHECK-DAG:       %[[SA_T:.*]] = torch_c.to_builtin_tensor %[[SA]] : !torch.vtensor<[64,4],f8E8M0FNU> -> tensor<64x4xf8E8M0FNU>
// CHECK-DAG:       %[[SB_T:.*]] = torch_c.to_builtin_tensor %[[SB]] : !torch.vtensor<[256,4],f8E8M0FNU> -> tensor<256x4xf8E8M0FNU>
// CHECK:           %[[OUT:.*]] = stablehlo.custom_call @__op$block_scaled_dot(%[[LHS_T]], %[[RHS_T]], %[[SA_T]], %[[SB_T]])
// CHECK-SAME:        {backend_config = "{\22scaled_dot_backend_config\22: {\22lhs_batch_dimensions\22: [], \22rhs_batch_dimensions\22: [], \22lhs_contracting_dimensions\22: [1], \22rhs_contracting_dimensions\22: [1], \22dequantize_type\22: \22BF16\22, \22element_dtype\22: \22float8_e5m2\22}}"}
// CHECK-SAME:        : (tensor<64x128xf8E5M2>, tensor<256x128xf8E5M2>, tensor<64x4xf8E8M0FNU>, tensor<256x4xf8E8M0FNU>) -> tensor<64x256xbf16>
// CHECK:           %[[RES:.*]] = torch_c.from_builtin_tensor %[[OUT]] : tensor<64x256xbf16> -> !torch.vtensor<[64,256],bf16>
// CHECK:           return %[[RES]] : !torch.vtensor<[64,256],bf16>
func.func @test_scaled_mm_basic(
    %arg0: !torch.vtensor<[64,128],f8E5M2>,
    %arg1: !torch.vtensor<[64,4],f8E8M0FNU>,
    %arg2: !torch.vtensor<[256,128],f8E5M2>,
    %arg3: !torch.vtensor<[256,4],f8E8M0FNU>) -> !torch.vtensor<[64,256],bf16> {
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %int23 = torch.constant.int 23
    %0 = torch.operator "torch.neuron._scaled_mm"(%arg0, %arg1, %arg2, %arg3, %none, %none, %int15, %int23)
        : (!torch.vtensor<[64,128],f8E5M2>, !torch.vtensor<[64,4],f8E8M0FNU>,
           !torch.vtensor<[256,128],f8E5M2>, !torch.vtensor<[256,4],f8E8M0FNU>,
           !torch.none, !torch.none, !torch.int, !torch.int)
        -> !torch.vtensor<[64,256],bf16>
    return %0 : !torch.vtensor<[64,256],bf16>
}

// -----

// Test _scaled_mm: out_dtype=None defaults to bf16, fp8_dtype=None defaults to float8_e4m3fn
// mat1: [16, 32] f8E4M3FN (M=16, K=32), mat2: [64, 32] f8E4M3FN (N=64, K=32, transposed)
// lhs_scale: [16, 1] (K/32=1), rhs_scale: [64, 1]
// result: [16, 64] bf16
// CHECK-LABEL:   func.func @test_scaled_mm_out_dtype_none(
// CHECK:           %[[OUT:.*]] = stablehlo.custom_call @__op$block_scaled_dot
// CHECK-SAME:        {backend_config = "{\22scaled_dot_backend_config\22: {\22lhs_batch_dimensions\22: [], \22rhs_batch_dimensions\22: [], \22lhs_contracting_dimensions\22: [1], \22rhs_contracting_dimensions\22: [1], \22dequantize_type\22: \22BF16\22, \22element_dtype\22: \22float8_e4m3fn\22}}"}
// CHECK-SAME:        : (tensor<16x32xf8E4M3FN>, tensor<64x32xf8E4M3FN>, tensor<16x1xf8E8M0FNU>, tensor<64x1xf8E8M0FNU>) -> tensor<16x64xbf16>
func.func @test_scaled_mm_out_dtype_none(
    %arg0: !torch.vtensor<[16,32],f8E4M3FN>,
    %arg1: !torch.vtensor<[16,1],f8E8M0FNU>,
    %arg2: !torch.vtensor<[64,32],f8E4M3FN>,
    %arg3: !torch.vtensor<[64,1],f8E8M0FNU>) -> !torch.vtensor<[16,64],bf16> {
    %none = torch.constant.none
    %0 = torch.operator "torch.neuron._scaled_mm"(%arg0, %arg1, %arg2, %arg3, %none, %none, %none, %none)
        : (!torch.vtensor<[16,32],f8E4M3FN>, !torch.vtensor<[16,1],f8E8M0FNU>,
           !torch.vtensor<[64,32],f8E4M3FN>, !torch.vtensor<[64,1],f8E8M0FNU>,
           !torch.none, !torch.none, !torch.none, !torch.none)
        -> !torch.vtensor<[16,64],bf16>
    return %0 : !torch.vtensor<[16,64],bf16>
}

// -----

// Test _scaled_mm: 3D batched case, fp8_dtype=e5m2
// mat1: [4, 64, 128] f8E5M2 (M=64, K=128), mat2: [4, 256, 128] f8E5M2 (N=256, K=128, transposed)
// lhs_scale: [4, 64, 4] (K/32=4), rhs_scale: [4, 256, 4]
// result: [4, 64, 256] bf16
// CHECK-LABEL:   func.func @test_scaled_mm_3d(
// CHECK:           %[[OUT:.*]] = stablehlo.custom_call @__op$block_scaled_dot
// CHECK-SAME:        {backend_config = "{\22scaled_dot_backend_config\22: {\22lhs_batch_dimensions\22: [0], \22rhs_batch_dimensions\22: [0], \22lhs_contracting_dimensions\22: [2], \22rhs_contracting_dimensions\22: [2], \22dequantize_type\22: \22BF16\22, \22element_dtype\22: \22float8_e5m2\22}}"}
// CHECK-SAME:        : (tensor<4x64x128xf8E5M2>, tensor<4x256x128xf8E5M2>, tensor<4x64x4xf8E8M0FNU>, tensor<4x256x4xf8E8M0FNU>) -> tensor<4x64x256xbf16>
func.func @test_scaled_mm_3d(
    %arg0: !torch.vtensor<[4,64,128],f8E5M2>,
    %arg1: !torch.vtensor<[4,64,4],f8E8M0FNU>,
    %arg2: !torch.vtensor<[4,256,128],f8E5M2>,
    %arg3: !torch.vtensor<[4,256,4],f8E8M0FNU>) -> !torch.vtensor<[4,64,256],bf16> {
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %int23 = torch.constant.int 23
    %0 = torch.operator "torch.neuron._scaled_mm"(%arg0, %arg1, %arg2, %arg3, %none, %none, %int15, %int23)
        : (!torch.vtensor<[4,64,128],f8E5M2>, !torch.vtensor<[4,64,4],f8E8M0FNU>,
           !torch.vtensor<[4,256,128],f8E5M2>, !torch.vtensor<[4,256,4],f8E8M0FNU>,
           !torch.none, !torch.none, !torch.int, !torch.int)
        -> !torch.vtensor<[4,64,256],bf16>
    return %0 : !torch.vtensor<[4,64,256],bf16>
}

// -----

// Test _scaled_mm: explicit e4m3fn fp8_dtype with bf16 output
// mat1: [64, 128] f8E4M3FN, mat2: [256, 128] f8E4M3FN (transposed)
// scale: [64, 4] / [256, 4] (K/32=4), result: [64, 256] bf16
// CHECK-LABEL:   func.func @test_scaled_mm_e4m3fn_explicit(
// CHECK:           %[[OUT:.*]] = stablehlo.custom_call @__op$block_scaled_dot
// CHECK-SAME:        \22element_dtype\22: \22float8_e4m3fn\22
// CHECK-SAME:        : (tensor<64x128xf8E4M3FN>, tensor<256x128xf8E4M3FN>, tensor<64x4xf8E8M0FNU>, tensor<256x4xf8E8M0FNU>) -> tensor<64x256xbf16>
func.func @test_scaled_mm_e4m3fn_explicit(
    %arg0: !torch.vtensor<[64,128],f8E4M3FN>,
    %arg1: !torch.vtensor<[64,4],f8E8M0FNU>,
    %arg2: !torch.vtensor<[256,128],f8E4M3FN>,
    %arg3: !torch.vtensor<[256,4],f8E8M0FNU>) -> !torch.vtensor<[64,256],bf16> {
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %int24 = torch.constant.int 24
    %0 = torch.operator "torch.neuron._scaled_mm"(%arg0, %arg1, %arg2, %arg3, %none, %none, %int15, %int24)
        : (!torch.vtensor<[64,128],f8E4M3FN>, !torch.vtensor<[64,4],f8E8M0FNU>,
           !torch.vtensor<[256,128],f8E4M3FN>, !torch.vtensor<[256,4],f8E8M0FNU>,
           !torch.none, !torch.none, !torch.int, !torch.int)
        -> !torch.vtensor<[64,256],bf16>
    return %0 : !torch.vtensor<[64,256],bf16>
}

// -----

// CHECK-LABEL:   func.func @test_neuron_argmax_dim(
// CHECK-SAME:                    %[[VAL_0:.*]]: !torch.vtensor<[4,5],f32>) -> !torch.vtensor<[4],si64> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,5],f32> -> tensor<4x5xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.custom_call @AwsNeuronArgMax(%[[VAL_1]]) {backend_config = "1"} : (tensor<4x5xf32>) -> tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<4xi64> -> !torch.vtensor<[4],si64>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[4],si64>
// CHECK:         }
func.func @test_neuron_argmax_dim(%arg0: !torch.vtensor<[4,5],f32>) -> !torch.vtensor<[4],si64> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %int1, %false : !torch.vtensor<[4,5],f32>, !torch.int, !torch.bool -> !torch.vtensor<[4],si64>
  return %0 : !torch.vtensor<[4],si64>
}

// -----

// A negative dim is normalised before it reaches the backend config.
// CHECK-LABEL:   func.func @test_neuron_argmax_negative_dim(
// CHECK:           stablehlo.custom_call @AwsNeuronArgMax(%{{.*}}) {backend_config = "1"} : (tensor<4x5xf32>) -> tensor<4xi64>
func.func @test_neuron_argmax_negative_dim(%arg0: !torch.vtensor<[4,5],f32>) -> !torch.vtensor<[4],si64> {
  %int-1 = torch.constant.int -1
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %int-1, %false : !torch.vtensor<[4,5],f32>, !torch.int, !torch.bool -> !torch.vtensor<[4],si64>
  return %0 : !torch.vtensor<[4],si64>
}

// -----

// dim=None reduces over the flattened tensor: reshape to rank 1, dim 0.
// CHECK-LABEL:   func.func @test_neuron_argmax_no_dim(
// CHECK:           %[[FLAT:.*]] = stablehlo.reshape %{{.*}} : (tensor<4x5xf32>) -> tensor<20xf32>
// CHECK:           stablehlo.custom_call @AwsNeuronArgMax(%[[FLAT]]) {backend_config = "0"} : (tensor<20xf32>) -> tensor<i64>
func.func @test_neuron_argmax_no_dim(%arg0: !torch.vtensor<[4,5],f32>) -> !torch.vtensor<[],si64> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %none, %false : !torch.vtensor<[4,5],f32>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}

// -----

// keepdim reshapes the result so the reduced dimension comes back as 1.
// CHECK-LABEL:   func.func @test_neuron_argmax_keepdim(
// CHECK:           %[[CC:.*]] = stablehlo.custom_call @AwsNeuronArgMax(%{{.*}}) {backend_config = "1"} : (tensor<4x5xf32>) -> tensor<4xi64>
// CHECK:           stablehlo.reshape %[[CC]] : (tensor<4xi64>) -> tensor<4x1xi64>
func.func @test_neuron_argmax_keepdim(%arg0: !torch.vtensor<[4,5],f32>) -> !torch.vtensor<[4,1],si64> {
  %int1 = torch.constant.int 1
  %true = torch.constant.bool true
  %0 = torch.aten.argmax %arg0, %int1, %true : !torch.vtensor<[4,5],f32>, !torch.int, !torch.bool -> !torch.vtensor<[4,1],si64>
  return %0 : !torch.vtensor<[4,1],si64>
}

// -----

// s32 is supported, and a rank-1 input with dim=None needs no reshape.
// CHECK-LABEL:   func.func @test_neuron_argmax_int32(
// CHECK-NOT:       stablehlo.reshape
// CHECK:           stablehlo.custom_call @AwsNeuronArgMax(%{{.*}}) {backend_config = "0"} : (tensor<8xi32>) -> tensor<i64>
func.func @test_neuron_argmax_int32(%arg0: !torch.vtensor<[8],si32>) -> !torch.vtensor<[],si64> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %none, %false : !torch.vtensor<[8],si32>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}

// -----

// An unsupported element type falls back to the max.dim lowering.
// CHECK-LABEL:   func.func @test_neuron_argmax_unsupported_dtype(
// CHECK-NOT:       AwsNeuronArgMax
func.func @test_neuron_argmax_unsupported_dtype(%arg0: !torch.vtensor<[8],si8>) -> !torch.vtensor<[],si64> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %none, %false : !torch.vtensor<[8],si8>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}

// -----

// A dynamic shape falls back too.
// CHECK-LABEL:   func.func @test_neuron_argmax_dynamic_shape(
// CHECK-NOT:       AwsNeuronArgMax
func.func @test_neuron_argmax_dynamic_shape(%arg0: !torch.vtensor<[?,5],f32>) -> !torch.vtensor<[?],si64> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %int1, %false : !torch.vtensor<[?,5],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?],si64>
  return %0 : !torch.vtensor<[?],si64>
}

// -----

// A non-trailing reduced dimension.
// CHECK-LABEL:   func.func @test_neuron_argmax_dim0_rank2(
// CHECK:           stablehlo.custom_call @AwsNeuronArgMax(%{{.*}}) {backend_config = "0"} : (tensor<4x5xf32>) -> tensor<5xi64>
func.func @test_neuron_argmax_dim0_rank2(%arg0: !torch.vtensor<[4,5],f32>) -> !torch.vtensor<[5],si64> {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %int0, %false : !torch.vtensor<[4,5],f32>, !torch.int, !torch.bool -> !torch.vtensor<[5],si64>
  return %0 : !torch.vtensor<[5],si64>
}

// -----

// A middle dimension, reduced away from both ends.
// CHECK-LABEL:   func.func @test_neuron_argmax_dim1_rank3(
// CHECK:           stablehlo.custom_call @AwsNeuronArgMax(%{{.*}}) {backend_config = "1"} : (tensor<2x3x4xf32>) -> tensor<2x4xi64>
func.func @test_neuron_argmax_dim1_rank3(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,4],si64> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %int1, %false : !torch.vtensor<[2,3,4],f32>, !torch.int, !torch.bool -> !torch.vtensor<[2,4],si64>
  return %0 : !torch.vtensor<[2,4],si64>
}

// -----

// dim=None with keepdim: torch declares the result as all-ones of the input rank.
// CHECK-LABEL:   func.func @test_neuron_argmax_no_dim_keepdim(
// CHECK:           %[[FLAT:.*]] = stablehlo.reshape %{{.*}} : (tensor<4x5xf32>) -> tensor<20xf32>
// CHECK:           %[[CC:.*]] = stablehlo.custom_call @AwsNeuronArgMax(%[[FLAT]]) {backend_config = "0"} : (tensor<20xf32>) -> tensor<i64>
// CHECK:           stablehlo.reshape %[[CC]] : (tensor<i64>) -> tensor<1x1xi64>
func.func @test_neuron_argmax_no_dim_keepdim(%arg0: !torch.vtensor<[4,5],f32>) -> !torch.vtensor<[1,1],si64> {
  %none = torch.constant.none
  %true = torch.constant.bool true
  %0 = torch.aten.argmax %arg0, %none, %true : !torch.vtensor<[4,5],f32>, !torch.none, !torch.bool -> !torch.vtensor<[1,1],si64>
  return %0 : !torch.vtensor<[1,1],si64>
}

// -----

// A rank-0 input with an explicit dim. The max.dim decomposition also declines
// rank 0, so nothing would lower it if this pattern did too.
// CHECK-LABEL:   func.func @test_neuron_argmax_rank0(
// CHECK:           %[[FLAT:.*]] = stablehlo.reshape %{{.*}} : (tensor<f32>) -> tensor<1xf32>
// CHECK:           stablehlo.custom_call @AwsNeuronArgMax(%[[FLAT]]) {backend_config = "0"} : (tensor<1xf32>) -> tensor<i64>
func.func @test_neuron_argmax_rank0(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],si64> {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %int0, %false : !torch.vtensor<[],f32>, !torch.int, !torch.bool -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}

// -----

// f16 is supported.
// CHECK-LABEL:   func.func @test_neuron_argmax_f16(
// CHECK:           stablehlo.custom_call @AwsNeuronArgMax(%{{.*}}) {backend_config = "1"} : (tensor<4x5xf16>) -> tensor<4xi64>
func.func @test_neuron_argmax_f16(%arg0: !torch.vtensor<[4,5],f16>) -> !torch.vtensor<[4],si64> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %int1, %false : !torch.vtensor<[4,5],f16>, !torch.int, !torch.bool -> !torch.vtensor<[4],si64>
  return %0 : !torch.vtensor<[4],si64>
}

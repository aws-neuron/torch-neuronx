// RUN: torch-mlir-opt -pass-pipeline='builtin.module(torch-raw-to-neuron-backend-pipeline)' %s -verify-diagnostics

module {

  func.func @test_invalid_topk_sorted_false(%arg0: !torch.vtensor<[24],f32>) -> !torch.vtensor<[3],f32> {
    %int3 = torch.constant.int 3
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    // expected-error @+1 {{failed to legalize operation 'torch.aten.topk'}}
    %values, %indices = torch.aten.topk %arg0, %int3, %int-1, %true, %false : !torch.vtensor<[24],f32>, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[3],f32>, !torch.vtensor<[3],si64>
    return %values : !torch.vtensor<[3],f32>
  }


  func.func @test_aten.sort$basic(%arg0: !torch.vtensor<[10,18000],f32>) -> (!torch.vtensor<[10,18000],f32>, !torch.vtensor<[10,18000],si64>) {
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    // expected-error @+1 {{failed to legalize operation 'torch.aten.sort'}}
    %values, %indices = torch.aten.sort %arg0, %int-1, %false : !torch.vtensor<[10,18000],f32>, !torch.int, !torch.bool -> !torch.vtensor<[10,18000],f32>, !torch.vtensor<[10,18000],si64>
    return %values, %indices : !torch.vtensor<[10,18000],f32>, !torch.vtensor<[10,18000],si64>
  }

  func.func @test_quantize_mx_invalid_scale_mode(%arg0: !torch.vtensor<[890,128],bf16>) -> (!torch.vtensor<[890,128],f8E5M2>, !torch.vtensor<[890,4],f8E8M0FNU>) {
    %int-1 = torch.constant.int -1
    %int32 = torch.constant.int 32
    %int23 = torch.constant.int 23
    %int1 = torch.constant.int 1
    // expected-error @+1 {{failed to legalize operation 'torch.aten.quantize_mx'}}
    %output, %scale = torch.aten.quantize_mx %arg0, %int-1, %int32, %int23, %int1 : !torch.vtensor<[890,128],bf16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[890,128],f8E5M2>, !torch.vtensor<[890,4],f8E8M0FNU>
    return %output, %scale : !torch.vtensor<[890,128],f8E5M2>, !torch.vtensor<[890,4],f8E8M0FNU>
  }

  func.func @test_quantize_mx_invalid_block_size(%arg0: !torch.vtensor<[890,128],bf16>) -> (!torch.vtensor<[890,128],f8E5M2>, !torch.vtensor<[890,2],f8E8M0FNU>) {
    %int-1 = torch.constant.int -1
    %int64 = torch.constant.int 64
    %int23 = torch.constant.int 23
    %int0 = torch.constant.int 0
    // expected-error @+1 {{failed to legalize operation 'torch.aten.quantize_mx'}}
    %output, %scale = torch.aten.quantize_mx %arg0, %int-1, %int64, %int23, %int0 : !torch.vtensor<[890,128],bf16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[890,128],f8E5M2>, !torch.vtensor<[890,2],f8E8M0FNU>
    return %output, %scale : !torch.vtensor<[890,128],f8E5M2>, !torch.vtensor<[890,2],f8E8M0FNU>
  }

  func.func @test_quantize_mx_invalid_dim(%arg0: !torch.vtensor<[4,890,128],bf16>) -> (!torch.vtensor<[4,890,128],f8E5M2>, !torch.vtensor<[4,890,4],f8E8M0FNU>) {
    %int0 = torch.constant.int 0
    %int32 = torch.constant.int 32
    %int23 = torch.constant.int 23
    %int0_mode = torch.constant.int 0
    // expected-error @+1 {{failed to legalize operation 'torch.aten.quantize_mx'}}
    %output, %scale = torch.aten.quantize_mx %arg0, %int0, %int32, %int23, %int0_mode : !torch.vtensor<[4,890,128],bf16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,890,128],f8E5M2>, !torch.vtensor<[4,890,4],f8E8M0FNU>
    return %output, %scale : !torch.vtensor<[4,890,128],f8E5M2>, !torch.vtensor<[4,890,4],f8E8M0FNU>
  }

  // Invalid fp8_dtype (not 23 or 24)
  func.func @test_fused_quantize_mx_scaled_mm_invalid_fp8_dtype(
      %arg0: !torch.vtensor<[64,128],bf16>,
      %arg1: !torch.vtensor<[256,128],bf16>) -> !torch.vtensor<[64,256],bf16> {
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %int5 = torch.constant.int 5
    // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
    %0 = torch.operator "torch.neuron.fused_quantize_mx_scaled_mm"(
        %arg0, %arg1, %none, %none, %int15, %int5)
        : (!torch.vtensor<[64,128],bf16>, !torch.vtensor<[256,128],bf16>,
           !torch.none, !torch.none, !torch.int, !torch.int)
        -> !torch.vtensor<[64,256],bf16>
    return %0 : !torch.vtensor<[64,256],bf16>
  }

  // Invalid: bias is not None
  func.func @test_fused_quantize_mx_scaled_mm_bias_not_none(
      %arg0: !torch.vtensor<[64,128],bf16>,
      %arg1: !torch.vtensor<[256,128],bf16>,
      %arg2: !torch.vtensor<[256],bf16>) -> !torch.vtensor<[64,256],bf16> {
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %int23 = torch.constant.int 23
    // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
    %0 = torch.operator "torch.neuron.fused_quantize_mx_scaled_mm"(
        %arg0, %arg1, %arg2, %none, %int15, %int23)
        : (!torch.vtensor<[64,128],bf16>, !torch.vtensor<[256,128],bf16>,
           !torch.vtensor<[256],bf16>, !torch.none, !torch.int, !torch.int)
        -> !torch.vtensor<[64,256],bf16>
    return %0 : !torch.vtensor<[64,256],bf16>
  }

  // Invalid: input element type is f32 (not bf16 or f16)
  func.func @test_fused_quantize_mx_scaled_mm_invalid_input_dtype(
      %arg0: !torch.vtensor<[64,128],f32>,
      %arg1: !torch.vtensor<[256,128],f32>) -> !torch.vtensor<[64,256],bf16> {
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %int23 = torch.constant.int 23
    // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
    %0 = torch.operator "torch.neuron.fused_quantize_mx_scaled_mm"(
        %arg0, %arg1, %none, %none, %int15, %int23)
        : (!torch.vtensor<[64,128],f32>, !torch.vtensor<[256,128],f32>,
           !torch.none, !torch.none, !torch.int, !torch.int)
        -> !torch.vtensor<[64,256],bf16>
    return %0 : !torch.vtensor<[64,256],bf16>
  }

  // Invalid: rank < 2
  func.func @test_fused_quantize_mx_scaled_mm_rank1(
      %arg0: !torch.vtensor<[128],bf16>,
      %arg1: !torch.vtensor<[128],bf16>) -> !torch.vtensor<[128],bf16> {
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %int23 = torch.constant.int 23
    // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
    %0 = torch.operator "torch.neuron.fused_quantize_mx_scaled_mm"(
        %arg0, %arg1, %none, %none, %int15, %int23)
        : (!torch.vtensor<[128],bf16>, !torch.vtensor<[128],bf16>,
           !torch.none, !torch.none, !torch.int, !torch.int)
        -> !torch.vtensor<[128],bf16>
    return %0 : !torch.vtensor<[128],bf16>
  }

  // Invalid: dynamic K dimension
  func.func @test_fused_quantize_mx_scaled_mm_dynamic_k(
      %arg0: !torch.vtensor<[64,?],bf16>,
      %arg1: !torch.vtensor<[256,?],bf16>) -> !torch.vtensor<[64,256],bf16> {
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %int23 = torch.constant.int 23
    // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
    %0 = torch.operator "torch.neuron.fused_quantize_mx_scaled_mm"(
        %arg0, %arg1, %none, %none, %int15, %int23)
        : (!torch.vtensor<[64,?],bf16>, !torch.vtensor<[256,?],bf16>,
           !torch.none, !torch.none, !torch.int, !torch.int)
        -> !torch.vtensor<[64,256],bf16>
    return %0 : !torch.vtensor<[64,256],bf16>
  }

  // Invalid: non-FP8 input type (bf16 instead of f8E5M2/f8E4M3FN).
  // Expected to fail at the FP8 element-type check in transformScaledMm.
  func.func @test_scaled_mm_invalid_input_dtype(
      %arg0: !torch.vtensor<[64,128],bf16>,
      %arg1: !torch.vtensor<[64,4],f8E8M0FNU>,
      %arg2: !torch.vtensor<[256,128],bf16>,
      %arg3: !torch.vtensor<[256,4],f8E8M0FNU>) -> !torch.vtensor<[64,256],bf16> {
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %int23 = torch.constant.int 23
    // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
    %0 = torch.operator "torch.neuron._scaled_mm"(%arg0, %arg1, %arg2, %arg3, %none, %none, %int15, %int23)
        : (!torch.vtensor<[64,128],bf16>, !torch.vtensor<[64,4],f8E8M0FNU>,
           !torch.vtensor<[256,128],bf16>, !torch.vtensor<[256,4],f8E8M0FNU>,
           !torch.none, !torch.none, !torch.int, !torch.int)
        -> !torch.vtensor<[64,256],bf16>
    return %0 : !torch.vtensor<[64,256],bf16>
  }

  // Invalid: bias is not None
  func.func @test_scaled_mm_bias_not_none(
      %arg0: !torch.vtensor<[64,128],f8E5M2>,
      %arg1: !torch.vtensor<[64,4],f8E8M0FNU>,
      %arg2: !torch.vtensor<[256,128],f8E5M2>,
      %arg3: !torch.vtensor<[256,4],f8E8M0FNU>,
      %arg4: !torch.vtensor<[256],bf16>) -> !torch.vtensor<[64,256],bf16> {
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %int23 = torch.constant.int 23
    // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
    %0 = torch.operator "torch.neuron._scaled_mm"(%arg0, %arg1, %arg2, %arg3, %arg4, %none, %int15, %int23)
        : (!torch.vtensor<[64,128],f8E5M2>, !torch.vtensor<[64,4],f8E8M0FNU>,
           !torch.vtensor<[256,128],f8E5M2>, !torch.vtensor<[256,4],f8E8M0FNU>,
           !torch.vtensor<[256],bf16>, !torch.none, !torch.int, !torch.int)
        -> !torch.vtensor<[64,256],bf16>
    return %0 : !torch.vtensor<[64,256],bf16>
  }

  // Invalid: out_dtype=Int (si32 = 3) is not a supported output type for block_scaled_dot;
  // only BFloat16 (15), Float (6), and Half (5) are accepted
  func.func @test_scaled_mm_invalid_out_dtype(
      %arg0: !torch.vtensor<[64,128],f8E5M2>,
      %arg1: !torch.vtensor<[64,4],f8E8M0FNU>,
      %arg2: !torch.vtensor<[256,128],f8E5M2>,
      %arg3: !torch.vtensor<[256,4],f8E8M0FNU>) -> !torch.vtensor<[64,256],si32> {
    %none = torch.constant.none
    %int3 = torch.constant.int 3
    %int23 = torch.constant.int 23
    // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
    %0 = torch.operator "torch.neuron._scaled_mm"(%arg0, %arg1, %arg2, %arg3, %none, %none, %int3, %int23)
        : (!torch.vtensor<[64,128],f8E5M2>, !torch.vtensor<[64,4],f8E8M0FNU>,
           !torch.vtensor<[256,128],f8E5M2>, !torch.vtensor<[256,4],f8E8M0FNU>,
           !torch.none, !torch.none, !torch.int, !torch.int)
        -> !torch.vtensor<[64,256],si32>
    return %0 : !torch.vtensor<[64,256],si32>
  }

  // Invalid: lhs_scale and rhs_scale are f32 instead of f8E8M0FNU
  func.func @test_scaled_mm_invalid_scale_dtype(
      %arg0: !torch.vtensor<[64,128],f8E5M2>,
      %arg1: !torch.vtensor<[64,4],f32>,
      %arg2: !torch.vtensor<[256,128],f8E5M2>,
      %arg3: !torch.vtensor<[256,4],f32>) -> !torch.vtensor<[64,256],bf16> {
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %int23 = torch.constant.int 23
    // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
    %0 = torch.operator "torch.neuron._scaled_mm"(%arg0, %arg1, %arg2, %arg3, %none, %none, %int15, %int23)
        : (!torch.vtensor<[64,128],f8E5M2>, !torch.vtensor<[64,4],f32>,
           !torch.vtensor<[256,128],f8E5M2>, !torch.vtensor<[256,4],f32>,
           !torch.none, !torch.none, !torch.int, !torch.int)
        -> !torch.vtensor<[64,256],bf16>
    return %0 : !torch.vtensor<[64,256],bf16>
  }

  // Invalid: lhs_scale contracting dim is 2 but expected K/32 = 128/32 = 4
  func.func @test_scaled_mm_invalid_scale_shape(
      %arg0: !torch.vtensor<[64,128],f8E5M2>,
      %arg1: !torch.vtensor<[64,2],f8E8M0FNU>,
      %arg2: !torch.vtensor<[256,128],f8E5M2>,
      %arg3: !torch.vtensor<[256,4],f8E8M0FNU>) -> !torch.vtensor<[64,256],bf16> {
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %int23 = torch.constant.int 23
    // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
    %0 = torch.operator "torch.neuron._scaled_mm"(%arg0, %arg1, %arg2, %arg3, %none, %none, %int15, %int23)
        : (!torch.vtensor<[64,128],f8E5M2>, !torch.vtensor<[64,2],f8E8M0FNU>,
           !torch.vtensor<[256,128],f8E5M2>, !torch.vtensor<[256,4],f8E8M0FNU>,
           !torch.none, !torch.none, !torch.int, !torch.int)
        -> !torch.vtensor<[64,256],bf16>
    return %0 : !torch.vtensor<[64,256],bf16>
  }

  // Invalid: K=33 is not divisible by block_size (32).
  func.func @test_scaled_mm_k_not_divisible_by_32(
      %arg0: !torch.vtensor<[64,33],f8E5M2>,
      %arg1: !torch.vtensor<[64,1],f8E8M0FNU>,
      %arg2: !torch.vtensor<[256,33],f8E5M2>,
      %arg3: !torch.vtensor<[256,1],f8E8M0FNU>) -> !torch.vtensor<[64,256],bf16> {
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %int23 = torch.constant.int 23
    // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
    %0 = torch.operator "torch.neuron._scaled_mm"(%arg0, %arg1, %arg2, %arg3, %none, %none, %int15, %int23)
        : (!torch.vtensor<[64,33],f8E5M2>, !torch.vtensor<[64,1],f8E8M0FNU>,
           !torch.vtensor<[256,33],f8E5M2>, !torch.vtensor<[256,1],f8E8M0FNU>,
           !torch.none, !torch.none, !torch.int, !torch.int)
        -> !torch.vtensor<[64,256],bf16>
    return %0 : !torch.vtensor<[64,256],bf16>
  }

  // Invalid: dynamic K is rejected (symmetric with the fused op).
  func.func @test_scaled_mm_dynamic_k(
      %arg0: !torch.vtensor<[64,?],f8E5M2>,
      %arg1: !torch.vtensor<[64,?],f8E8M0FNU>,
      %arg2: !torch.vtensor<[256,?],f8E5M2>,
      %arg3: !torch.vtensor<[256,?],f8E8M0FNU>) -> !torch.vtensor<[64,256],bf16> {
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %int23 = torch.constant.int 23
    // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
    %0 = torch.operator "torch.neuron._scaled_mm"(%arg0, %arg1, %arg2, %arg3, %none, %none, %int15, %int23)
        : (!torch.vtensor<[64,?],f8E5M2>, !torch.vtensor<[64,?],f8E8M0FNU>,
           !torch.vtensor<[256,?],f8E5M2>, !torch.vtensor<[256,?],f8E8M0FNU>,
           !torch.none, !torch.none, !torch.int, !torch.int)
        -> !torch.vtensor<[64,256],bf16>
    return %0 : !torch.vtensor<[64,256],bf16>
  }

}

// RUN: torch-mlir-opt -pass-pipeline='builtin.module(torch-raw-to-neuron-backend-pipeline)' -split-input-file %s | FileCheck %s



module {
  func.func @aliased_func(%arg0: !torch.vtensor<[1024,1024],bf16>) -> (!torch.vtensor<[1024,1024],bf16>, !torch.vtensor<[1024,1024],bf16>) {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %none_1 = torch.constant.none
    %false = torch.constant.bool false
    %none_2 = torch.constant.none
    %0 = torch.aten.ones_like %arg0, %none, %none_0, %none_1, %false, %none_2 : !torch.vtensor<[1024,1024],bf16>, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1024,1024],bf16>

    // CHECK: {{.*}}:2 = stablehlo.custom_call @AwsNeuronCustomNativeKernel({{.*}}, {{.*}}) {backend_config = "eyMSJ9", output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 1, operand_tuple_indices = []>]} : (tensor<1024x1024xbf16>, tensor<1024x1024xbf16>) -> (tensor<1024x1024xbf16>, tensor<1024x1024xbf16>)
    %2:2 = torch.operator "stablehlo.custom_call"(%0, %arg0) {backend_config = "eyMSJ9", call_target_name = "AwsNeuronCustomNativeKernel", operand_output_aliases = [[1, 0]]} : (!torch.vtensor<[1024,1024],bf16>, !torch.vtensor<[1024,1024],bf16>) -> (!torch.vtensor<[1024,1024],bf16>, !torch.vtensor<[1024,1024],bf16>)
    return %2#1, %2#0 : !torch.vtensor<[1024,1024],bf16>, !torch.vtensor<[1024,1024],bf16>
  }
}

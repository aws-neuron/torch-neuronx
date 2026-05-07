// RUN: torch-mlir-opt --torch-decompose-complex-ops -split-input-file %s | FileCheck %s

// Test that convolution_backward doesn't hang.

// CHECK-LABEL: @convolution_backward_with_dilation
// CHECK: return
func.func @convolution_backward_with_dilation(
    %grad_output: !torch.vtensor<[2,2,10,10],f32>,
    %input: !torch.vtensor<[2,3,14,14],f32>,
    %weight: !torch.vtensor<[2,3,3,3],f32>
) -> (!torch.vtensor<[2,3,14,14],f32>, !torch.vtensor<[2,3,3,3],f32>, !torch.vtensor<[2],f32>) {
    %int2 = torch.constant.int 2
    %bias_sizes = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
    %int1 = torch.constant.int 1
    %stride = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %padding = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %int3 = torch.constant.int 3
    %dilation = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %int0 = torch.constant.int 0
    %output_padding = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %groups = torch.constant.int 1
    %true = torch.constant.bool true
    %output_mask = torch.prim.ListConstruct %true, %true, %true : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>

    %result0, %result1, %result2 = torch.aten.convolution_backward %grad_output, %input, %weight, %bias_sizes, %stride, %padding, %dilation, %false, %output_padding, %groups, %output_mask : !torch.vtensor<[2,2,10,10],f32>, !torch.vtensor<[2,3,14,14],f32>, !torch.vtensor<[2,3,3,3],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int, !torch.list<bool> -> !torch.vtensor<[2,3,14,14],f32>, !torch.vtensor<[2,3,3,3],f32>, !torch.vtensor<[2],f32>

    return %result0, %result1, %result2 : !torch.vtensor<[2,3,14,14],f32>, !torch.vtensor<[2,3,3,3],f32>, !torch.vtensor<[2],f32>
}

# RUN: %PYTHON %s | FileCheck %s

import torch
import torch.nn as nn
from torch_mlir import fx
from torch_mlir.compiler_utils import (
    run_pipeline_with_repro_report,
    TorchMlirCompilerError,
)
from torch_mlir.compiler_utils import OutputType
from torch._dynamo.backends.common import aot_autograd


def compile(module, enable_ir_printing):
    def backend(gm, _):
        module = fx.stateless_fx_import(
            gm, output_type=OutputType.RAW, verbose=False, enable_ir_printing=False
        )
        # First pipeline call - same as backends.py
        run_pipeline_with_repro_report(
            module,
            "builtin.module(func.func(torch-match-quantized-custom-ops), torchdynamo-export-to-torch-backend-pipeline)",
            "Lowering Torch FX IR (RAW) -> Torch Backend IR",
            enable_ir_printing=enable_ir_printing,
        )
        # Second pipeline call - same as backends.py
        run_pipeline_with_repro_report(
            module,
            "builtin.module(torch-backend-to-stablehlo-backend-pipeline)",
            "Lowering Torch Backend IR -> StableHLO IR",
            enable_ir_printing=enable_ir_printing,
        )
        return gm

    with torch.no_grad():
        compiled = torch.compile(module, backend=aot_autograd(fw_compiler=backend))
    return compiled


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


class HardtanhBackwardModule(torch.nn.Module):
    def forward(self, grad_out, input):
        return torch.ops.aten.hardtanh_backward(
            grad_out, input, min_val=0.2, max_val=0.5
        )


@run
# CHECK-LABEL: test_repro_report_with_ir_printing_enabled
# CHECK: backend='compiler_fn' raised:
# CHECK: TorchMlirCompilerError:{{.*}}
# CHECK: Failed Torch-MLIR Lowering Torch Backend IR -> StableHLO IR: loc("{{.*}}/test_compiler_utils.py":{{[0-9]+}}:{{[0-9]+}}): error: failed to legalize operation 'torch.constant.float'
# CHECK: // -----// IR Dump After FinalizingBackendTypeConversionForStablehlo Failed (torch-finalizing-backend-type-conversion-for-stablehlo) //----- //
# CHECK: func.func @main(%arg0: tensor<10x20xf32>, %arg1: tensor<10x20xf32>) -> tensor<10x20xf32> {
# CHECK: %float5.000000e-01 = torch.constant.float 5.000000e-01
# CHECK: %float2.000000e-01 = torch.constant.float 2.000000e-01
# CHECK: %[[VAR0:.*]] = torch_c.from_builtin_tensor %arg1 : tensor<10x20xf32> -> !torch.vtensor<[10,20],f32>
# CHECK: %[[VAR1:.*]] = torch_c.from_builtin_tensor %arg0 : tensor<10x20xf32> -> !torch.vtensor<[10,20],f32>
# CHECK: %[[VAR2:.*]] = torch.aten.hardtanh_backward %[[VAR1]], %[[VAR0]], %float2.000000e-01, %float5.000000e-01 : !torch.vtensor<[10,20],f32>, !torch.vtensor<[10,20],f32>, !torch.float, !torch.float -> !torch.vtensor<[10,20],f32>
# CHECK: %[[VAR3:.*]] = torch_c.to_builtin_tensor %[[VAR2]] : !torch.vtensor<[10,20],f32> -> tensor<10x20xf32>
# CHECK: return %[[VAR3]] : tensor<10x20xf32>
# CHECK: }
# CHECK: For Torch-MLIR developers, the error can be reproduced with:
# CHECK: $ torch-mlir-opt -pass-pipeline='builtin.module(torch-backend-to-stablehlo-backend-pipeline)' {{.*}}.mlir
# CHECK: Add '-mlir-print-ir-after-all -mlir-disable-threading' to get the IR dump for debugging purpose.
def test_repro_report_with_ir_printing_enabled():
    """Test that run_pipeline_with_repro_report with enable_ir_printing=True includes reproduction instructions"""

    try:
        grad_out = torch.rand(10, 20)
        input_tensor = torch.rand(10, 20)
        m = compile(HardtanhBackwardModule(), True)
        m(grad_out, input_tensor)
    except Exception as e:
        error_message = str(e)
        print(error_message)


@run
# CHECK-LABEL: test_repro_report_with_ir_printing_disabled
# CHECK: backend='compiler_fn' raised:
# CHECK: TorchMlirCompilerError:{{.*}}
# CHECK: Failed Torch-MLIR Lowering Torch Backend IR -> StableHLO IR: loc("{{.*}}/test_compiler_utils.py":{{[0-9]+}}:{{[0-9]+}}): error: failed to legalize operation 'torch.constant.float'
# CHECK-NOT: // -----// IR Dump After FinalizingBackendTypeConversionForStablehlo Failed (torch-finalizing-backend-type-conversion-for-stablehlo) //----- //
# CHECK-NOT: func.func @main
# CHECK-NOT: For Torch-MLIR developers, the error can be reproduced with:
# CHECK-NOT: $ torch-mlir-opt -pass-pipeline='builtin.module(torch-backend-to-stablehlo-backend-pipeline)' {{.*}}.mlir
# CHECK-NOT: Add '-mlir-print-ir-after-all -mlir-disable-threading' to get the IR dump for debugging purpose.
def test_repro_report_with_ir_printing_disabled():
    """Test that run_pipeline_with_repro_report with enable_ir_printing=False does not include reproduction instructions"""

    try:
        grad_out = torch.rand(10, 20)
        input_tensor = torch.rand(10, 20)
        m = compile(HardtanhBackwardModule(), False)
        m(grad_out, input_tensor)
    except Exception as e:
        error_message = str(e)
        print(error_message)

// Force-link MLIR CAPI and torch-mlir symbols into _C.so.
// Uses targeted dialect/pass registration instead of RegisterAll*
// to avoid pulling in all LLVM backends (saves ~350MB).

#include "mlir-c/IR.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/Pass.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

#include "torch-mlir-c/Registration.h"
#include "torch-mlir-c/Dialects.h"

extern "C" {

void mlirRegisterAllDialects(MlirDialectRegistry registry) {
    mlir::DialectRegistry &reg = *unwrap(registry);
    // Register only dialects used by torch-mlir
    reg.insert<
        mlir::func::FuncDialect,
        mlir::arith::ArithDialect,
        mlir::tensor::TensorDialect,
        mlir::scf::SCFDialect,
        mlir::linalg::LinalgDialect,
        mlir::math::MathDialect,
        mlir::cf::ControlFlowDialect,
        mlir::complex::ComplexDialect,
        mlir::shape::ShapeDialect,
        mlir::bufferization::BufferizationDialect,
        mlir::memref::MemRefDialect,
        mlir::index::IndexDialect,
        mlir::LLVM::LLVMDialect,
        mlir::ml_program::MLProgramDialect,
        mlir::sparse_tensor::SparseTensorDialect,
        mlir::vector::VectorDialect,
        mlir::affine::AffineDialect,
        mlir::ub::UBDialect
    >();
}

void mlirRegisterAllPasses() {
    static bool registered = false;
    if (!registered) {
        mlir::registerTransformsPasses();
        mlir::registerConversionPasses();
        registered = true;
    }
}

void mlirRegisterAllLLVMTranslations(MlirContext context) {
    // No-op: torch-mlir doesn't need LLVM IR translations
}

} // extern "C"

// Force-link torch-mlir symbols needed by _torchMlir.so
volatile const void *_force_link_symbols[] = {
    (const void *)&torchMlirRegisterAllPasses,
    (const void *)&torchMlirRegisterAllDialects,
    (const void *)&mlirGetDialectHandle__torch__,
};

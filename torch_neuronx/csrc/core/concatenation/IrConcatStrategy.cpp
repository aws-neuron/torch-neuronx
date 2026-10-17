#include "IrConcatStrategy.h"

#include <memory>
#include <stdexcept>
#include <unordered_set>

#include "torch_neuronx/csrc/core/NeuronLogging.h"

namespace torch_neuronx {

bool MatMulToMatMulStrategy::IsFusibleBoundaryOperation(const std::string& op_name) {
  static const std::unordered_set<std::string> fusible_ops = {
      "aten::linear",     "aten::linear_backward",
      "aten::matmul",     "aten::matmul_backward",
      "aten::mm",         "aten::bmm",
      "aten::addmm",      "aten::baddbmm",
      "aten::addbmm",     "aten::conv2d",
      "nki_kernel_global"};

  return fusible_ops.count(op_name) > 0;
}

}  // namespace torch_neuronx

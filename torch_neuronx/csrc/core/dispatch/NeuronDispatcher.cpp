#include <ATen/native/CPUFallback.h>
#include <ATen/native/transformers/sdp_utils.h>
#include <ATen/ops/nonzero_numpy.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <limits>
#include <numeric>
#include <sstream>

#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/NeuronOpTracking.h"
#include "torch_neuronx/csrc/core/NeuronOpsApi.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/cache/CacheKeyGenerator.h"
#include "torch_neuronx/csrc/core/dispatch/DispatchUtils.h"
#include "torch_neuronx/csrc/core/dispatch/DtypeUtils.h"
#include "torch_neuronx/csrc/core/dispatch/OutputProcessing.h"
#include "torch_neuronx/csrc/core/dispatch/ParseLoweringResult.h"
#include "torch_neuronx/csrc/core/dispatch/PreprocessArgs.h"
#include "torch_neuronx/csrc/ops/InternalOps.h"
#include "torch_neuronx/csrc/utils/CopyUtils.h"

namespace torch_neuronx {
namespace dispatch {

// RAII guard for setting/clearing the dispatch cache key on the Python
// threading.local context.  Guarantees cleanup on all exit paths.
struct DispatchContextGuard {
  py::object ctx_;
  DispatchContextGuard(py::object ctx, const std::string& key) : ctx_(std::move(ctx)) {
    ctx_.attr("cache_key") = py::cast(key);
  }
  ~DispatchContextGuard() { ctx_.attr("cache_key") = py::none(); }
};

// Flag: when set, ops that cannot run on Neuron raise an error instead of
// falling back to CPU.  Used in multiple dispatch functions.
static const bool kFallbackOnlyForUnimplemented = []() {
  const char* env = std::getenv("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS");
  return env != nullptr && std::string(env) == "1";
}();

// Cached Python callable for hierarchy capture. Stored as a raw PyObject*
// (no C++ destructor) to avoid Py_XDECREF during static destruction after
// the Python interpreter has been finalized.
static PyObject* g_hierarchy_fn = nullptr;

// ═══════════════════════════════════════════════════════════════
//  Singleton
// ═══════════════════════════════════════════════════════════════

NeuronDispatcher& NeuronDispatcher::Instance() {
  static NeuronDispatcher instance;
  return instance;
}

void NeuronDispatcher::Cleanup() {
  op_registrations_.clear();
  dispatch_lib_.reset();
  dispatch_cache_.clear();
  // Release cached Python callable before interpreter finalization.
  if (g_hierarchy_fn) {
    if (Py_IsInitialized()) {
      py::gil_scoped_acquire gil;
      PyObject* tmp = g_hierarchy_fn;
      g_hierarchy_fn = nullptr;
      Py_DECREF(tmp);
    } else {
      g_hierarchy_fn = nullptr;
    }
  }
}

torch::Library& NeuronDispatcher::GetDispatchLib() {
  if (!dispatch_lib_) {
    dispatch_lib_ = std::make_unique<torch::Library>(torch::Library::IMPL, "aten",
                                                     c10::DispatchKey::PrivateUse1,
                                                     /*file=*/"NeuronDispatcher.cpp", /*line=*/0);
  }
  return *dispatch_lib_;
}

// ═══════════════════════════════════════════════════════════════
//  Registration
// ═══════════════════════════════════════════════════════════════

void NeuronDispatcher::RegisterOp(const std::string& op_name, py::object python_wrapper,
                                  py::object lower_func, std::vector<int64_t> static_argnums_vec,
                                  std::vector<std::string> static_argnames_vec, bool skip_cache) {
  std::unordered_set<size_t> sa_set(static_argnums_vec.begin(), static_argnums_vec.end());
  std::unordered_set<std::string> sn_set(static_argnames_vec.begin(), static_argnames_vec.end());

  bool has_lower = !lower_func.is_none();
  auto reg = std::make_shared<OpRegistration>(
      OpRegistration{std::move(python_wrapper), std::move(lower_func), std::move(sa_set),
                     std::move(sn_set), skip_cache, has_lower});

  op_registrations_[op_name] = reg;

  auto& lib = GetDispatchLib();

  // Ops that bypass the full lowering pipeline and use a dedicated handler.
  static const std::unordered_set<std::string> kScalarTensorOps = {
      "scalar_tensor",
  };
  static const std::unordered_set<std::string> kCopyOps = {
      "copy_",
  };
  static const std::unordered_set<std::string> kToCopyOps = {
      "_to_copy",
  };
  static const std::unordered_set<std::string> kContiguousOps = {
      "contiguous",
  };
  static const std::unordered_set<std::string> kNonzeroOps = {
      "nonzero",
      "nonzero.out",
  };
  static const std::unordered_set<std::string> kIndexOps = {
      "index.Tensor",
      "index.Tensor_out",
  };
  static const std::unordered_set<std::string> kSdpaFwdOps = {
      "_scaled_dot_product_fused_attention_overrideable",
  };
  static const std::unordered_set<std::string> kSdpaBwdOps = {
      "_scaled_dot_product_fused_attention_overrideable_backward",
  };

  // Strip "aten::" prefix if present for the lookup.
  std::string bare_name = op_name;
  if (bare_name.rfind("aten::", 0) == 0) {
    bare_name = bare_name.substr(6);
  }

  // Each handler captures the shared_ptr<OpRegistration> via BoxedDispatchFunctor,
  // so the dispatch hot path accesses it directly — no map lookup,
  // no py::object copy/refcount overhead.
  if (kScalarTensorOps.count(bare_name)) {
    TORCH_NEURONX_DISPATCH_DEBUG(
        "[NeuronDispatcher::RegisterOp] registered DispatchScalarTensor handler", " op=", op_name);
    // scalar_tensor doesn't use the registration, keep static function.
    lib.impl(op_name.c_str(),
             torch::CppFunction::makeFromBoxedFunction<&NeuronDispatcher::DispatchScalarTensor>());
  } else if (kCopyOps.count(bare_name)) {
    TORCH_NEURONX_DISPATCH_DEBUG("[NeuronDispatcher::RegisterOp] registered DispatchCopy handler",
                                 " op=", op_name);
    lib.impl(op_name.c_str(),
             torch::CppFunction::makeFromBoxedFunctor(
                 std::make_unique<BoxedDispatchFunctor>(reg, &NeuronDispatcher::DispatchCopyImpl)));
  } else if (kToCopyOps.count(bare_name)) {
    TORCH_NEURONX_DISPATCH_DEBUG("[NeuronDispatcher::RegisterOp] registered DispatchToCopy handler",
                                 " op=", op_name);
    lib.impl(op_name.c_str(),
             torch::CppFunction::makeFromBoxedFunctor(std::make_unique<BoxedDispatchFunctor>(
                 reg, &NeuronDispatcher::DispatchToCopyImpl)));
  } else if (kContiguousOps.count(bare_name)) {
    TORCH_NEURONX_DISPATCH_DEBUG(
        "[NeuronDispatcher::RegisterOp] registered DispatchContiguous handler", " op=", op_name);
    lib.impl(op_name.c_str(),
             torch::CppFunction::makeFromBoxedFunctor(std::make_unique<BoxedDispatchFunctor>(
                 reg, &NeuronDispatcher::DispatchContiguousImpl)));
  } else if (kNonzeroOps.count(bare_name)) {
    TORCH_NEURONX_DISPATCH_DEBUG(
        "[NeuronDispatcher::RegisterOp] registered DispatchNonzero handler", " op=", op_name);
    lib.impl(op_name.c_str(),
             torch::CppFunction::makeFromBoxedFunctor(std::make_unique<BoxedDispatchFunctor>(
                 reg, &NeuronDispatcher::DispatchNonzeroImpl)));
  } else if (kIndexOps.count(bare_name)) {
    TORCH_NEURONX_DISPATCH_DEBUG("[NeuronDispatcher::RegisterOp] registered DispatchIndex handler",
                                 " op=", op_name);
    lib.impl(op_name.c_str(),
             torch::CppFunction::makeFromBoxedFunctor(std::make_unique<BoxedDispatchFunctor>(
                 reg, &NeuronDispatcher::DispatchIndexImpl)));
  } else if (kSdpaFwdOps.count(bare_name)) {
    TORCH_NEURONX_DISPATCH_DEBUG(
        "[NeuronDispatcher::RegisterOp] registered DispatchSdpaFwd handler", " op=", op_name);
    lib.impl(op_name.c_str(),
             torch::CppFunction::makeFromBoxedFunctor(std::make_unique<BoxedDispatchFunctor>(
                 reg, &NeuronDispatcher::DispatchSdpaFwdImpl)));
  } else if (kSdpaBwdOps.count(bare_name)) {
    TORCH_NEURONX_DISPATCH_DEBUG(
        "[NeuronDispatcher::RegisterOp] registered DispatchSdpaBwd handler", " op=", op_name);
    lib.impl(op_name.c_str(),
             torch::CppFunction::makeFromBoxedFunctor(std::make_unique<BoxedDispatchFunctor>(
                 reg, &NeuronDispatcher::DispatchSdpaBwdImpl)));
  } else {
    TORCH_NEURONX_DISPATCH_DEBUG("[NeuronDispatcher::RegisterOp] registered C++ boxed handler",
                                 " op=", op_name, " has_lower=", !reg->lower_func.is_none(),
                                 " total_registered=", op_registrations_.size());
    lib.impl(op_name.c_str(),
             torch::CppFunction::makeFromBoxedFunctor(
                 std::make_unique<BoxedDispatchFunctor>(reg, &NeuronDispatcher::DispatchImpl)));
  }
}

std::vector<std::string> NeuronDispatcher::GetRegisteredOps() const {
  std::vector<std::string> ops;
  ops.reserve(op_registrations_.size());
  for (const auto& [name, _] : op_registrations_) {
    ops.push_back(name);
  }
  return ops;
}

// ═══════════════════════════════════════════════════════════════
//  Scalar tensor shortcircuit: create on CPU, copy to Neuron
// ═══════════════════════════════════════════════════════════════

void NeuronDispatcher::DispatchScalarTensor(const c10::OperatorHandle& op,
                                            c10::DispatchKeySet dispatch_keys,
                                            torch::jit::Stack* stack) {
  // Schema: aten::scalar_tensor(Scalar s, *, ScalarType? dtype=None,
  //           Layout? layout=None, Device? device=None, bool? pin_memory=None)
  auto schema = op.schema();
  size_t num_args = schema.arguments().size();
  auto args = torch::jit::pop(*stack, num_args);

  auto scalar = args[0].toScalar();

  // Capture the target Neuron device (from the device kwarg, or current device)
  torch::Device target_device(c10::DeviceType::PrivateUse1,
                              static_cast<c10::DeviceIndex>(at::neuron::current_device()));
  if (args.size() > 3 && !args[3].isNone()) {
    target_device = args[3].toDevice();
  }

  // Build CPU tensor options, forwarding all original kwargs except device
  auto opts = torch::TensorOptions().device(torch::kCPU);
  if (args.size() > 1 && !args[1].isNone()) {
    opts = opts.dtype(args[1].toScalarType());
  }
  if (args.size() > 2 && !args[2].isNone()) {
    opts = opts.layout(args[2].toLayout());
  }
  if (args.size() > 4 && !args[4].isNone()) {
    opts = opts.pinned_memory(args[4].toBool());
  }

  // Delegate to CPU – let PyTorch handle all dtype inference
  auto cpu_tensor = at::scalar_tensor(scalar, opts);

  // Copy CPU → Neuron (non-blocking to allow overlap with other work)
  auto neuron_tensor = cpu_tensor.to(target_device, /*non_blocking=*/true);

  // Log for op tracking so test framework sees the op as executed.
  logOperationExecuted("aten::scalar_tensor");

  TORCH_NEURONX_DISPATCH_DEBUG(
      "[DispatchScalarTensor] scalar=", scalar.toDouble(), " cpu_dtype=", cpu_tensor.scalar_type(),
      " target_device=", target_device.str(), " neuron_dtype=", neuron_tensor.scalar_type());

  torch::jit::push(*stack, neuron_tensor);
}

// ═══════════════════════════════════════════════════════════════
//  Helper: copy Neuron tensor to contiguous CPU tensor via raw NRT.
//  Makes src contiguous on device (if needed), then copies to CPU.
//  Allocates pinned memory when non_blocking=true for async DMA.
// ═══════════════════════════════════════════════════════════════
static torch::Tensor NeuronToContiguousCPU(const at::Tensor& src, bool non_blocking = false) {
  auto opts = torch::TensorOptions().dtype(src.scalar_type()).device(torch::kCPU);
  if (non_blocking) opts = opts.pinned_memory(true);
  if (src.is_contiguous()) {
    auto cpu = torch::empty(src.sizes(), opts);
    torch_neuronx::utils::copy_neuron_to_cpu(src, cpu, non_blocking);
    return cpu;
  }

  // Non-contiguous: copy full storage to CPU, restride there.
  // Cannot use contiguous_internal on device here — it may recurse back
  // through _to_copy → NeuronToContiguousCPU for non-lowerable patterns.
  auto storage_bytes = src.storage().nbytes();
  auto elem_size = src.element_size();
  int64_t storage_numel = static_cast<int64_t>(storage_bytes / elem_size);

  auto flat_neuron = src.as_strided({storage_numel}, {1}, 0);
  auto flat_cpu = torch::empty({storage_numel},
                               torch::TensorOptions().dtype(src.scalar_type()).device(torch::kCPU));
  torch_neuronx::utils::copy_neuron_to_cpu(flat_neuron, flat_cpu, /*non_blocking=*/false);

  auto result = flat_cpu.as_strided(src.sizes().vec(), src.strides().vec(), src.storage_offset())
                    .contiguous();
  if (non_blocking) result = result.pin_memory(at::kPrivateUse1);
  return result;
}

// ═══════════════════════════════════════════════════════════════
//  Shared NRT copy helpers — used by DispatchCopyImpl and
//  DispatchToCopyImpl.  All follow the Copy{Src}To{Dst} pattern.
// ═══════════════════════════════════════════════════════════════

/// Copy a Neuron tensor to a contiguous CPU tensor, casting to target_dtype.
static torch::Tensor CopyNeuronToCPU(const at::Tensor& src, c10::ScalarType target_dtype,
                                     bool non_blocking = false) {
  auto cpu = NeuronToContiguousCPU(src, non_blocking);
  if (cpu.scalar_type() != target_dtype) {
    // Synchronize before reading data for the dtype cast — the non-blocking
    // copy may still be in flight.
    auto stream = at::neuron::getCurrentNeuronStream(src.device().index());
    stream.synchronize();
    cpu = cpu.to(target_dtype);
  }
  return cpu;
}

/// Copy a CPU tensor into a new Neuron tensor, returning the new tensor.
static torch::Tensor CopyCPUToNewNeuron(const at::Tensor& cpu_src, torch::Device device,
                                        bool non_blocking) {
  auto dst = torch::empty(cpu_src.sizes(),
                          torch::TensorOptions().dtype(cpu_src.scalar_type()).device(device));
  torch_neuronx::utils::copy_cpu_to_neuron(cpu_src, dst, non_blocking);
  return dst;
}

/// Copy a CPU tensor into an existing Neuron tensor (handles non-contiguous dst).
static void CopyCPUIntoNeuron(const at::Tensor& cpu_src, at::Tensor& dst, bool non_blocking) {
  if (dst.is_contiguous()) {
    torch_neuronx::utils::copy_cpu_to_neuron(cpu_src, dst, non_blocking);
  } else {
    // Upload to a contiguous temp on Neuron, then use on-device stride matching.
    // Note: non_blocking is intentionally not forwarded for the temp upload —
    // CopyToNonContiguous depends on the data being present in the temp tensor,
    // and the temp is transient (not visible to the caller).
    auto temp = torch::empty(
        cpu_src.sizes(), torch::TensorOptions().dtype(cpu_src.scalar_type()).device(dst.device()));
    torch_neuronx::utils::copy_cpu_to_neuron(cpu_src, temp, /*non_blocking=*/false);
    torch_neuronx::utils::CopyToNonContiguous(dst, temp);
  }
}

/// Cast a CPU tensor to target_dtype and make contiguous — prepares for NRT upload.
static torch::Tensor CastContiguousCPU(const at::Tensor& src, c10::ScalarType target_dtype) {
  auto out = src;
  if (out.scalar_type() != target_dtype) {
    out = out.to(target_dtype);
  }
  if (!out.is_contiguous()) {
    out = out.contiguous();
  }
  return out;
}

/// Dtype supported by the MLIR aten::copy_ lowering (excludes Bool).
static bool IsCopyMlirDtypeSupported(c10::ScalarType dt) {
  return IsSupportedDtype(dt) && dt != c10::ScalarType::Bool && !c10::isComplexType(dt);
}

/// Dtype supported by the MLIR aten::_to_copy lowering.
/// Only float16, bfloat16, float32 (matches Python to_copy.py unsupported_dtypes).
static bool IsToCopyMlirDtypeSupported(c10::ScalarType dt) {
  static const bool native_int64 = IsNativeInt64Enabled();
  return dt == c10::ScalarType::Half || dt == c10::ScalarType::BFloat16 ||
         dt == c10::ScalarType::Float || (native_int64 && dt == c10::ScalarType::Long);
}

/// Returns true when on a non-default stream with host-cc enabled.
/// Mirrors Python can_handle() guard: `not (is_host_cc_enabled() and
/// current_stream() != default_stream())`.
static bool IsNonDefaultStreamWithHostCC(c10::DeviceIndex device_index) {
  static const bool host_cc_enabled = [] {
    const char* env = std::getenv("TORCH_NEURONX_ENABLE_HOST_CC");
    return env != nullptr && std::string(env) == "1";
  }();
  if (!host_cc_enabled) return false;
  auto current = at::neuron::getCurrentNeuronStream(device_index);
  auto default_s = at::neuron::getDefaultNeuronStream(device_index);
  return current != default_s;
}

// ═══════════════════════════════════════════════════════════════
//  aten::copy_ — CPU↔Neuron via NRT, Neuron→Neuron via MLIR or NRT
// ═══════════════════════════════════════════════════════════════

void NeuronDispatcher::DispatchCopyImpl(const c10::OperatorHandle& op,
                                        c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack,
                                        const OpRegistration& reg) {
  // Schema: aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
  auto schema = op.schema();
  size_t num_args = schema.arguments().size();
  auto args = torch::jit::pop(*stack, num_args);

  auto self_tensor = args[0].toTensor();
  auto src_ivalue = args[1];
  bool non_blocking = (args.size() > 2 && !args[2].isNone()) ? args[2].toBool() : false;

  // Handle scalar src → CPU tensor
  torch::Tensor src;
  if (src_ivalue.isScalar()) {
    auto scalar = src_ivalue.toScalar();
    src = at::scalar_tensor(
              scalar, torch::TensorOptions().dtype(self_tensor.scalar_type()).device(torch::kCPU))
              .expand(self_tensor.sizes())
              .contiguous();
  } else {
    src = src_ivalue.toTensor();
  }

  TORCH_NEURONX_DISPATCH_DEBUG("[DispatchCopy] ENTER", " dst_device=", self_tensor.device(),
                               " dst_dtype=", self_tensor.scalar_type(),
                               " dst_shape=", self_tensor.sizes(), " src_device=", src.device(),
                               " src_dtype=", src.scalar_type(), " src_shape=", src.sizes(),
                               " non_blocking=", non_blocking);

  // ── Empty tensor shortcircuit: no-op ──
  if (self_tensor.numel() == 0) {
    logOperationExecuted("aten::copy_");
    torch::jit::push(*stack, self_tensor);
    return;
  }

  // ── Self-copy shortcircuit ──
  if (self_tensor.data_ptr() == src.data_ptr() &&
      self_tensor.storage_offset() == src.storage_offset() && self_tensor.sizes() == src.sizes()) {
    logOperationExecuted("aten::copy_");
    torch::jit::push(*stack, self_tensor);
    return;
  }

  bool needs_broadcast = (self_tensor.sizes() != src.sizes());
  bool src_is_neuron = src.is_privateuseone();
  bool dst_is_neuron = self_tensor.is_privateuseone();

  // ── CPU → Neuron ──
  if (src.device().is_cpu() && dst_is_neuron) {
    // Broadcast on CPU (free and correct), then copy to Neuron
    auto cpu_src = needs_broadcast ? src.expand_as(self_tensor).contiguous() : src;
    auto cpu_prepared = CastContiguousCPU(cpu_src, self_tensor.scalar_type());
    CopyCPUIntoNeuron(cpu_prepared, self_tensor, non_blocking);
    logOperationExecuted("aten::copy_");
    torch::jit::push(*stack, self_tensor);
    return;
  }

  // ── Neuron → CPU ──
  if (src_is_neuron && self_tensor.device().is_cpu()) {
    // Copy small src to CPU first, then let CPU handle broadcast natively
    auto src_prepared = src.is_contiguous() ? src
                                            : torch_neuronx::ops::contiguous_internal(
                                                  src, c10::MemoryFormat::Contiguous);
    auto tmp_cpu = CopyNeuronToCPU(src_prepared, self_tensor.scalar_type(), /*non_blocking=*/false);
    // CPU copy_ handles broadcasting natively
    self_tensor.copy_(tmp_cpu, non_blocking);
    logOperationExecuted("aten::copy_");
    torch::jit::push(*stack, self_tensor);
    return;
  }

  // ── Neuron → Neuron ──
  //
  // Three tiers, most optimal first:
  //   1. MLIR pipeline  — contiguous dst, supported dtypes, default stream
  //   2. NRT on-device  — non-contiguous dst, same dtype, same shape
  //   3. NRT CPU bounce — everything else
  if (src_is_neuron && dst_is_neuron) {
    bool mlir_can_handle = self_tensor.dim() > 0 && self_tensor.is_contiguous() &&
                           IsCopyMlirDtypeSupported(src.scalar_type()) &&
                           IsCopyMlirDtypeSupported(self_tensor.scalar_type()) &&
                           !IsNonDefaultStreamWithHostCC(self_tensor.device().index());

    // ── 1. MLIR lowering pipeline (most optimal) ──
    // Mirrors Python CopyMLIRImpl.can_handle(): contiguous dst, supported
    // dtypes, on default stream (or host-cc disabled).
    if (mlir_can_handle) {
      if (needs_broadcast) {
        if (!src.is_contiguous()) {
          src = torch_neuronx::ops::contiguous_internal(src, c10::MemoryFormat::Contiguous);
        }
        src = src.expand_as(self_tensor).contiguous();
        args[1] = c10::IValue(src);
      }
      // Replace non_blocking (arg[2]) with None — it's a bool scalar that
      // PreprocessArgs would convert to a 0-dim Bool tensor on Neuron.
      // copy_lower only uses args[0] (dest) and args[1] (src).
      if (args.size() > 2) {
        args[2] = c10::IValue();
      }
      for (auto& arg : args) {
        torch::jit::push(*stack, arg);
      }
      NeuronDispatcher::DispatchImpl(op, dispatch_keys, stack, reg);
      return;
    }

    // ── 2. Non-contiguous dst: on-device stride matching via NRT ──
    if (!self_tensor.is_contiguous()) {
      // Dtype mismatch or unsupported dtype → CPU bounce
      if (src.scalar_type() != self_tensor.scalar_type() ||
          !IsCopyMlirDtypeSupported(src.scalar_type()) ||
          !IsCopyMlirDtypeSupported(self_tensor.scalar_type())) {
        auto tmp_cpu = CopyNeuronToCPU(src, self_tensor.scalar_type());
        if (needs_broadcast) {
          tmp_cpu = tmp_cpu.expand_as(self_tensor).contiguous();
        }
        CopyCPUIntoNeuron(tmp_cpu, self_tensor, /*non_blocking=*/false);
        logOperationExecuted("aten::copy_");
        torch::jit::push(*stack, self_tensor);
        return;
      }

      // Broadcast + non-contiguous dst → CPU bounce
      if (needs_broadcast) {
        auto tmp_cpu = CopyNeuronToCPU(src, self_tensor.scalar_type());
        tmp_cpu = tmp_cpu.expand_as(self_tensor).contiguous();
        CopyCPUIntoNeuron(tmp_cpu, self_tensor, /*non_blocking=*/false);
        logOperationExecuted("aten::copy_");
        torch::jit::push(*stack, self_tensor);
        return;
      }

      // Same dtype, same shape — hot path (SDPA attention).
      // On-device transpose + flat D2D copy.
      auto src_prepared = src.is_contiguous() ? src
                                              : torch_neuronx::ops::contiguous_internal(
                                                    src, c10::MemoryFormat::Contiguous);
      torch_neuronx::utils::CopyToNonContiguous(self_tensor, src_prepared);
      logOperationExecuted("aten::copy_");
      torch::jit::push(*stack, self_tensor);
      return;
    }

    // ── 3. NRT CPU bounce (least optimal) ──
    // Contiguous dst but MLIR unavailable: unsupported dtype or non-default
    // stream.  CPU bounce handles dtype cast + broadcast.
    TORCH_NEURONX_DISPATCH_DEBUG("[DispatchCopy] Neuron→Neuron NRT CPU bounce");
    auto tmp_cpu = CopyNeuronToCPU(src, self_tensor.scalar_type());
    if (needs_broadcast) {
      tmp_cpu = tmp_cpu.expand_as(self_tensor).contiguous();
    }
    CopyCPUIntoNeuron(tmp_cpu, self_tensor, /*non_blocking=*/false);
    logOperationExecuted("aten::copy_");
    torch::jit::push(*stack, self_tensor);
    return;
  }

  // ── Unhandled case ──
  // If we reach here, the dispatch branches above have a gap.  Fail loudly
  // instead of calling cpu_fallback, which would trigger recursive copy_
  // dispatch.
  TORCH_CHECK(false, "DispatchCopy: unhandled case — src_device=", src.device(),
              " src_dtype=", src.scalar_type(), " dst_device=", self_tensor.device(),
              " dst_dtype=", self_tensor.scalar_type(), " non_blocking=", non_blocking,
              " needs_broadcast=", needs_broadcast);
}

// ═══════════════════════════════════════════════════════════════
//  aten::_to_copy — CPU↔Neuron via NRT, Neuron→Neuron via MLIR or NRT
// ═══════════════════════════════════════════════════════════════

void NeuronDispatcher::DispatchToCopyImpl(const c10::OperatorHandle& op,
                                          c10::DispatchKeySet dispatch_keys,
                                          torch::jit::Stack* stack, const OpRegistration& reg) {
  // Schema: aten::_to_copy(Tensor self, *, ScalarType? dtype=None,
  //           Layout? layout=None, Device? device=None,
  //           bool? pin_memory=None, bool non_blocking=False,
  //           MemoryFormat? memory_format=None) -> Tensor
  auto schema = op.schema();
  size_t num_args = schema.arguments().size();
  auto args = torch::jit::pop(*stack, num_args);

  auto self_tensor = args[0].toTensor();

  // Resolve kwargs by schema position
  c10::optional<c10::ScalarType> dtype_opt;
  c10::optional<torch::Device> device_opt;
  c10::optional<c10::MemoryFormat> memory_format_opt;
  bool non_blocking = false;

  const auto& schema_args = schema.arguments();

  for (size_t i = 1; i < args.size() && i < schema_args.size(); ++i) {
    const auto& name = schema_args[i].name();
    if (name == "dtype" && !args[i].isNone()) {
      dtype_opt = args[i].toScalarType();
    } else if (name == "device" && !args[i].isNone()) {
      device_opt = args[i].toDevice();
    } else if (name == "non_blocking" && !args[i].isNone()) {
      non_blocking = args[i].toBool();
    } else if (name == "memory_format" && !args[i].isNone()) {
      memory_format_opt = args[i].toMemoryFormat();
    }
  }

  auto target_dtype = dtype_opt.value_or(self_tensor.scalar_type());
  auto target_device = device_opt.value_or(self_tensor.device());

  TORCH_NEURONX_DISPATCH_DEBUG(
      "[DispatchToCopy] ENTER", " src_device=", self_tensor.device(),
      " src_dtype=", self_tensor.scalar_type(), " src_shape=", self_tensor.sizes(),
      " target_device=", target_device, " target_dtype=", target_dtype,
      " non_blocking=", non_blocking, " memory_format=",
      (memory_format_opt.has_value() ? c10::toString(*memory_format_opt) : "None"));

  bool src_is_neuron = self_tensor.is_privateuseone();
  bool dst_is_neuron = (target_device.type() == c10::DeviceType::PrivateUse1);

  // ── No-op fast path ──
  // Return the tensor directly instead of cloning — _to_copy with identical
  // dtype and device is a no-op.  Matches Python to_copy.py:89-95 which
  // returns self_tensor without copying.
  // IMPORTANT: Only take this shortcut when memory_format is not requesting a
  // layout change (None or Preserve), OR the tensor is already contiguous in
  // the requested format.  If a concrete format like contiguous_format is
  // requested on a non-contiguous tensor, we must fall through so the tensor
  // is made contiguous — otherwise callers (e.g. DDP gradient bucketing in
  // FSDP+TP 2D composability) that expect contiguous data will access
  // out-of-bounds memory.
  bool no_format_change = !memory_format_opt.has_value() ||
                          *memory_format_opt == c10::MemoryFormat::Preserve ||
                          self_tensor.is_contiguous(*memory_format_opt);
  if (target_dtype == self_tensor.scalar_type() && target_device == self_tensor.device() &&
      no_format_change) {
    logOperationExecuted("aten::_to_copy");
    torch::jit::push(*stack, self_tensor);
    return;
  }

  // ── Empty tensor shortcircuit ──
  if (self_tensor.numel() == 0) {
    auto result = torch::empty(self_tensor.sizes(),
                               torch::TensorOptions().dtype(target_dtype).device(target_device));
    logOperationExecuted("aten::_to_copy");
    torch::jit::push(*stack, result);
    return;
  }

  // ── CPU → Neuron ──
  if (self_tensor.device().is_cpu() && dst_is_neuron) {
    auto src_strides = self_tensor.strides().vec();
    auto cpu_prepared = CastContiguousCPU(self_tensor, target_dtype);
    auto dst = CopyCPUToNewNeuron(cpu_prepared, target_device, non_blocking);
    // Preserve original strides (e.g. from transpose)
    if (!self_tensor.is_contiguous() && dst.strides().vec() != src_strides) {
      auto strided =
          torch::empty_strided(self_tensor.sizes(), src_strides,
                               torch::TensorOptions().dtype(target_dtype).device(target_device));
      torch_neuronx::utils::copy_strided_neuron_to_neuron(dst, strided);
      dst = strided;
    }
    logOperationExecuted("aten::_to_copy");
    torch::jit::push(*stack, dst);
    return;
  }

  // ── Neuron → CPU ──
  if (src_is_neuron && target_device.is_cpu()) {
    auto cpu_result = CopyNeuronToCPU(self_tensor, target_dtype, non_blocking);
    logOperationExecuted("aten::_to_copy");
    torch::jit::push(*stack, cpu_result);
    return;
  }

  // ── Neuron → Neuron ──
  //
  // Two tiers, most optimal first:
  //   1. MLIR pipeline  — supported dtypes, default stream
  //   2. NRT CPU bounce — unsupported dtypes or non-default stream
  if (src_is_neuron && dst_is_neuron) {
    bool mlir_can_handle = IsToCopyMlirDtypeSupported(self_tensor.scalar_type()) &&
                           IsToCopyMlirDtypeSupported(target_dtype) &&
                           !IsNonDefaultStreamWithHostCC(self_tensor.device().index());

    // ── 1. MLIR lowering pipeline (most optimal) ──
    if (mlir_can_handle) {
      TORCH_NEURONX_DISPATCH_DEBUG(
          "[DispatchToCopy] N2N MLIR", " src_dtype=", self_tensor.scalar_type(),
          " target_dtype=", target_dtype, " src_shape=", self_tensor.sizes());
      for (auto& arg : args) {
        torch::jit::push(*stack, arg);
      }
      NeuronDispatcher::DispatchImpl(op, dispatch_keys, stack, reg);
      return;
    }

    // ── 2. NRT CPU bounce ──
    TORCH_NEURONX_DISPATCH_DEBUG("[DispatchToCopy] Neuron→Neuron NRT CPU bounce");
    auto cpu_tmp = CopyNeuronToCPU(self_tensor, target_dtype);
    auto dst = CopyCPUToNewNeuron(cpu_tmp, target_device, /*non_blocking=*/false);
    logOperationExecuted("aten::_to_copy");
    torch::jit::push(*stack, dst);
    return;
  }

  // ── Unhandled case ──
  // If we reach here, the dispatch branches above have a gap.  Fail loudly
  // instead of calling cpu_fallback, which would trigger recursive _to_copy
  // dispatch.
  TORCH_CHECK(false, "DispatchToCopy: unhandled case — src_device=", self_tensor.device(),
              " src_dtype=", self_tensor.scalar_type(), " target_device=", target_device,
              " target_dtype=", target_dtype, " non_blocking=", non_blocking);
}

// ═══════════════════════════════════════════════════════════════
//  Main Dispatch Pipeline
// ═══════════════════════════════════════════════════════════════

void NeuronDispatcher::DispatchImpl(const c10::OperatorHandle& op,
                                    c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack,
                                    const OpRegistration& reg) {
  auto op_name = c10::toString(op.schema().operator_name());
  std::string lookup_name = op_name;
  if (lookup_name.rfind("aten::", 0) == 0) {
    lookup_name = lookup_name.substr(6);
  }

  auto& dispatcher = NeuronDispatcher::Instance();

  TORCH_NEURONX_DISPATCH_DEBUG("[NeuronDispatcher::Dispatch] ENTER", " op=", op_name,
                               " lookup=", lookup_name);

  auto schema = op.schema();
  auto arguments = schema.arguments();
  size_t num_args = arguments.size();
  auto args = torch::jit::pop(*stack, num_args);

  TORCH_NEURONX_DISPATCH_DEBUG("[NeuronDispatcher::Dispatch] popped args", " op=", op_name,
                               " num_args=", num_args, " has_lower=", !reg.lower_func.is_none());

  // Empty tensors (numel()==0) cannot be compiled by the Neuron compiler.
  if (TryHandleEmptyTensors(op, stack, args, schema, op_name)) {
    return;
  }

  // Complex dtype allowlist: only ops verified to lower and execute correctly
  // with complex tensors are allowed through. All others fall back to CPU.
  // With TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1, raise only when
  // a complex tensor is at an input position (not just an out= tensor).
  if (HasComplexInput(args)) {
    static const std::unordered_set<std::string> kComplexSupportedOps = {
        "mul.Tensor", "mul.Scalar", "mul.out",      "mul.Scalar_out", "mul_",
        "div.Tensor", "div.out",    "div.out_mode", "div_.Tensor",
    };
    if (kComplexSupportedOps.find(lookup_name) == kComplexSupportedOps.end()) {
      if (kFallbackOnlyForUnimplemented && HasComplexNonOutputInput(args, schema)) {
        RaiseComplexFallbackDisabled(op_name);
      }
      TORCH_NEURONX_DISPATCH_DEBUG(
          "[NeuronDispatcher::Dispatch] complex input not in allowlist → CPU fallback",
          " op=", op_name);
      torch_neuronx::markFallbackLogged(op_name);
      for (auto& arg : args) {
        torch::jit::push(*stack, arg);
      }
      at::native::cpu_fallback(op, stack);
      return;
    }
  }

  TORCH_CHECK(
      !reg.lower_func.is_none(), "NeuronDispatcher: operation '", op_name,
      "' was registered without a "
      "lower_func. All ops must have a C++ lowering path when TORCH_NEURONX_CPP_DISPATCH=1.");

  // ── 1. Generate cache key on RAW args (before any downcast) ──
  std::string cache_key = GenerateCacheKey(lookup_name, schema, args, reg);
  TORCH_NEURONX_DISPATCH_DEBUG("[NeuronDispatcher::Dispatch] cache_key", " op=", op_name,
                               " cache_key=", cache_key);

  // ── 2. Dispatch with casting ──
  // Dtype inference is unified into the lowering step: on cache miss,
  // lower_func receives both downcasted and original args, and the
  // LoweringResult contains inferred_output_dtypes for upcasting.
  DispatchWithCasting(op, dispatch_keys, stack, args, schema, reg, dispatcher, op_name, cache_key);
}

// ═══════════════════════════════════════════════════════════════
//  MakeChannelsLast — NRT strided copy into channels_last layout
// ═══════════════════════════════════════════════════════════════

/// Reorder a tensor to channels_last (or channels_last_3d) layout on-device
/// using NRT strided copy.  Works for any input layout — broadcast, transposed,
/// sliced, etc. — without MLIR compilation.
static torch::Tensor MakeChannelsLast(const torch::Tensor& self_tensor,
                                      c10::MemoryFormat memory_format) {
  auto src = self_tensor;

  // Materialize broadcast sources (stride=0) to avoid O(numel) element-by-element
  // NRT copy calls.  The strided-copy helper recurses per element when a stride is 0.
  for (auto s : src.strides()) {
    if (s == 0) {
      src = src.clone();
      break;
    }
  }

  auto output = torch::empty_like(src, src.options().memory_format(memory_format));
  torch_neuronx::utils::copy_strided_neuron_to_neuron(src, output);
  return output;
}

// ═══════════════════════════════════════════════════════════════
//  aten::contiguous — prologue-driven: the identity kernel just
//  clones, and ProcessInputTransformations handles all non-contiguous
//  patterns (broadcast, transpose, slice, compound).
//
//  No recursion risk: ProcessInputTransformations uses NRT strided
//  copy (not contiguous_internal) for unsupported patterns, so it
//  never re-enters the dispatch pipeline.
// ═══════════════════════════════════════════════════════════════

void NeuronDispatcher::DispatchContiguousImpl(const c10::OperatorHandle& op,
                                              c10::DispatchKeySet dispatch_keys,
                                              torch::jit::Stack* stack, const OpRegistration& reg) {
  auto op_name = c10::toString(op.schema().operator_name());
  std::string lookup_name = op_name;
  if (lookup_name.rfind("aten::", 0) == 0) {
    lookup_name = lookup_name.substr(6);
  }

  auto& dispatcher = NeuronDispatcher::Instance();

  TORCH_NEURONX_DISPATCH_DEBUG("[DispatchContiguous] ENTER", " op=", op_name);

  auto schema = op.schema();
  auto args = torch::jit::pop(*stack, schema.arguments().size());

  if (TryHandleEmptyTensors(op, stack, args, schema, op_name)) {
    return;
  }

  auto self_tensor = args[0].toTensor();
  auto memory_format = c10::MemoryFormat::Contiguous;
  if (args.size() > 1 && args[1].isInt()) {
    memory_format = static_cast<c10::MemoryFormat>(args[1].toInt());
  }

  // Fast exit: already contiguous in the requested format.
  if (self_tensor.is_contiguous(memory_format)) {
    TORCH_NEURONX_DISPATCH_DEBUG("[DispatchContiguous] already contiguous — returning as-is",
                                 " op=", op_name);
    torch::jit::push(*stack, self_tensor);
    return;
  }

  // Complex dtypes: fall back to CPU
  if (HasComplexInput(args)) {
    if (kFallbackOnlyForUnimplemented && HasComplexNonOutputInput(args, schema)) {
      RaiseComplexFallbackDisabled(op_name);
    }
    TORCH_NEURONX_DISPATCH_DEBUG("[DispatchContiguous] complex dtype → CPU fallback",
                                 " op=", op_name);
    for (auto& arg : args) {
      torch::jit::push(*stack, arg);
    }
    at::native::cpu_fallback(op, stack);
    logOperationExecuted(op_name);
    return;
  }

  // Channels-last: use NRT strided copy for all cases (any input layout →
  // channels_last output).  The prologue/MLIR pipeline doesn't handle
  // channels_last output strides.
  bool is_channels_last = (memory_format == c10::MemoryFormat::ChannelsLast ||
                           memory_format == c10::MemoryFormat::ChannelsLast3d);
  if (is_channels_last) {
    TORCH_NEURONX_DISPATCH_DEBUG("[DispatchContiguous] channels_last via NRT strided copy",
                                 " op=", op_name);
    auto output = MakeChannelsLast(self_tensor, memory_format);
    logOperationExecuted(op_name);
    torch::jit::push(*stack, output);
    return;
  }

  if (reg.lower_func.is_none()) {
    TORCH_CHECK(false, "[DispatchContiguous] op '", op_name,
                "' registered without lower_func — this should not happen");
  }

  auto original_args = args;

  std::string cache_key = GenerateCacheKey(lookup_name, schema, args, reg);

  // Append storage_offset to cache key: two views with identical shape/strides
  // but different offsets need distinct NEFFs (the prologue's slice transformation
  // uses offset).
  if (self_tensor.storage_offset() != 0) {
    cache_key += ":so=" + std::to_string(self_tensor.storage_offset());
  }

  try {
    // skip_contiguous=true: we ARE the contiguous implementation — asking
    // PreprocessArgs to call t.contiguous() would re-dispatch back into
    // DispatchContiguousImpl and cause infinite recursion (stack overflow).
    // ProcessInputTransformations handles the non-contiguous input by
    // prepending transformations (broadcast, transpose, slice, compound)
    // to the kernel. The kernel itself is a simple clone.
    DispatchWithCasting(op, dispatch_keys, stack, args, schema, reg, dispatcher, op_name, cache_key,
                        /*skip_contiguous=*/true);
  } catch (const std::exception& e) {
    static bool fatal = std::getenv("TORCH_NEURONX_CONTIGUOUS_FALLBACK_FATAL") != nullptr;
    if (fatal) {
      TORCH_CHECK(false, "[DispatchContiguous] lowering failed for op '", op_name, "': ", e.what(),
                  ". Contiguous CPU fallback is fatal because "
                  "TORCH_NEURONX_CONTIGUOUS_FALLBACK_FATAL is set.");
    }
    TORCH_NEURONX_DISPATCH_DEBUG("[DispatchContiguous] lowering failed → CPU fallback",
                                 " op=", op_name, " error=", e.what());
    for (auto& arg : original_args) {
      torch::jit::push(*stack, arg);
    }
    at::native::cpu_fallback(op, stack);
    logOperationExecuted(op_name);
    return;
  }
}

// ═══════════════════════════════════════════════════════════════
//  aten::nonzero — uses nonzero_with_count kernel which returns
//  (indices, count) as 2 tensors. Post-processes: indices[:count].to(int64).
// ═══════════════════════════════════════════════════════════════

void NeuronDispatcher::DispatchNonzeroImpl(const c10::OperatorHandle& op,
                                           c10::DispatchKeySet dispatch_keys,
                                           torch::jit::Stack* stack, const OpRegistration& reg) {
  auto op_name = c10::toString(op.schema().operator_name());
  std::string lookup_name = op_name;
  if (lookup_name.rfind("aten::", 0) == 0) {
    lookup_name = lookup_name.substr(6);
  }

  auto& dispatcher = NeuronDispatcher::Instance();

  TORCH_NEURONX_DISPATCH_DEBUG("[DispatchNonzero] ENTER", " op=", op_name);

  auto schema = op.schema();
  auto args = torch::jit::pop(*stack, schema.arguments().size());

  // Validate out= dtype if present (nonzero.out requires int64 output)
  for (size_t i = 0; i < args.size() && i < schema.arguments().size(); ++i) {
    const auto& alias_info = schema.arguments()[i].alias_info();
    if (alias_info && alias_info->isWrite() && args[i].isTensor()) {
      auto out_tensor = args[i].toTensor();
      if (out_tensor.defined() && out_tensor.scalar_type() != c10::ScalarType::Long) {
        TORCH_CHECK(false,
                    "nonzero: Expected out tensor to have scalar type Long but got scalar type",
                    c10::toString(out_tensor.scalar_type()));
      }
    }
  }

  if (TryHandleEmptyTensors(op, stack, args, schema, op_name)) {
    return;
  }

  if (HasComplexInput(args)) {
    TORCH_NEURONX_DISPATCH_DEBUG("[DispatchNonzero] complex dtype → CPU fallback", " op=", op_name);
    torch_neuronx::markFallbackLogged(op_name);
    for (auto& arg : args) {
      torch::jit::push(*stack, arg);
    }
    at::native::cpu_fallback(op, stack);
    return;
  }

  TORCH_CHECK(!reg.lower_func.is_none(), "NeuronDispatcher: operation '", op_name,
              "' was registered without a lower_func.");

  std::string cache_key = GenerateCacheKey(lookup_name, schema, args, reg);

  // ── Snapshot + downcast + preprocess (same as DispatchWithCasting steps 1-4) ──
  auto fallback_ctx = BuildCPUFallbackContext(args, schema);
  auto original_args = args;
  for (size_t i = 0; i < original_args.size(); ++i) {
    if (original_args[i].isTensorList()) {
      original_args[i] = c10::IValue(original_args[i].toTensorList().copy());
    } else if (original_args[i].isList()) {
      original_args[i] = c10::IValue(original_args[i].toList().copy());
    }
  }
  DowncastArgs(args, schema);

  auto effective_static_argnums = reg.static_argnums;
  if (!reg.static_argnames.empty()) {
    const auto& sa = schema.arguments();
    for (size_t i = 0; i < sa.size(); ++i) {
      if (reg.static_argnames.count(sa[i].name())) {
        effective_static_argnums.insert(i);
      }
    }
  }
  PreprocessArgs(args, schema, effective_static_argnums, /*skip_contiguous=*/false);

  // ── Run DispatchCore — returns 2 tensors: (indices, count) ──
  auto exec_outputs = DispatchCore(op_name, cache_key, op, dispatch_keys, stack, args, schema, reg,
                                   dispatcher, std::move(fallback_ctx), original_args);

  if (exec_outputs.empty()) {
    // DispatchCore already handled fallback and pushed to stack
    return;
  }

  TORCH_CHECK(exec_outputs.size() == 2,
              "[DispatchNonzero] expected 2 outputs (indices, count) from nonzero_with_count, got ",
              exec_outputs.size());

  // ── Post-process: slice indices[:count] and cast to int64 ──
  auto& indices = exec_outputs[0];
  auto& count_tensor = exec_outputs[1];

  // count is a scalar tensor on Neuron — synchronize and read
  auto stream = at::neuron::getCurrentNeuronStream(count_tensor.device().index());
  stream.synchronize();
  auto count_cpu = count_tensor.cpu();
  int64_t count = count_cpu.item<int64_t>();

  auto result = indices.slice(0, 0, count);

  // Cast to int64 — nonzero always returns Long dtype
  if (result.scalar_type() != c10::ScalarType::Long) {
    result = NeuronSafeCast(result, c10::ScalarType::Long);
  }

  // Handle out= parameter if present
  // Schema: nonzero(Tensor self) -> Tensor
  // Schema: nonzero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  const auto& schema_args = schema.arguments();
  for (size_t i = 0; i < original_args.size() && i < schema_args.size(); ++i) {
    const auto& alias_info = schema_args[i].alias_info();
    if (!alias_info || !alias_info->isWrite()) continue;
    if (!original_args[i].isTensor()) continue;
    auto out_tensor = original_args[i].toTensor();
    if (out_tensor.defined()) {
      ResizeOutputWithWarning(out_tensor, result.sizes());
      out_tensor.copy_(result);
      result = out_tensor;
      break;
    }
  }

  // Push single tensor result
  torch::jit::push(*stack, result);

  TORCH_NEURONX_DISPATCH_DEBUG("[DispatchNonzero] EXIT", " op=", op_name, " count=", count,
                               " result_shape=", result.sizes());
}

// ═══════════════════════════════════════════════════════════════
//  aten::index.Tensor — convert boolean indices to integer indices
//  BEFORE cache key generation (data-dependent shapes).
// ═══════════════════════════════════════════════════════════════

void NeuronDispatcher::DispatchIndexImpl(const c10::OperatorHandle& op,
                                         c10::DispatchKeySet dispatch_keys,
                                         torch::jit::Stack* stack, const OpRegistration& reg) {
  auto op_name = c10::toString(op.schema().operator_name());
  std::string lookup_name = op_name;
  if (lookup_name.rfind("aten::", 0) == 0) {
    lookup_name = lookup_name.substr(6);
  }

  auto& dispatcher = NeuronDispatcher::Instance();

  TORCH_NEURONX_DISPATCH_DEBUG("[DispatchIndex] ENTER", " op=", op_name);

  auto schema = op.schema();
  auto args = torch::jit::pop(*stack, schema.arguments().size());

  if (TryHandleEmptyTensors(op, stack, args, schema, op_name)) {
    return;
  }

  if (HasComplexInput(args)) {
    TORCH_NEURONX_DISPATCH_DEBUG("[DispatchIndex] complex dtype → CPU fallback", " op=", op_name);
    torch_neuronx::markFallbackLogged(op_name);
    for (auto& arg : args) {
      torch::jit::push(*stack, arg);
    }
    at::native::cpu_fallback(op, stack);
    return;
  }

  TORCH_CHECK(!reg.lower_func.is_none(), "NeuronDispatcher: operation '", op_name,
              "' was registered without a lower_func.");

  // Validate index dtypes — must be long, int, byte, or bool.
  // Invalid dtypes fall back to CPU (matches PyTorch's index validation).
  if (args.size() > 1 && args[1].isList()) {
    auto idx_list_check = args[1].toListRef();
    for (const auto& elem : idx_list_check) {
      if (!elem.isNone() && elem.isTensor()) {
        auto idx = elem.toTensor();
        if (idx.defined()) {
          auto dt = idx.scalar_type();
          if (dt != c10::ScalarType::Long && dt != c10::ScalarType::Int &&
              dt != c10::ScalarType::Byte && dt != c10::ScalarType::Bool) {
            torch_neuronx::markFallbackLogged(op_name);
            for (auto& arg : args) {
              torch::jit::push(*stack, arg);
            }
            at::native::cpu_fallback(op, stack);
            return;
          }
        }
      }
    }
  }

  // ── Convert boolean indices to integer indices ──
  // Schema: index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
  // args[1] is the indices list (c10::List<c10::optional<at::Tensor>>).
  // Boolean indices produce data-dependent shapes via nonzero, so this
  // conversion must happen BEFORE cache key generation.
  // Also move CPU indices to Neuron device.
  auto self_device = args[0].toTensor().device();
  if (args.size() > 1 && args[1].isList()) {
    auto indices_list = args[1].toListRef();
    c10::List<c10::optional<at::Tensor>> converted;
    bool has_bool_index = false;
    bool needs_update = false;

    for (const auto& elem : indices_list) {
      if (elem.isNone()) {
        converted.push_back(c10::nullopt);
      } else if (elem.isTensor()) {
        auto idx = elem.toTensor();
        if (idx.defined() && idx.scalar_type() == c10::ScalarType::Bool) {
          has_bool_index = true;
          needs_update = true;
          auto nonzero_indices = at::nonzero_numpy(idx);
          for (auto& nz_idx : nonzero_indices) {
            converted.push_back(std::move(nz_idx));
          }
        } else if (idx.defined() && !idx.is_privateuseone()) {
          // Move CPU index to Neuron device
          needs_update = true;
          converted.push_back(idx.to(self_device));
        } else {
          converted.push_back(idx);
        }
      } else {
        converted.push_back(c10::nullopt);
      }
    }

    if (needs_update) {
      args[1] = c10::IValue(converted);
      TORCH_NEURONX_DISPATCH_DEBUG("[DispatchIndex] converted boolean indices to integer",
                                   " op=", op_name, " num_converted_indices=", converted.size());

      // Check for empty indices after conversion (e.g., all-false boolean mask).
      // Return empty result without executing the kernel.
      bool has_empty_index = false;
      for (size_t i = 0; i < converted.size(); ++i) {
        auto elem = converted.get(i);
        if (elem.has_value() && elem->defined() && elem->numel() == 0) {
          has_empty_index = true;
          break;
        }
      }
      if (has_empty_index) {
        auto self = args[0].toTensor();
        auto result = torch::empty({0}, self.options());
        torch::jit::push(*stack, result);
        return;
      }
    }
  }

  // ── Validate index bounds (respects TORCH_NEURONX_SKIP_OP_PRECONDITIONS) ──
  static bool skip_preconditions = [] {
    const char* env = std::getenv("TORCH_NEURONX_SKIP_OP_PRECONDITIONS");
    return env != nullptr && std::string(env) == "1";
  }();
  if (!skip_preconditions && args.size() > 1 && args[1].isList()) {
    auto self = args[0].toTensor();
    auto indices_list = args[1].toListRef();
    int dim = 0;
    for (const auto& elem : indices_list) {
      if (!elem.isNone() && elem.isTensor()) {
        auto idx = elem.toTensor();
        if (idx.defined() && idx.numel() > 0 && idx.scalar_type() != c10::ScalarType::Bool &&
            idx.is_privateuseone()) {
          int64_t dim_size = self.size(dim);
          // Synchronize to read index values for bounds check
          auto stream = at::neuron::getCurrentNeuronStream(idx.device().index());
          stream.synchronize();
          auto idx_cpu = idx.cpu();
          auto idx_max = idx_cpu.max().item<int64_t>();
          auto idx_min = idx_cpu.min().item<int64_t>();
          if (idx_max >= dim_size || idx_min < -dim_size) {
            // Fall back to CPU which raises IndexError (not RuntimeError)
            torch_neuronx::markFallbackLogged(op_name);
            for (auto& arg : args) {
              torch::jit::push(*stack, arg);
            }
            at::native::cpu_fallback(op, stack);
            return;
          }
        }
      }
      dim++;
    }
  }

  // ── Generate cache key with converted indices ──
  std::string cache_key = GenerateCacheKey(lookup_name, schema, args, reg);

  TORCH_NEURONX_DISPATCH_DEBUG("[DispatchIndex] cache_key", " op=", op_name,
                               " cache_key=", cache_key);

  // ── Standard dispatch pipeline ──
  DispatchWithCasting(op, dispatch_keys, stack, args, schema, reg, dispatcher, op_name, cache_key);
}

// ═══════════════════════════════════════════════════════════════
//  NKI dispatch core — shared pipeline for SDPA ops
//
//  Bypasses DowncastArgs/UpcastOutputs entirely — NKI ops only
//  accept bf16/fp16/fp32, never float64/int64, so dtype casting
//  is unnecessary and causes regressions (CPU round-trips via
//  NeuronSafeCast on backward logsumexp float32).
// ═══════════════════════════════════════════════════════════════

SdpaCoreResult NeuronDispatcher::DispatchSdpaCore(const c10::OperatorHandle& op,
                                                  c10::DispatchKeySet dispatch_keys,
                                                  torch::jit::Stack* stack,
                                                  const OpRegistration& reg) {
  auto op_name = c10::toString(op.schema().operator_name());
  auto& dispatcher = NeuronDispatcher::Instance();
  auto schema = op.schema();
  auto args = torch::jit::pop(*stack, schema.arguments().size());

  // Early exit: empty tensors
  if (TryHandleEmptyTensors(op, stack, args, schema, op_name)) {
    return {{}, {}, torch::Device(torch::kCPU), true};
  }

  // Complex dtypes: fall back to CPU
  if (HasComplexInput(args)) {
    if (kFallbackOnlyForUnimplemented && HasComplexNonOutputInput(args, schema)) {
      RaiseComplexFallbackDisabled(op_name);
    }
    for (auto& arg : args) torch::jit::push(*stack, arg);
    at::native::cpu_fallback(op, stack);
    logOperationExecuted(op_name);
    return {{}, {}, torch::Device(torch::kCPU), true};
  }

  std::string lookup_name = op_name;
  if (lookup_name.rfind("aten::", 0) == 0) lookup_name = lookup_name.substr(6);

  auto cache_key = GenerateCacheKey(lookup_name, schema, args, reg);
  auto original_args = args;

  // PreprocessArgs only — NO DowncastArgs.
  auto effective_static_argnums = reg.static_argnums;
  if (!reg.static_argnames.empty()) {
    const auto& sa = schema.arguments();
    for (size_t i = 0; i < sa.size(); ++i) {
      if (reg.static_argnames.count(sa[i].name())) {
        effective_static_argnums.insert(i);
      }
    }
  }
  PreprocessArgs(args, schema, effective_static_argnums, /*skip_contiguous=*/false);

  // GetOrLower
  auto entry = dispatcher.GetOrLowerOperation(cache_key, reg, args, original_args, schema);
  if (!entry) {
    if (kFallbackOnlyForUnimplemented) {
      TORCH_CHECK(false, "NeuronDispatcher: CPU fallback disabled. Operation '", op_name,
                  "' could not be lowered to Neuron.");
    }
    torch_neuronx::markFallbackLogged(op_name);
    for (const auto& arg : original_args) torch::jit::push(*stack, arg);
    at::native::cpu_fallback(op, stack);
    logOperationExecuted(op_name);
    return {{}, {}, torch::Device(torch::kCPU), true};
  }

  // Empty-tensor lowering result
  if (entry->is_empty_tensor) {
    auto device = ExtractDevice(args);
    std::vector<torch::Tensor> empty_outputs;
    for (const auto& spec : entry->original_output_specs) {
      if (spec.dtype == c10::ScalarType::Undefined) {
        empty_outputs.push_back(torch::Tensor());
      } else {
        empty_outputs.push_back(
            torch::empty(spec.shape, torch::TensorOptions().dtype(spec.dtype).device(device)));
      }
    }
    PushOutputsToStack(stack, empty_outputs, schema);
    logOperationExecuted(op_name);
    return {{}, {}, device, true};
  }

  // Execute
  auto device = ExtractDevice(args);
  auto filtered_inputs = FlattenAndFilterInputs(args, schema, entry->kept_input_indices, op_name);
  auto exec_outputs = AllocateExecutionOutputs(*entry, args, filtered_inputs, device, schema);
  int device_id = c10_neuron::GetTargetDeviceId(filtered_inputs, exec_outputs);
  auto fallback_ctx = BuildCPUFallbackContext(original_args, schema);

  EnqueueXlaOperation(entry->op_name, filtered_inputs, exec_outputs, entry->stablehlo_bytes,
                      entry->neff_cache_key, entry->has_collectives, device_id,
                      std::move(fallback_ctx));

  auto final_outputs = ReconstructOutputs(exec_outputs, *entry, args, device);

  return {std::move(final_outputs), std::move(original_args), device, false};
}

// ═══════════════════════════════════════════════════════════════
//  SDPA Forward — C++ handler with stride fixup and 9-tuple output
// ═══════════════════════════════════════════════════════════════

void NeuronDispatcher::DispatchSdpaFwdImpl(const c10::OperatorHandle& op,
                                           c10::DispatchKeySet dispatch_keys,
                                           torch::jit::Stack* stack, const OpRegistration& reg) {
  auto op_name = c10::toString(op.schema().operator_name());

  // Validate return_debug_mask (arg index 6 in the ATen schema)
  auto schema = op.schema();
  size_t num_args = schema.arguments().size();
  if (num_args > 6) {
    const auto& debug_mask_arg = (*stack)[stack->size() - num_args + 6];
    if (debug_mask_arg.isBool() && debug_mask_arg.toBool()) {
      TORCH_CHECK(false, "return_debug_mask=True is not supported in Neuron SDPA implementation");
    }
  }

  // Capture query tensor reference before DispatchSdpaCore pops args from stack
  {
    // Validate SDPA input dtypes and shapes
    const auto& q = (*stack)[stack->size() - num_args].toTensor();
    const auto& k = (*stack)[stack->size() - num_args + 1].toTensor();
    const auto& v = (*stack)[stack->size() - num_args + 2].toTensor();

    // Dtype validation: must be float32, float16, or bfloat16, and all must match
    auto dt = q.scalar_type();
    TORCH_CHECK(dt == c10::ScalarType::Float || dt == c10::ScalarType::Half ||
                    dt == c10::ScalarType::BFloat16,
                "SDPA: query dtype must be float32, float16, or bfloat16, got ", c10::toString(dt));
    TORCH_CHECK(k.scalar_type() == dt && v.scalar_type() == dt,
                "SDPA: query, key, value must have the same dtype");

    // Shape validation: must be 4D (B, H, S, D)
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "SDPA: query, key, value must be 4D tensors (B, H, S, D)");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "SDPA: batch sizes must match");
    TORCH_CHECK(k.size(1) == v.size(1), "SDPA: key and value must have the same number of heads");
    TORCH_CHECK(k.size(2) == v.size(2), "SDPA: key and value must have the same sequence length");
    TORCH_CHECK(q.size(1) % k.size(1) == 0,
                "SDPA: Number of heads in key and value must divide the number of heads in query");
  }
  // We need query (args[0]) and value (args[2]) for stride fixup after execution
  const auto& query_ref = (*stack)[stack->size() - num_args].toTensor();
  const auto& value_ref = (*stack)[stack->size() - num_args + 2].toTensor();
  // Copy shape info before stack is popped
  auto query_sizes = query_ref.sizes().vec();

  auto result = DispatchSdpaCore(op, dispatch_keys, stack, reg);
  if (result.handled) return;

  auto& final_outputs = result.final_outputs;
  TORCH_CHECK(final_outputs.size() == 7, "SDPA forward expected 7 outputs, got ",
              final_outputs.size());

  // Stride fixup: use alloc_with_matching_layout
  // to produce output with strides matching query's layout.
  if (!final_outputs.empty() && final_outputs[0].defined()) {
    const auto& query = result.original_args[0].toTensor();
    const auto& value = result.original_args[2].toTensor();
    std::vector<int64_t> res_shape = {query.size(0), query.size(1), query.size(2), value.size(3)};
    at::Tensor fixed;
    at::native::alloc_with_matching_layout(query, fixed, res_shape);
    if (fixed.strides().vec() != final_outputs[0].strides().vec()) {
      fixed.copy_(final_outputs[0]);
      final_outputs[0] = fixed;
    }
  }

  // Push 9-tuple with SymInt and None slots
  int64_t seq_len_q = result.original_args[0].toTensor().size(2);
  int64_t seq_len_k = result.original_args[1].toTensor().size(2);
  torch::jit::push(*stack, final_outputs[0]);  // output
  torch::jit::push(*stack, final_outputs[1]);  // logsumexp
  torch::jit::push(*stack, at::Tensor());      // cum_seq_q = None
  torch::jit::push(*stack, at::Tensor());      // cum_seq_k = None
  torch::jit::push(*stack, seq_len_q);         // max_q
  torch::jit::push(*stack, seq_len_k);         // max_k
  torch::jit::push(*stack, final_outputs[4]);  // philox_seed
  torch::jit::push(*stack, final_outputs[5]);  // philox_offset
  torch::jit::push(*stack, at::Tensor());      // debug_mask = None

  logOperationExecuted(op_name);
}

// ═══════════════════════════════════════════════════════════════
//  SDPA Backward — C++ handler with grad stride fixup
// ═══════════════════════════════════════════════════════════════

void NeuronDispatcher::DispatchSdpaBwdImpl(const c10::OperatorHandle& op,
                                           c10::DispatchKeySet dispatch_keys,
                                           torch::jit::Stack* stack, const OpRegistration& reg) {
  auto op_name = c10::toString(op.schema().operator_name());

  auto result = DispatchSdpaCore(op, dispatch_keys, stack, reg);
  if (result.handled) return;

  auto& final_outputs = result.final_outputs;

  // Fix grad strides to match input strides
  auto fixGradStrides = [](torch::Tensor& grad, const at::Tensor& ref) {
    if (grad.defined() && grad.strides().vec() != ref.strides().vec()) {
      auto fixed = torch::empty_strided(grad.sizes(), ref.strides(), grad.options());
      fixed.copy_(grad);
      grad = fixed;
    }
  };
  // grad_query matches query (arg 1), grad_key matches key (arg 2), grad_value matches value (arg
  // 3)
  if (final_outputs.size() > 0)
    fixGradStrides(final_outputs[0], result.original_args[1].toTensor());
  if (final_outputs.size() > 1)
    fixGradStrides(final_outputs[1], result.original_args[2].toTensor());
  if (final_outputs.size() > 2)
    fixGradStrides(final_outputs[2], result.original_args[3].toTensor());

  // If attn_bias was None, set grad_attn_bias to None
  if (final_outputs.size() > 3) {
    auto attn_bias = result.original_args[4];
    bool attn_bias_is_none = !attn_bias.isTensor() || !attn_bias.toTensor().defined();
    if (attn_bias_is_none) {
      final_outputs[3] = at::Tensor();
    }
  }

  PushOutputsToStack(stack, final_outputs, op.schema());
  logOperationExecuted(op_name);
}

// ═══════════════════════════════════════════════════════════════
//  DispatchCore — shared execution pipeline (no casting awareness)
// ═══════════════════════════════════════════════════════════════

std::vector<torch::Tensor> NeuronDispatcher::DispatchCore(
    const std::string& op_name, const std::string& cache_key, const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack, std::vector<c10::IValue>& args,
    const c10::FunctionSchema& schema, const OpRegistration& reg, NeuronDispatcher& dispatcher,
    at::neuron::CPUFallbackContext fallback_ctx, const std::vector<c10::IValue>& original_args,
    std::shared_ptr<DispatchCacheEntry>* out_entry) {
  TORCH_NEURONX_DISPATCH_DEBUG("[DispatchCore] using cache_key", " op=", op_name,
                               " cache_key=", cache_key);

  LogArgs("[DispatchCore]", args);

  auto entry = dispatcher.GetOrLowerOperation(cache_key, reg, args, original_args, schema);

  if (out_entry) *out_entry = entry;

  if (!entry) {
    if (kFallbackOnlyForUnimplemented) {
      TORCH_CHECK(false,
                  "NeuronDispatcher: CPU fallback disabled because "
                  "TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1. "
                  "Operation '",
                  op_name, "' could not be lowered to Neuron.");
    }
    TORCH_NEURONX_DISPATCH_DEBUG("[DispatchCore] lowering returned null → CPU fallback",
                                 " op=", op_name);
    torch_neuronx::markFallbackLogged(op_name);
    // Use original (pre-downcast) args for CPU fallback when available,
    // so that dtypes (e.g. int64) are preserved instead of using the
    // downcasted args (e.g. int32) from DowncastArgs.
    const auto& fallback_args = original_args.empty() ? args : original_args;
    for (const auto& arg : fallback_args) {
      torch::jit::push(*stack, arg);
    }
    at::native::cpu_fallback(op, stack);
    return {};
  }

  TORCH_NEURONX_DISPATCH_DEBUG("[DispatchCore] cache entry obtained", " op=", op_name,
                               " neff_cache_key=", entry->neff_cache_key,
                               " num_output_specs=", entry->output_specs.size());

  // ── Zero-sized outputs: skip compilation and NRT execution entirely ──
  if (entry->is_empty_tensor) {
    logOperationExecuted(op_name);
    auto device = ExtractDevice(args);
    std::vector<torch::Tensor> empty_outputs;
    for (const auto& spec : entry->original_output_specs) {
      if (spec.dtype == c10::ScalarType::Undefined) {
        empty_outputs.push_back(torch::Tensor());
      } else {
        auto opts = torch::TensorOptions().dtype(spec.dtype).device(device);
        empty_outputs.push_back(torch::empty(spec.shape, opts));
      }
    }
    return empty_outputs;
  }

  TORCH_CHECK(!entry->stablehlo_bytes.empty(),
              "[DispatchCore] StableHLO bytes are empty for non-empty-tensor op '", op_name,
              "' — lowering produced no HLO");

  if (NeuronLogger::shouldLog(LogLevel::DEBUG) || dispatch_debug_enabled()) {
    std::string hlo_str(reinterpret_cast<const char*>(entry->stablehlo_bytes.data()),
                        entry->stablehlo_bytes.size());
    TORCH_NEURONX_DISPATCH_DEBUG("[DispatchCore] StableHLO for ", op_name, ":\n", hlo_str);
  }

  auto device = ExtractDevice(args);

  auto filtered_inputs = FlattenAndFilterInputs(args, schema, entry->kept_input_indices, op_name);

  TORCH_NEURONX_DISPATCH_DEBUG("[DispatchCore] filtered inputs ready", " op=", op_name,
                               " num_filtered_inputs=", filtered_inputs.size());

  LogTensors("[DispatchCore] filtered_inputs", filtered_inputs);

  // Non-contiguous inplace tensors: AllocateExecutionOutputs will skip the
  // alias (is_contiguous check) and allocate a contiguous temp instead.
  // CopyBackToOriginal scatters the result back via .copy_().
  if (entry->is_inplace && !args.empty() && args[0].isTensor()) {
    auto first_tensor = args[0].toTensor();
    if (first_tensor.defined() && !first_tensor.is_contiguous()) {
      TORCH_NEURONX_DISPATCH_DEBUG("[DispatchCore] non-contiguous inplace tensor detected",
                                   " op=", op_name, " — will allocate contiguous temp");
    }
  }

  auto exec_outputs = AllocateExecutionOutputs(*entry, args, filtered_inputs, device, schema);

  LogTensors("[DispatchCore] exec_outputs", exec_outputs);

  int device_id = c10_neuron::GetTargetDeviceId(filtered_inputs, exec_outputs);

  // Capture hierarchy string from the live Python call stack.
  // Only when NEURON_HIERARCHY_DEBUG=1 to avoid GIL overhead in production.
  // Optimization: cache the Python function object and logger to avoid
  // py::module_::import + attr lookup on every call (~5-10µs saved per op).
  std::string hierarchy_str;
  {
    static const bool hierarchy_enabled = []() {
      const char* env = std::getenv("NEURON_HIERARCHY_DEBUG");
      return env != nullptr && std::string(env) == "1";
    }();
    if (hierarchy_enabled) {
      py::gil_scoped_acquire gil;
      try {
        if (!g_hierarchy_fn) {
          auto mod = py::module_::import("torch_neuronx.python_ops.torch_mlir.eager_hierarchy");
          g_hierarchy_fn = mod.attr("get_hierarchy_from_stack").ptr();
          Py_XINCREF(g_hierarchy_fn);
        }
        auto result =
            py::reinterpret_borrow<py::object>(g_hierarchy_fn)(py::arg("op_name") = entry->op_name);
        hierarchy_str = result.cast<std::string>();
      } catch (const py::error_already_set& e) {
        TORCH_WARN_ONCE("[DispatchCore] hierarchy capture failed (Python): ", e.what());
        hierarchy_str = "";
      } catch (const std::exception& e) {
        TORCH_WARN_ONCE("[DispatchCore] hierarchy capture failed: ", e.what());
        hierarchy_str = "";
      }
    }
  }

  EnqueueXlaOperation(entry->op_name, filtered_inputs, exec_outputs, entry->stablehlo_bytes,
                      entry->neff_cache_key, entry->has_collectives, device_id,
                      std::move(fallback_ctx), hierarchy_str);

  auto final_outputs = ReconstructOutputs(exec_outputs, *entry, args, device);

  LogTensors("[DispatchCore] final_outputs", final_outputs);

  return final_outputs;
}

// ═══════════════════════════════════════════════════════════════
//  DispatchWithCasting — 64-bit dtype casting path
// ═══════════════════════════════════════════════════════════════

void NeuronDispatcher::DispatchWithCasting(const c10::OperatorHandle& op,
                                           c10::DispatchKeySet dispatch_keys,
                                           torch::jit::Stack* stack, std::vector<c10::IValue>& args,
                                           const c10::FunctionSchema& schema,
                                           const OpRegistration& reg, NeuronDispatcher& dispatcher,
                                           const std::string& op_name, const std::string& cache_key,
                                           bool skip_contiguous) {
  try {
    // ── 1. Snapshot out tensors before any mutation ──
    auto original_out_tensors = SnapshotOutTensors(schema, args);

    // ── 1.5. Build CPU fallback context on RAW args (before downcast/preprocess) ──
    // The fallback executor needs original dtypes and scalar types to reconstruct
    // the op correctly on CPU. Building after DowncastArgs/PreprocessArgs would
    // capture mutated types (int64→int32, scalars→tensors) causing type mismatches.
    auto fallback_ctx = BuildCPUFallbackContext(args, schema);

    // ── 2. Snapshot original args before downcast ──
    // Original args are passed to lower_func on cache miss for dtype inference
    // at original precision, and used for CPU fallback with original dtypes.
    // Deep-copy TensorList/GenericList args because DowncastArgs mutates them
    // in-place via shared intrusive_ptr storage.
    auto original_args = args;
    for (size_t i = 0; i < original_args.size(); ++i) {
      if (original_args[i].isTensorList()) {
        original_args[i] = c10::IValue(original_args[i].toTensorList().copy());
      } else if (original_args[i].isList()) {
        original_args[i] = c10::IValue(original_args[i].toList().copy());
      }
    }

    // ── 3. Downcast args (tensors + dtype kwargs) to supported types ──
    DowncastArgs(args, schema);

    // ── 4. PreprocessArgs (scalar→tensor, contiguous, device — NO dtype cast) ──
    auto effective_static_argnums = reg.static_argnums;
    if (!reg.static_argnames.empty()) {
      const auto& sa = schema.arguments();
      for (size_t i = 0; i < sa.size(); ++i) {
        if (reg.static_argnames.count(sa[i].name())) {
          effective_static_argnums.insert(i);
        }
      }
    }
    PreprocessArgs(args, schema, effective_static_argnums, skip_contiguous);

    // ── 5. Run core dispatch pipeline ──
    // DispatchCore calls GetOrLowerOperation which, on cache miss, passes both
    // downcasted args (for HLO compilation) and original args (for dtype inference)
    // to the Python lower_func in a single GIL acquisition.
    std::shared_ptr<DispatchCacheEntry> cache_entry;
    auto exec_outputs =
        DispatchCore(op_name, cache_key, op, dispatch_keys, stack, args, schema, reg, dispatcher,
                     std::move(fallback_ctx), original_args, &cache_entry);

    if (exec_outputs.empty()) {
      // DispatchCore already handled fallback and pushed to stack
      return;
    }

    // ── 6. Upcast outputs + copy-back into original out/inplace tensors ──
    // UpcastExecutionOutputs uses inferred_output_dtypes from the cache entry
    // (exact per-output dtypes at original precision) and handles out-tensor
    // copy-back atomically.
    if (cache_entry) {
      exec_outputs = UpcastExecutionOutputs(exec_outputs, *cache_entry, original_out_tensors);
    }

    // ── 7. Push to stack ──
    PushOutputsToStack(stack, exec_outputs, schema);

    TORCH_NEURONX_DISPATCH_DEBUG("[NeuronDispatcher::Dispatch] EXIT (CASTING PATH)",
                                 " op=", op_name);

  } catch (const std::exception& e) {
    TORCH_NEURONX_DISPATCH_DEBUG("[DispatchWithCasting] Failed EXCEPTION (re-throwing)",
                                 " op=", op_name, " error=", e.what());
    throw;
  }
}

// ═══════════════════════════════════════════════════════════════
//  GetOrLowerOperation (cache hit or miss)
// ═══════════════════════════════════════════════════════════════

std::shared_ptr<DispatchCacheEntry> NeuronDispatcher::GetOrLowerOperation(
    const std::string& cache_key, const OpRegistration& reg, const std::vector<c10::IValue>& args,
    const std::vector<c10::IValue>& original_args, const c10::FunctionSchema& schema) {
  TORCH_NEURONX_DISPATCH_DEBUG("[GetOrLowerOperation] looking up cache", " cache_key=", cache_key,
                               " skip_cache=", reg.skip_cache);

  if (!reg.skip_cache) {
    auto entry = CacheLookup(cache_key);
    if (entry) {
      TORCH_NEURONX_DISPATCH_DEBUG("[GetOrLowerOperation] CACHE HIT (no GIL)",
                                   " cache_key=", cache_key);
      return entry;
    }
  }

  TORCH_NEURONX_DISPATCH_DEBUG("[GetOrLowerOperation] ",
                               reg.skip_cache ? "skip_cache set" : "CACHE MISS", ", acquiring GIL",
                               " cache_key=", cache_key);

  py::gil_scoped_acquire gil;

  if (!reg.skip_cache) {
    auto entry = CacheLookup(cache_key);
    if (entry) {
      TORCH_NEURONX_DISPATCH_DEBUG("[GetOrLowerOperation] CACHE HIT (after GIL)",
                                   " cache_key=", cache_key);
      return entry;
    }
  }

  auto [py_args, py_kwargs] = IValuesToPyArgsKwargs(args, schema);

  // Pass original (pre-downcast) args to lower_func for dtype inference.
  // lower_func uses these to infer output dtypes at original precision.
  if (!original_args.empty()) {
    auto [orig_py_args, orig_py_kwargs] = IValuesToPyArgsKwargs(original_args, schema);
    py_kwargs["original_args"] = orig_py_args;
    py_kwargs["original_kwargs"] = orig_py_kwargs;
  }

  // Set the dispatch cache key on Python threading.local so the lowering code
  // can reuse it.  RAII guard guarantees cleanup on all exit paths.
  py::module_ tnx = py::module_::import("torch_neuronx");
  DispatchContextGuard dispatch_ctx_guard(tnx.attr("_dispatch_context"), cache_key);

  TORCH_NEURONX_DISPATCH_DEBUG("[GetOrLowerOperation] calling Python lower_func",
                               " cache_key=", cache_key, " num_positional=", py_args.size(),
                               " num_kwargs=", py::len(py_kwargs));

  try {
    py::object result = reg.lower_func(*py_args, **py_kwargs);

    if (result.is_none()) {
      // Composite lower signals no impl could lower — the caller (DispatchCore)
      // catches the nullptr and falls back to CPU or raises based on the flag.
      TORCH_NEURONX_DISPATCH_DEBUG("[GetOrLowerOperation] lower_func returned None",
                                   " cache_key=", cache_key);
      return nullptr;
    }

    TORCH_NEURONX_DISPATCH_DEBUG("[GetOrLowerOperation] parsing LoweringResult",
                                 " cache_key=", cache_key);
    auto entry = ParseLoweringResult(result);

    if (entry) {
      if (!reg.skip_cache) {
        TORCH_NEURONX_DISPATCH_DEBUG("[GetOrLowerOperation] caching result",
                                     " cache_key=", cache_key, " op_name=", entry->op_name,
                                     " neff_key=", entry->neff_cache_key);
        CacheInsert(cache_key, entry);
      } else {
        TORCH_NEURONX_DISPATCH_DEBUG("[GetOrLowerOperation] skip_cache, not caching",
                                     " cache_key=", cache_key, " op_name=", entry->op_name);
      }
    }

    return entry;
  } catch (const py::error_already_set& e) {
    TORCH_NEURONX_WARN("[GetOrLowerOperation] Python lower_func failed: ", e.what(),
                       " cache_key=", cache_key);
    throw;
  }
}

// ═══════════════════════════════════════════════════════════════
//  Cache operations
// ═══════════════════════════════════════════════════════════════

std::shared_ptr<DispatchCacheEntry> NeuronDispatcher::CacheLookup(const std::string& cache_key) {
  std::shared_lock lock(cache_mutex_);
  auto it = dispatch_cache_.find(cache_key);
  if (it != dispatch_cache_.end()) {
    return it->second;
  }
  return nullptr;
}

void NeuronDispatcher::CacheInsert(const std::string& cache_key,
                                   std::shared_ptr<DispatchCacheEntry> entry) {
  std::unique_lock lock(cache_mutex_);
  dispatch_cache_[cache_key] = std::move(entry);
}

// ═══════════════════════════════════════════════════════════════
//  Cache key generation — delegates to CacheKeyGenerator
// ═══════════════════════════════════════════════════════════════

std::string NeuronDispatcher::GenerateCacheKey(const std::string& op_name,
                                               const c10::FunctionSchema& schema,
                                               const std::vector<c10::IValue>& args,
                                               const OpRegistration& reg) {
  return at::neuron::CacheKeyGenerator::Generate(op_name, schema, args, reg.static_argnums,
                                                 reg.static_argnames);
}

// ═══════════════════════════════════════════════════════════════
//  Input helpers
// ═══════════════════════════════════════════════════════════════

// Extract all defined tensors from a single IValue into `out`.
// Handles Tensor, TensorList, and GenericList (Tensor?[]).
static void FlattenTensorsFromIValue(const c10::IValue& val, std::vector<torch::Tensor>& out) {
  if (val.isTensor()) {
    auto t = val.toTensor();
    if (t.defined()) out.push_back(t);
  } else if (val.isTensorList()) {
    for (const auto& t_ref : val.toTensorList()) {
      at::Tensor t = t_ref;
      if (t.defined()) out.push_back(t);
    }
  } else if (val.isList()) {
    // GenericList — covers Tensor?[] (e.g. _index_put_impl_ indices).
    // Elements may be Tensor or None; skip non-tensor entries.
    for (const auto& elem : val.toListRef()) {
      if (elem.isTensor()) {
        auto t = elem.toTensor();
        if (t.defined()) out.push_back(t);
      }
    }
  }
}

std::vector<torch::Tensor> NeuronDispatcher::FlattenAndFilterInputs(
    const std::vector<c10::IValue>& args, const c10::FunctionSchema& schema,
    const std::vector<int>& kept_input_indices, const std::string& op_name) {
  const auto& schema_args = schema.arguments();
  std::vector<torch::Tensor> flattened;

  // Positional (non-kwarg-only) tensor args
  for (size_t i = 0; i < args.size() && i < schema_args.size(); ++i) {
    if (schema_args[i].kwarg_only()) continue;
    FlattenTensorsFromIValue(args[i], flattened);
  }
  // Kwarg-only tensor args
  for (size_t i = 0; i < args.size() && i < schema_args.size(); ++i) {
    if (!schema_args[i].kwarg_only()) continue;
    FlattenTensorsFromIValue(args[i], flattened);
  }

  TORCH_NEURONX_DISPATCH_DEBUG("[FlattenAndFilterInputs] flattened", " op=", op_name,
                               " num_flattened=", flattened.size(),
                               " num_kept=", kept_input_indices.size());

  if (kept_input_indices.empty()) return flattened;

  std::vector<torch::Tensor> filtered;
  filtered.reserve(kept_input_indices.size());
  for (int idx : kept_input_indices) {
    if (idx >= 0 && static_cast<size_t>(idx) < flattened.size()) {
      filtered.push_back(flattened[idx]);
    }
  }

  TORCH_NEURONX_DISPATCH_DEBUG("[FlattenAndFilterInputs] filtered", " op=", op_name,
                               " num_filtered=", filtered.size());

  return filtered;
}

// ═══════════════════════════════════════════════════════════════
//  Device extraction
// ═══════════════════════════════════════════════════════════════

torch::Device NeuronDispatcher::ExtractDevice(const std::vector<c10::IValue>& args) {
  torch::Device result(c10::DeviceType::PrivateUse1,
                       static_cast<c10::DeviceIndex>(at::neuron::current_device()));
  bool found = false;
  // First, check input tensors for device info.
  ForEachTensor(args, [&](const at::Tensor& t) {
    if (!found && t.is_privateuseone()) {
      result = t.device();
      found = true;
    }
  });
  // For creation ops (ones, zeros, etc.) there are no input tensors.
  // Fall back to the explicit device kwarg if present.
  if (!found) {
    for (const auto& arg : args) {
      if (arg.isDevice()) {
        auto dev = arg.toDevice();
        if (dev.type() == c10::DeviceType::PrivateUse1) {
          result = dev;
          break;
        }
      }
    }
  }
  return result;
}

// ═══════════════════════════════════════════════════════════════
//  Empty-tensor handling — two-layer approach
//
//  Layer 1: C++ mirror of Python _handle_empty_tensors (kernel.py).
//    Uses Meta dispatch for output shape/dtype inference, then
//    fills outputs with correct values based on op type:
//      - Reductions → identity value (0, 1, NaN, or error)
//      - Index ops  → return input tensor unchanged
//      - General    → zeros
//    Tensors are created on CPU then moved to Neuron to avoid
//    reentrancy (torch::zeros on Neuron would re-enter via fill_).
//
//  Layer 2: CPU fallback (safety net if Meta dispatch fails).
//    Falls through to cpu_fallback for ops without Meta kernels.
// ═══════════════════════════════════════════════════════════════

bool NeuronDispatcher::TryHandleEmptyTensors(const c10::OperatorHandle& op,
                                             torch::jit::Stack* stack,
                                             const std::vector<c10::IValue>& args,
                                             const c10::FunctionSchema& schema,
                                             const std::string& op_name) {
  // ════════════════════════════════════════════════════════════
  //  Detection: does this op have empty tensors that we should handle?
  //  Each op category checks a different subset of args.
  // ════════════════════════════════════════════════════════════

  auto* index_cfg = GetIndexOpConfig(op_name);

  if (index_cfg) {
    // ── Category 1: GetIndexOpConfig ops (scatter, scatter_add, _index_put_impl_) ──
    // Check ALL args including lists for empty tensors.
    bool has_empty = false;
    for (const auto& arg : args) {
      if (arg.isTensor()) {
        if (arg.toTensor().defined() && arg.toTensor().numel() == 0) {
          has_empty = true;
          break;
        }
      } else if (arg.isList()) {
        for (const auto& elem : arg.toListRef()) {
          if (elem.isTensor() && elem.toTensor().defined() && elem.toTensor().numel() == 0) {
            has_empty = true;
            break;
          }
        }
        if (has_empty) break;
      }
    }
    if (!has_empty) return false;

  } else if (IsIndexOp(op_name)) {
    // ── Category 2: IsIndexOp ops (gather, index_copy, index_select, ...) ──
    // Meta dispatch does NOT validate shapes for these.
    // Only treat as empty if the tensor has an integer/bool dtype — CPU
    // validates index dtype before checking emptiness, so a float empty
    // index should NOT be short-circuited (must raise RuntimeError).
    bool has_empty = false;
    for (const auto& arg : args) {
      if (arg.isTensor()) {
        auto t = arg.toTensor();
        if (t.defined() && t.is_privateuseone() && t.numel() == 0 && t.dim() > 0 &&
            at::isIntegralType(t.scalar_type(), /*includeBool=*/true)) {
          has_empty = true;
          break;
        }
      }
    }
    if (!has_empty) return false;

    // Empty self (arg 0) → normal lowering validates shapes (Meta doesn't).
    if (!args.empty() && args[0].isTensor() && args[0].toTensor().defined() &&
        args[0].toTensor().numel() == 0) {
      TORCH_NEURONX_DISPATCH_DEBUG("[TryHandleEmpty] index op empty self → skip", " op=", op_name);
      return false;
    }
    // Self is non-empty, some other arg (index) is empty → fall through to Meta + zeros.

  } else {
    // ── Category 3+4: Non-index ops (reductions, general) ──
    // For top-level tensors: handle if any is empty.
    // For list args (e.g. cat): handle only if ALL tensors in the list are empty.
    // Mixed empty + non-empty lists are handled by the torch_fn which filters empties.
    bool has_empty = false;
    const auto& schema_args = schema.arguments();
    for (size_t i = 0; i < args.size() && i < schema_args.size(); ++i) {
      if (schema_args[i].kwarg_only()) continue;
      if (args[i].isTensor()) {
        auto t = args[i].toTensor();
        if (t.defined() && t.is_privateuseone() && t.numel() == 0 && t.dim() > 0) {
          has_empty = true;
          break;
        }
      } else if (args[i].isList()) {
        bool all_empty = true, has_tensors = false;
        for (const auto& elem : args[i].toListRef()) {
          if (!elem.isTensor()) continue;
          auto t = elem.toTensor();
          if (!t.defined() || !t.is_privateuseone()) continue;
          has_tensors = true;
          if (t.numel() != 0 || t.dim() == 0) {
            all_empty = false;
            break;
          }
        }
        if (has_tensors && all_empty) {
          has_empty = true;
          break;
        }
      }
    }
    if (!has_empty) return false;
  }

  // ════════════════════════════════════════════════════════════
  //  Handling: produce correct outputs for empty tensor cases.
  // ════════════════════════════════════════════════════════════
  TORCH_NEURONX_DISPATCH_DEBUG("[TryHandleEmpty] ENTER handling", " op=", op_name);

  try {
    auto device = ExtractDevice(args);

    // ── 1a. Infer output shapes/dtypes via Meta dispatch ──
    auto meta_specs = InferOutputsViaMeta(op, args);

    TORCH_NEURONX_DISPATCH_DEBUG("[TryHandleEmpty] Meta inferred", " op=", op_name,
                                 " num_outputs=", meta_specs.size());

    // Helpers: create tensors on CPU then move to Neuron to avoid reentrancy.
    // torch::zeros/full on Neuron would re-enter the dispatcher via fill_.
    // torch::empty on Neuron is safe — goes through allocator only.
    auto make_zeros = [&](const MetaOutputSpec& spec) {
      auto cpu_opts = torch::TensorOptions().dtype(spec.dtype).device(c10::kCPU);
      return torch::zeros(spec.shape, cpu_opts).to(device);
    };
    auto make_empty = [&](const MetaOutputSpec& spec) {
      auto opts = torch::TensorOptions().dtype(spec.dtype).device(device);
      return torch::empty(spec.shape, opts);
    };
    auto make_full = [&](const MetaOutputSpec& spec, double fill_value) {
      auto cpu_opts = torch::TensorOptions().dtype(spec.dtype).device(c10::kCPU);
      return torch::full(spec.shape, fill_value, cpu_opts).to(device);
    };

    // ── 1b. Create output tensors with correct fill values ──
    std::vector<torch::Tensor> outputs;
    outputs.reserve(meta_specs.size());

    auto* reduction_cfg = GetReductionConfig(op_name);

    if (reduction_cfg) {
      // ── Reduction: fill with identity value ──
      TORCH_NEURONX_DISPATCH_DEBUG("[TryHandleEmpty] reduction op", " op=", op_name);

      if (reduction_cfg->fill_type == ReductionConfig::FillType::kError) {
        TORCH_CHECK(false, "Cannot perform ", op_name, " on empty tensor (no identity value)");
      }

      double fill_value = 0.0;
      if (reduction_cfg->fill_type == ReductionConfig::FillType::kNaN) {
        fill_value = std::numeric_limits<double>::quiet_NaN();
      } else {
        fill_value = reduction_cfg->identity_value;
      }

      for (size_t i = 0; i < meta_specs.size(); ++i) {
        const auto& spec = meta_specs[i];
        if (spec.is_undefined) {
          outputs.push_back(torch::Tensor());
        } else if (i == 0) {
          // First output gets the identity value
          outputs.push_back(make_full(spec, fill_value));
        } else {
          // Additional outputs (like indices) get zeros
          outputs.push_back(make_zeros(spec));
        }
      }

    } else if (index_cfg) {
      // ── Index op: return input tensor if index is empty ──
      TORCH_NEURONX_DISPATCH_DEBUG("[TryHandleEmpty] index op", " op=", op_name);

      bool returned_input = false;
      int idx_argnum = index_cfg->index_argnum;
      int inp_argnum = index_cfg->input_argnum;

      if (idx_argnum < static_cast<int>(args.size()) &&
          inp_argnum < static_cast<int>(args.size()) && args[inp_argnum].isTensor()) {
        auto input_tensor = args[inp_argnum].toTensor();

        if (input_tensor.defined() && input_tensor.numel() > 0) {
          // Check if the index arg is empty
          bool index_is_empty = false;
          if (args[idx_argnum].isTensor()) {
            index_is_empty = args[idx_argnum].toTensor().numel() == 0;
          } else if (args[idx_argnum].isList()) {
            index_is_empty = true;
            for (const auto& elem : args[idx_argnum].toListRef()) {
              if (elem.isTensor() && elem.toTensor().defined() && elem.toTensor().numel() != 0) {
                index_is_empty = false;
                break;
              }
            }
          }

          if (index_is_empty) {
            outputs.push_back(input_tensor);
            returned_input = true;
          }
        }
      }

      if (!returned_input) {
        // Fallback: create zeros
        for (const auto& spec : meta_specs) {
          if (spec.is_undefined) {
            outputs.push_back(torch::Tensor());
          } else {
            bool has_zero_dim =
                std::any_of(spec.shape.begin(), spec.shape.end(), [](int64_t d) { return d == 0; });
            outputs.push_back(has_zero_dim ? make_empty(spec) : make_zeros(spec));
          }
        }
      }

    } else {
      // ── General case: create zeros ──
      for (const auto& spec : meta_specs) {
        if (spec.is_undefined) {
          outputs.push_back(torch::Tensor());
        } else {
          bool has_zero_dim =
              std::any_of(spec.shape.begin(), spec.shape.end(), [](int64_t d) { return d == 0; });
          outputs.push_back(has_zero_dim ? make_empty(spec) : make_zeros(spec));
        }
      }
    }

    // ── 1c. Handle out= / inplace tensors ──
    const auto& schema_args = schema.arguments();
    const auto& returns = schema.returns();
    for (size_t i = 0; i < args.size() && i < schema_args.size(); ++i) {
      const auto& alias_info = schema_args[i].alias_info();
      if (!alias_info || !alias_info->isWrite()) continue;
      if (!args[i].isTensor()) continue;

      auto out_tensor = args[i].toTensor();
      if (!out_tensor.defined()) continue;

      for (size_t ret_idx = 0; ret_idx < returns.size(); ++ret_idx) {
        const auto& ret_alias = returns[ret_idx].alias_info();
        if (!ret_alias || !ret_alias->isWrite()) continue;
        if (ret_alias->beforeSets() != alias_info->beforeSets()) continue;
        if (ret_idx >= outputs.size() || !outputs[ret_idx].defined()) break;

        ResizeOutputWithWarning(out_tensor, outputs[ret_idx].sizes());
        out_tensor.copy_(outputs[ret_idx]);
        outputs[ret_idx] = out_tensor;
        break;
      }
    }

    // ── 1d. Push outputs to stack ──
    PushOutputsToStack(stack, outputs, schema);
    logOperationExecuted(op_name);

    TORCH_NEURONX_DISPATCH_DEBUG("[TryHandleEmpty] EXIT (Layer 1 — Meta + fill)", " op=", op_name);
    return true;

  } catch (const std::exception& e) {
    TORCH_NEURONX_DISPATCH_DEBUG("[TryHandleEmpty] Layer 1 failed, falling back to CPU",
                                 " op=", op_name, " error=", e.what());
  }

  // ════════════════════════════════════════════════
  //  Layer 2: CPU fallback (safety net)
  // ════════════════════════════════════════════════
  for (const auto& arg : args) {
    torch::jit::push(*stack, arg);
  }
  at::native::cpu_fallback(op, stack);
  logOperationExecuted(op_name);

  TORCH_NEURONX_DISPATCH_DEBUG("[TryHandleEmpty] EXIT (Layer 2 — CPU fallback)", " op=", op_name);
  return true;
}

// ═══════════════════════════════════════════════════════════════
//  EnqueueXlaOperation — dispatches to KernelExecution
// ═══════════════════════════════════════════════════════════════

void NeuronDispatcher::EnqueueXlaOperation(
    const std::string& op_name, const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& outputs, const std::vector<uint8_t>& stablehlo_bytes,
    const std::string& neff_cache_key, bool has_collectives, int device_id,
    at::neuron::CPUFallbackContext fallback_ctx, const std::string& hierarchy_string) {
  TORCH_NEURONX_DISPATCH_DEBUG("[EnqueueXlaOperation] op=", op_name, " inputs=", inputs.size(),
                               " outputs=", outputs.size(), " neff_key=", neff_cache_key,
                               " has_collectives=", has_collectives, " device_id=", device_id);

  auto transform_result =
      torch_neuronx::ProcessInputTransformations(op_name, inputs, has_collectives);

  auto stream = at::neuron::getCurrentNeuronStream();

  auto [input_refs, input_contexts] =
      torch_neuronx::PrepareKernelTensors(transform_result.final_inputs);
  auto [output_refs, output_contexts] = torch_neuronx::PrepareKernelTensors(outputs);

  auto xla_kernel = std::make_unique<at::neuron::XLACompilableKernelExecution>(
      op_name, std::move(input_refs), std::move(output_refs), input_contexts, output_contexts,
      neff_cache_key, stablehlo_bytes, has_collectives, device_id);

  if (transform_result.has_input_transforms) {
    xla_kernel->SetInputTransformations(std::move(transform_result.kernel_transforms));
  }

  at::neuron::CPUFallbackContext fb_ctx = std::move(fallback_ctx);
  auto context = std::make_unique<at::neuron::OperationContext>(
      std::move(xla_kernel), /*python_stack_trace=*/"", std::move(fb_ctx), hierarchy_string);
  at::neuron::SubmitOperationContext(stream, std::move(context));
}

}  // namespace dispatch
}  // namespace torch_neuronx

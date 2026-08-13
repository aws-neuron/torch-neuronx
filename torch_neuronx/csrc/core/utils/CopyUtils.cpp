#include "CopyUtils.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/Logging.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <numeric>

#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/NeuronOpTracking.h"
#include "torch_neuronx/csrc/core/NeuronStorageImpl.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/allocator/NeuronHostAllocator.h"
#include "torch_neuronx/csrc/core/utils/NonTemporalMemcpy.h"
#include "torch_neuronx/csrc/ops/InternalOps.h"

namespace torch_neuronx {
namespace utils {

nrt_tensor_t* get_nrt_tensor(const at::Tensor& tensor, bool non_blocking) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
              "Expected Neuron tensor, got ", tensor.device());

  // Get the storage implementation
  auto storage_impl = tensor.storage().unsafeGetStorageImpl();
  auto neuron_storage = static_cast<c10_neuron::NeuronStorageImpl*>(storage_impl);
  return neuron_storage->neuron_tensor();
}

void copy_cpu_to_neuron(const at::Tensor& src, at::Tensor& dst, bool non_blocking) {
  TORCH_CHECK(src.is_cpu(), "Source tensor must be CPU tensor");
  TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1,
              "Destination must be Neuron tensor");

  TORCH_CHECK(src.is_contiguous(),
              "Source CPU tensor must be contiguous. "
              "Please call .contiguous() on CPU tensor before copying to Neuron");
  TORCH_CHECK(dst.is_contiguous(), "Destination Neuron tensor must be contiguous");

  size_t size_bytes = src.nbytes();
  // Skip copy if there are no bytes to copy (empty tensor)
  if (size_bytes == 0) {
    return;
  }
  int device = dst.device().index();
  size_t byte_offset = dst.storage_offset() * dst.element_size();

  void* dst_ptr = dst.storage().mutable_data();
  auto dst_tensor_ptr = c10_neuron::NeuronCachingAllocator::findTensorPtr(dst_ptr);

  auto write_kernel = std::make_unique<at::neuron::WriteDirectKernelExecution>(
      "neuron::copy::cpu_to_neuron", src.data_ptr(),
      at::neuron::TensorDataRef{std::move(dst_tensor_ptr), dst_ptr}, byte_offset, size_bytes,
      device);

  auto stream = at::neuron::getCurrentNeuronStream(device);
  bool use_bounce = non_blocking || src.numel() == 1;

  if (use_bounce) {
    auto* host_alloc = torch_neuronx::GetNeuronHostAllocatorAsHostAllocator();
    bool recorded = host_alloc->record_event(src.data_ptr(), src.storage().data_ptr().get_context(),
                                             c10::Stream(stream.unwrap()));
    // record_event returns false when src wasn't allocated by NeuronHostAllocator
    // (e.g. regular CPU tensor), so fall back to bounce buffer for safe async copy.
    if (!recorded) {
      write_kernel->AllocateBounceBuffer(size_bytes);
    }
  }

  auto context = std::make_unique<at::neuron::OperationContext>(std::move(write_kernel));
  at::neuron::SubmitOperationContext(stream, std::move(context));
  if (!use_bounce) {
    stream.synchronize();
  }
}

void copy_neuron_to_cpu(const at::Tensor& src, at::Tensor& dst, bool non_blocking) {
  TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1, "Source must be Neuron tensor");
  TORCH_CHECK(dst.is_cpu(), "Destination tensor must be CPU tensor");

  TORCH_CHECK(src.is_contiguous(),
              "Source Neuron tensor must be contiguous for direct copy to CPU");
  TORCH_CHECK(dst.is_contiguous(),
              "Destination CPU tensor must be contiguous for direct copy from Neuron");

  size_t size_bytes = dst.nbytes();
  TORCH_CHECK(src.nbytes() == size_bytes, "Source and destination byte counts must match (source ",
              src.nbytes(), " bytes, destination ", size_bytes, " bytes)");
  // Skip copy if there are no bytes to copy (empty tensor)
  if (size_bytes == 0) {
    return;
  }

  int device = src.device().index();
  void* dst_data_ptr = dst.data_ptr();
  // Calculate offset based on tensor's storage offset
  const auto storage_offset = src.storage_offset();
  const size_t element_size = src.element_size();
  TORCH_CHECK(storage_offset >= 0, "Source storage offset must be non-negative");
  TORCH_CHECK(
      static_cast<uint64_t>(storage_offset) <= std::numeric_limits<size_t>::max() / element_size,
      "Source storage offset overflows byte offset");
  size_t byte_offset = static_cast<size_t>(storage_offset) * element_size;
  const size_t storage_bytes = src.storage().nbytes();
  TORCH_CHECK(byte_offset <= storage_bytes && size_bytes <= storage_bytes - byte_offset,
              "Direct Neuron-to-CPU copy exceeds source storage bounds");

  auto stream = at::neuron::getCurrentNeuronStream(device);
  void* src_ptr = src.storage().mutable_data();
  auto src_tensor_ptr = c10_neuron::NeuronCachingAllocator::findTensorPtr(src_ptr);
  auto read_kernel = std::make_unique<at::neuron::ReadDirectKernelExecution>(
      "neuron::copy::neuron_to_cpu", at::neuron::TensorDataRef{std::move(src_tensor_ptr), src_ptr},
      dst_data_ptr, byte_offset, size_bytes, device);
  auto context = std::make_unique<at::neuron::OperationContext>(std::move(read_kernel));

  // record_event registers the stream on the host block; the allocator records
  // the actual completion event if the block is released. Register before
  // enqueue so an exception cannot leave accepted DMA untracked.
  bool async_lifetime_registered = false;
  if (non_blocking) {
    auto* host_alloc = torch_neuronx::GetNeuronHostAllocatorAsHostAllocator();
    async_lifetime_registered = host_alloc->record_event(
        dst_data_ptr, dst.storage().data_ptr().get_context(), c10::Stream(stream.unwrap()));
  }

  try {
    at::neuron::SubmitOperationContext(stream, std::move(context));
  } catch (...) {
    auto submit_error = std::current_exception();
    try {
      stream.synchronize();
    } catch (...) {
      // Preserve the submission exception after draining any accepted work.
    }
    std::rethrow_exception(submit_error);
  }

  // Pageable/foreign destinations cannot be protected by the host allocator,
  // so non_blocking is best-effort and must synchronize before returning.
  if (!async_lifetime_registered) {
    stream.synchronize();
  }
}

void nrt_copy_neuron_to_cpu(nrt_tensor_t* src_data_ptr, void* dst_data_ptr, size_t src_offset,
                            size_t size) {
  TORCH_CHECK(src_data_ptr != nullptr, "Source data pointer must not be null");
  TORCH_CHECK(size > 0, "Destination size must be valid");
  NRT_STATUS status = nrt_tensor_read(src_data_ptr, dst_data_ptr, src_offset, size);
  if (status != NRT_SUCCESS) {
    throw std::runtime_error("Failed to copy data from Neuron device. Status: " +
                             std::to_string(status));
  }
}

void nrt_copy_cpu_to_neuron(void* src_data_ptr, nrt_tensor_t* dst_data_ptr, size_t dst_offset,
                            size_t size) {
  TORCH_CHECK(dst_data_ptr != nullptr, "Destination data pointer must not be null");
  TORCH_CHECK(size > 0, "Destination size must be valid");
  // Skip copy if there are no bytes to copy (empty tensor)
  NRT_STATUS status = nrt_tensor_write(dst_data_ptr, src_data_ptr, dst_offset, size);
  if (status != NRT_SUCCESS) {
    throw std::runtime_error("Failed to copy data to Neuron device. Status: " +
                             std::to_string(status));
  }
}

void copy_neuron_to_neuron(const at::Tensor& src, at::Tensor& dst) {
  TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1, "Source must be Neuron tensor");
  TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1,
              "Destination must be Neuron tensor");

  // Check if on same device (required by nrt_tensor_copy for D2D)
  TORCH_CHECK(src.device().index() == dst.device().index(),
              "Device-to-device copy requires tensors on same Neuron core");

  // Callers must ensure src is contiguous or handle layout themselves.
  TORCH_CHECK(dst.is_contiguous(), "Destination Neuron tensor must be contiguous");

  // Ensure tensors have same size
  TORCH_CHECK(src.nbytes() == dst.nbytes(),
              "Source and destination must have same number of bytes");

  int device = src.device().index();
  size_t size_bytes = src.nbytes();
  // Skip copy if there are no bytes to copy (empty tensor)
  if (size_bytes == 0) {
    return;
  }

  // Calculate offsets based on tensor's storage offset
  size_t src_byte_offset = src.storage_offset() * src.element_size();
  size_t dst_byte_offset = dst.storage_offset() * dst.element_size();

  void* src_ptr = src.storage().mutable_data();
  void* dst_ptr = dst.storage().mutable_data();
  auto src_tensor_ptr = c10_neuron::NeuronCachingAllocator::findTensorPtr(src_ptr);
  auto dst_tensor_ptr = c10_neuron::NeuronCachingAllocator::findTensorPtr(dst_ptr);
  auto copy_kernel = std::make_unique<at::neuron::CopyDirectKernelExecution>(
      "neuron::copy::neuron_to_neuron",
      at::neuron::TensorDataRef{std::move(src_tensor_ptr), src_ptr},
      at::neuron::TensorDataRef{std::move(dst_tensor_ptr), dst_ptr}, src_byte_offset,
      dst_byte_offset, size_bytes, device);
  auto context = std::make_unique<at::neuron::OperationContext>(std::move(copy_kernel));
  auto stream = at::neuron::getCurrentNeuronStream(device);
  at::neuron::SubmitOperationContext(stream, std::move(context));
}

namespace {

// Return S if `t` has a uniform stride-S layout over its storage: strides ==
// S * contiguous_strides(shape) with S >= 2 (S is the innermost stride), i.e.
// its elements sit at offsets base, base+S, base+2S, ... — exactly what the NKI
// stride-S gather/scatter kernel treats as a group-stream. Returns 0 otherwise
// (contiguous, overlapping, or non-uniform, e.g. an odd-row-strided slice).
// Size-1 dims are ignored. Mirrors _uniform_stride in nki_strided_copy_op.py.
int64_t UniformStride(const at::Tensor& t) {
  if (t.dim() == 0) return 0;
  const int64_t stride = t.stride(-1);
  if (stride < 2) return 0;
  int64_t expected = stride;
  for (int64_t d = t.dim() - 1; d >= 0; --d) {
    const int64_t n = t.size(d);
    if (n == 1) continue;  // stride immaterial for a size-1 dim
    if (t.stride(d) != expected) return 0;
    expected *= n;
  }
  return stride;
}

// Whether this copy is a uniform stride-S gather/scatter the on-device kernel
// handles: one side contiguous, the other a uniform stride-S (S>=2) lane view.
// Any stride is SBUF-safe (the kernel bounds its loaded staging independently of
// stride) and any storage span is safe (the kernel reads the last, possibly
// truncated, group only up to the lane), so there is no stride cap or full-group
// requirement. Must stay a subset of what nki_strided_copy_ handles so the op
// never falls back into copy_strided (no recursion).
// Dtypes the strided-copy kernel handles correctly — fixed-width 1/2/4-byte int
// and float types.
//
// Deliberately a SEPARATE list from IsSupportedDtype (DispatchUtils.cpp): it
// includes Bool/ComplexFloat and EXCLUDES the narrow ints (Short/Char/Byte)
// because the compiler mishandles arithmetic on them — but this kernel is pure
// data movement (byte copy, no compute), so narrow ints are safe here. What must
// NOT go through the kernel: bool/float64/complex (fail to lower), and int64
// (lowers WITHOUT error but the Vector-Engine lane-select copy is not bit-exact
// beyond ~2^24 — it SILENTLY CORRUPTS large int64). Mirrors
// _STRIDED_COPY_KERNEL_DTYPES in nki_strided_copy_op.py.
//
// TODO: strided-copy-specific allowlist; remove once a shared copy-supported-
// dtype helper covers exactly these types.
bool StridedCopyKernelSupportsDtype(c10::ScalarType dt) {
  switch (dt) {
    case c10::ScalarType::Float:
    case c10::ScalarType::Half:
    case c10::ScalarType::BFloat16:
    case c10::ScalarType::Int:
    case c10::ScalarType::Short:
    case c10::ScalarType::Char:
    case c10::ScalarType::Byte:
      return true;
    default:
      return false;
  }
}

bool IsNkiStridedCopyEligible(const at::Tensor& src, const at::Tensor& dst) {
  // Only dtypes the strided-copy kernel handles correctly take the kernel path;
  // everything else (bool, float64, int64, complex, ...) stays on per-element.
  if (!StridedCopyKernelSupportsDtype(src.scalar_type())) return false;
  if (src.is_contiguous() && !dst.is_contiguous()) {
    return UniformStride(dst) >= 2;
  }
  if (dst.is_contiguous() && !src.is_contiguous()) {
    return UniformStride(src) >= 2;
  }
  return false;
}

// Invoke the Python-registered NKI stride-S gather/scatter op. Mirrors
// run_nki_strided_copy in PackedInt64Utils.cpp (typed handle resolved once).
void RunNkiStridedCopy(const at::Tensor& src, at::Tensor& dst) {
  static const auto op = c10::Dispatcher::singleton()
                             .findSchemaOrThrow("neuron::nki_strided_copy_", "")
                             .typed<at::Tensor(const at::Tensor&, at::Tensor&)>();
  op.call(src, dst);
}

}  // namespace

void copy_strided_neuron_to_neuron(const at::Tensor& src, at::Tensor& dst) {
  TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1,
              "copy_strided: expected src to be a neuron tensor");
  TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1,
              "copy_strided: expected dst to be a neuron tensor");
  TORCH_CHECK(src.sizes().equals(dst.sizes()),
              "copy_strided: src and dst must have the same shape");
  TORCH_CHECK(src.dtype() == dst.dtype(), "copy_strided: src and dst must have the same dtype");
  TORCH_CHECK(src.device().index() == dst.device().index(),
              "copy_strided: src and dst must be on the same Neuron core");

  if (src.numel() == 0) {
    return;
  }

  // Fast path: a uniform stride-S gather/scatter (one side contiguous, the other
  // a uniform stride-S lane view) runs on-device via the NKI kernel in a single
  // coalesced, stream-ordered launch — instead of the per-element, host-blocking
  // nrt_tensor_copy loop below. Only kernel-eligible patterns are routed, so
  // the op never falls back into this function (no recursion).
  if (IsNkiStridedCopyEligible(src, dst)) {
    RunNkiStridedCopy(src, dst);
    return;
  }

  // Drain the stream so all pending operations complete before we issue
  // direct nrt_tensor_copy calls outside the stream.
  auto stream = at::neuron::getCurrentNeuronStream(src.device().index());
  stream.synchronize();

  const auto element_size = static_cast<size_t>(src.element_size());
  const auto ndim = static_cast<int64_t>(src.dim());

  // Resolve NRT handles
  nrt_tensor_t* src_nrt = get_nrt_tensor(src);
  nrt_tensor_t* dst_nrt = get_nrt_tensor(dst);
  size_t src_base_offset = src.storage_offset() * element_size;
  size_t dst_base_offset = dst.storage_offset() * element_size;

  // Fast path: both contiguous — single bulk copy
  if (src.is_contiguous() && dst.is_contiguous()) {
    size_t nbytes = static_cast<size_t>(src.numel()) * element_size;
    NRT_STATUS status = nrt_tensor_copy(src_nrt, src_base_offset, dst_nrt, dst_base_offset, nbytes);
    TORCH_CHECK(status == NRT_SUCCESS,
                "copy_strided: nrt_tensor_copy failed (status=", static_cast<int>(status), ")");
    return;
  }

  // Recursive strided copy — walks dimensions and copies contiguous spans
  auto sizes = src.sizes();
  auto src_strides = src.strides();
  auto dst_strides = dst.strides();

  std::function<void(int64_t, size_t, size_t)> recurse = [&](int64_t dim, size_t src_off,
                                                             size_t dst_off) {
    if (dim == ndim) {
      // Scalar element — copy single element
      NRT_STATUS status = nrt_tensor_copy(src_nrt, src_off, dst_nrt, dst_off, element_size);
      TORCH_CHECK(status == NRT_SUCCESS, "copy_strided: nrt_tensor_copy failed");
      return;
    }

    const int64_t n = sizes[dim];
    if (n == 0) return;

    const int64_t src_step = src_strides[dim];
    const int64_t dst_step = dst_strides[dim];

    // If innermost dim and both strides are 1, copy the whole span at once
    if (dim == ndim - 1 && src_step == 1 && dst_step == 1) {
      size_t nbytes = static_cast<size_t>(n) * element_size;
      NRT_STATUS status = nrt_tensor_copy(src_nrt, src_off, dst_nrt, dst_off, nbytes);
      TORCH_CHECK(status == NRT_SUCCESS, "copy_strided: nrt_tensor_copy failed");
      return;
    }

    for (int64_t i = 0; i < n; ++i) {
      recurse(dim + 1, src_off + static_cast<size_t>(i * src_step) * element_size,
              dst_off + static_cast<size_t>(i * dst_step) * element_size);
    }
  };

  recurse(0, src_base_offset, dst_base_offset);
}

void write_to_noncontiguous_neuron(at::Tensor& dst, const at::Tensor& src) {
  // Mirrors Python _write_to_noncontiguous_neuron from cast_policy.py.
  // Writes contiguous src into non-contiguous dst via CPU bounce:
  //   1. Read dst's entire storage to CPU
  //   2. Do the strided write on CPU
  //   3. Write the storage back to Neuron
  TORCH_CHECK(src.is_contiguous(), "write_to_noncontiguous_neuron expects contiguous src");
  TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1,
              "write_to_noncontiguous_neuron expects Neuron dst");
  TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1,
              "write_to_noncontiguous_neuron expects Neuron src");

  if (dst.is_contiguous()) {
    copy_neuron_to_neuron(src, dst);
    return;
  }

  // CPU bounce: read dst's full storage, scatter-write src on CPU, write back.
  // Uses flat 1D view of dst's storage to satisfy contiguity preconditions
  // of the existing copy APIs (avoids manual kernel construction).
  //
  // NOTE: this is NOT a whole-op CPU fallback — the copy_ still runs on device;
  // only the scatter into the non-contiguous destination round-trips through the
  // host. It is reported under a distinct name so it does not shadow the bare
  // "aten::copy_" that legitimately appears in the executed-ops list (which
  // tests such as test_pad_decomposition_to_correct_ops key on).
  torch_neuronx::markFallbackLogged("aten::copy_[noncontiguous_scatter_cpu_bounce]");

  size_t storage_bytes = dst.storage().nbytes();
  int64_t storage_elems = static_cast<int64_t>(storage_bytes / dst.element_size());

  // Flat 1D contiguous view of dst's entire storage.
  auto dst_flat = dst.as_strided({storage_elems}, {1}, 0);

  // 1. Read dst's entire storage to CPU (preserves data in gaps between view elements)
  auto cpu_buffer = torch::empty(
      {storage_elems}, torch::TensorOptions().dtype(dst.scalar_type()).device(torch::kCPU));
  copy_neuron_to_cpu(dst_flat, cpu_buffer, /*non_blocking=*/false);

  // 2. Read src to CPU
  auto src_cpu = torch::empty(src.sizes().vec(),
                              torch::TensorOptions().dtype(src.scalar_type()).device(torch::kCPU));
  copy_neuron_to_cpu(src, src_cpu, /*non_blocking=*/false);

  // 3. Strided write on CPU — scatter src into dst's storage layout
  auto cpu_strided =
      cpu_buffer.as_strided(dst.sizes().vec(), dst.strides().vec(), dst.storage_offset());
  cpu_strided.copy_(src_cpu);

  // 4. Write storage back to Neuron
  copy_cpu_to_neuron(cpu_buffer, dst_flat, /*non_blocking=*/false);
}

// ════════════════════════════════════════════════════════════════════════════
// Permutation analysis — detect simple transpose from strides
// ════════════════════════════════════════════════════════════════════════════

/// Determine whether `strides` represent a simple permutation of a contiguous
/// layout for the given `shape`.
///
/// Returns the permutation vector (e.g. [0,2,1,3] for a dim-1/dim-2 swap).
/// Returns empty vector when:
///   - tensor is already contiguous (identity permutation — no work needed)
///   - any stride is 0 (broadcast)
///   - strides don't match any permutation of contiguous strides (sliced views,
///     non-dense storage with gaps)
static std::vector<int64_t> GetPermutationFromStrides(at::IntArrayRef shape,
                                                      at::IntArrayRef strides) {
  const int64_t ndim = static_cast<int64_t>(shape.size());
  if (ndim == 0) return {};

  // Reject broadcast (stride == 0)
  for (int64_t i = 0; i < ndim; ++i) {
    if (strides[i] == 0) return {};
  }

  // Sort dimension indices by stride descending.
  // For a dense transposed tensor, this recovers the permutation.
  std::vector<int64_t> perm(ndim);
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(),
            [&](int64_t a, int64_t b) { return strides[a] > strides[b]; });

  // Verify the permutation reproduces the strides exactly.
  // Walk from innermost permuted dim outward: stride must equal the running
  // product of inner dimension sizes.  If not, the view is non-dense (e.g.
  // tensor[::2] has stride 2*element on the sliced dim, but shape/2 elements).
  int64_t expected_stride = 1;
  for (int64_t i = ndim - 1; i >= 0; --i) {
    const int64_t dim = perm[i];
    if (strides[dim] != expected_stride) return {};
    expected_stride *= shape[dim];
  }

  // Identity check — already contiguous, nothing to do.
  bool is_identity = true;
  for (int64_t i = 0; i < ndim; ++i) {
    if (perm[i] != i) {
      is_identity = false;
      break;
    }
  }
  if (is_identity) return {};

  return perm;
}

// ════════════════════════════════════════════════════════════════════════════
// CopyToNonContiguous — on-device fast path + CPU bounce fallback
// ════════════════════════════════════════════════════════════════════════════

void CopyToNonContiguous(at::Tensor& dst, const at::Tensor& src) {
  TORCH_CHECK(src.is_contiguous(), "CopyToNonContiguous: src must be contiguous");
  TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1,
              "CopyToNonContiguous: dst must be a Neuron tensor");
  TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1,
              "CopyToNonContiguous: src must be a Neuron tensor");

  // Contract: callers must ensure dtype match. A mismatch would silently
  // produce garbage data via the flat D2D copy (no element-wise conversion).
  TORCH_INTERNAL_ASSERT(src.scalar_type() == dst.scalar_type(),
                        "CopyToNonContiguous: dtype mismatch — src is ", src.scalar_type(),
                        " but dst is ", dst.scalar_type(),
                        ". Caller must cast before calling CopyToNonContiguous.");

  // Fast path: dst is actually contiguous
  if (dst.is_contiguous()) {
    copy_neuron_to_neuron(src, dst);
    return;
  }

  if (src.numel() == 0) return;

  // ── On-device path: transpose + flat D2D copy ──
  auto perm = GetPermutationFromStrides(dst.sizes(), dst.strides());

  // The on-device path flattens dst's entire storage to a 1D view and does a
  // bulk D2D copy.  This is only valid when:
  //   1. storage_offset == 0 — otherwise we'd write to the wrong region
  //   2. storage size == numel * element_size — otherwise dst is a view into
  //      a larger allocation (e.g. base[:3,:].transpose(0,1) where base is 4x2:
  //      storage has 8 elements but dst.numel() is only 6)
  // GetPermutationFromStrides validates that the strides form a dense permutation
  // of dst's shape, but cannot detect these two conditions.
  size_t dst_logical_bytes = static_cast<size_t>(dst.numel()) * dst.element_size();
  bool storage_is_exact =
      (dst.storage_offset() == 0) && (dst.storage().nbytes() == dst_logical_bytes);
  if (!perm.empty() && storage_is_exact) {
    TORCH_NEURONX_DEBUG(
        "CopyToNonContiguous: on-device transpose path", " perm=[",
        [&]() {
          std::string s;
          for (size_t i = 0; i < perm.size(); ++i) {
            if (i > 0) s += ",";
            s += std::to_string(perm[i]);
          }
          return s;
        }(),
        "]", " dst_shape=", dst.sizes(), " dst_strides=", dst.strides());

    // src.permute(perm) creates a VIEW — reorders dimensions, no data movement.
    // contiguous_internal materializes the new layout on-device via MLIR
    // transpose kernel (compiled once per shape+perm, cached forever).
    //
    // After this, src_permuted is contiguous and its raw bytes are in the same
    // physical order as dst's storage.
    auto src_permuted = torch_neuronx::ops::contiguous_internal(src.permute(at::IntArrayRef(perm)),
                                                                c10::MemoryFormat::Contiguous);

    // Flatten dst's underlying storage to 1D and do a single bulk D2D copy.
    // This is safe because storage_is_exact guarantees:
    //   - storage_offset == 0 (no base offset)
    //   - storage.nbytes() == numel * element_size (no extra storage from slicing)
    int64_t storage_elems = static_cast<int64_t>(dst.storage().nbytes() / dst.element_size());
    auto dst_flat = dst.as_strided({storage_elems}, {1}, 0);
    auto src_flat = src_permuted.reshape({-1});

    copy_neuron_to_neuron(src_flat, dst_flat);
    return;
  }

  // ── Fallback: complex stride pattern → CPU bounce ──
  // Handles: broadcast (stride=0), sliced views (tensor[::2]), non-dense storage,
  // nonzero storage_offset on transposed views.
  TORCH_NEURONX_DEBUG("CopyToNonContiguous: CPU bounce fallback", " dst_shape=", dst.sizes(),
                      " dst_strides=", dst.strides(), " storage_offset=", dst.storage_offset());
  write_to_noncontiguous_neuron(dst, src);
}

}  // namespace utils
}  // namespace torch_neuronx

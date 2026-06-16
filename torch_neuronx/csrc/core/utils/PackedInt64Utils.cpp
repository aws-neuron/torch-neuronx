#include "torch_neuronx/csrc/core/utils/PackedInt64Utils.h"

#include <ATen/core/dispatch/Dispatcher.h>

#include <cstdlib>
#include <string>

#include "torch_neuronx/csrc/core/utils/CopyUtils.h"

namespace torch_neuronx {
namespace packed_int64 {

namespace {

// Gather the stride-2 low-word `alias` into the contiguous int32 `dst` using
// the NKI strided-copy kernel (registered in Python as neuron::nki_strided_copy_).
//
// The typed operator handle is resolved once and cached in a function-local
// static: findSchemaOrThrow hashes the schema string and walks the dispatcher
// table, which is wasteful to repeat on every cast. Mirrors the pattern used in
// the runtime's neuron_tensor_ops (static auto op = ...).
void run_nki_strided_copy(const torch::Tensor& src, torch::Tensor& dst) {
  static const auto op = c10::Dispatcher::singleton()
                             .findSchemaOrThrow("neuron::nki_strided_copy_", "")
                             .typed<at::Tensor(const at::Tensor&, at::Tensor&)>();
  op.call(src, dst);
}

// Whether the packed-int64 overflow check is enabled. Read once from
// TORCH_NEURONX_PACKED_INT64_CHECK_OVERFLOW (default off — the check roughly
// doubles the gather cost and forces a device sync, so it is a debug aid).
bool overflow_check_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("TORCH_NEURONX_PACKED_INT64_CHECK_OVERFLOW");
    return env != nullptr && std::string(env) == "1";
  }();
  return enabled;
}

// Verify every packed int64 element is representable in int32: the stored high
// lane must equal the sign extension of the low lane (low >> 31). Raises if any
// element is out of range. `low` is the already-gathered contiguous low words;
// `t` is the source int64 tensor (device, non-empty).
//
// Mirrors the Python check in utils.convert_for_neuron so both dispatch paths
// behave identically. Validates *inputs* at downcast time, not results after
// compute (a value that fits going in can still overflow during the op).
void check_no_overflow(const torch::Tensor& t, const torch::Tensor& low) {
  auto high_alias = high_word_alias(t);
  auto high = torch::empty_like(low);
  run_nki_strided_copy(high_alias, high);
  // Arithmetic right shift by 31 yields 0 for non-negative, -1 for negative —
  // the expected high lane for an in-range int32 value.
  auto expected_high = at::bitwise_right_shift(low, 31);
  // Mirror the Python decomposition (eq -> all -> item) rather than at::equal,
  // which keeps the reduction on device and matches the supported op set.
  bool ok = at::eq(high, expected_high).all().item<bool>();
  TORCH_CHECK(ok,
              "int64 tensor contains values outside int32 range "
              "[-2147483648, 2147483647]. Packed int64 mode only supports "
              "values representable in 32 bits. Set "
              "TORCH_NEURONX_INT64_MODE=cpu_bounce for full range.");
}

}  // namespace

torch::Tensor execution_view(const torch::Tensor& t) {
  TORCH_CHECK(t.scalar_type() == c10::ScalarType::Long,
              "packed_int64::execution_view: expected int64 tensor, got ", t.scalar_type());

  // Allocate contiguous int32 output with same logical shape.
  auto dst = torch::empty(t.sizes().vec(), t.options().dtype(c10::ScalarType::Int));

  if (t.numel() == 0) {
    return dst;
  }

  // low_word_alias: int32, stride-2 (lane 0), zero-copy metadata view.
  auto alias = low_word_alias(t);

  if (t.device().type() == c10::DeviceType::PrivateUse1) {
    // On-device: gather low words via the NKI strided-copy kernel. This routes
    // to the burst-load + on-chip SBUF select kernel (~10x faster than a
    // strided HBM read through copy_strided_neuron_to_neuron).
    run_nki_strided_copy(alias, dst);
    if (overflow_check_enabled()) {
      check_no_overflow(t, dst);
    }
  } else {
    // CPU fallback (for testing).
    dst.copy_(alias);
  }

  return dst;
}

torch::Tensor materialize_from_i32(const torch::Tensor& low_words) {
  TORCH_CHECK(low_words.scalar_type() == c10::ScalarType::Int,
              "packed_int64::materialize_from_i32: expected int32 input, got ",
              low_words.scalar_type());

  // Allocate the int64 output. NOTE: torch::empty maps directly to the
  // empty_strided_neuron factory and does NOT redispatch through the Neuron
  // dispatcher. Using a composite op such as torch::zeros here would re-enter
  // the dispatcher, get downcast Long->Int, execute, and then upcast Int->Long
  // via UpcastExecutionOutputs -> NeuronSafeCast -> materialize_from_i32 again,
  // recursing infinitely (stack overflow).
  auto dst =
      torch::empty(low_words.sizes().vec(), low_words.options().dtype(c10::ScalarType::Long));

  if (low_words.numel() == 0) {
    return dst;
  }

  // Ensure a contiguous int32 source for a clean bulk copy.
  auto low = low_words.is_contiguous() ? low_words : low_words.contiguous();

  // Sign-extend the high lane: high = low >> 31. Arithmetic right shift on a
  // signed int32 yields 0 for non-negative values and -1 for negative values,
  // which is exactly the upper 32 bits of the sign-extended int64. A zero high
  // lane (the previous behaviour) would corrupt negative results.
  auto high = at::bitwise_right_shift(low, 31);

  // Interleave into [low, high] pairs: contiguous int32 tensor of shape (*, 2).
  // This matches the packed physical layout (little-endian low word first).
  auto packed_words = at::stack({low, high}, /*dim=*/-1).contiguous();

  // physical_view shares dst's storage as an int32 (*, 2) view. For a freshly
  // allocated contiguous int64 dst it is contiguous, so a single bulk D2D copy
  // writes the packed pairs straight into the int64 storage.
  auto physical = physical_view(dst);
  if (dst.device().type() == c10::DeviceType::PrivateUse1) {
    utils::copy_neuron_to_neuron(packed_words, physical);
  } else {
    // CPU fallback (for testing)
    physical.copy_(packed_words);
  }

  return dst;
}

}  // namespace packed_int64
}  // namespace torch_neuronx

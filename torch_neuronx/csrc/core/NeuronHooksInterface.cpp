#include "torch_neuronx/csrc/core/NeuronHooksInterface.h"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>
#include <ATen/EmptyTensor.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/util/CallOnce.h>

#include "torch_neuronx/csrc/Stream.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"

namespace torch_neuronx {

/**
 * NeuronGeneratorImpl - Philox-based generator for Neuron devices (16-byte state)
 *
 * See Note [Acquire lock when using random generators] in c10/core/GeneratorImpl.h
 * This generator is NOT thread-safe. Callers must use mutex_ for concurrent access.
 */
class NeuronGeneratorImpl : public c10::GeneratorImpl {
 public:
  explicit NeuronGeneratorImpl(c10::DeviceIndex device_index = -1)
      : c10::GeneratorImpl(c10::Device(c10::DeviceType::PrivateUse1, device_index),
                           c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)),
        seed_(c10::default_rng_seed_val),
        philox_offset_(0) {}

  ~NeuronGeneratorImpl() override = default;

  void set_current_seed(uint64_t seed) override {
    seed_ = seed;
    philox_offset_ = 0;
  }

  void set_offset(uint64_t offset) override { philox_offset_ = offset; }

  uint64_t get_offset() const override { return philox_offset_; }

  uint64_t current_seed() const override { return seed_; }

  uint64_t seed() override {
    auto random = c10::detail::getNonDeterministicRandom(true);
    set_current_seed(random);
    return random;
  }

  c10::intrusive_ptr<c10::TensorImpl> get_state() const override {
    constexpr size_t seed_size = sizeof(uint64_t);
    constexpr size_t offset_size = sizeof(int64_t);
    constexpr size_t total_size = seed_size + offset_size;

    auto state_tensor =
        at::detail::empty_cpu({static_cast<int64_t>(total_size)}, at::ScalarType::Byte,
                              std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    auto rng_state = static_cast<uint8_t*>(state_tensor.data_ptr());
    auto offset = static_cast<int64_t>(philox_offset_);
    memcpy(rng_state, &seed_, seed_size);
    memcpy(rng_state + seed_size, &offset, offset_size);
    return state_tensor.getIntrusivePtr();
  }

  void set_state(const c10::TensorImpl& new_state) override {
    constexpr size_t seed_size = sizeof(uint64_t);
    constexpr size_t offset_size = sizeof(int64_t);
    constexpr size_t total_size = seed_size + offset_size;

    auto new_state_size = new_state.numel();
    TORCH_CHECK(new_state_size == total_size || new_state_size == seed_size,
                "RNG state is wrong size");

    auto new_rng_state = static_cast<const uint8_t*>(new_state.data());
    memcpy(&seed_, new_rng_state, seed_size);

    if (new_state_size == total_size) {
      int64_t offset = 0;
      memcpy(&offset, new_rng_state + seed_size, offset_size);
      philox_offset_ = static_cast<uint64_t>(offset);
    } else {
      philox_offset_ = 0;
    }
  }

  static c10::DeviceType device_type() { return c10::DeviceType::PrivateUse1; }

 private:
  NeuronGeneratorImpl* clone_impl() const override {
    auto gen = new NeuronGeneratorImpl(device_.index());
    gen->seed_ = seed_;
    gen->philox_offset_ = philox_offset_;
    return gen;
  }

  uint64_t seed_;
  uint64_t philox_offset_;
};

// Per-device default generators
static std::vector<at::Generator> default_neuron_generators;
static std::unique_ptr<c10::once_flag[]> neuron_gens_init_flag;
static c10::once_flag num_devices_init_flag;
static int64_t num_devices = -1;

static void initNeuronGenVector() {
  c10::call_once(num_devices_init_flag, [] {
    num_devices = c10_neuron::device_count();
    default_neuron_generators.resize(num_devices);
    neuron_gens_init_flag = std::make_unique<c10::once_flag[]>(num_devices);
  });
}

const at::Generator& getDefaultNeuronGenerator(c10::DeviceIndex device_index) {
  initNeuronGenVector();
  c10::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = c10_neuron::current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < num_devices);
  }
  c10::call_once(neuron_gens_init_flag[idx], [&] {
    default_neuron_generators[idx] = at::make_generator<NeuronGeneratorImpl>(idx);
    default_neuron_generators[idx].seed();
  });
  return default_neuron_generators[idx];
}

// Initialize the Neuron runtime if needed
void NeuronHooksInterface::init() const {
  // The Neuron runtime is initialized when the extension is loaded
  // in register_neuron_extension(). We don't need to do anything here.
}

// Check if Neuron device has a primary context
bool NeuronHooksInterface::hasPrimaryContext(c10::DeviceIndex device_index) const {
  // For Neuron, we consider the device active if it's within the valid range
  return device_index >= 0 && device_index < c10_neuron::device_count();
}

// Get default generator for the device
at::Generator NeuronHooksInterface::getDefaultGenerator(c10::DeviceIndex device_index) {
  return getDefaultNeuronGenerator(device_index);
}

// Get a new generator for a specific device
at::Generator NeuronHooksInterface::getNewGenerator(c10::DeviceIndex device_index) const {
  return at::make_generator<NeuronGeneratorImpl>(device_index);
}

// Resize storage for Neuron tensors
void NeuronHooksInterface::resizePrivateUse1Bytes(const c10::Storage& storage,
                                                  size_t new_bytes) const {
  // Check if storage is resizable
  TORCH_CHECK(storage.resizable(), "Cannot resize a non-resizable storage");

  // Get current size
  size_t old_bytes = storage.nbytes();

  // If sizes are the same, nothing to do
  if (old_bytes == new_bytes) {
    return;
  }

  c10::StorageImpl* storage_impl = storage.unsafeGetStorageImpl();

  // If new_bytes is 0, do not need to allocate
  if (new_bytes == 0) {
    storage_impl->set_data_ptr_noswap(at::DataPtr(nullptr, storage.device()));
    storage_impl->set_nbytes(0);
    return;
  }

  // Allocate new memory using the storage's allocator
  c10::Allocator* allocator = storage.allocator();
  TORCH_CHECK(allocator != nullptr, "Storage allocator is null");

  at::DataPtr new_data = allocator->allocate(new_bytes);

  int device_id = storage.device().index();

  // Get reference to the old data
  const at::DataPtr& old_data = storage_impl->data_ptr();

  // Copy old data to new allocation if there was data
  if (old_data.get() != nullptr && new_bytes > 0 && old_bytes > 0) {
    size_t copy_bytes = std::min(old_bytes, new_bytes);

    // Use NeuronCachingAllocator's copy function for device-to-device copy
    bool copy_success = c10_neuron::NeuronCachingAllocator::copyTensorData(
        new_data.get(), old_data.get(), copy_bytes, device_id);
    TORCH_CHECK(copy_success, "Failed to copy data during storage resize");
  }

  // Use PyTorch's noswap approach to avoid replacing the storage address
  storage_impl->set_data_ptr_noswap(std::move(new_data));

  // Update storage size
  storage_impl->set_nbytes(new_bytes);
}

// Check if Neuron runtime is available
bool NeuronHooksInterface::isAvailable() const { return c10_neuron::device_count() > 0; }

// Get the singleton instance of NeuronHooksInterface
at::PrivateUse1HooksInterface* get_neuron_hooks() {
  static at::PrivateUse1HooksInterface* neuron_hooks = nullptr;
  static c10::once_flag once;
  c10::call_once(once, [] { neuron_hooks = new NeuronHooksInterface(); });
  return neuron_hooks;
}

}  // namespace torch_neuronx

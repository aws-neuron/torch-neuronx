#include "NeuronDevice.h"

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>

#include <atomic>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <nrt/nrt.h>
}

namespace c10_neuron {

namespace {
// Process-global default device for worker threads that haven't called set_device().
// This is set during distributed init via set_local_device_start_index().
// Uses -1 to indicate "not yet set" vs 0 which is a valid device.
std::atomic<int> global_default_device{-1};

// Thread-local current device override.
// -1 means "use global_default_device", >= 0 means explicit per-thread device.
// This allows worker threads to inherit the process default while still
// allowing explicit per-thread overrides via set_device().
thread_local int current_device_index = -1;

int local_world_size = -1;
int local_device_start_index = -1;
int world_size = -1;
int rank = -1;
int rank_vnc_count = -1;

// Atomic, not mutex-guarded: mark_in_bad_fork() runs from a pthread_atfork
// child handler, where a lock the parent held at fork() would never unlock.
std::atomic<bool> g_initialized{false};
std::atomic<bool> g_in_bad_fork{false};

// Lazily initialized from TORCH_NEURONX_COMPILE_ONLY on first read.
std::atomic<int> g_compile_only{-1};  // -1 = not yet resolved, 0/1 = value
}  // namespace

bool is_initialized() { return g_initialized.load(std::memory_order_acquire); }

void set_initialized(bool value) { g_initialized.store(value, std::memory_order_release); }

bool is_in_bad_fork() { return g_in_bad_fork.load(std::memory_order_relaxed); }

void mark_in_bad_fork() {
  g_in_bad_fork.store(true, std::memory_order_relaxed);
  g_initialized.store(false, std::memory_order_release);
}

bool is_compile_only() {
  int v = g_compile_only.load(std::memory_order_acquire);
  if (v < 0) {
    const char* env = std::getenv("TORCH_NEURONX_COMPILE_ONLY");
    v = (env && (env[0] == '1' || env[0] == 't' || env[0] == 'T')) ? 1 : 0;
    int expected = -1;
    g_compile_only.compare_exchange_strong(expected, v, std::memory_order_acq_rel);
    v = g_compile_only.load(std::memory_order_acquire);
  }
  return v != 0;
}

void set_compile_only(bool value) {
  g_compile_only.store(value ? 1 : 0, std::memory_order_release);
}

// Helper to check if a device is a Neuron device
bool IsNeuronDevice(const c10::Device& device) {
  return device.type() == c10::DeviceType::PrivateUse1;
}

// Helper function to extract device ID from a kernel's storage impls.
// Iterates through all storages to ensure they're on the same device and returns that device ID.
int GetTargetDeviceId(const std::vector<at::Tensor>& inputs,
                      const std::vector<at::Tensor>& outputs) {
  int device_id = -1;
  bool device_found = false;

  // Check all input tensors
  for (const auto& input : inputs) {
    c10::Device device = input.storage().device();
    TORCH_CHECK(IsNeuronDevice(device), "input tensor is on cpu device, expected neuron");
    int current_device = device.index();
    if (!device_found) {
      device_id = current_device;
      device_found = true;
    } else if (device_id != current_device) {
      throw std::runtime_error("Inconsistent device IDs found in kernel inputs: expected device " +
                               std::to_string(device_id) + " but found device " +
                               std::to_string(current_device));
    }
  }

  // Check all output tensors
  for (const auto& output : outputs) {
    c10::Device device = output.storage().device();
    TORCH_CHECK(IsNeuronDevice(device),
                "Expected all tensors to be on the same device, but the output tensor is on ",
                device.str(), " and the operation was dispatched to the Neuron device");
    int current_device = device.index();
    if (!device_found) {
      device_id = current_device;
      device_found = true;
    } else if (device_id != current_device) {
      throw std::runtime_error("Inconsistent device IDs found in kernel outputs: expected device " +
                               std::to_string(device_id) + " but found device " +
                               std::to_string(current_device));
    }
  }

  TORCH_CHECK(device_found, "No device found in kernel inputs and outputs");
  return device_id;
}

int current_device() {
  // If this thread has explicitly set a device, use that
  if (current_device_index >= 0) {
    return current_device_index;
  }
  // Otherwise, use the process-global default (set during distributed init)
  int global_dev = global_default_device.load(std::memory_order_acquire);
  if (global_dev >= 0) {
    return global_dev;
  }
  // Fallback for non-distributed case: device 0
  return 0;
}

int get_local_device_start_index() {
  return local_device_start_index == -1 ? 0 : local_device_start_index;
}

std::vector<c10::DeviceIndex> get_visible_device_indices() {
  int start = get_local_device_start_index();
  int count = vnc_count();
  std::vector<c10::DeviceIndex> indices;
  indices.reserve(count);
  for (int i = 0; i < count; i++) {
    indices.push_back(static_cast<c10::DeviceIndex>(start + i));
  }
  return indices;
}

void set_device(int device) {
  // Validate device index
  int start_index = get_local_device_start_index();
  int count = vnc_count();
  if (device < start_index || device >= (start_index + count)) {
    throw std::invalid_argument("Device index " + std::to_string(device) +
                                " is out of range. Valid range is " + std::to_string(start_index) +
                                " to " + std::to_string(start_index + count - 1) +
                                " for this process");
  }

  // Set thread-local device
  current_device_index = device;

  // Also set global default if this is the first set_device call in the process.
  // This handles the case where user code calls set_device() before distributed init.
  // Uses compare_exchange to only set once (first caller wins).
  int expected = -1;
  global_default_device.compare_exchange_strong(expected, device, std::memory_order_acq_rel);
}

void set_local_world_size(int size) {
  // Local world_size is the number of devices per node.
  // Its different from vnc_count. vnc_count is number
  // of devices per process within the node
  if (size < 0) {
    throw std::invalid_argument("local_world_size must be non-negative, got " +
                                std::to_string(size));
  }
  if (local_world_size != -1) {
    throw std::runtime_error("Attempted to reset local_world_size");
  }
  local_world_size = size;
}

void set_local_device_start_index(int id) {
  // local_start index is used to keep track of the local device
  // which pytorch uses, its different from vnc_id.
  if (id < 0) {
    throw std::invalid_argument("local start index must be non-negative, got " +
                                std::to_string(id));
  }

  if (local_world_size == -1) {
    throw std::runtime_error(
        "Attempted to set local_device_start_index before setting the local_world_size");
  }

  if (id >= local_world_size) {
    throw std::invalid_argument("local device start index " + std::to_string(id) +
                                " is out of range. Valid range is 0 to " +
                                std::to_string(local_world_size - 1));
  }
  local_device_start_index = id;
  current_device_index = id;

  // CRITICAL: Set the process-global default device for worker threads.
  // This is the primary initialization point during distributed setup.
  // Worker threads (autograd, etc.) that haven't called set_device() will
  // inherit this default via current_device().
  global_default_device.store(id, std::memory_order_release);
}

int get_vnc_id(int device_id) {
  // VNC Id is going to be different from pytorch's device id.
  // PyTorch Device ID will always be equal to the local rank
  // of the process, but vnc id for a given process will always start
  // from 0 and go upto the number of devices allocated for a given process
  if (local_world_size == -1) {
    return device_id;
  }

  if (device_id >= local_world_size) {
    throw std::invalid_argument("Device index " + std::to_string(device_id) +
                                " is out of range. Valid range is 0 to " +
                                std::to_string(local_world_size - 1));
  }
  if (device_id < local_device_start_index) {
    throw std::invalid_argument(
        "Attempted to access device that doesn't attach to this process, received " +
        std::to_string(device_id) + " expected to be equal or greater than " +
        std::to_string(local_device_start_index));
  }

  return device_id - local_device_start_index;
}

bool get_vnc_memory_stats(int device_id, uint64_t* bytes_limit, uint64_t* bytes_used) {
  if (device_id < 0) {
    return false;
  }
  nrt_vnc_memory_stats_t stats;
  size_t out_size = 0;
  if (nrt_get_vnc_memory_stats(static_cast<uint32_t>(get_vnc_id(device_id)), &stats, sizeof(stats),
                               &out_size) != NRT_SUCCESS) {
    return false;
  }
  if (bytes_limit) *bytes_limit = stats.bytes_limit;
  if (bytes_used) *bytes_used = stats.bytes_used;
  return true;
}

int vnc_count() {
  if (rank_vnc_count != -1) {
    return rank_vnc_count;
  }
  uint32_t count = 0;
  NRT_STATUS status = nrt_get_visible_vnc_count(&count);
  if (status != NRT_SUCCESS) {
    return 0;
  }
  rank_vnc_count = static_cast<int>(count);
  return rank_vnc_count;
}

void reset_vnc_count() { rank_vnc_count = -1; }

int device_count() {
  // device count in pytorch is equal to the local_world_size
  // its different from vnc_count. vnc_count is equal to the number of
  // devices a process sees. Note: In distributed setting, this is contrainted to 1
  if (local_world_size != -1) {
    return local_world_size;
  }
  return vnc_count();
}

void set_world_size(int size) {
  if (size < 0) {
    throw std::invalid_argument("world_size must be non-negative, got " + std::to_string(size));
  }
  if (world_size != -1) {
    throw std::runtime_error("Attempted to reset world_size");
  }
  world_size = size;
}

void set_rank(int r) {
  if (r < 0) {
    throw std::invalid_argument("rank must be non-negative, got " + std::to_string(r));
  }
  if (rank != -1) {
    throw std::runtime_error("Attempted to reset rank");
  }
  rank = r;
}

int get_world_size() {
  if (world_size == -1) {
    throw std::runtime_error("world_size has not been set");
  }
  return world_size;
}

int get_rank() {
  if (rank == -1) {
    throw std::runtime_error("rank has not been set");
  }
  return rank;
}

void reset_distributed_state() {
  world_size = -1;
  rank = -1;
  local_world_size = -1;
  local_device_start_index = -1;
  // Reset global default device so next distributed init can set it fresh.
  // Note: This doesn't reset thread-local current_device_index - each thread
  // retains its explicit device setting if any.
  global_default_device.store(-1, std::memory_order_release);
  rank_vnc_count = -1;
}

bool is_distributed_initialized() { return world_size != -1; }

}  // namespace c10_neuron

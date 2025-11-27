#include "NeuronDevice.h"

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>

#include <stdexcept>
#include <string>

extern "C" {
#include <nrt/nrt.h>
}

namespace c10_neuron {

namespace {
// Thread-local current device
thread_local int current_device_index = 0;
int local_world_size = -1;
int local_device_start_index = -1;
int world_size = -1;
int rank = -1;
}  // namespace

int current_device() { return current_device_index; }

void set_device(int device) {
  // Validate device index
  int start_index = local_device_start_index == -1 ? 0 : local_device_start_index;
  int count = vnc_count();
  if (device < start_index || device >= (start_index + count)) {
    throw std::invalid_argument("Device index " + std::to_string(device) +
                                " is out of range. Valid range is " + std::to_string(start_index) +
                                " to " + std::to_string(start_index + count - 1) +
                                " for this process");
  }

  current_device_index = device;
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

int vnc_count() {
  uint32_t vnc_count = 0;
  NRT_STATUS status = nrt_get_visible_vnc_count(&vnc_count);

  if (status != NRT_SUCCESS) {
    return 0;
  }

  return static_cast<int>(vnc_count);
}

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
}

}  // namespace c10_neuron

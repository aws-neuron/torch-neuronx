#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace c10_neuron {

// Helper to check if a device is a Neuron device
bool IsNeuronDevice(const c10::Device& device);

// Helper function to extract device ID from the inputs and outputs
int GetTargetDeviceId(const std::vector<at::Tensor>& inputs,
                      const std::vector<at::Tensor>& outputs);

// Get the current Neuron device index
int current_device();

// Set the current Neuron device
void set_device(int device);

void set_local_world_size(int size);

void set_local_device_start_index(int id);

int get_vnc_id(int device_id);

int vnc_count();

// Get device count
int device_count();

void set_world_size(int size);

int get_world_size();

void set_rank(int r);

int get_rank();

// Mainly used for testing
void reset_distributed_state();

}  // namespace c10_neuron

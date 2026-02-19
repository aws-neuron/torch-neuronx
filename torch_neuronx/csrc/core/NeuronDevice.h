#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <vector>

namespace c10_neuron {

// Check if Neuron runtime is initialized
bool is_initialized();
void set_initialized(bool value);

// Helper to check if a device is a Neuron device
bool IsNeuronDevice(const c10::Device& device);

// Helper function to extract device ID from the inputs and outputs
int GetTargetDeviceId(const std::vector<at::Tensor>& inputs,
                      const std::vector<at::Tensor>& outputs);

// Get the current Neuron device index
int current_device();

// Set the current Neuron device
void set_device(int device);

// Check if a device index is valid for this process

void set_local_world_size(int size);

void set_local_device_start_index(int id);

int get_local_device_start_index();

std::vector<c10::DeviceIndex> get_visible_device_indices();

int get_vnc_id(int device_id);

int vnc_count();

void reset_vnc_count();

// Get device count
int device_count();

void set_world_size(int size);

int get_world_size();

void set_rank(int r);

int get_rank();

// Mainly used for testing
void reset_distributed_state();

}  // namespace c10_neuron

namespace torch_neuronx {

// Lazy initialization helper - initializes Neuron runtime if not already initialized
void maybe_lazy_init();

}  // namespace torch_neuronx

#include "MockNRT.h"

#include <cstring>

// Flag to check if mocks are initialized
static bool g_mocks_initialized = false;

// Override NRT symbols
// NRT symbols by default are sourced via shared libraries and are late binding.
// For tests we early bind NRT symbols by statically linking them into the test binary.
extern "C" {

#pragma GCC visibility push(default)

NRT_STATUS nrt_init(nrt_framework_type_t framework, const char* fw_version,
                    const char* fal_version) {
  if (g_mocks_initialized) {
    return neuron::testing::MockNRT::GetInstance()->nrt_init(framework, fw_version, fal_version);
  }
  return NRT_FAILURE;
}

void nrt_close() {
  if (g_mocks_initialized) {
    neuron::testing::MockNRT::GetInstance()->nrt_close();
    return;
  }
}

NRT_STATUS nrt_get_total_nc_count(uint32_t* nc_count) {
  if (g_mocks_initialized) {
    return neuron::testing::MockNRT::GetInstance()->nrt_get_total_nc_count(nc_count);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_get_visible_nc_count(uint32_t* nc_count) {
  if (g_mocks_initialized) {
    return neuron::testing::MockNRT::GetInstance()->nrt_get_visible_nc_count(nc_count);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_tensor_allocate(nrt_tensor_placement_t placement, int32_t logical_nc_id, size_t size,
                               const char* name, nrt_tensor_t** tensor) {
  if (g_mocks_initialized) {
    return neuron::testing::MockNRT::GetInstance()->nrt_tensor_allocate(placement, logical_nc_id,
                                                                        size, name, tensor);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_tensor_allocate_slice(const nrt_tensor_t* base_tensor, size_t offset, size_t size,
                                     const char* name, nrt_tensor_t** slice_tensor) {
  if (g_mocks_initialized) {
    return neuron::testing::MockNRT::GetInstance()->nrt_tensor_allocate_slice(
        base_tensor, offset, size, name, slice_tensor);
  }
  return NRT_FAILURE;
}

size_t nrt_tensor_get_size(const nrt_tensor_t* tensor) {
  if (g_mocks_initialized) {
    return neuron::testing::MockNRT::GetInstance()->nrt_tensor_get_size(tensor);
  }
  return 0;
}

void nrt_tensor_free(nrt_tensor_t** tensor) {
  if (g_mocks_initialized) {
    neuron::testing::MockNRT::GetInstance()->nrt_tensor_free(tensor);
    return;
  }
}

void* nrt_tensor_get_va(const nrt_tensor_t* tensor) {
  if (g_mocks_initialized) {
    return neuron::testing::MockNRT::GetInstance()->nrt_tensor_get_va(tensor);
  }
  return nullptr;
}

NRT_STATUS nrt_tensor_write(nrt_tensor_t* tensor, const void* buf, size_t offset, size_t size) {
  if (g_mocks_initialized) {
    return neuron::testing::MockNRT::GetInstance()->nrt_tensor_write(tensor, buf, offset, size);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_tensor_read(const nrt_tensor_t* tensor, void* buf, size_t offset, size_t size) {
  if (g_mocks_initialized) {
    return neuron::testing::MockNRT::GetInstance()->nrt_tensor_read(tensor, buf, offset, size);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_allocate_tensor_set(nrt_tensor_set_t** tensor_set) {
  if (g_mocks_initialized) {
    return neuron::testing::MockNRT::GetInstance()->nrt_allocate_tensor_set(tensor_set);
  }
  return NRT_FAILURE;
}

void nrt_destroy_tensor_set(nrt_tensor_set_t** tensor_set) {
  if (g_mocks_initialized) {
    neuron::testing::MockNRT::GetInstance()->nrt_destroy_tensor_set(tensor_set);
    return;
  }
}

NRT_STATUS nrt_add_tensor_to_tensor_set(nrt_tensor_set_t* tensor_set, const char* tensor_name,
                                        nrt_tensor_t* tensor) {
  if (g_mocks_initialized) {
    return neuron::testing::MockNRT::GetInstance()->nrt_add_tensor_to_tensor_set(
        tensor_set, tensor_name, tensor);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_load(const void* neff_bytes, size_t size, int32_t start_nc, int32_t nc_count,
                    nrt_model_t** model) {
  if (g_mocks_initialized) {
    return neuron::testing::MockNRT::GetInstance()->nrt_load(neff_bytes, size, start_nc, nc_count,
                                                             model);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_load_collectives(const void* neff_bytes, size_t size, int32_t start_vnc,
                                int32_t vnc_count, uint32_t ctx_device_id,
                                uint32_t ctx_device_count, nrt_model_t** model) {
  if (g_mocks_initialized) {
    return neuron::testing::MockNRT::GetInstance()->nrt_load_collectives(
        neff_bytes, size, start_vnc, vnc_count, ctx_device_id, ctx_device_count, model);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_unload(nrt_model_t* model) {
  if (g_mocks_initialized) {
    return neuron::testing::MockNRT::GetInstance()->nrt_unload(model);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_execute(nrt_model_t* model, const nrt_tensor_set_t* input_set,
                       nrt_tensor_set_t* output_set) {
  if (g_mocks_initialized) {
    return neuron::testing::MockNRT::GetInstance()->nrt_execute(model, input_set, output_set);
  }
  return NRT_FAILURE;
}

#pragma GCC visibility pop

}  // extern "C"

namespace neuron {
namespace testing {

MockNRTSession::MockNRTSession() {
  if (!g_mocks_initialized) {
    g_mocks_initialized = true;
    initialized_ = true;
  }
}

MockNRTSession::~MockNRTSession() {
  if (initialized_) {
    g_mocks_initialized = false;
    // Clear any expectations
    ::testing::Mock::VerifyAndClearExpectations(MockNRT::GetInstance());
  }
}

}  // namespace testing
}  // namespace neuron

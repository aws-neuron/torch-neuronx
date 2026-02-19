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
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_init(framework, fw_version,
                                                                    fal_version);
  }
  return NRT_FAILURE;
}

void nrt_close() {
  if (g_mocks_initialized) {
    torch_neuronx::testing::MockNRT::GetInstance()->nrt_close();
    return;
  }
}

NRT_STATUS nrt_get_total_vnc_count(uint32_t* vnc_count) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_get_total_vnc_count(vnc_count);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_get_visible_vnc_count(uint32_t* vnc_count) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_get_visible_vnc_count(vnc_count);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_tensor_allocate(nrt_tensor_placement_t placement, int32_t logical_nc_id, size_t size,
                               const char* name, nrt_tensor_t** tensor) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_tensor_allocate(
        placement, logical_nc_id, size, name, tensor);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_tensor_allocate_slice(const nrt_tensor_t* base_tensor, size_t offset, size_t size,
                                     const char* name, nrt_tensor_t** slice_tensor) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_tensor_allocate_slice(
        base_tensor, offset, size, name, slice_tensor);
  }
  return NRT_FAILURE;
}

size_t nrt_tensor_get_size(const nrt_tensor_t* tensor) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_tensor_get_size(tensor);
  }
  return 0;
}

void nrt_tensor_free(nrt_tensor_t** tensor) {
  if (g_mocks_initialized) {
    torch_neuronx::testing::MockNRT::GetInstance()->nrt_tensor_free(tensor);
    return;
  }
}

void* nrt_tensor_get_va(const nrt_tensor_t* tensor) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_tensor_get_va(tensor);
  }
  return nullptr;
}

NRT_STATUS nrt_tensor_write(nrt_tensor_t* tensor, const void* buf, size_t offset, size_t size) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_tensor_write(tensor, buf, offset,
                                                                            size);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_tensor_read(const nrt_tensor_t* tensor, void* buf, size_t offset, size_t size) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_tensor_read(tensor, buf, offset,
                                                                           size);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_tensor_copy(const nrt_tensor_t* src, size_t src_offset, nrt_tensor_t* dst,
                           size_t dst_offset, size_t size) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_tensor_copy(src, src_offset, dst,
                                                                           dst_offset, size);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_allocate_tensor_set(nrt_tensor_set_t** tensor_set) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_allocate_tensor_set(tensor_set);
  }
  return NRT_FAILURE;
}

void nrt_destroy_tensor_set(nrt_tensor_set_t** tensor_set) {
  if (g_mocks_initialized) {
    torch_neuronx::testing::MockNRT::GetInstance()->nrt_destroy_tensor_set(tensor_set);
    return;
  }
}

NRT_STATUS nrt_add_tensor_to_tensor_set(nrt_tensor_set_t* tensor_set, const char* tensor_name,
                                        nrt_tensor_t* tensor) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_add_tensor_to_tensor_set(
        tensor_set, tensor_name, tensor);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_load(const void* neff_bytes, size_t size, int32_t start_nc, int32_t vnc_count,
                    nrt_model_t** model) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_load(neff_bytes, size, start_nc,
                                                                    vnc_count, model);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_load_collectives(const void* neff_bytes, size_t size, int32_t start_vnc,
                                int32_t vvnc_count, uint32_t ctx_device_id,
                                uint32_t ctx_device_count, nrt_model_t** model) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_load_collectives(
        neff_bytes, size, start_vnc, vvnc_count, ctx_device_id, ctx_device_count, model);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_unload(nrt_model_t* model) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_unload(model);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_execute(nrt_model_t* model, const nrt_tensor_set_t* input_set,
                       nrt_tensor_set_t* output_set) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_execute(model, input_set,
                                                                       output_set);
  }
  return NRT_FAILURE;
}

// Profiler APIs
NRT_STATUS nrt_inspect_config_allocate(nrt_inspect_config_t** config) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_inspect_config_allocate(config);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_barrier(int32_t vnc, uint32_t g_device_id, uint32_t g_device_count) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_barrier(vnc, g_device_id,
                                                                       g_device_count);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_inspect_config_set_defaults(nrt_inspect_config_t* config) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_inspect_config_set_defaults(config);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_inspect_config_free(nrt_inspect_config_t* config) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_inspect_config_free(config);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_inspect_config_set_enable_inspect(nrt_inspect_config_t* config, bool enable) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_inspect_config_set_enable_inspect(
        config, enable);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_inspect_config_set_activity(nrt_inspect_config_t* config, const char* activity,
                                           bool enable) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_inspect_config_set_activity(
        config, activity, enable);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_inspect_config_set_sys_trace_max_events_per_nc(nrt_inspect_config_t* config,
                                                              uint64_t max_events) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()
        ->nrt_inspect_config_set_sys_trace_max_events_per_nc(config, max_events);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_inspect_config_set_capture_enabled_for_nc(nrt_inspect_config_t* config,
                                                         uint32_t nc_idx, bool enable) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()
        ->nrt_inspect_config_set_capture_enabled_for_nc(config, nc_idx, enable);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_inspect_config_set_inspect_device_profile_mode(
    nrt_inspect_config_t* config, nrt_inspect_device_profile_mode_t mode) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()
        ->nrt_inspect_config_set_inspect_device_profile_mode(config, mode);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_inspect_config_set_output_dir(nrt_inspect_config_t* config, const char* output_dir) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_inspect_config_set_output_dir(
        config, output_dir);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_inspect_begin_with_options(nrt_inspect_config_t* config) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_inspect_begin_with_options(config);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrt_inspect_stop() {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrt_inspect_stop();
  }
  return NRT_FAILURE;
}
// End Profiler APIs

NRT_STATUS nrta_get_sequence(uint32_t lnc, nrta_xu_t xu, int queue, nrta_seq_t* seq) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrta_get_sequence(lnc, xu, queue, seq);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrta_tensor_write(nrt_tensor_t* tensor, const void* buf, uint64_t offset, uint64_t size,
                             int queue, nrta_error_tracker_t* err_tracker, nrta_seq_t* req_seq) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrta_tensor_write(
        tensor, buf, offset, size, queue, err_tracker, req_seq);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrta_tensor_read(void* buf, nrt_tensor_t* tensor, uint64_t offset, uint64_t size,
                            int queue, nrta_error_tracker_t* err_tracker, nrta_seq_t* req_seq) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrta_tensor_read(
        buf, tensor, offset, size, queue, err_tracker, req_seq);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrta_tensor_copy(nrt_tensor_t* src, uint64_t src_offset, nrt_tensor_t* dst,
                            uint64_t dst_offset, uint64_t size, int queue,
                            nrta_error_tracker_t* err_tracker, nrta_seq_t* req_seq) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrta_tensor_copy(
        src, src_offset, dst, dst_offset, size, queue, err_tracker, req_seq);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrta_is_completed(nrta_seq_t seq, bool* is_completed) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrta_is_completed(seq, is_completed);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrta_error_tracker_create(uint32_t lnc_idx, nrta_error_tracker_t** tracker) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrta_error_tracker_create(lnc_idx,
                                                                                     tracker);
  }
  return NRT_FAILURE;
}

void nrta_error_tracker_destroy(nrta_error_tracker_t* tracker) {
  if (g_mocks_initialized) {
    torch_neuronx::testing::MockNRT::GetInstance()->nrta_error_tracker_destroy(tracker);
  }
}

NRT_STATUS nrta_error_tracker_get_list(nrta_error_tracker_t* tracker, const nrta_error_t** errors,
                                       size_t* count) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrta_error_tracker_get_list(
        tracker, errors, count);
  }
  return NRT_FAILURE;
}

NRT_STATUS nrta_execute_schedule(nrt_model_t* model, const nrt_tensor_set_t* input_set,
                                 nrt_tensor_set_t* output_set, int queue,
                                 nrta_error_tracker_t* err_tracker, nrta_seq_t* req_seq) {
  if (g_mocks_initialized) {
    return torch_neuronx::testing::MockNRT::GetInstance()->nrta_execute_schedule(
        model, input_set, output_set, queue, err_tracker, req_seq);
  }
  return NRT_FAILURE;
}

#pragma GCC visibility pop

}  // extern "C"

namespace torch_neuronx {
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
}  // namespace torch_neuronx

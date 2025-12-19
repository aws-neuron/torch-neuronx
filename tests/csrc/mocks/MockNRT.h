#pragma once

#include <gmock/gmock.h>

extern "C" {
#include <nrt/nrt.h>
}

namespace torch_neuronx {
namespace testing {

class MockNRT {
 public:
  static MockNRT* GetInstance() {
    static MockNRT instance;
    return &instance;
  }

  // Core NRT functions
  MOCK_METHOD(NRT_STATUS, nrt_init, (nrt_framework_type_t, const char*, const char*));
  MOCK_METHOD(void, nrt_close, ());
  MOCK_METHOD(NRT_STATUS, nrt_get_total_vnc_count, (uint32_t*));
  MOCK_METHOD(NRT_STATUS, nrt_get_visible_vnc_count, (uint32_t*));

  // Tensor management
  MOCK_METHOD(NRT_STATUS, nrt_tensor_allocate,
              (nrt_tensor_placement_t, int32_t, size_t, const char*, nrt_tensor_t**));
  MOCK_METHOD(NRT_STATUS, nrt_tensor_allocate_slice,
              (const nrt_tensor_t*, size_t, size_t, const char*, nrt_tensor_t**));
  MOCK_METHOD(size_t, nrt_tensor_get_size, (const nrt_tensor_t*));
  MOCK_METHOD(void, nrt_tensor_free, (nrt_tensor_t**));
  MOCK_METHOD(void*, nrt_tensor_get_va, (const nrt_tensor_t*));
  MOCK_METHOD(NRT_STATUS, nrt_tensor_write, (nrt_tensor_t*, const void*, size_t, size_t));
  MOCK_METHOD(NRT_STATUS, nrt_tensor_read, (const nrt_tensor_t*, void*, size_t, size_t));
  MOCK_METHOD(NRT_STATUS, nrt_tensor_copy,
              (const nrt_tensor_t*, size_t, nrt_tensor_t*, size_t, size_t));

  // Tensor set management
  MOCK_METHOD(NRT_STATUS, nrt_allocate_tensor_set, (nrt_tensor_set_t**));
  MOCK_METHOD(void, nrt_destroy_tensor_set, (nrt_tensor_set_t**));
  MOCK_METHOD(NRT_STATUS, nrt_add_tensor_to_tensor_set,
              (nrt_tensor_set_t*, const char*, nrt_tensor_t*));

  // Model management
  MOCK_METHOD(NRT_STATUS, nrt_load, (const void*, size_t, int32_t, int32_t, nrt_model_t**));
  MOCK_METHOD(NRT_STATUS, nrt_load_collectives,
              (const void*, size_t, int32_t, int32_t, uint32_t, uint32_t, nrt_model_t**));
  MOCK_METHOD(NRT_STATUS, nrt_unload, (nrt_model_t*));

  // Execution
  MOCK_METHOD(NRT_STATUS, nrt_execute, (nrt_model_t*, const nrt_tensor_set_t*, nrt_tensor_set_t*));

 private:
  MockNRT() = default;
};

// Helper class to manage mock session lifecycle
class MockNRTSession {
 public:
  MockNRTSession();
  ~MockNRTSession();
  MockNRTSession(const MockNRTSession&) = delete;
  MockNRTSession& operator=(const MockNRTSession&) = delete;

 private:
  bool initialized_ = false;
};

}  // namespace testing
}  // namespace torch_neuronx

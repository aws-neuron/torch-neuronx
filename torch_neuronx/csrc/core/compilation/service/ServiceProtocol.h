#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace at::neuron {

// Wire protocol for the compilation service.
// All messages: [type(u8)][payload_len(u32 LE)][payload]
// Payloads are null-delimited strings (small — only paths and keys).

enum class ServiceMsgType : uint8_t {
  kCompileRequest = 0x01,   // client→service
  kCompileResponse = 0x02,  // service→client
  kShutdown = 0xFF,
};

struct CompileRequest {
  uint64_t req_id;
  std::string hlo_file_path;
  std::string ir_type;             // "StableHLO"
  std::string op_name;             // operation name for IR dumping
  std::string cache_key;           // cache key for IR dumping
  std::string optimization_level;  // per-op from kernel_execution.GetOptimizationLevel()
  std::string additional_args;  // per-op compiler flags from kernel_execution.GetAdditionalArgs()

  std::vector<uint8_t> Serialize() const;
  static CompileRequest Deserialize(const uint8_t* data, size_t len);
};

struct CompileResponse {
  uint64_t req_id;
  bool success;
  std::string neff_cache_path;  // path to compiled NEFF temp file
  std::string error_message;    // compiler error message on failure

  std::vector<uint8_t> Serialize() const;
  static CompileResponse Deserialize(const uint8_t* data, size_t len);
};

// Blocking send/recv helpers for Unix domain sockets.
bool SendMessage(int fd, ServiceMsgType type, const std::vector<uint8_t>& payload);
bool SendMessage(int fd, ServiceMsgType type, const uint8_t* data, size_t len);

// Returns false on connection close. Sets out_type and fills payload.
bool RecvMessage(int fd, ServiceMsgType* out_type, std::vector<uint8_t>* payload);

}  // namespace at::neuron

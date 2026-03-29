#include "ServiceProtocol.h"

#include <sys/socket.h>

#include <cerrno>

namespace at::neuron {

namespace {

bool SendExact(int fd, const void* buf, size_t len) {
  const uint8_t* p = static_cast<const uint8_t*>(buf);
  size_t remaining = len;
  while (remaining > 0) {
    ssize_t n = send(fd, p, remaining, MSG_NOSIGNAL);
    if (n <= 0) {
      if (n < 0 && errno == EINTR) continue;
      return false;
    }
    p += n;
    remaining -= static_cast<size_t>(n);
  }
  return true;
}

bool RecvExact(int fd, void* buf, size_t len) {
  uint8_t* p = static_cast<uint8_t*>(buf);
  size_t remaining = len;
  while (remaining > 0) {
    ssize_t n = recv(fd, p, remaining, 0);
    if (n <= 0) {
      if (n < 0 && errno == EINTR) continue;
      return false;
    }
    p += n;
    remaining -= static_cast<size_t>(n);
  }
  return true;
}

void AppendString(std::vector<uint8_t>& buf, const std::string& s) {
  buf.insert(buf.end(), s.begin(), s.end());
  buf.push_back('\0');
}

std::string ReadString(const uint8_t*& p, const uint8_t* end) {
  const uint8_t* start = p;
  while (p < end && *p != '\0') ++p;
  std::string result(reinterpret_cast<const char*>(start), p - start);
  if (p < end) ++p;  // skip null
  return result;
}

}  // namespace

std::vector<uint8_t> CompileRequest::Serialize() const {
  std::vector<uint8_t> buf;
  buf.resize(sizeof(uint64_t));
  std::memcpy(buf.data(), &req_id, sizeof(uint64_t));
  AppendString(buf, hlo_file_path);
  AppendString(buf, ir_type);
  AppendString(buf, optimization_level);
  AppendString(buf, additional_args);
  return buf;
}

CompileRequest CompileRequest::Deserialize(const uint8_t* data, size_t len) {
  CompileRequest req;
  const uint8_t* p = data;
  const uint8_t* end = data + len;
  if (static_cast<size_t>(end - p) < sizeof(uint64_t)) return req;
  std::memcpy(&req.req_id, p, sizeof(uint64_t));
  p += sizeof(uint64_t);
  req.hlo_file_path = ReadString(p, end);
  req.ir_type = ReadString(p, end);
  req.optimization_level = ReadString(p, end);
  req.additional_args = ReadString(p, end);
  return req;
}

std::vector<uint8_t> CompileResponse::Serialize() const {
  std::vector<uint8_t> buf;
  buf.resize(sizeof(uint64_t) + 1);
  std::memcpy(buf.data(), &req_id, sizeof(uint64_t));
  buf[sizeof(uint64_t)] = success ? 0 : 1;
  AppendString(buf, neff_cache_path);
  AppendString(buf, error_message);
  return buf;
}

CompileResponse CompileResponse::Deserialize(const uint8_t* data, size_t len) {
  CompileResponse resp;
  const uint8_t* p = data;
  const uint8_t* end = data + len;
  if (static_cast<size_t>(end - p) < sizeof(uint64_t) + 1) return resp;
  std::memcpy(&resp.req_id, p, sizeof(uint64_t));
  p += sizeof(uint64_t);
  resp.success = (*p == 0);
  ++p;
  resp.neff_cache_path = ReadString(p, end);
  resp.error_message = ReadString(p, end);
  return resp;
}

bool SendMessage(int fd, ServiceMsgType type, const std::vector<uint8_t>& payload) {
  return SendMessage(fd, type, payload.data(), payload.size());
}

bool SendMessage(int fd, ServiceMsgType type, const uint8_t* data, size_t len) {
  uint8_t header[5];
  header[0] = static_cast<uint8_t>(type);
  uint32_t payload_len = static_cast<uint32_t>(len);
  std::memcpy(header + 1, &payload_len, sizeof(uint32_t));
  if (!SendExact(fd, header, sizeof(header))) return false;
  if (len > 0 && !SendExact(fd, data, len)) return false;
  return true;
}

bool RecvMessage(int fd, ServiceMsgType* out_type, std::vector<uint8_t>* payload) {
  uint8_t header[5];
  if (!RecvExact(fd, header, sizeof(header))) return false;
  *out_type = static_cast<ServiceMsgType>(header[0]);
  uint32_t payload_len;
  std::memcpy(&payload_len, header + 1, sizeof(uint32_t));
  // Sanity cap: 1MB is more than enough for paths and keys
  if (payload_len > 1048576) return false;
  payload->resize(payload_len);
  if (payload_len > 0 && !RecvExact(fd, payload->data(), payload_len)) return false;
  return true;
}

}  // namespace at::neuron

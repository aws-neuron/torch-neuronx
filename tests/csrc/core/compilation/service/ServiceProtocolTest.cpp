#include <gtest/gtest.h>
#include <sys/socket.h>
#include <unistd.h>

#include "torch_neuronx/csrc/core/compilation/NeuronCompiler.h"
#include "torch_neuronx/csrc/core/compilation/service/ServiceProtocol.h"

using namespace at::neuron;

TEST(ServiceProtocolTest, CompileRequestRoundTrip) {
  CompileRequest req;
  req.req_id = 42;
  req.hlo_file_path = "/tmp/hlo_test.pb";
  req.ir_type = "StableHLO";
  req.optimization_level = "-O2";
  req.additional_args = "--model-type transformer --auto-cast=none";

  auto bytes = req.Serialize();
  auto decoded = CompileRequest::Deserialize(bytes.data(), bytes.size());

  EXPECT_EQ(decoded.req_id, 42);
  EXPECT_EQ(decoded.hlo_file_path, "/tmp/hlo_test.pb");
  EXPECT_EQ(decoded.ir_type, "StableHLO");
  EXPECT_EQ(decoded.optimization_level, "-O2");
  EXPECT_EQ(decoded.additional_args, "--model-type transformer --auto-cast=none");
}

TEST(ServiceProtocolTest, CompileResponseRoundTrip) {
  CompileResponse resp;
  resp.req_id = 99;
  resp.success = true;
  resp.neff_cache_path = "/cache/ab/cd/abcd1234.neff";

  auto bytes = resp.Serialize();
  auto decoded = CompileResponse::Deserialize(bytes.data(), bytes.size());

  EXPECT_EQ(decoded.req_id, 99);
  EXPECT_TRUE(decoded.success);
  EXPECT_EQ(decoded.neff_cache_path, "/cache/ab/cd/abcd1234.neff");
}

TEST(ServiceProtocolTest, EmptyStringsRoundTrip) {
  CompileRequest req;
  req.req_id = 0;
  req.hlo_file_path = "";
  req.ir_type = "";
  req.optimization_level = "";
  req.additional_args = "";

  auto bytes = req.Serialize();
  auto decoded = CompileRequest::Deserialize(bytes.data(), bytes.size());

  EXPECT_EQ(decoded.req_id, 0);
  EXPECT_TRUE(decoded.ir_type.empty());
  EXPECT_TRUE(decoded.optimization_level.empty());
  EXPECT_TRUE(decoded.additional_args.empty());
}

TEST(ServiceProtocolTest, SocketSendRecvPair) {
  int fds[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

  CompileRequest req;
  req.req_id = 123;
  req.hlo_file_path = "/tmp/test.pb";
  req.ir_type = "StableHLO";
  req.optimization_level = "-O1";
  req.additional_args = "--model-type transformer";
  auto payload = req.Serialize();

  ASSERT_TRUE(SendMessage(fds[0], ServiceMsgType::kCompileRequest, payload));

  ServiceMsgType recv_type;
  std::vector<uint8_t> recv_payload;
  ASSERT_TRUE(RecvMessage(fds[1], &recv_type, &recv_payload));

  EXPECT_EQ(recv_type, ServiceMsgType::kCompileRequest);
  auto decoded = CompileRequest::Deserialize(recv_payload.data(), recv_payload.size());
  EXPECT_EQ(decoded.req_id, 123);
  EXPECT_EQ(decoded.optimization_level, "-O1");
  EXPECT_EQ(decoded.additional_args, "--model-type transformer");

  close(fds[0]);
  close(fds[1]);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

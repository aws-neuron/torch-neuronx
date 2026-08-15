// RdmaManager unit tests: store-based control plane and P2P handling.
//
// Only the control plane is testable here, and only for batches whose entries are
// all zero-size. Anything touching a real transfer — MR registration, rkey
// exchange, the same-rank TORCH_CHECK, transfer_id uniqueness, CQ completion
// and error attribution — needs hardware plus a peer process and belongs in the
// multi-rank Python P2P tests.

#include <gtest/gtest.h>

#include <chrono>
#include <map>
#include <string>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <vector>

#include "torch_neuronx/csrc/core/rdma/RdmaManager.h"
#include "torch_neuronx/csrc/core/rdma/TransferDescriptor.pb.h"

namespace {

// Non-zero so rank is actually serialized
constexpr int kSelfRank = 3;
constexpr int kPeerRank = 1;

// In-memory Store recording the calls PrepareBatch makes. multiGet does not
// block on a missing key, so test that reaches a blocking multiGet has already
// failed, and returning empty keeps the suite from hanging.
class FakeStore : public c10d::Store {
 public:
  void set(const std::string& key, const std::vector<uint8_t>& value) override {
    data_[key] = value;
  }

  std::vector<uint8_t> get(const std::string& key) override {
    auto it = data_.find(key);
    return it == data_.end() ? std::vector<uint8_t>{} : it->second;
  }

  void multiSet(const std::vector<std::string>& keys,
                const std::vector<std::vector<uint8_t>>& values) override {
    multi_set_calls_++;
    for (size_t i = 0; i < keys.size(); ++i) {
      data_[keys[i]] = values[i];
      set_keys_.push_back(keys[i]);
    }
  }

  std::vector<std::vector<uint8_t>> multiGet(const std::vector<std::string>& keys) override {
    std::vector<std::vector<uint8_t>> out;
    out.reserve(keys.size());
    for (const auto& k : keys) {
      requested_keys_.push_back(k);
      out.push_back(get(k));
    }
    return out;
  }

  bool hasExtendedApi() const override { return true; }

  // Pure virtual on c10d::Store; RdmaManager never calls it. Returns a second
  // handle to this same object (refcount bumped) rather than a fresh copy, so
  // recorded calls stay observable through either handle.
  c10::intrusive_ptr<c10d::Store> clone() override {
    return c10::intrusive_ptr<c10d::Store>::reclaim_copy(this);
  }

  int64_t add(const std::string&, int64_t) override { return 0; }
  bool deleteKey(const std::string& key) override { return data_.erase(key) > 0; }
  bool check(const std::vector<std::string>& keys) override {
    for (const auto& k : keys) {
      if (data_.find(k) == data_.end()) return false;
    }
    return true;
  }
  int64_t getNumKeys() override { return static_cast<int64_t>(data_.size()); }
  void wait(const std::vector<std::string>&) override {}
  void wait(const std::vector<std::string>&, const std::chrono::milliseconds&) override {}

  bool Has(const std::string& key) const { return data_.find(key) != data_.end(); }
  int multi_set_calls() const { return multi_set_calls_; }
  // Keys this rank published (phase 1) and looked up (phase 2), in order.
  const std::vector<std::string>& set_keys() const { return set_keys_; }
  const std::vector<std::string>& requested_keys() const { return requested_keys_; }

 private:
  std::map<std::string, std::vector<uint8_t>> data_;
  std::vector<std::string> set_keys_;
  std::vector<std::string> requested_keys_;
  int multi_set_calls_ = 0;
};

// Mirrors RdmaManager::MakeStoreKey (private). This key
// format is the cross-rank wire contract, so changing it should fail a test here
// rather than desync peers at runtime.
std::string StoreKey(int src, int dst, int tag, int seq) {
  return "rdma_" + std::to_string(src) + "_" + std::to_string(dst) + "_" + std::to_string(tag) +
         "_" + std::to_string(seq);
}

neuron::P2PEntry ZeroSizeRecv(int tag, int peer = kPeerRank) {
  return neuron::P2PEntry{neuron::P2PEntry::Kind::kRecv, /*user_va=*/nullptr, /*size=*/0, peer,
                          tag};
}

neuron::P2PEntry ZeroSizeSend(int tag, int peer = kPeerRank) {
  return neuron::P2PEntry{neuron::P2PEntry::Kind::kSend, /*user_va=*/nullptr, /*size=*/0, peer,
                          tag};
}

// Serialized descriptor as a peer's zero-size recv publishes it: no regions.
std::vector<uint8_t> RegionlessDescriptor() {
  neuron::rdma::TransferDescriptor desc;
  desc.set_transfer_id(0);
  desc.set_rank(kPeerRank);
  std::string s;
  EXPECT_TRUE(desc.SerializeToString(&s));
  return std::vector<uint8_t>(s.begin(), s.end());
}

// Descriptor for a real (non-zero-size) recv: carries one memory region.
std::vector<uint8_t> DescriptorWithRegion() {
  neuron::rdma::TransferDescriptor desc;
  desc.set_transfer_id(7);
  desc.set_rank(kPeerRank);
  auto* region = desc.add_regions();
  region->set_remote_addr(0xdeadbeef);
  region->set_size(64);
  region->set_rkey(1234);
  std::string s;
  EXPECT_TRUE(desc.SerializeToString(&s));
  return std::vector<uint8_t>(s.begin(), s.end());
}

class RdmaManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    store_ = c10::make_intrusive<FakeStore>();
    auto& mgr = neuron::RdmaManager::Instance();
    mgr.SetStore(store_);
    mgr.SetRank(kSelfRank);
  }

  FakeStore& store() { return *store_; }
  neuron::RdmaManager& mgr() { return neuron::RdmaManager::Instance(); }

  c10::intrusive_ptr<FakeStore> store_;
};

// ===========================================================================
// Descriptor wire format
// ===========================================================================

TEST(TransferDescriptor, RoundTripsRegionFields) {
  // The descriptor is the only thing crossing between ranks; a dropped or
  // narrowed field silently corrupts the remote write's target.
  neuron::rdma::TransferDescriptor desc;
  desc.set_receiver_endpoint(std::string("\x01\x00\x02", 3));  // binary fi_getname blob
  desc.set_transfer_id(0xFFFFFFFFu);
  desc.set_rank(7);
  auto* region = desc.add_regions();
  region->set_remote_addr(0x7F0000001000ULL);
  region->set_size(1ULL << 40);
  region->set_rkey(0xABCDEF0123456789ULL);

  std::string bytes;
  ASSERT_TRUE(desc.SerializeToString(&bytes));
  neuron::rdma::TransferDescriptor parsed;
  ASSERT_TRUE(parsed.ParseFromString(bytes));

  EXPECT_EQ(parsed.receiver_endpoint(), std::string("\x01\x00\x02", 3));
  EXPECT_EQ(parsed.transfer_id(), 0xFFFFFFFFu);
  EXPECT_EQ(parsed.rank(), 7);
  ASSERT_EQ(parsed.regions_size(), 1);
  EXPECT_EQ(parsed.regions(0).remote_addr(), 0x7F0000001000ULL);
  EXPECT_EQ(parsed.regions(0).size(), 1ULL << 40);
  EXPECT_EQ(parsed.regions(0).rkey(), 0xABCDEF0123456789ULL);
}

TEST(TransferDescriptor, EmptyBytesParseAsRegionlessMessage) {
  // proto3 accepts empty input as a valid empty message, so "absent key" and
  // "regionless descriptor" are indistinguishable after parsing. The seq handshake
  // (not parse failure) is what guarantees the key was actually published by the peer.
  neuron::rdma::TransferDescriptor parsed;
  EXPECT_TRUE(parsed.ParseFromString(""));
  EXPECT_EQ(parsed.regions_size(), 0);
}

// ===========================================================================
// Control plane: store keys and sequence numbering
// ===========================================================================

TEST_F(RdmaManagerTest, RecvPublishesUnderSenderKeyedTuple) {
  constexpr int kTag = 4101;
  std::vector<neuron::P2PEntry> entries{ZeroSizeRecv(kTag)};

  mgr().PrepareBatch(entries);

  // A recv publishes under (peer=src, self=dst): the key the paired sender
  // computes as (self=src, peer=dst).
  ASSERT_EQ(store().set_keys().size(), 1u);
  EXPECT_EQ(store().set_keys()[0], StoreKey(kPeerRank, kSelfRank, kTag, 0));
}

TEST_F(RdmaManagerTest, SendLooksUpSelfKeyedTuple) {
  constexpr int kTag = 4102;
  store().set(StoreKey(kSelfRank, kPeerRank, kTag, 0), RegionlessDescriptor());

  std::vector<neuron::P2PEntry> entries{ZeroSizeSend(kTag)};
  ASSERT_NO_THROW(mgr().PrepareBatch(entries));

  ASSERT_EQ(store().requested_keys().size(), 1u);
  EXPECT_EQ(store().requested_keys()[0], StoreKey(kSelfRank, kPeerRank, kTag, 0));
}

TEST_F(RdmaManagerTest, SeqCountersAreIndependentPerTag) {
  constexpr int kTagA = 4103;
  constexpr int kTagB = 4104;

  std::vector<neuron::P2PEntry> entries{ZeroSizeRecv(kTagA), ZeroSizeRecv(kTagB)};
  mgr().PrepareBatch(entries);

  // Distinct tags are distinct streams: both start at seq 0. A shared counter
  // would put one of them at seq 1 and strand the peer waiting on seq 0.
  EXPECT_TRUE(store().Has(StoreKey(kPeerRank, kSelfRank, kTagA, 0)));
  EXPECT_TRUE(store().Has(StoreKey(kPeerRank, kSelfRank, kTagB, 0)));
}

TEST_F(RdmaManagerTest, SeqCountersAreIndependentPerPeer) {
  constexpr int kTag = 4105;
  constexpr int kOtherPeer = 2;

  std::vector<neuron::P2PEntry> entries{ZeroSizeRecv(kTag, kPeerRank),
                                        ZeroSizeRecv(kTag, kOtherPeer)};
  mgr().PrepareBatch(entries);

  // Same tag, different peers.
  EXPECT_TRUE(store().Has(StoreKey(kPeerRank, kSelfRank, kTag, 0)));
  EXPECT_TRUE(store().Has(StoreKey(kOtherPeer, kSelfRank, kTag, 0)));
}

TEST_F(RdmaManagerTest, SeqCounterAdvancesAcrossBatches) {
  constexpr int kTag = 4106;

  std::vector<neuron::P2PEntry> first{ZeroSizeRecv(kTag)};
  mgr().PrepareBatch(first);
  std::vector<neuron::P2PEntry> second{ZeroSizeRecv(kTag)};
  mgr().PrepareBatch(second);

  // Consecutive transfers on one tuple must occupy consecutive seq slots.
  EXPECT_EQ(store().set_keys().size(), 2u);
  EXPECT_EQ(store().set_keys()[0], StoreKey(kPeerRank, kSelfRank, kTag, 0));
  EXPECT_EQ(store().set_keys()[1], StoreKey(kPeerRank, kSelfRank, kTag, 1));
}

TEST_F(RdmaManagerTest, SendSeqCounterAdvancesAcrossBatches) {
  constexpr int kTag = 4107;
  store().set(StoreKey(kSelfRank, kPeerRank, kTag, 0), RegionlessDescriptor());
  store().set(StoreKey(kSelfRank, kPeerRank, kTag, 1), RegionlessDescriptor());

  std::vector<neuron::P2PEntry> first{ZeroSizeSend(kTag)};
  ASSERT_NO_THROW(mgr().PrepareBatch(first));
  std::vector<neuron::P2PEntry> second{ZeroSizeSend(kTag)};
  ASSERT_NO_THROW(mgr().PrepareBatch(second));

  // Asserted on the requested keys, not on a throw: a missing key would parse
  // as a regionless descriptor and be silently accepted by the zero-size path
  // so key identity is the only real evidence the seq advanced.
  ASSERT_EQ(store().requested_keys().size(), 2u);
  EXPECT_EQ(store().requested_keys()[0], StoreKey(kSelfRank, kPeerRank, kTag, 0));
  EXPECT_EQ(store().requested_keys()[1], StoreKey(kSelfRank, kPeerRank, kTag, 1));
}

TEST_F(RdmaManagerTest, SendAndRecvOnSameTagUseSeparateStreams) {
  constexpr int kTag = 4108;
  store().set(StoreKey(kSelfRank, kPeerRank, kTag, 0), RegionlessDescriptor());

  // (src,dst) ordering distinguishes the two directions, so a bidirectional
  // exchange on one tag does not collide - both sit at seq 0.
  std::vector<neuron::P2PEntry> entries{ZeroSizeRecv(kTag), ZeroSizeSend(kTag)};
  ASSERT_NO_THROW(mgr().PrepareBatch(entries));

  ASSERT_EQ(store().set_keys().size(), 1u);
  EXPECT_EQ(store().set_keys()[0], StoreKey(kPeerRank, kSelfRank, kTag, 0));
  ASSERT_EQ(store().requested_keys().size(), 1u);
  EXPECT_EQ(store().requested_keys()[0], StoreKey(kSelfRank, kPeerRank, kTag, 0));
}

TEST_F(RdmaManagerTest, TagExceedingFieldWidthIsRejected) {
  // NextSeq packs (src, dst, tag) into 20 bits each. dist.send/recv tags are
  // user-controlled, so an oversized tag must fail loudly rather than bleed
  // into the adjacent field and alias another tuple's counter.
  constexpr int kOversizedTag = 1 << 20;
  std::vector<neuron::P2PEntry> entries{ZeroSizeRecv(kOversizedTag)};
  EXPECT_THROW(mgr().PrepareBatch(entries), c10::Error);
}

TEST_F(RdmaManagerTest, LargestValidTagAccepted) {
  constexpr int kMaxTag = (1 << 20) - 1;
  std::vector<neuron::P2PEntry> entries{ZeroSizeRecv(kMaxTag)};
  ASSERT_NO_THROW(mgr().PrepareBatch(entries));
  EXPECT_TRUE(store().Has(StoreKey(kPeerRank, kSelfRank, kMaxTag, 0)));
}

// ===========================================================================
// Zero-size recv
// ===========================================================================

TEST_F(RdmaManagerTest, ZeroSizeRecvPublishesRegionlessDescriptor) {
  constexpr int kTag = 4201;
  std::vector<neuron::P2PEntry> entries{ZeroSizeRecv(kTag)};

  mgr().PrepareBatch(entries);

  auto raw = store().get(StoreKey(kPeerRank, kSelfRank, kTag, 0));
  neuron::rdma::TransferDescriptor desc;
  ASSERT_TRUE(desc.ParseFromArray(raw.data(), static_cast<int>(raw.size())));

  // No regions is the signal the sender keys off to skip fi_writedata.
  EXPECT_EQ(desc.regions_size(), 0);
  EXPECT_EQ(desc.rank(), kSelfRank);
  // transfer_id stays 0: a zero-size recv must not consume an id from
  // next_transfer_id_, or it could collide with a real recv's CQ immediate.
  EXPECT_EQ(desc.transfer_id(), 0u);
}

TEST_F(RdmaManagerTest, ZeroSizeRecvSkipsVaResolutionAndCompletesImmediately) {
  constexpr int kTag = 4202;
  std::vector<neuron::P2PEntry> entries{ZeroSizeRecv(kTag)};

  // Would throw if the entry went through ResolveHbmVA (null user_va is not
  // NRT-resolvable) or EnsureInitialized (no EFA device in this binary).
  ASSERT_NO_THROW(mgr().PrepareBatch(entries));

  EXPECT_EQ(entries[0].nrt_va, nullptr);
  EXPECT_TRUE(entries[0].prepared);
  // No FI_REMOTE_CQ_DATA will ever arrive for a zero-byte remote write, so the
  // recv is marked consumed up front; otherwise DrainAndPending reports it
  // pending forever and the op spins until the P2P timeout.
  EXPECT_TRUE(entries[0].recv_consumed);
  EXPECT_FALSE(mgr().DrainAndPending(entries));
}

// ===========================================================================
// Zero-size send
// ===========================================================================

TEST_F(RdmaManagerTest, ZeroSizeSendConsumesRegionlessDescriptorWithoutPosting) {
  constexpr int kTag = 4301;
  store().set(StoreKey(kSelfRank, kPeerRank, kTag, 0), RegionlessDescriptor());

  std::vector<neuron::P2PEntry> entries{ZeroSizeSend(kTag)};
  ASSERT_NO_THROW(mgr().PrepareBatch(entries));

  EXPECT_TRUE(entries[0].prepared);
  // No fi_writedata was posted, so there is no completion context. A null
  // send_ctx is what DrainAndPending treats as already-complete.
  EXPECT_EQ(entries[0].send_ctx, nullptr);
  EXPECT_FALSE(mgr().DrainAndPending(entries));
}

TEST_F(RdmaManagerTest, ZeroSizeSendRejectsDescriptorCarryingRegions) {
  constexpr int kTag = 4302;
  // Peer published a real (regionful) descriptor while this rank thinks the
  // transfer is zero-size: the ranks disagree on the tensor's size, so the seq
  // streams are already skewed. Fail loudly instead of silently dropping a
  // transfer the peer is waiting on.
  store().set(StoreKey(kSelfRank, kPeerRank, kTag, 0), DescriptorWithRegion());

  std::vector<neuron::P2PEntry> entries{ZeroSizeSend(kTag)};
  EXPECT_THROW(mgr().PrepareBatch(entries), c10::Error);
}

TEST_F(RdmaManagerTest, ZeroSizeSendRejectsUnparseableDescriptor) {
  constexpr int kTag = 4303;
  // Field 1 (regions), wire type 2, declared length 127, no payload: truncated
  // and unparseable. Parsing happens before the size check, so corruption must
  // surface as an error rather than degrade into a plausible "regionless, nothing to write".
  store().set(StoreKey(kSelfRank, kPeerRank, kTag, 0), std::vector<uint8_t>{0x0A, 0x7F});

  std::vector<neuron::P2PEntry> entries{ZeroSizeSend(kTag)};
  EXPECT_THROW(mgr().PrepareBatch(entries), c10::Error);
}

// ===========================================================================
// Batch-level behavior
// ===========================================================================

TEST_F(RdmaManagerTest, MultipleZeroSizeRecvsShareOneMultiSet) {
  constexpr int kTagA = 4401;
  constexpr int kTagB = 4402;

  std::vector<neuron::P2PEntry> entries{ZeroSizeRecv(kTagA), ZeroSizeRecv(kTagB)};
  mgr().PrepareBatch(entries);

  // Both descriptors go out in the single phase-1 multiSet, matching the
  // batching a coalesced batch_isend_irecv group relies on.
  EXPECT_EQ(store().multi_set_calls(), 1);
  for (const auto& e : entries) EXPECT_TRUE(e.recv_consumed);
  EXPECT_FALSE(mgr().DrainAndPending(entries));
}

TEST_F(RdmaManagerTest, ZeroSizeSendAndRecvInOneBatch) {
  constexpr int kSendTag = 4403;
  constexpr int kRecvTag = 4404;
  store().set(StoreKey(kSelfRank, kPeerRank, kSendTag, 0), RegionlessDescriptor());

  // Mirrors a coalesced group exchanging empty tensors both ways, as a pipeline
  // stage does when a microbatch carries no data for this rank.
  std::vector<neuron::P2PEntry> entries{ZeroSizeSend(kSendTag), ZeroSizeRecv(kRecvTag)};
  ASSERT_NO_THROW(mgr().PrepareBatch(entries));

  EXPECT_EQ(entries[0].send_ctx, nullptr);
  EXPECT_TRUE(entries[1].recv_consumed);
  EXPECT_FALSE(mgr().DrainAndPending(entries));
}

TEST_F(RdmaManagerTest, PrepareBatchIsIdempotent) {
  constexpr int kTag = 4405;
  std::vector<neuron::P2PEntry> entries{ZeroSizeRecv(kTag)};

  mgr().PrepareBatch(entries);
  const int after_first = store().multi_set_calls();

  // The kHost closure re-enters PrepareBatch while polling; the prepared
  // early-return must keep it from publishing again under a fresh seq, which
  // would strand the peer's send on the original key.
  mgr().PrepareBatch(entries);

  EXPECT_EQ(store().multi_set_calls(), after_first);
  EXPECT_FALSE(store().Has(StoreKey(kPeerRank, kSelfRank, kTag, 1)));
}

TEST_F(RdmaManagerTest, DrainAndPendingIsStableAcrossRepeatedCalls) {
  constexpr int kTag = 4406;
  std::vector<neuron::P2PEntry> entries{ZeroSizeRecv(kTag)};
  mgr().PrepareBatch(entries);

  // The poll loop calls DrainAndPending repeatedly. recv_consumed must keep an
  // already-satisfied recv from flipping back to pending once the completed-id
  // set is empty.
  for (int i = 0; i < 3; ++i) {
    EXPECT_FALSE(mgr().DrainAndPending(entries)) << "call " << i;
  }
}

TEST_F(RdmaManagerTest, EmptyBatchIsANoOp) {
  std::vector<neuron::P2PEntry> entries;

  ASSERT_NO_THROW(mgr().PrepareBatch(entries));
  EXPECT_EQ(store().multi_set_calls(), 0);
  EXPECT_FALSE(mgr().DrainAndPending(entries));
}

}  // namespace

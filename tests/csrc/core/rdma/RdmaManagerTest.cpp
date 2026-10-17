// RdmaManager unit tests: descriptor wire format, the pure seq/channel-key
// logic, and the local (fabric-free) zero-size P2P path.
//
// Testable without Neuron hardware or a peer process:
//   1. the TransferDescriptor protobuf wire format (pure serialization);
//   2. NextSeq (per-(src,dst,tag) seq lockstep) and TagChannelKey (the channel-map
//      key), reached via a friend seam; and
//   3. the zero-size (empty-tensor) send/recv path, which by design touches
//      neither the store nor the fabric — it only advances the per-(src,dst,tag)
//      NextSeq counter to stay in lockstep with the peer.
// The per-(peer,tag) channel setup + RMA round-trip — MR registration, the
// store coordinate/imm-token exchange, fi_writedata, CQ completion and error
// attribution — needs hardware and a peer, and belongs in the multi-rank Python
// P2P tests.

#include <gtest/gtest.h>

#include <chrono>
#include <map>
#include <set>
#include <string>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <vector>

#include "torch_neuronx/csrc/core/rdma/RdmaManager.h"
#include "torch_neuronx/csrc/core/rdma/TransferDescriptor.pb.h"

namespace neuron {
namespace {

constexpr int kSelfRank = 3;
constexpr int kPeerRank = 1;

// In-memory Store standing in for the TCPStore. The zero-size path must NOT touch
// it (only the one-time endpoint/mailbox bootstrap does, and that needs hardware),
// so the tests assert these recording vectors stay empty. multiGet does not block
// on a missing key — a test that reached a blocking multiGet has already failed,
// and returning empty keeps the suite from hanging.
class FakeStore : public c10d::Store {
 public:
  void set(const std::string& key, const std::vector<uint8_t>& value) override {
    set_keys_.push_back(key);
    data_[key] = value;
  }

  std::vector<uint8_t> get(const std::string& key) override {
    requested_keys_.push_back(key);
    auto it = data_.find(key);
    return it == data_.end() ? std::vector<uint8_t>{} : it->second;
  }

  void multiSet(const std::vector<std::string>& keys,
                const std::vector<std::vector<uint8_t>>& values) override {
    for (size_t i = 0; i < keys.size(); ++i) {
      set_keys_.push_back(keys[i]);
      data_[keys[i]] = values[i];
    }
  }

  std::vector<std::vector<uint8_t>> multiGet(const std::vector<std::string>& keys) override {
    std::vector<std::vector<uint8_t>> out;
    out.reserve(keys.size());
    for (const auto& k : keys) {
      requested_keys_.push_back(k);
      auto it = data_.find(k);
      out.push_back(it == data_.end() ? std::vector<uint8_t>{} : it->second);
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

  // Keys this rank published and looked up, in order. Both stay empty for the
  // zero-size path.
  const std::vector<std::string>& set_keys() const { return set_keys_; }
  const std::vector<std::string>& requested_keys() const { return requested_keys_; }

 private:
  std::map<std::string, std::vector<uint8_t>> data_;
  std::vector<std::string> set_keys_;
  std::vector<std::string> requested_keys_;
};

neuron::P2PEntry ZeroSizeRecv(int tag, int peer = kPeerRank) {
  return neuron::P2PEntry{neuron::P2PEntry::Kind::kRecv, /*storage_va=*/nullptr, /*size=*/0, peer,
                          tag};
}

neuron::P2PEntry ZeroSizeSend(int tag, int peer = kPeerRank) {
  return neuron::P2PEntry{neuron::P2PEntry::Kind::kSend, /*storage_va=*/nullptr, /*size=*/0, peer,
                          tag};
}

// Allocation from the allocator registry
constexpr uintptr_t kStorageVA = 0x1000;
// Mapped NRT HBM VA
constexpr uintptr_t kHbmVA = 0x7F0000000000ULL;

// Stub for ResolveHbmVA
void* FakeResolveHbmVA(void* storage_va) {
  EXPECT_EQ(reinterpret_cast<uintptr_t>(storage_va), kStorageVA);
  return reinterpret_cast<void*>(kHbmVA);
}

// A send that starts offset_bytes into kStorageVA
neuron::P2PEntry SendAtOffset(size_t offset_bytes, size_t size) {
  return neuron::P2PEntry{neuron::P2PEntry::Kind::kSend,
                          reinterpret_cast<void*>(kStorageVA),
                          size,
                          kPeerRank,
                          /*tag=*/0,
                          offset_bytes};
}

}  // namespace

// ===========================================================================
// Descriptor wire format — the only bytes that cross between ranks.
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
  // proto3 accepts empty input as a valid empty message. The mailbox seq
  // handshake (not parse success) is what proves a slot was actually written.
  neuron::rdma::TransferDescriptor parsed;
  EXPECT_TRUE(parsed.ParseFromString(""));
  EXPECT_EQ(parsed.regions_size(), 0);
}

class RdmaManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    store_ = c10::make_intrusive<FakeStore>();
    auto& mgr = RdmaManager::Instance();
    mgr.SetStore(store_);
    mgr.SetRank(kSelfRank);
  }

  FakeStore& store() { return *store_; }
  RdmaManager& mgr() { return RdmaManager::Instance(); }

  // --- private-member seams (friendship-gated) ---
  uint32_t nextSeq(int src, int dst, int tag) { return mgr().NextSeq(src, dst, tag); }
  static uint64_t tagChannelKey(int peer, int tag) { return RdmaManager::TagChannelKey(peer, tag); }
  static uint32_t ringDepth(uint32_t s, uint32_t r) { return RdmaManager::RingDepth(s, r); }
  static void* resolveEntryVAs(std::vector<P2PEntry>& entries, void* (*resolve)(void*)) {
    return RdmaManager::ResolveEntryVAs(entries, resolve);
  }
  static constexpr uint32_t kMailboxSlots = RdmaManager::kMailboxSlots;

  c10::intrusive_ptr<FakeStore> store_;
};

// ===========================================================================
// NextSeq: per-(src,dst,tag) counters kept in lockstep across ranks.
// The manager is a singleton with persistent counters, so each test uses a
// tuple range no other test touches (src/dst >= 40, tags >= 500).
// ===========================================================================

TEST_F(RdmaManagerTest, NextSeqAdvancesPerTuple) {
  // Consecutive transfers on one tuple must occupy consecutive seq slots, or the
  // sender's expected_seq guard would reject the receiver's descriptor.
  EXPECT_EQ(nextSeq(40, 41, 500), 0u);
  EXPECT_EQ(nextSeq(40, 41, 500), 1u);
  EXPECT_EQ(nextSeq(40, 41, 500), 2u);
}

TEST_F(RdmaManagerTest, NextSeqIsIndependentPerTag) {
  // Distinct tags are distinct streams: both start at 0. A shared counter would
  // strand the peer waiting on the seq the other tag consumed.
  EXPECT_EQ(nextSeq(42, 43, 501), 0u);
  EXPECT_EQ(nextSeq(42, 43, 502), 0u);
}

TEST_F(RdmaManagerTest, NextSeqIsIndependentPerEndpoint) {
  // Same tag, different src or dst — each (src,dst) pair is its own stream.
  EXPECT_EQ(nextSeq(44, 45, 503), 0u);
  EXPECT_EQ(nextSeq(46, 45, 503), 0u);  // different src
  EXPECT_EQ(nextSeq(44, 47, 503), 0u);  // different dst
}

TEST_F(RdmaManagerTest, NextSeqSeparatesSendAndRecvDirections) {
  // (src,dst) ordering distinguishes the two directions, so a bidirectional
  // exchange on one tag does not collide — both sit at seq 0.
  EXPECT_EQ(nextSeq(48, 49, 504), 0u);
  EXPECT_EQ(nextSeq(49, 48, 504), 0u);  // reversed direction is a separate stream
}

TEST_F(RdmaManagerTest, NextSeqRejectsFieldOverflow) {
  // (src,dst,tag) each pack into 20 bits; an out-of-range value would bleed into
  // the adjacent field and alias another tuple's counter.
  constexpr int kOverflow = 1 << 20;
  EXPECT_THROW(nextSeq(kOverflow, 1, 1), c10::Error);
  EXPECT_THROW(nextSeq(1, kOverflow, 1), c10::Error);
  EXPECT_THROW(nextSeq(1, 1, kOverflow), c10::Error);
}

// ===========================================================================
// TagChannelKey: packs (peer, tag) into the lazily-created channel-map key.
// Distinct channels must map to distinct keys, or two channels would alias one
// buffer + imm-token set.
// ===========================================================================

TEST_F(RdmaManagerTest, TagChannelKeyIsInjective) {
  std::set<uint64_t> seen;
  for (int peer = 0; peer < 40; ++peer) {
    for (int tag = 0; tag < 40; ++tag) {
      EXPECT_TRUE(seen.insert(tagChannelKey(peer, tag)).second)
          << "channel-key collision at peer=" << peer << " tag=" << tag;
    }
  }
}

TEST_F(RdmaManagerTest, TagChannelKeySeparatesPeerAndTag) {
  // peer and tag occupy disjoint halves of the 64-bit key.
  EXPECT_NE(tagChannelKey(1, 0), tagChannelKey(0, 1));
  EXPECT_EQ(tagChannelKey(0, 5), tagChannelKey(0, 5));
}

// ===========================================================================
// RingDepth: per-batch (peer, tag) ring depth = max(kMailboxSlots, sends, recvs).
// The depth must be identical on both ranks with no negotiation, which holds
// because my {send,recv} counts on a channel are the peer's {recv,send} counts and
// max is order-independent — so an asymmetric (e.g. ZBV) batch still agrees.
// ===========================================================================

TEST_F(RdmaManagerTest, RingDepthFloorsAtMailboxSlots) {
  // A single transfer each way still gets the double-buffering floor.
  EXPECT_EQ(ringDepth(0, 0), kMailboxSlots);
  EXPECT_EQ(ringDepth(1, 1), kMailboxSlots);
  EXPECT_EQ(ringDepth(1, 0), kMailboxSlots);
}

TEST_F(RdmaManagerTest, RingDepthGrowsToBatchCount) {
  // N same-(peer,tag) transfers one way need N distinct slots (above the floor).
  EXPECT_EQ(ringDepth(5, 1), 5u);
  EXPECT_EQ(ringDepth(1, 8), 8u);
  EXPECT_EQ(ringDepth(3, 7), 7u);
}

TEST_F(RdmaManagerTest, RingDepthIsSymmetricUnderAsymmetricCounts) {
  // The crux: a ZBV batch may have send_count != recv_count on a channel, but the
  // two ranks see mirrored counts. max() must return the same depth either way, or
  // the ranks would disagree on the ring size and read/write mismatched slots.
  for (uint32_t s = 0; s < 12; ++s) {
    for (uint32_t r = 0; r < 12; ++r) {
      EXPECT_EQ(ringDepth(s, r), ringDepth(r, s)) << "asymmetry at s=" << s << " r=" << r;
    }
  }
}

// ===========================================================================
// Entry addressing: storage base VA + view offset -> nrt_va.
// ===========================================================================

TEST_F(RdmaManagerTest, EntryAtStorageBaseAddressesHbmBase) {
  std::vector<neuron::P2PEntry> entries{SendAtOffset(/*offset_bytes=*/0, /*size=*/32)};
  EXPECT_EQ(resolveEntryVAs(entries, FakeResolveHbmVA), reinterpret_cast<void*>(kHbmVA));
  EXPECT_EQ(entries[0].nrt_va, reinterpret_cast<void*>(kHbmVA));
}

TEST_F(RdmaManagerTest, EntryWithViewOffsetAddressesWithOffset) {
  constexpr size_t kOffset = 8;
  std::vector<neuron::P2PEntry> entries{SendAtOffset(kOffset, /*size=*/8)};
  resolveEntryVAs(entries, FakeResolveHbmVA);
  EXPECT_EQ(entries[0].nrt_va, reinterpret_cast<void*>(kHbmVA + kOffset));
}

TEST_F(RdmaManagerTest, EachEntryInABatchKeepsOwnOffset) {
  std::vector<neuron::P2PEntry> entries{SendAtOffset(0, 8), SendAtOffset(8, 8),
                                        SendAtOffset(56, 8)};
  resolveEntryVAs(entries, FakeResolveHbmVA);
  EXPECT_EQ(entries[0].nrt_va, reinterpret_cast<void*>(kHbmVA));
  EXPECT_EQ(entries[1].nrt_va, reinterpret_cast<void*>(kHbmVA + 8));
  EXPECT_EQ(entries[2].nrt_va, reinterpret_cast<void*>(kHbmVA + 56));
}

TEST_F(RdmaManagerTest, LazyFabricInitGetsBaseVA) {
  std::vector<neuron::P2PEntry> entries{SendAtOffset(/*offset_bytes=*/64, /*size=*/16)};
  EXPECT_EQ(resolveEntryVAs(entries, FakeResolveHbmVA), reinterpret_cast<void*>(kHbmVA));
}

TEST_F(RdmaManagerTest, ZeroSizeEntryKeepsNullVAAndNeverResolved) {
  std::vector<neuron::P2PEntry> entries{ZeroSizeSend(/*tag=*/600)};
  void* first = resolveEntryVAs(entries, [](void*) -> void* {
    ADD_FAILURE() << "a zero-size entry has no VA to resolve";
    return nullptr;
  });
  EXPECT_EQ(first, nullptr);
  EXPECT_EQ(entries[0].nrt_va, nullptr);
}

TEST_F(RdmaManagerTest, FabricInitVAComesFromTheFirstRealVA) {
  std::vector<neuron::P2PEntry> entries{ZeroSizeSend(/*tag=*/601),
                                        SendAtOffset(/*offset_bytes=*/16, /*size=*/4)};
  EXPECT_EQ(resolveEntryVAs(entries, FakeResolveHbmVA), reinterpret_cast<void*>(kHbmVA));
  EXPECT_EQ(entries[0].nrt_va, nullptr);
  EXPECT_EQ(entries[1].nrt_va, reinterpret_cast<void*>(kHbmVA + 16));
}

// ===========================================================================
// Zero-size (empty-tensor) path: local, fabric-free, store-free.
// ===========================================================================

TEST_F(RdmaManagerTest, ZeroSizeRecvCompletesLocallyWithoutStoreOrFabric) {
  constexpr int kTag = 1;
  std::vector<neuron::P2PEntry> entries{ZeroSizeRecv(kTag)};
  ASSERT_NO_THROW(mgr().PrepareBatch(entries));

  EXPECT_TRUE(entries[0].prepared);
  EXPECT_TRUE(entries[0].recv_consumed);
  EXPECT_EQ(entries[0].transfer_id, 0u);
  // The descriptor exchange is skipped entirely: nothing crossed the store.
  EXPECT_TRUE(store().set_keys().empty());
  EXPECT_TRUE(store().requested_keys().empty());
  // No data completion will ever arrive, yet the recv is already satisfied.
  EXPECT_FALSE(mgr().DrainAndPending(entries));
}

TEST_F(RdmaManagerTest, ZeroSizeSendCompletesLocallyWithoutStoreOrFabric) {
  constexpr int kTag = 2;
  std::vector<neuron::P2PEntry> entries{ZeroSizeSend(kTag)};
  ASSERT_NO_THROW(mgr().PrepareBatch(entries));

  EXPECT_TRUE(entries[0].prepared);
  EXPECT_EQ(entries[0].send_ctx, nullptr);  // no fi_writedata was posted
  EXPECT_TRUE(store().set_keys().empty());
  EXPECT_TRUE(store().requested_keys().empty());
  EXPECT_FALSE(mgr().DrainAndPending(entries));
}

TEST_F(RdmaManagerTest, AllZeroSizeBatchNeverTouchesUninitializedFabric) {
  // Regression: an all-zero-size batch has no HBM VA to lazily initialize the
  // fabric with, so it must complete without ever calling into libfabric. Routed
  // through the mailbox this segfaulted on a null endpoint/AV; the local path
  // never reaches the fabric.
  constexpr int kTag = 3;
  std::vector<neuron::P2PEntry> entries{ZeroSizeRecv(kTag), ZeroSizeSend(kTag, /*peer=*/2)};
  ASSERT_NO_THROW(mgr().PrepareBatch(entries));
  EXPECT_TRUE(entries[0].prepared);
  EXPECT_TRUE(entries[1].prepared);
  EXPECT_FALSE(mgr().DrainAndPending(entries));
  EXPECT_TRUE(store().set_keys().empty());
  EXPECT_TRUE(store().requested_keys().empty());
}

TEST_F(RdmaManagerTest, MixedZeroSizeSendAndRecvInOneBatchBothComplete) {
  constexpr int kTag = 4;
  std::vector<neuron::P2PEntry> entries{ZeroSizeRecv(kTag), ZeroSizeSend(kTag)};
  ASSERT_NO_THROW(mgr().PrepareBatch(entries));
  EXPECT_FALSE(mgr().DrainAndPending(entries));
  for (const auto& e : entries) EXPECT_TRUE(e.prepared);
}

TEST_F(RdmaManagerTest, ZeroSizeAcceptsLargeTag) {
  // Tags are unbounded (channels are created lazily per (peer, tag)); zero-size
  // in particular is bounded only by NextSeq's 20-bit field. A large tag that a
  // fixed per-peer slot table would have rejected must be accepted.
  constexpr int kLargeTag = (1 << 20) - 7;
  std::vector<neuron::P2PEntry> entries{ZeroSizeRecv(kLargeTag)};
  EXPECT_NO_THROW(mgr().PrepareBatch(entries));
}

// ===========================================================================
// Batch bookkeeping.
// ===========================================================================

TEST_F(RdmaManagerTest, PrepareBatchIsIdempotent) {
  constexpr int kTag = 5;
  std::vector<neuron::P2PEntry> entries{ZeroSizeRecv(kTag)};
  ASSERT_NO_THROW(mgr().PrepareBatch(entries));
  // Second call is a no-op early-return on entries.front().prepared.
  ASSERT_NO_THROW(mgr().PrepareBatch(entries));
  EXPECT_TRUE(store().set_keys().empty());
  EXPECT_TRUE(store().requested_keys().empty());
}

TEST_F(RdmaManagerTest, EmptyBatchIsANoOp) {
  std::vector<neuron::P2PEntry> entries;
  ASSERT_NO_THROW(mgr().PrepareBatch(entries));
  EXPECT_FALSE(mgr().DrainAndPending(entries));
}

}  // namespace neuron

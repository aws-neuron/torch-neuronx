/* Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
==============================================================================*/
#pragma once

#include <rdma/fabric.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace neuron {

// Minimal status type for RDMA operations
struct RdmaStatus {
  std::string error;
  bool ok() const { return error.empty(); }
  static RdmaStatus Ok() { return {}; }
  static RdmaStatus Error(std::string msg) { return {std::move(msg)}; }
};

// RAII wrapper for libfabric fid objects
struct FidDeleter {
  template <typename T>
  void operator()(T* ptr) const {
    if (ptr) fi_close(&ptr->fid);
  }
};
template <typename T>
using FidPtr = std::unique_ptr<T, FidDeleter>;

// Per-send completion context. Hands itself to libfabric via fi_writedata's
// op_context argument; the CQ event returns the same pointer when the write
// completes. Lifetime is owned by RdmaManager::send_contexts_ and persists
// until DrainAndPending observes ctx->complete and erases the entry.
struct SendContext : fi_context2 {
  void* addr{nullptr};
  bool complete{false};
  bool error{false};
  bool is_mailbox{false};
};

// One entry of a P2P batch. storage_va is resolved to nrt_va inside PrepareBatch
// using NeuronCachingAllocator::findTensor + nrt_tensor_get_va — required
// under lazy allocation, where data_ptr() is an mmap placeholder until the
// kAlloc op binds the slice.
struct P2PEntry {
  enum class Kind { kSend, kRecv };
  Kind kind;
  // Base VA of the tensor's storage
  void* storage_va;
  size_t size;
  int peer;
  int tag;
  // Byte offset of the tensor in that storage: storage_offset * element_size
  size_t offset_bytes = 0;

  // Filled by PrepareBatch:
  void* nrt_va = nullptr;    // resolved VA: storage HBM VA + offset_bytes
  uint32_t transfer_id = 0;  // recv only — matches FI_REMOTE_CQ_DATA imm
  bool prepared = false;
  // Set true by DrainAndPending the first time a recv's transfer_id is found
  // (and consumed) in completed_recv_ids_. Subsequent drain calls skip it.
  // Without this, the engine would re-check an already-consumed recv against
  // the now-empty completed set and report pending=true forever.
  // Asymmetric with send completion (which is ctx-identified) because
  // FI_REMOTE_CQ_DATA carries no per-recv ctx.
  bool recv_consumed = false;
  // Per-send completion context. Set by PrepareBatch when fi_writedata is
  // posted; consulted by DrainAndPending to test completion. Multiple
  // in-flight sends may share the same nrt_va (looped/interleaved
  // schedules), so completion is keyed by this pointer rather than by
  // nrt_va. The pointer is owned (via unique_ptr) by RdmaManager's
  // send_contexts_ pool.
  SendContext* send_ctx = nullptr;
};

// Singleton managing RDMA over EFA for P2P send/recv.
// Owns all libfabric state (fabric, domain, endpoint, CQ, AV, MR).
//
// Threading (single-owner-thread invariant — load-bearing):
//   - SetRank / SetStore run once on the main thread during the first
//     dist.send/recv (under NeuronBackend::p2p_mutex_), published via the rank_
//     atomic + setup_mutex_/initialized_ fence.
//   - PrepareBatch and DrainAndPending both run on the one HostWorkerPool worker
//     (the P2P kHost op's RunHost body; see NeuronBackend.cpp SubmitP2POp). With
//     kHostWorkerThreads == 1 (static_assert'd there), every mutable member below
//     is touched by a single thread and needs no lock.
//   - findTensor (in PrepareBatch) is safe off the engine: the tensor's AllocBind
//     is ordered before this op's enqueue and Submit→WorkerLoop is a
//     release/acquire edge, so the bind happens-before the read; NeuronWork keeps
//     the tensor alive until the op retires.
//   - libfabric calls are FI_THREAD_SAFE regardless.
class RdmaManager {
  friend class RdmaManagerTest;

 public:
  static RdmaManager& Instance();

  // --- Main thread: one-time setup ---
  void SetRank(int rank);
  void SetStore(c10::intrusive_ptr<c10d::Store> store);

  // --- Host worker: per-batch ---
  // First call (per process) runs libfabric init using the HBM VA resolved from
  // the first entry, publishes our endpoint + mailbox coordinates to the store,
  // and eagerly AV-inserts every peer (one-time bootstrap). Every call: resolve
  // VAs, exchange recv descriptors via the mailbox (receiver RMA-writes into the
  // sender's inbox; sender reads locally), then post fi_writedata for each send.
  // Idempotent — re-entry hits the entries[i].prepared early-return. Throws on
  // libfabric / store errors.
  void PrepareBatch(std::vector<P2PEntry>& entries, int64_t timeout_ms = 0);

  // Drains the CQ once and returns true if any entry is still pending.
  bool DrainAndPending(std::vector<P2PEntry>& entries);

 private:
  RdmaManager() = default;
  ~RdmaManager();
  RdmaManager(const RdmaManager&) = delete;
  RdmaManager& operator=(const RdmaManager&) = delete;

  // --- Init helpers ---
  struct HBMInfo {
    void* base;
    size_t size;
    int device_id;
    std::string efa_bdf;
  };
  // Engine-thread one-time fabric init. nrt_va must be a real HBM VA (resolved
  // via findTensor + nrt_tensor_get_va by the caller); nrt_get_attached_efa_bdf
  // is what turns it into the EFA NIC selection.
  void EnsureInitialized(void* nrt_va);

  void DiscoverHBMRegions();
  RdmaStatus DiscoverEFAInfo();
  RdmaStatus InitFabric();
  RdmaStatus CreateEndpoint();
  RdmaStatus EnsureRegistered(const HBMInfo& info);
  RdmaStatus EnsureBufferRegistered(void* addr, size_t size, int device_id = 0);
  fid_mr* FindMR(void* addr, size_t size);
  // Fill nrt_va for every entry of a batch: resolve the storage base, then add the
  // view offset. Returns the HBM base VA of the first entry that moves bytes or
  // nullptr for an all-zero-size batch.
  static void* ResolveEntryVAs(std::vector<P2PEntry>& entries, void* (*resolve)(void*));
  // Non-blocking AV insert: returns false (defer) if the peer hasn't published its
  // endpoint yet. On success caches + returns the fi_addr via *out. Uses store
  // check() (never blocks) before the one-time get()/fi_av_insert.
  bool TryGetOrInsertPeer(int peer, fi_addr_t* out);

  // --- State ---
  // initialized_ is engine-thread-only after the move to lazy init; atomic
  // is harmless overkill kept for tear-down readers.
  std::atomic<bool> initialized_{false};
  // Sticky fatal-CQ flag. Set when fi_cq_read returns an error that cannot be
  // attributed to a specific SendContext (no per-op error entry). Once set it
  // is never cleared — every subsequent batch's DrainAndPending throws rather
  // than busy-loop forever on recvs whose completions can no longer arrive (a
  // fabric-level CQ error is unrecoverable for the process). Touched only on the
  // single host worker thread (see the single-owner-thread invariant above).
  bool cq_fatal_{false};
  // rank_ is written by SetRank (user thread) and read by EnsureInitialized
  // (engine thread). atomic + seq_cst provides the missing happens-before edge
  // (SubmitOperationContext between SetRank and engine wake-up is incidental
  // ordering, not guaranteed).
  std::atomic<int> rank_{-1};
  std::string efa_bdf_;
  std::string efa_interface_;
  std::string local_endpoint_addr_;

  // Libfabric handles
  FidPtr<fid_fabric> fabric_;
  FidPtr<fid_domain> domain_;
  FidPtr<fid_ep> endpoint_;
  FidPtr<fid_cq> cq_;
  FidPtr<fid_av> av_;
  fi_info* hints_{nullptr};
  fi_info* info_{nullptr};

  // Memory registrations
  struct RegisteredMemory {
    int device_id;
    void* base_address;
    size_t size;
    FidPtr<fid_mr> mr;
  };
  std::vector<RegisteredMemory> registered_memory_;

  // AV cache: peer global rank -> resolved fi_addr_t.
  std::unordered_map<int, fi_addr_t> av_cache_;

  // A mailbox slot is a small fixed frame RMA-written from one rank's outbox into
  // another rank's inbox: a raw {seq, payload_len} header followed by a serialized
  // neuron::rdma::TransferDescriptor. Each (peer, tag) channel owns a ring of
  // TagChannel::ring_depth such slots, indexed by seq % ring_depth, so a receiver
  // can write step k+1's descriptor without clobbering step k's before the sender
  // consumes it. The depth is sized per batch to max(kMailboxSlots, #sends, #recvs)
  // on the channel, so a single coalesced batch with N same-(peer,tag) transfers
  // gets N distinct slots; kMailboxSlots is just the floor.
  static constexpr uint32_t kMailboxSlots = 2;              // minimum ring depth (floor)
  static constexpr uint32_t kMailboxImmFlag = 0x80000000u;  // imm bit: descriptor write
  static constexpr uint32_t kMailboxPayloadBytes = 128;     // max serialized descriptor
  struct MailboxSlot {
    uint32_t seq;                           // gen stamp; sender accepts when == expected
    uint32_t payload_len;                   // valid bytes in payload
    uint8_t payload[kMailboxPayloadBytes];  // serialized neuron::rdma::TransferDescriptor
  };
  static_assert(sizeof(MailboxSlot) == 8 + kMailboxPayloadBytes, "MailboxSlot frame changed");
  static_assert(std::is_trivially_copyable<MailboxSlot>::value,
                "MailboxSlot must be trivially copyable (it is RMA-written raw)");
  static constexpr size_t kMailboxSlotBytes = sizeof(MailboxSlot);

  // A (peer, tag) descriptor channel, created lazily on first use so tags are
  // unbounded.
  // Layout of buf: [ INBOX (peer writes our recv-descriptors here) | OUTBOX (we
  // stage our recv-descriptors before RMA-writing them to the peer) ], each half
  // ring_depth slots. One MR over the whole buffer. ring_depth is chosen per batch
  // and is provably identical on both ends (max is symmetric over the mirrored
  // send/recv counts; see PrepareBatch), so no depth negotiation is needed.
  struct TagChannel {
    std::vector<uint8_t> buf;  // [inbox | outbox]; each half = ring_depth * kMailboxSlotBytes
    FidPtr<fid_mr> mr;
    uint32_t ring_depth = 0;         // slots per half (our inbox + our outbox)
    uint64_t inbox_rkey = 0;         // rkey of our inbox half (published to the peer)
    uint32_t inbox_base_token = 0;   // imm token WE assign; ring r arrival carries base+r
    bool remote_ready = false;       // peer inbox coords fetched?
    uint64_t remote_inbox_addr = 0;  // peer's inbox half base (we write here)
    uint64_t remote_rkey = 0;        // rkey for the peer's inbox
    uint32_t remote_base_token = 0;  // imm token the peer assigned; we stamp base+r
    uint32_t remote_ring_depth = 0;  // peer inbox depth (== ring_depth once resolved)
  };
  std::unordered_map<uint64_t, TagChannel> tag_channels_;  // key = TagChannelKey(peer, tag)
  // Old channel buffers retired by a ring-depth grow. A grow allocates a fresh
  // buffer + MR and replaces the channel; the previous buffer may still have an
  // in-flight mailbox write referencing it, so rather than free it under a
  // possibly-pending write we park it here (its MR stays registered) until
  // teardown. Grows are rare — for a fixed schedule the first batch on a channel
  // sizes it to the max and it never grows again — so this stays tiny.
  std::vector<TagChannel> retired_channels_;
  // Dense imm-token allocator. Each channel's inbox reserves ring_depth tokens (one
  // per ring slot); the token uniquely identifies a physical inbox slot so the
  // sender can match a CQ arrival to the exact pending send. A grow reserves a new
  // block and abandons the old one (2^31 token space; never exhausted in practice).
  // Must stay below kMailboxImmFlag so it never collides with the flag bit.
  std::atomic<uint32_t> next_imm_token_{1};
  static uint64_t TagChannelKey(int peer, int tag) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(peer)) << 32) | static_cast<uint32_t>(tag);
  }
  // Ring depth for a channel this batch: max(kMailboxSlots, send_count, recv_count).
  // The max is symmetric across the two ranks (my {send,recv} == peer's {recv,send}
  // per tag), so both ends size the ring identically from their own local batch.
  static uint32_t RingDepth(uint32_t send_count, uint32_t recv_count);
  // Non-blocking: lazily create (or GROW to needed_depth) the (peer, tag) channel
  // and PUBLISH our inbox coordinates (addr + rkey + imm token + depth). Returns
  // nullptr if the peer's endpoint isn't published yet (bilateral AV: we must
  // fi_av_insert the peer before writing), so the caller defers this op instead of
  // blocking the whole batch. Idempotent once the channel is at >= needed_depth.
  TagChannel* TryEnsureLocalChannel(int peer, int tag, uint32_t needed_depth);
  // Non-blocking one-time (per depth) fetch of the peer's inbox coordinates, used
  // only by the recv side (which writes into the peer's inbox). Returns false
  // (defer) if the peer hasn't yet published a channel of at least needed_depth —
  // check() never blocks server-side, so a not-yet-ready peer just re-checks next
  // iteration. Idempotent once resolved at >= needed_depth (ch.remote_ready).
  bool TryResolveRemoteChannel(TagChannel& ch, int peer, int tag, uint32_t needed_depth);

  // In-flight send tracking. ctx pointers are stable for the lifetime of the
  // send (until DrainAndPending consumes the completion); we hold ownership
  // here and stash a raw pointer in P2PEntry::send_ctx. Looped/interleaved
  // schedules can have multiple in-flight sends sharing the same nrt_va,
  // which is why we cannot key by nrt_va. Single-host-worker-owned (see above).
  std::unordered_set<std::unique_ptr<SendContext>> send_contexts_;

  // Recv completions: transfer_ids that arrived via CQ. Single-host-worker-owned.
  std::unordered_set<uint32_t> completed_recv_ids_;

  // Descriptor-mailbox arrivals confirmed on the CQ, keyed by imm (the per-slot
  // token, flag bit set). Consuming the remote-write completion that records an
  // entry here is the delivery barrier for that slot's payload; PrepareBatch
  // phase 2 waits on the matching imm before reading the slot, then erases it.
  // Single-host-worker-owned.
  std::unordered_set<uint32_t> mailbox_arrived_;

  // Transfer ID counter
  std::atomic<uint32_t> next_transfer_id_{1};

  // HBM discovery
  std::vector<HBMInfo> hbm_infos_;

  // --- Store-based control plane state ---
  // store_ is set by SetStore (user thread), read by EnsureInitialized /
  // PrepareBatch / GetOrInsertPeer (engine thread). One-time-write protection
  // via setup_mutex_; reads after EnsureInitialized are race-free because
  // initialized_ acts as the publication fence.
  c10::intrusive_ptr<c10d::Store> store_;
  mutable std::mutex setup_mutex_;
  std::unordered_map<uint64_t, uint32_t> seq_counters_;  // (src,dst,tag) -> next seq
  uint32_t NextSeq(int src, int dst, int tag);
};

}  // namespace neuron

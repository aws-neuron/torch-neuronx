/* Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
==============================================================================*/
#pragma once

#include <rdma/fabric.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <torch/csrc/distributed/c10d/Store.hpp>
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
};

// One entry of a P2P batch. user_va is resolved to nrt_va inside PrepareBatch
// using NeuronCachingAllocator::findTensor + nrt_tensor_get_va — required
// under lazy allocation, where data_ptr() is an mmap placeholder until the
// kAlloc op binds the slice.
struct P2PEntry {
  enum class Kind { kSend, kRecv };
  Kind kind;
  void* user_va;
  size_t size;
  int peer;
  int tag;

  // Filled by PrepareBatch:
  void* nrt_va = nullptr;
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
 public:
  static RdmaManager& Instance();

  // --- Main thread: one-time setup ---
  void SetRank(int rank);
  void SetStore(c10::intrusive_ptr<c10d::Store> store);

  // --- Host worker: per-batch ---
  // First call (per process) runs libfabric init using the HBM VA resolved from
  // the first entry and publishes our endpoint to the store. Every call: resolve
  // VAs, AV-insert new peers, exchange descriptors (multiSet/multiGet), post
  // fi_writedata. Idempotent — re-entry hits the entries[i].prepared early-return.
  // Throws on libfabric / store errors.
  void PrepareBatch(std::vector<P2PEntry>& entries);

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
  fi_addr_t GetOrInsertPeer(int peer);  // store->get + fi_av_insert + cache

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

  // In-flight send tracking. ctx pointers are stable for the lifetime of the
  // send (until DrainAndPending consumes the completion); we hold ownership
  // here and stash a raw pointer in P2PEntry::send_ctx. Looped/interleaved
  // schedules can have multiple in-flight sends sharing the same nrt_va,
  // which is why we cannot key by nrt_va. Single-host-worker-owned (see above).
  std::unordered_set<std::unique_ptr<SendContext>> send_contexts_;

  // Recv completions: transfer_ids that arrived via CQ. Single-host-worker-owned.
  std::unordered_set<uint32_t> completed_recv_ids_;

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
  std::unordered_map<uint64_t, int> seq_counters_;  // (src,dst,tag) -> next seq
  int NextSeq(int src, int dst, int tag);
  std::string MakeStoreKey(int src, int dst, int tag, int seq);
};

}  // namespace neuron

/* Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
==============================================================================*/
#include "torch_neuronx/csrc/core/rdma/RdmaManager.h"

#include <dirent.h>
#include <dlfcn.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_ext.h>
#include <rdma/fi_rma.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <thread>

extern "C" {
#include <nrt/nrt.h>
}

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/rdma/TransferDescriptor.pb.h"

namespace neuron {

// ---------------------------------------------------------------------------
// NRT dlsym wrappers
// ---------------------------------------------------------------------------
namespace {

struct NrtRdmaSymbols {
  using GetHbmMmapVaFn = NRT_STATUS (*)(uint32_t, uint32_t, void**, size_t*);
  using GetAttachedEfaBdfFn = NRT_STATUS (*)(void*, char*, size_t*);

  GetHbmMmapVaFn nrt_get_hbm_mmap_va = nullptr;
  GetAttachedEfaBdfFn nrt_get_attached_efa_bdf = nullptr;

  static const NrtRdmaSymbols& Instance() {
    static NrtRdmaSymbols syms;
    return syms;
  }

 private:
  NrtRdmaSymbols() {
    void* handle = dlopen("libnrt.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!handle) handle = dlopen("libnrt.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!handle) return;
    nrt_get_hbm_mmap_va = reinterpret_cast<GetHbmMmapVaFn>(dlsym(handle, "nrt_get_hbm_mmap_va"));
    nrt_get_attached_efa_bdf =
        reinterpret_cast<GetAttachedEfaBdfFn>(dlsym(handle, "nrt_get_attached_efa_bdf"));
  }
};

// ---------------------------------------------------------------------------
// Libfabric dlsym wrappers - loaded lazily so import torch_neuronx succeeds
// without EFA installed. Only bootstrap functions are true exports; all other
// fi_* calls are static inline vtable dispatches defined in the headers.
// ---------------------------------------------------------------------------
struct LibfabricSymbols {
  decltype(&::fi_getinfo) fi_getinfo = nullptr;
  decltype(&::fi_freeinfo) fi_freeinfo = nullptr;
  decltype(&::fi_dupinfo) fi_dupinfo = nullptr;
  decltype(&::fi_fabric) fi_fabric = nullptr;
  decltype(&::fi_strerror) fi_strerror = nullptr;

  bool loaded = false;

  static const LibfabricSymbols& Instance() {
    static LibfabricSymbols syms;
    return syms;
  }

 private:
  LibfabricSymbols() {
    void* handle = dlopen("libfabric.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!handle) return;

    auto load = [handle](auto& dest, const char* name) { *(void**)(&dest) = dlsym(handle, name); };
    load(fi_getinfo, "fi_getinfo");
    load(fi_freeinfo, "fi_freeinfo");
    load(fi_dupinfo, "fi_dupinfo");
    load(fi_fabric, "fi_fabric");
    load(fi_strerror, "fi_strerror");

    loaded = (fi_getinfo && fi_freeinfo && fi_dupinfo && fi_fabric && fi_strerror);
  }
};

bool TensorRdmaSupport() {
  const auto& syms = NrtRdmaSymbols::Instance();
  if (!syms.nrt_get_hbm_mmap_va || !syms.nrt_get_attached_efa_bdf) return false;
  static const bool hbm_mapped = [] {
    const char* val = std::getenv("NEURON_RT_MAP_HBM");
    if (!val || val[0] == '\0' || std::strcmp(val, "0") == 0) return false;
    return strcasecmp(val, "false") != 0;
  }();
  return hbm_mapped;
}

RdmaStatus NrtGetHbmMmapVa(uint32_t device_id, uint32_t hbm_idx, void** hbm_base,
                           size_t* hbm_size) {
  const auto& syms = NrtRdmaSymbols::Instance();
  if (!syms.nrt_get_hbm_mmap_va) return RdmaStatus::Error("nrt_get_hbm_mmap_va not available");
  NRT_STATUS rt = syms.nrt_get_hbm_mmap_va(device_id, hbm_idx, hbm_base, hbm_size);
  if (rt == NRT_INVALID_HANDLE) return RdmaStatus::Error("No more HBM regions");
  if (rt != NRT_SUCCESS)
    return RdmaStatus::Error("nrt_get_hbm_mmap_va failed: " + std::to_string(rt));
  return RdmaStatus::Ok();
}

RdmaStatus NrtGetAttachedEfaBdf(void* hbm_base, std::string* out_bdf) {
  const auto& syms = NrtRdmaSymbols::Instance();
  if (!syms.nrt_get_attached_efa_bdf)
    return RdmaStatus::Error("nrt_get_attached_efa_bdf not available");
  char buf[256];
  size_t len = sizeof(buf);
  NRT_STATUS rt = syms.nrt_get_attached_efa_bdf(hbm_base, buf, &len);
  if (rt != NRT_SUCCESS)
    return RdmaStatus::Error("nrt_get_attached_efa_bdf failed: " + std::to_string(rt));
  *out_bdf = std::string(buf, strnlen(buf, len));
  return RdmaStatus::Ok();
}

constexpr int kCQDepth = 1024;
constexpr int kCQBatchSize = 32;

}  // namespace

// ---------------------------------------------------------------------------
// Singleton & lifecycle
// ---------------------------------------------------------------------------
RdmaManager& RdmaManager::Instance() {
  static RdmaManager instance;
  return instance;
}

RdmaManager::~RdmaManager() {
  // Close the endpoint/CQ before freeing SendContexts: libfabric owns each
  // op_context until completion, so fi_close must quiesce in-flight writes
  // (which reference these contexts) before we destroy them.
  endpoint_.reset();
  cq_.reset();
  av_.reset();
  registered_memory_.clear();
  tag_channels_.clear();
  retired_channels_.clear();
  domain_.reset();
  fabric_.reset();
  if (info_) LibfabricSymbols::Instance().fi_freeinfo(info_);
  if (hints_) LibfabricSymbols::Instance().fi_freeinfo(hints_);
  info_ = nullptr;
  hints_ = nullptr;
  // Safe to clear: at process teardown the engine has already joined the host
  // worker (OperationExecutionEngine::Shutdown), so no PrepareBatch/
  // DrainAndPending can be touching these containers.
  send_contexts_.clear();
  completed_recv_ids_.clear();
  mailbox_arrived_.clear();
  av_cache_.clear();
}

// ---------------------------------------------------------------------------
// HBM discovery
// ---------------------------------------------------------------------------
namespace {
std::vector<int> EnumerateNeuronDevices() {
  std::vector<int> ids;
  DIR* dir = opendir("/dev");
  if (!dir) return ids;
  while (struct dirent* entry = readdir(dir)) {
    int id;
    if (sscanf(entry->d_name, "neuron%d", &id) == 1) {
      ids.push_back(id);
    }
  }
  closedir(dir);
  std::sort(ids.begin(), ids.end());
  return ids;
}
}  // namespace

void RdmaManager::DiscoverHBMRegions() {
  if (!TensorRdmaSupport()) {
    TORCH_NEURONX_RDMA_INFO("DiscoverHBMRegions: TensorRdmaSupport=false, skipping");
    return;
  }
  auto enumerated = EnumerateNeuronDevices();
  std::string ids;
  for (int id : enumerated) {
    if (!ids.empty()) ids += ",";
    ids += std::to_string(id);
  }
  TORCH_NEURONX_RDMA_INFO("DiscoverHBMRegions: enumerated /dev/neuron* device_ids=[", ids, "]");
  for (int device_id : enumerated) {
    for (int hbm_idx = 0; hbm_idx < 4; hbm_idx++) {
      void* base = nullptr;
      size_t size = 0;
      auto va_status = NrtGetHbmMmapVa(device_id, hbm_idx, &base, &size);
      if (!va_status.ok()) {
        TORCH_NEURONX_RDMA_INFO("DiscoverHBMRegions: nrt_get_hbm_mmap_va(device_id=", device_id,
                                ", hbm_idx=", hbm_idx, ") failed: ", va_status.error,
                                " (stop scanning this device)");
        break;
      }
      std::string bdf;
      auto bdf_status = NrtGetAttachedEfaBdf(base, &bdf);
      if (bdf_status.ok()) {
        TORCH_NEURONX_RDMA_INFO("DiscoverHBMRegions: device_id=", device_id, " hbm_idx=", hbm_idx,
                                " base=", base, " size=", size, " bdf=", bdf, " -> registered");
        hbm_infos_.push_back({base, size, device_id, bdf});
      } else {
        TORCH_NEURONX_RDMA_INFO("DiscoverHBMRegions: device_id=", device_id, " hbm_idx=", hbm_idx,
                                " base=", base, " size=", size,
                                " bdf-lookup failed: ", bdf_status.error, " -> skipped");
      }
    }
  }
  TORCH_NEURONX_RDMA_INFO("DiscoverHBMRegions: total ", hbm_infos_.size(), " HBM regions across ",
                          enumerated.size(), " enumerated devices");
}

RdmaStatus RdmaManager::DiscoverEFAInfo() {
  fi_info* hints = LibfabricSymbols::Instance().fi_dupinfo(nullptr);
  if (!hints) return RdmaStatus::Error("fi_allocinfo failed");
  hints->fabric_attr->prov_name = strdup("efa");

  fi_info* info = nullptr;
  int ret =
      LibfabricSymbols::Instance().fi_getinfo(FI_VERSION(1, 9), nullptr, nullptr, 0, hints, &info);
  LibfabricSymbols::Instance().fi_freeinfo(hints);
  if (ret)
    return RdmaStatus::Error(std::string("fi_getinfo failed: ") +
                             LibfabricSymbols::Instance().fi_strerror(-ret));

  unsigned int domain, bus, device, function;
  if (sscanf(efa_bdf_.c_str(), "%x:%x:%x.%x", &domain, &bus, &device, &function) != 4) {
    LibfabricSymbols::Instance().fi_freeinfo(info);
    return RdmaStatus::Error(std::string("Invalid BDF format: ") + efa_bdf_);
  }

  bool found = false;
  for (fi_info* cur = info; cur; cur = cur->next) {
    if (cur->domain_attr && cur->domain_attr->name && cur->nic && cur->nic->bus_attr &&
        cur->nic->bus_attr->bus_type == FI_BUS_PCI) {
      auto& pci = cur->nic->bus_attr->attr.pci;
      if (pci.domain_id == domain && pci.bus_id == bus && pci.device_id == device &&
          pci.function_id == function) {
        efa_interface_ = cur->domain_attr->name;
        found = true;
        break;
      }
    }
  }
  LibfabricSymbols::Instance().fi_freeinfo(info);

  if (!found) return RdmaStatus::Error(std::string("No EFA device found for BDF: ") + efa_bdf_);
  TORCH_NEURONX_RDMA_INFO("Discovered EFA domain: ", efa_interface_, " for BDF: ", efa_bdf_);
  return RdmaStatus::Ok();
}

RdmaStatus RdmaManager::InitFabric() {
  hints_ = LibfabricSymbols::Instance().fi_dupinfo(nullptr);
  if (!hints_) return RdmaStatus::Error("fi_allocinfo failed");

  hints_->caps = FI_MSG | FI_RMA | FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_HMEM;
  hints_->mode = FI_CONTEXT2;
  hints_->domain_attr->mr_mode =
      FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_HMEM;
  hints_->domain_attr->threading = FI_THREAD_SAFE;
  hints_->domain_attr->name = strdup(efa_interface_.c_str());
  hints_->ep_attr->type = FI_EP_RDM;
  hints_->fabric_attr->prov_name = strdup("efa");
  hints_->fabric_attr->name = strdup("efa-direct");

  int ret = LibfabricSymbols::Instance().fi_getinfo(FI_VERSION(1, 9), nullptr, nullptr, 0, hints_,
                                                    &info_);
  if (ret)
    return RdmaStatus::Error(std::string("fi_getinfo failed: ") +
                             LibfabricSymbols::Instance().fi_strerror(-ret));

  if (info_->domain_attr->cq_data_size < 4)
    return RdmaStatus::Error("EFA provider does not support 4-byte immediate data");

  fid_fabric* fabric_raw = nullptr;
  ret = LibfabricSymbols::Instance().fi_fabric(info_->fabric_attr, &fabric_raw, nullptr);
  if (ret)
    return RdmaStatus::Error(std::string("fi_fabric failed: ") +
                             LibfabricSymbols::Instance().fi_strerror(-ret));
  fabric_.reset(fabric_raw);

  fid_domain* domain_raw = nullptr;
  ret = fi_domain(fabric_.get(), info_, &domain_raw, nullptr);
  if (ret)
    return RdmaStatus::Error(std::string("fi_domain failed: ") +
                             LibfabricSymbols::Instance().fi_strerror(-ret));
  domain_.reset(domain_raw);

  return CreateEndpoint();
}

RdmaStatus RdmaManager::CreateEndpoint() {
  fi_cq_attr cq_attr = {};
  cq_attr.format = FI_CQ_FORMAT_DATA;
  // efa-direct does not support FI_WAIT_FD; CQ is polled in DrainAndPending.
  cq_attr.wait_obj = FI_WAIT_NONE;
  cq_attr.size = kCQDepth;

  fid_cq* cq_raw = nullptr;
  int ret = fi_cq_open(domain_.get(), &cq_attr, &cq_raw, nullptr);
  if (ret)
    return RdmaStatus::Error(std::string("fi_cq_open failed: ") +
                             LibfabricSymbols::Instance().fi_strerror(-ret));
  cq_.reset(cq_raw);

  fi_av_attr av_attr = {};
  av_attr.type = FI_AV_MAP;
  fid_av* av_raw = nullptr;
  ret = fi_av_open(domain_.get(), &av_attr, &av_raw, nullptr);
  if (ret)
    return RdmaStatus::Error(std::string("fi_av_open failed: ") +
                             LibfabricSymbols::Instance().fi_strerror(-ret));
  av_.reset(av_raw);

  fid_ep* ep_raw = nullptr;
  ret = fi_endpoint(domain_.get(), info_, &ep_raw, nullptr);
  if (ret)
    return RdmaStatus::Error(std::string("fi_endpoint failed: ") +
                             LibfabricSymbols::Instance().fi_strerror(-ret));
  endpoint_.reset(ep_raw);

  ret = fi_ep_bind(endpoint_.get(), &cq_->fid, FI_TRANSMIT | FI_RECV);
  if (ret)
    return RdmaStatus::Error(std::string("fi_ep_bind(cq) failed: ") +
                             LibfabricSymbols::Instance().fi_strerror(-ret));
  ret = fi_ep_bind(endpoint_.get(), &av_->fid, 0);
  if (ret)
    return RdmaStatus::Error(std::string("fi_ep_bind(av) failed: ") +
                             LibfabricSymbols::Instance().fi_strerror(-ret));

  // Disable EFA's intra-node SHM transfer. The default SHM path does not
  // support FI_HMEM_NEURON writes between processes sharing the same NIC, so
  // intra-node sends silently fail at the CQ. The env var FI_EFA_ENABLE_SHM_TRANSFER=0
  // achieves the same thing at process scope, but is fragile (depends on the
  // launcher propagating it). Setting it explicitly on the endpoint is the
  // canonical fix (matches the Kaena PJRT reference).
  {
    bool shm_permitted = false;
    int sret = fi_setopt(&endpoint_->fid, FI_OPT_ENDPOINT, FI_OPT_SHARED_MEMORY_PERMITTED,
                         &shm_permitted, sizeof(shm_permitted));
    if (sret && sret != -FI_ENOSYS && sret != -FI_ENOPROTOOPT) {
      TORCH_NEURONX_RDMA_WARN("fi_setopt FI_OPT_SHARED_MEMORY_PERMITTED failed: ",
                              LibfabricSymbols::Instance().fi_strerror(-sret),
                              " (continuing — SHM may be active)");
    } else {
      TORCH_NEURONX_RDMA_INFO("Disabled FI_OPT_SHARED_MEMORY_PERMITTED on endpoint");
    }
  }

  ret = fi_enable(endpoint_.get());
  if (ret)
    return RdmaStatus::Error(std::string("fi_enable failed: ") +
                             LibfabricSymbols::Instance().fi_strerror(-ret));

  char ep_addr[256];
  size_t ep_addr_len = sizeof(ep_addr);
  ret = fi_getname(&endpoint_->fid, ep_addr, &ep_addr_len);
  if (ret)
    return RdmaStatus::Error(std::string("fi_getname failed: ") +
                             LibfabricSymbols::Instance().fi_strerror(-ret));
  local_endpoint_addr_ = std::string(ep_addr, ep_addr_len);

  TORCH_NEURONX_RDMA_INFO("Local endpoint created on EFA ", efa_bdf_);
  return RdmaStatus::Ok();
}

// ---------------------------------------------------------------------------
// Memory registration
// ---------------------------------------------------------------------------
RdmaStatus RdmaManager::EnsureRegistered(const HBMInfo& info) {
  if (!domain_) return RdmaStatus::Error("Domain not initialized");

  uintptr_t base_addr = reinterpret_cast<uintptr_t>(info.base);
  for (const auto& r : registered_memory_)
    if (reinterpret_cast<uintptr_t>(r.base_address) == base_addr) return RdmaStatus::Ok();

  fi_mr_attr mr_attr = {};
  mr_attr.access = FI_READ | FI_REMOTE_WRITE | FI_REMOTE_READ;
  mr_attr.iface = FI_HMEM_NEURON;
  mr_attr.device.neuron = info.device_id;
  struct iovec iov = {info.base, info.size};
  mr_attr.mr_iov = &iov;
  mr_attr.iov_count = 1;

  fid_mr* mr_raw = nullptr;
  int ret = fi_mr_regattr(domain_.get(), &mr_attr, 0, &mr_raw);
  if (ret)
    return RdmaStatus::Error(std::string("fi_mr_regattr failed: ") +
                             LibfabricSymbols::Instance().fi_strerror(-ret));

  registered_memory_.push_back({info.device_id, info.base, info.size, FidPtr<fid_mr>(mr_raw)});
  TORCH_NEURONX_RDMA_INFO("Registered MR: base=", info.base, ", size=", info.size / (1024 * 1024),
                          "MB");
  return RdmaStatus::Ok();
}

RdmaStatus RdmaManager::EnsureBufferRegistered(void* addr, size_t size, int device_id) {
  if (!domain_) return RdmaStatus::Error("Domain not initialized");

  uintptr_t a = reinterpret_cast<uintptr_t>(addr);
  for (const auto& r : registered_memory_) {
    uintptr_t start = reinterpret_cast<uintptr_t>(r.base_address);
    if (a >= start && a + size <= start + r.size) return RdmaStatus::Ok();
  }

  fi_mr_attr mr_attr = {};
  mr_attr.access = FI_READ | FI_REMOTE_WRITE | FI_REMOTE_READ;
  mr_attr.iface = FI_HMEM_NEURON;
  mr_attr.device.neuron = device_id;
  struct iovec iov = {addr, size};
  mr_attr.mr_iov = &iov;
  mr_attr.iov_count = 1;

  fid_mr* mr_raw = nullptr;
  int ret = fi_mr_regattr(domain_.get(), &mr_attr, 0, &mr_raw);
  if (ret)
    return RdmaStatus::Error(std::string("fi_mr_regattr failed: ") +
                             LibfabricSymbols::Instance().fi_strerror(-ret));

  registered_memory_.push_back({device_id, addr, size, FidPtr<fid_mr>(mr_raw)});
  return RdmaStatus::Ok();
}

fid_mr* RdmaManager::FindMR(void* addr, size_t size) {
  uintptr_t a = reinterpret_cast<uintptr_t>(addr);
  for (const auto& r : registered_memory_) {
    uintptr_t start = reinterpret_cast<uintptr_t>(r.base_address);
    if (a >= start && a + size <= start + r.size) return r.mr.get();
  }
  return nullptr;
}

uint32_t RdmaManager::RingDepth(uint32_t send_count, uint32_t recv_count) {
  return std::max(kMailboxSlots, std::max(send_count, recv_count));
}

RdmaManager::TagChannel* RdmaManager::TryEnsureLocalChannel(int peer, int tag,
                                                            uint32_t needed_depth) {
  const uint64_t key = TagChannelKey(peer, tag);
  auto it = tag_channels_.find(key);
  if (it != tag_channels_.end() && it->second.ring_depth >= needed_depth) {
    return &it->second;  // existing channel already deep enough
  }

  // Bilateral AV: EFA RDM requires the peer inserted before we publish/write.
  // Non-blocking — if the peer hasn't published its endpoint yet, defer (return
  // nullptr) and let the progress loop retry, rather than block the whole batch.
  fi_addr_t peer_fi;
  if (!TryGetOrInsertPeer(peer, &peer_fi)) return nullptr;

  const int rank = rank_.load();
  const uint32_t depth = std::max(needed_depth, kMailboxSlots);

  // Grow path: an earlier, shallower channel exists. Retire its buffer+MR (a
  // prior mailbox write may still reference it) rather than free under a possibly
  // in-flight write, then build a fresh, deeper one. Safe because each coalesced
  // batch fully drains before the next PrepareBatch runs (SubmitP2POp drains to
  // completion), so no live transfer targets this channel when we resize.
  if (it != tag_channels_.end()) {
    TORCH_NEURONX_RDMA_INFO("TryEnsureLocalChannel: GROW peer=", peer, " tag=", tag, " depth ",
                            it->second.ring_depth, " -> ", depth);
    retired_channels_.push_back(std::move(it->second));
    tag_channels_.erase(it);
  }

  TagChannel ch;
  ch.ring_depth = depth;
  const size_t inbox_bytes = static_cast<size_t>(depth) * kMailboxSlotBytes;
  ch.buf.assign(2 * inbox_bytes, 0);  // zeroed: a reader never sees a stale seq

  fi_mr_attr mr_attr = {};
  mr_attr.access = FI_REMOTE_WRITE | FI_WRITE | FI_READ | FI_REMOTE_READ;
  mr_attr.iface = FI_HMEM_SYSTEM;
  struct iovec iov = {ch.buf.data(), ch.buf.size()};
  mr_attr.mr_iov = &iov;
  mr_attr.iov_count = 1;
  fid_mr* mr_raw = nullptr;
  int ret = fi_mr_regattr(domain_.get(), &mr_attr, 0, &mr_raw);
  TORCH_CHECK(ret == 0, "RdmaManager: tag-channel fi_mr_regattr failed (peer=", peer, " tag=", tag,
              "): ", LibfabricSymbols::Instance().fi_strerror(-ret));
  ch.mr.reset(mr_raw);
  ch.inbox_rkey = fi_mr_key(mr_raw);
  ch.inbox_base_token = next_imm_token_.fetch_add(depth);
  TORCH_CHECK(ch.inbox_base_token + depth <= kMailboxImmFlag,
              "RdmaManager: imm token space exhausted at ", ch.inbox_base_token);

  // Publish OUR inbox coords under (owner=rank, writer=peer, tag): value =
  // 8B addr || 8B rkey || 4B base token || 4B ring depth, little-endian.
  const std::string pub_key =
      "rdma_slot_" + std::to_string(rank) + "_" + std::to_string(peer) + "_" + std::to_string(tag);
  const uint64_t inbox_addr = reinterpret_cast<uint64_t>(ch.buf.data());
  std::vector<uint8_t> val(24);
  std::memcpy(val.data(), &inbox_addr, 8);
  std::memcpy(val.data() + 8, &ch.inbox_rkey, 8);
  std::memcpy(val.data() + 16, &ch.inbox_base_token, 4);
  std::memcpy(val.data() + 20, &depth, 4);
  store_->set(pub_key, val);
  TORCH_NEURONX_RDMA_INFO("TryEnsureLocalChannel: peer=", peer, " tag=", tag, " depth=", depth,
                          " inbox=0x", std::hex, inbox_addr, " rkey=0x", ch.inbox_rkey, std::dec,
                          " token=", ch.inbox_base_token, " (published ", pub_key, ")");
  auto res = tag_channels_.emplace(key, std::move(ch));
  return &res.first->second;
}

bool RdmaManager::TryResolveRemoteChannel(TagChannel& ch, int peer, int tag,
                                          uint32_t needed_depth) {
  if (ch.remote_ready && ch.remote_ring_depth >= needed_depth) return true;
  const int rank = rank_.load();
  // Peer's inbox under (owner=peer, writer=rank, tag).
  const std::string fetch_key =
      "rdma_slot_" + std::to_string(peer) + "_" + std::to_string(rank) + "_" + std::to_string(tag);
  // Non-blocking: check() never blocks server-side. defer and re-check next iteration.
  if (!store_->check({fetch_key})) return false;
  auto bytes = store_->get(fetch_key);
  TORCH_CHECK(bytes.size() == 24, "RdmaManager: bad tag-channel value (peer=", peer, " tag=", tag,
              "): ", bytes.size(), " bytes, expected 24");
  std::memcpy(&ch.remote_inbox_addr, bytes.data(), 8);
  std::memcpy(&ch.remote_rkey, bytes.data() + 8, 8);
  std::memcpy(&ch.remote_base_token, bytes.data() + 16, 4);
  std::memcpy(&ch.remote_ring_depth, bytes.data() + 20, 4);
  // The peer may not have grown to needed_depth yet (its matching batch grows its
  // inbox and republishes; symmetric depth guarantees it will reach needed_depth).
  // Treat a still-shallow publication as not-ready and re-check next iteration.
  if (ch.remote_ring_depth < needed_depth) {
    ch.remote_ready = false;
    return false;
  }
  ch.remote_ready = true;
  TORCH_NEURONX_RDMA_INFO("TryResolveRemoteChannel: peer=", peer, " tag=", tag,
                          " rdepth=", ch.remote_ring_depth, " remote=0x", std::hex,
                          ch.remote_inbox_addr, " rrkey=0x", ch.remote_rkey, std::dec,
                          " rtoken=", ch.remote_base_token, " (fetched ", fetch_key, ")");
  return true;
}

void RdmaManager::SetRank(int rank) { rank_.store(rank); }

void RdmaManager::SetStore(c10::intrusive_ptr<c10d::Store> store) {
  std::lock_guard<std::mutex> lock(setup_mutex_);
  store_ = std::move(store);
}

// ---------------------------------------------------------------------------
// Lazy fabric init (engine thread)
// ---------------------------------------------------------------------------
//
// Called from PrepareBatch on the first batch. nrt_va must be a real HBM VA —
// nrt_get_attached_efa_bdf can resolve placement only for NRT-managed VAs, not
// the mmap placeholders the lazy allocator returns from data_ptr() before
// kAlloc binds.
void RdmaManager::EnsureInitialized(void* nrt_va) {
  if (initialized_) return;
  TORCH_CHECK(LibfabricSymbols::Instance().loaded,
              "RdmaManager: libfabric.so.1 not available. "
              "Install aws-efa-installer for RDMA/P2P support.");
  TORCH_NEURONX_RDMA_INFO("EnsureInitialized: starting (nrt_va=", nrt_va, ")");
  {
    std::lock_guard<std::mutex> lock(setup_mutex_);
    TORCH_CHECK(store_, "RdmaManager: SetStore must be called before first PrepareBatch");
  }
  const int rank = rank_.load();
  TORCH_CHECK(rank >= 0, "RdmaManager: SetRank must be called before first PrepareBatch");
  TORCH_NEURONX_RDMA_INFO("EnsureInitialized: rank=", rank);

  DiscoverHBMRegions();

  auto bdf_status = NrtGetAttachedEfaBdf(nrt_va, &efa_bdf_);
  if (!bdf_status.ok()) {
    TORCH_NEURONX_RDMA_ERROR("EnsureInitialized: nrt_get_attached_efa_bdf(nrt_va=", nrt_va,
                             ") failed: ", bdf_status.error,
                             "; this VA is not NRT-resolvable. hbm_infos_ has ", hbm_infos_.size(),
                             " entries.");
  }
  TORCH_CHECK(bdf_status.ok(), "RdmaManager: nrt_get_attached_efa_bdf failed: ", bdf_status.error);
  TORCH_NEURONX_RDMA_INFO("EnsureInitialized: nrt_va=", nrt_va, " -> efa_bdf=", efa_bdf_);

  auto status = DiscoverEFAInfo();
  TORCH_CHECK(status.ok(), "RdmaManager: DiscoverEFAInfo failed: ", status.error);
  status = InitFabric();
  TORCH_CHECK(status.ok(), "RdmaManager: InitFabric failed: ", status.error);

  // Register every HBM region on this process's EFA. Buffers carved out of
  // them are MR-cache hits in Send/Recv.
  size_t registered = 0;
  for (const auto& h : hbm_infos_) {
    if (h.efa_bdf != efa_bdf_) {
      TORCH_NEURONX_RDMA_DEBUG("EnsureInitialized: skipping HBM region device_id=", h.device_id,
                               " bdf=", h.efa_bdf, " (does not match our efa_bdf=", efa_bdf_, ")");
      continue;
    }
    auto reg = EnsureRegistered(h);
    TORCH_CHECK(reg.ok(), "RdmaManager: EnsureRegistered failed: ", reg.error);
    registered++;
  }
  TORCH_NEURONX_RDMA_INFO("EnsureInitialized: registered ", registered, " HBM regions on EFA ",
                          efa_bdf_);

  // Publish our endpoint so peers can find us, plus a sibling rdma_ep_bdf_<rank>
  // key carrying the EFA BDF so TryGetOrInsertPeer can log same- vs cross-NIC peers.
  const std::string ep_key = "rdma_ep_" + std::to_string(rank);
  store_->set(ep_key,
              std::vector<uint8_t>(local_endpoint_addr_.begin(), local_endpoint_addr_.end()));
  const std::string bdf_key = "rdma_ep_bdf_" + std::to_string(rank);
  store_->set(bdf_key, std::vector<uint8_t>(efa_bdf_.begin(), efa_bdf_.end()));
  TORCH_NEURONX_RDMA_INFO("EnsureInitialized: published ", ep_key, " (",
                          local_endpoint_addr_.size(), " bytes), ", bdf_key, "=", efa_bdf_);

  // TryGetOrInsertPeer runs on the first transfer that touches a peer and fetches
  // that peer's one-time rdma_ep_ key (non-blocking via check()); each (peer, tag)
  // channel is likewise set up on first use via the rdma_slot_ rendezvous.
  initialized_.store(true);
  TORCH_NEURONX_RDMA_INFO("EnsureInitialized: done; rank=", rank, " efa_bdf=", efa_bdf_);
}

bool RdmaManager::TryGetOrInsertPeer(int peer, fi_addr_t* out) {
  auto it = av_cache_.find(peer);
  if (it != av_cache_.end()) {
    TORCH_NEURONX_RDMA_DEBUG("TryGetOrInsertPeer: cache hit peer=", peer, " fi_addr=", it->second);
    *out = it->second;
    return true;
  }
  const std::string ep_key = "rdma_ep_" + std::to_string(peer);
  const std::string bdf_key = "rdma_ep_bdf_" + std::to_string(peer);
  // Non-blocking: only proceed once the peer has published BOTH keys
  if (!store_->check({ep_key, bdf_key})) {
    TORCH_NEURONX_RDMA_DEBUG("TryGetOrInsertPeer: peer=", peer, " endpoint not yet published");
    return false;
  }
  TORCH_NEURONX_RDMA_INFO("TryGetOrInsertPeer: cache miss peer=", peer, ", fetching ", ep_key);
  auto bytes = store_->get(ep_key);
  fi_addr_t addr;
  // Synchronous AV (FI_AV_MAP, no FI_EVENT): fi_av_insert returns the count of
  // addresses inserted, not a negative errno. On failure it is < 1 and addr is
  // set to FI_ADDR_NOTAVAIL — do not run it through fi_strerror.
  int ret = fi_av_insert(av_.get(), bytes.data(), 1, &addr, 0, nullptr);
  TORCH_CHECK(ret == 1, "fi_av_insert failed for peer ", peer, ": inserted ", ret,
              " of 1 addresses (", bytes.size(), "-byte endpoint)");
  av_cache_[peer] = addr;
  auto bdf_bytes = store_->get(bdf_key);
  std::string peer_bdf(bdf_bytes.begin(), bdf_bytes.end());
  TORCH_NEURONX_RDMA_INFO("TryGetOrInsertPeer: inserted peer=", peer, " fi_addr=", addr,
                          " peer_bdf=", peer_bdf, " self_bdf=", efa_bdf_,
                          peer_bdf != efa_bdf_ ? " (REMOTE NIC)" : " (same NIC)");
  *out = addr;
  return true;
}

uint32_t RdmaManager::NextSeq(int src, int dst, int tag) {
  // Pack (src, dst, tag) into 20 bits each. A tag (or rank) >= 2^20 would
  // bleed into the adjacent field and silently alias seq counters across
  // distinct tuples, matching the wrong descriptor. PP schedules use small
  // tags, but dist.send/recv tags are user-controlled, so guard explicitly.
  constexpr int kMaxField = 1 << 20;
  TORCH_CHECK(
      src >= 0 && src < kMaxField && dst >= 0 && dst < kMaxField && tag >= 0 && tag < kMaxField,
      "RdmaManager::NextSeq: (src=", src, ", dst=", dst, ", tag=", tag,
      ") exceeds the 20-bit field width; tags and ranks must be < ", kMaxField);
  uint64_t key = (uint64_t(src) << 40) | (uint64_t(dst) << 20) | uint64_t(tag);
  return seq_counters_[key]++;
}

// ---------------------------------------------------------------------------
// Engine-thread helpers
// ---------------------------------------------------------------------------
namespace {

// Resolve a user-visible VA to the HBM-backed NRT VA via NeuronCachingAllocator.
// findTensor() returns the bound nrt_tensor_t* (block->nrt_slice in the lazy
// path). nrt_tensor_get_va gives the actual HBM VA registered with libfabric.
//
// findTensor's "main thread only" comment refers to concurrent allocator
// mutations from kAlloc/kDealloc — both also run on the engine thread (single
// RuntimeWorker), so engine-thread reads are safe by sequential ordering.
void* ResolveHbmVA(void* user_va) {
  auto* tensor = c10_neuron::NeuronCachingAllocator::findTensor(user_va);
  if (!tensor) {
    TORCH_NEURONX_RDMA_ERROR("ResolveHbmVA: findTensor returned null for user_va=", user_va,
                             " (kAlloc may not have run; check stream ordering)");
    TORCH_CHECK(false, "RdmaManager: tensor not found for VA ", user_va);
  }
  void* hbm_va = nrt_tensor_get_va(tensor);
  TORCH_NEURONX_RDMA_DEBUG("ResolveHbmVA: user_va=", user_va, " -> nrt_tensor=", tensor,
                           " -> hbm_va=", hbm_va, " (same=", (user_va == hbm_va), ")");
  return hbm_va;
}

}  // namespace

// ---------------------------------------------------------------------------
// PrepareBatch
// ---------------------------------------------------------------------------
//
// The descriptor handshake has three conceptual phases:
//   1. every recv RMA-writes its descriptor into the SENDER's mailbox inbox;
//   2. every send reads the descriptor the receiver wrote into OUR inbox;
//   3. every send posts fi_writedata for the data, using that descriptor.
// They run in one loop with no barrier between them: each iteration advances the
// ops that are ready (peer endpoint published, channel coords resolved, descriptor
// arrived on the CQ) and skips the rest. No op blocks another — every store touch
// is a non-blocking check() and every descriptor read is gated on a CQ arrival
//
// seq stamps for all recvs then all sends are assigned up front, keeping NextSeq in
// lockstep with the peer regardless of the arrival order the loop observes.
//
// Zero-size (empty-tensor) entries are handled locally: they move no data, so they
// touch neither the fabric nor a mailbox slot — they only bump NextSeq to stay in
// lockstep with the peer's local-only pair. An all-zero-size batch thus never
// reaches an uninitialized endpoint (it has no HBM VA to init the fabric with).
//
// Same-rank pairs (peer == self) are filtered before reaching the engine —
// NeuronBackend::endCoalescing pairs them and resolves via at::Tensor::copy_.
// Non-coalesced same-rank send/recv is rejected at the user-thread entry.
//
// The per-(peer,tag) ring depth is max(kMailboxSlots, #sends, #recvs) on the
// channel, so a batch with N same-(peer,tag) transfers gets N distinct slots. The
// depth is identical on both ends (max is symmetric over the mirrored send/recv
// counts), so it needs no negotiation; a deeper batch grows the channel in place.
void RdmaManager::PrepareBatch(std::vector<P2PEntry>& entries, int64_t timeout_ms) {
  if (entries.empty()) return;
  if (entries.front().prepared) return;  // already prepared

  // Engine thread is the only reader after SetRank published it; load once.
  const int rank = rank_.load();
  TORCH_NEURONX_RDMA_INFO("PrepareBatch: ", entries.size(), " entries, rank=", rank);
  for (size_t i = 0; i < entries.size(); ++i) {
    const auto& e = entries[i];
    TORCH_NEURONX_RDMA_DEBUG(
        "PrepareBatch:   entry[", i, "] kind=", (e.kind == P2PEntry::Kind::kSend ? "SEND" : "RECV"),
        " user_va=", e.user_va, " size=", e.size, " peer=", e.peer, " tag=", e.tag);
  }

  // Resolve VAs first; the first one drives lazy fabric init.
  void* first_real_va = nullptr;
  for (auto& e : entries) {
    if (e.size == 0) {
      e.nrt_va = nullptr;
      continue;
    }
    e.nrt_va = ResolveHbmVA(e.user_va);
    if (first_real_va == nullptr) first_real_va = e.nrt_va;
  }
  // Lazy fabric init needs a real HBM VA. Only call when this batch has a non-zero-size entry.
  // An all-zero-size batch has no fabric work to do and no resolvable VA, so calling ResolveHbmVA
  // on its null user_va would fault.
  if (first_real_va != nullptr) {
    EnsureInitialized(first_real_va);
  }

  // Ring depth per (peer, tag) this batch = max(kMailboxSlots, #sends, #recvs) on
  // the channel. Both ranks compute the IDENTICAL depth from their own local batch:
  // my {send,recv} counts on (peer,tag) are the peer's {recv,send} counts on
  // (self,tag), and max is order-independent
  std::unordered_map<uint64_t, uint32_t> send_ct, recv_ct;
  for (const auto& e : entries) {
    if (e.size == 0) continue;
    const uint64_t k = TagChannelKey(e.peer, e.tag);
    if (e.kind == P2PEntry::Kind::kSend)
      ++send_ct[k];
    else
      ++recv_ct[k];
  }
  auto channel_depth = [&](int peer, int tag) -> uint32_t {
    const uint64_t k = TagChannelKey(peer, tag);
    const auto si = send_ct.find(k);
    const auto ri = recv_ct.find(k);
    return RingDepth(si == send_ct.end() ? 0 : si->second, ri == recv_ct.end() ? 0 : ri->second);
  };

  // Assign seq to every recv (entry order) then every send (entry order) up front,
  // so NextSeq advances in lockstep with the peer regardless of the arrival timing
  // the progress loop observes. Zero-size entries bump the counter and complete
  // locally to stay in lockstep with the peer's local-only pair.
  struct RecvWork {
    size_t idx = 0;
    uint32_t seq = 0;
    bool posted = false;
  };
  struct SendWork {
    size_t idx = 0;
    uint32_t seq = 0;
    bool have_desc = false;  // descriptor read from our inbox yet?
    bool posted = false;     // data write posted yet?
    uint32_t transfer_id = 0;
    uint64_t remote_addr = 0;
    uint64_t rkey = 0;
    int32_t peer_rank = 0;
  };
  std::vector<RecvWork> recvs;
  std::vector<SendWork> sends;
  for (size_t i = 0; i < entries.size(); ++i) {
    auto& e = entries[i];
    if (e.kind != P2PEntry::Kind::kRecv) continue;
    const uint32_t seq = NextSeq(e.peer, rank, e.tag);  // also validates 20-bit tag packing
    if (e.size == 0) {
      e.transfer_id = 0;
      e.recv_consumed = true;
      continue;
    }
    RecvWork rw;
    rw.idx = i;
    rw.seq = seq;
    recvs.push_back(rw);
  }
  for (size_t i = 0; i < entries.size(); ++i) {
    auto& e = entries[i];
    if (e.kind != P2PEntry::Kind::kSend) continue;
    const uint32_t seq = NextSeq(rank, e.peer, e.tag);
    if (e.size == 0) continue;
    SendWork sw;
    sw.idx = i;
    sw.seq = seq;
    sends.push_back(sw);
  }

  size_t posted_recvs = 0, posted_sends = 0;
  std::optional<std::chrono::steady_clock::time_point> deadline;
  while (posted_recvs < recvs.size() || posted_sends < sends.size()) {
    DrainAndPending(entries);  // drains CQ -> mailbox_arrived_ + completions
    bool progress = false;

    // Phase 1: RMA-write each recv's descriptor into the SENDER's inbox slot, once
    // our channel is published and the peer's inbox coords are resolved.
    for (auto& rw : recvs) {
      if (rw.posted) continue;
      auto& e = entries[rw.idx];
      const uint32_t depth = channel_depth(e.peer, e.tag);
      TagChannel* ch = TryEnsureLocalChannel(e.peer, e.tag, depth);
      if (!ch) continue;  // peer endpoint not published yet (bilateral AV) — defer
      if (!TryResolveRemoteChannel(*ch, e.peer, e.tag, depth)) continue;  // peer coords not ready
      fi_addr_t peer_fi;
      if (!TryGetOrInsertPeer(e.peer, &peer_fi)) continue;  // cache hit after the above
      const uint32_t ring = rw.seq % ch->ring_depth;

      // Build the recv descriptor: register the buffer and advertise its (addr,
      // size, rkey) so the sender can RMA-write straight into it.
      auto reg = EnsureBufferRegistered(e.nrt_va, e.size);
      TORCH_CHECK(reg.ok(), "RdmaManager: register recv buffer failed: ", reg.error);
      fid_mr* mr = FindMR(e.nrt_va, e.size);
      TORCH_CHECK(mr, "RdmaManager: recv VA not in registered memory: ", e.nrt_va);
      uint32_t tid = next_transfer_id_.fetch_add(1);
      if (tid == 0) tid = next_transfer_id_.fetch_add(1);
      e.transfer_id = tid;
      neuron::rdma::TransferDescriptor desc;
      desc.set_rank(rank);
      desc.set_transfer_id(tid);
      auto* region = desc.add_regions();
      region->set_remote_addr(reinterpret_cast<uint64_t>(e.nrt_va));
      region->set_size(e.size);
      region->set_rkey(fi_mr_key(mr));

      const size_t desc_len = desc.ByteSizeLong();
      TORCH_CHECK(desc_len <= kMailboxPayloadBytes, "RdmaManager: serialized descriptor (",
                  desc_len, "B) exceeds mailbox payload capacity ", kMailboxPayloadBytes);

      // Stage the {seq, len, proto-bytes} frame in our OUTBOX half at this ring slot.
      const size_t inbox_bytes = static_cast<size_t>(ch->ring_depth) * kMailboxSlotBytes;
      auto* out =
          reinterpret_cast<MailboxSlot*>(ch->buf.data() + inbox_bytes + ring * kMailboxSlotBytes);
      // The whole slot ships in one fi_writedata. The sender does NOT read it until
      // it consumes the write's remote CQ completion (see the send path + DrainAndPending).
      out->payload_len = static_cast<uint32_t>(desc_len);
      TORCH_CHECK(desc.SerializeToArray(out->payload, kMailboxPayloadBytes),
                  "RdmaManager: descriptor SerializeToArray failed");
      out->seq = rw.seq + 1;

      // Write into the peer's inbox slot for this ring. imm = the token the peer
      // assigned to that physical slot, so the sender matches this exact arrival.
      uint64_t dst = ch->remote_inbox_addr + ring * kMailboxSlotBytes;
      uint32_t imm = kMailboxImmFlag | (ch->remote_base_token + ring);

      auto ctx = std::make_unique<SendContext>();
      ctx->addr = out;
      ctx->is_mailbox = true;
      SendContext* ctx_raw = ctx.get();
      TORCH_NEURONX_RDMA_DEBUG("PrepareBatch: recv mailbox write -> sender=", e.peer,
                               " tag=", e.tag, " seq=", rw.seq, " ring=", ring, " dst=0x", std::hex,
                               dst, " rkey=0x", ch->remote_rkey, std::dec, " imm=", imm,
                               " tid=", tid);
      int ret = fi_writedata(endpoint_.get(), out, kMailboxSlotBytes, fi_mr_desc(ch->mr.get()), imm,
                             peer_fi, dst, ch->remote_rkey, ctx_raw);
      TORCH_CHECK(ret == 0,
                  "mailbox fi_writedata failed: ", LibfabricSymbols::Instance().fi_strerror(-ret));
      send_contexts_.insert(std::move(ctx));
      rw.posted = true;
      ++posted_recvs;
      progress = true;
    }

    // Phase 2: read each send's descriptor from OUR inbox once its CQ arrival is
    // confirmed. Phase 3: post the data write using that descriptor. Both happen
    // here so a send flows straight from descriptor to data in one pass.
    for (auto& sw : sends) {
      if (sw.posted) continue;
      auto& e = entries[sw.idx];
      const uint32_t depth = channel_depth(e.peer, e.tag);
      TagChannel* ch = TryEnsureLocalChannel(e.peer, e.tag, depth);
      if (!ch) continue;  // cannot publish our inbox until the peer is AV-insertable
      const uint32_t ring = sw.seq % ch->ring_depth;

      // Phase 2. The receiver stamps the token WE assigned this inbox slot, so the
      // CQ arrival carrying that imm is our signal the descriptor has fully landed.
      if (!sw.have_desc) {
        const uint32_t expected_imm = kMailboxImmFlag | (ch->inbox_base_token + ring);
        auto ait = mailbox_arrived_.find(expected_imm);
        if (ait == mailbox_arrived_.end()) continue;  // descriptor not yet arrived — defer
        mailbox_arrived_.erase(ait);
        const MailboxSlot* s =
            reinterpret_cast<const MailboxSlot*>(ch->buf.data() + ring * kMailboxSlotBytes);
        TORCH_CHECK(s->seq == sw.seq + 1,  // generation guard: receiver stamped seq+1
                    "RdmaManager: mailbox generation mismatch (slot seq=", s->seq, ", expected ",
                    sw.seq + 1, ") for peer=", e.peer, " tag=", e.tag);
        neuron::rdma::TransferDescriptor desc;
        TORCH_CHECK(s->payload_len <= kMailboxPayloadBytes &&
                        desc.ParseFromArray(s->payload, static_cast<int>(s->payload_len)),
                    "RdmaManager: descriptor ParseFromArray failed (len=", s->payload_len,
                    ") — mailbox corruption or version mismatch");
        // A written descriptor always carries exactly one region (zero-size
        // transfers never touch the mailbox).
        TORCH_CHECK(desc.regions_size() == 1, "RdmaManager: expected exactly 1 memory region, got ",
                    desc.regions_size());
        sw.peer_rank = desc.rank();
        sw.transfer_id = desc.transfer_id();
        sw.remote_addr = desc.regions(0).remote_addr();
        sw.rkey = desc.regions(0).rkey();
        sw.have_desc = true;
        progress = true;
      }

      // Phase 3. endCoalescing handles same-rank pairs on the user thread, so every
      // peer here is remote.
      fi_addr_t peer_fi;
      if (!TryGetOrInsertPeer(e.peer, &peer_fi)) continue;  // cache hit in practice
      TORCH_CHECK(sw.peer_rank != rank,
                  "RdmaManager: same-rank P2P should be handled before reaching the engine "
                  "(NeuronBackend::endCoalescing). Got peer rank == self (",
                  rank, ") for entry user_va=", e.user_va);
      auto reg = EnsureBufferRegistered(e.nrt_va, e.size);
      TORCH_CHECK(reg.ok(), "RdmaManager: register send buffer failed: ", reg.error);
      fid_mr* mr = FindMR(e.nrt_va, e.size);
      TORCH_CHECK(mr, "RdmaManager: send VA not in registered memory: ", e.nrt_va);
      // Completion context keyed by the ctx pointer (not nrt_va): looped / interleaved
      // schedules can have multiple in-flight sends from one buffer.
      auto ctx = std::make_unique<SendContext>();
      ctx->addr = e.nrt_va;
      ctx->complete = false;
      ctx->error = false;
      SendContext* ctx_raw = ctx.get();
      TORCH_NEURONX_RDMA_DEBUG("PrepareBatch: send fi_writedata local nrt_va=", e.nrt_va,
                               " size=", e.size, " peer=", e.peer, " peer_fi_addr=", peer_fi,
                               " remote_addr=0x", std::hex, sw.remote_addr, std::dec, " rkey=0x",
                               std::hex, sw.rkey, std::dec, " imm(transfer_id)=", sw.transfer_id);
      int ret = fi_writedata(endpoint_.get(), e.nrt_va, e.size, fi_mr_desc(mr), sw.transfer_id,
                             peer_fi, sw.remote_addr, sw.rkey, ctx_raw);
      if (ret != 0) {
        TORCH_NEURONX_RDMA_ERROR(
            "PrepareBatch: fi_writedata failed: ", LibfabricSymbols::Instance().fi_strerror(-ret),
            " (nrt_va=", e.nrt_va, " peer=", e.peer, ")");
      }
      TORCH_CHECK(ret == 0,
                  "fi_writedata failed: ", LibfabricSymbols::Instance().fi_strerror(-ret));
      send_contexts_.insert(std::move(ctx));
      e.send_ctx = ctx_raw;
      sw.posted = true;
      ++posted_sends;
      progress = true;
    }

    if (posted_recvs == recvs.size() && posted_sends == sends.size()) break;

    // Timeout is a dead-peer backstop only (default 0 = wait forever). Covers both
    // un-posted recvs (peer endpoint/channel never appeared) and un-posted sends
    // (descriptor never arrived).
    if (timeout_ms > 0) {
      if (!deadline) {
        deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
      } else if (std::chrono::steady_clock::now() > *deadline) {
        std::string stuck;
        for (const auto& rw : recvs) {
          if (rw.posted) continue;
          const auto& e = entries[rw.idx];
          if (!stuck.empty()) stuck += ", ";
          stuck += "recv<-" + std::to_string(e.peer) + " tag=" + std::to_string(e.tag);
        }
        for (const auto& sw : sends) {
          if (sw.posted) continue;
          const auto& e = entries[sw.idx];
          if (!stuck.empty()) stuck += ", ";
          stuck += "send->" + std::to_string(e.peer) + " tag=" + std::to_string(e.tag);
        }
        throw std::runtime_error("RDMA P2P prepare timeout after " + std::to_string(timeout_ms) +
                                 "ms; unfinished [" + stuck + "] (peer may have failed)");
      }
    }
    // Only yield when nothing advanced this pass — avoids a hot spin while still
    // polling the CQ every iteration (efa-direct has no wait-fd to block on).
    if (!progress) std::this_thread::yield();
  }

  for (auto& e : entries) e.prepared = true;
  TORCH_NEURONX_RDMA_INFO("PrepareBatch: prepared ", entries.size(), " entries (", posted_recvs,
                          " recvs posted, ", posted_sends, " sends posted)");
}

// ---------------------------------------------------------------------------
// DrainAndPending
// ---------------------------------------------------------------------------
bool RdmaManager::DrainAndPending(std::vector<P2PEntry>& entries) {
  // Runs on the HostWorkerPool worker thread (the kHost RunHost closure), same
  // thread as PrepareBatch (see the single-owner-thread invariant in the header
  // and the static_assert in NeuronBackend.cpp). No lock needed: send_contexts_
  // / completed_recv_ids_ are touched only by this one worker while
  // kHostWorkerThreads == 1.

  // Drain the CQ once, marking SendContexts complete and recording recv tids.
  if (cq_) {
    while (true) {
      fi_cq_data_entry cq_entries[kCQBatchSize];
      int ret = fi_cq_read(cq_.get(), cq_entries, kCQBatchSize);
      if (ret > 0) {
        for (int i = 0; i < ret; ++i) {
          if (cq_entries[i].flags & FI_WRITE) {
            auto* ctx = static_cast<SendContext*>(cq_entries[i].op_context);
            TORCH_NEURONX_RDMA_DEBUG("CQ: FI_WRITE complete ctx=", ctx,
                                     " addr=", (ctx ? ctx->addr : nullptr));
            if (ctx && ctx->is_mailbox) {
              auto cit = std::find_if(
                  send_contexts_.begin(), send_contexts_.end(),
                  [ctx](const std::unique_ptr<SendContext>& p) { return p.get() == ctx; });
              if (cit != send_contexts_.end()) send_contexts_.erase(cit);
            } else if (ctx) {
              ctx->complete = true;
            }
          } else if (cq_entries[i].flags & (FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA)) {
            uint32_t imm = static_cast<uint32_t>(cq_entries[i].data);
            if (imm & kMailboxImmFlag) {
              // A peer's descriptor write landed. Consuming this completion is the
              // delivery barrier: the slot bytes are now fully visible. Record the
              // arrival so PrepareBatch phase 2 knows it is safe to read that slot.
              mailbox_arrived_.insert(imm);
              TORCH_NEURONX_RDMA_DEBUG("CQ: mailbox descriptor arrived imm=0x", std::hex, imm,
                                       std::dec);
            } else {
              uint32_t tid = imm;
              TORCH_NEURONX_RDMA_DEBUG("CQ: FI_REMOTE_CQ_DATA arrived transfer_id=", tid);
              completed_recv_ids_.insert(tid);
            }
          } else {
            // Defensive: a CQ entry with neither FI_WRITE nor FI_REMOTE_WRITE
            // should not happen with the FI_CQ_FORMAT_DATA / efa-direct setup
            // we configure. Surface it loudly if it ever does.
            TORCH_NEURONX_RDMA_WARN("CQ unknown flags=0x", std::hex, cq_entries[i].flags, std::dec,
                                    " — dropping completion event");
          }
        }
      } else if (ret == -FI_EAGAIN) {
        break;
      } else {
        TORCH_NEURONX_RDMA_ERROR("fi_cq_read returned ", ret, " (",
                                 LibfabricSymbols::Instance().fi_strerror(-ret),
                                 ") — calling fi_cq_readerr for details");
        fi_cq_err_entry err = {};
        int err_ret = fi_cq_readerr(cq_.get(), &err, 0);
        if (err_ret > 0) {
          char prov_buf[256] = {};
          fi_cq_strerror(cq_.get(), err.prov_errno, err.err_data, prov_buf, sizeof(prov_buf));
          TORCH_NEURONX_RDMA_ERROR("CQ err_entry: err=", err.err, " (",
                                   LibfabricSymbols::Instance().fi_strerror(err.err),
                                   ") prov_errno=", err.prov_errno, " prov_err=\"", prov_buf,
                                   "\" flags=0x", std::hex, err.flags, std::dec, " len=", err.len,
                                   " data=", err.data, " op_context=", err.op_context);
          auto* ctx = static_cast<SendContext*>(err.op_context);
          if (ctx) {
            ctx->error = true;
            ctx->complete = true;
          } else {
            // Error with no recoverable per-op context (e.g. a remote-write /
            // recv failure, or CQ overrun): no SendContext to mark, so the
            // pending recvs in this batch can never complete. Fail hard rather
            // than busy-loop forever.
            cq_fatal_ = true;
          }
        } else {
          // fi_cq_readerr gave nothing (e.g. -FI_EOVERRUN with no entry): the
          // CQ is in an unrecoverable state; fail the batch instead of looping.
          TORCH_NEURONX_RDMA_ERROR("fi_cq_readerr returned ", err_ret,
                                   " — no error entry available; marking CQ fatal");
          cq_fatal_ = true;
        }
        break;
      }
    }
  }

  TORCH_CHECK(!cq_fatal_,
              "RdmaManager: fatal CQ error with no attributable operation; aborting batch "
              "(see preceding fi_cq_read/readerr log lines)");

  bool pending = false;
  for (auto& e : entries) {
    if (e.kind == P2PEntry::Kind::kSend) {
      SendContext* ctx = e.send_ctx;
      if (!ctx) continue;  // already consumed (cleared below)
      if (!ctx->complete) {
        pending = true;
        continue;
      }
      bool had_error = ctx->error;
      // Erase via O(n) find — std::unordered_set::erase keyed by the raw
      // pointer (rather than the unique_ptr identity) is the simplest path,
      // and n ≤ in-flight send count (~tens) so this is negligible.
      auto it =
          std::find_if(send_contexts_.begin(), send_contexts_.end(),
                       [ctx](const std::unique_ptr<SendContext>& p) { return p.get() == ctx; });
      if (it != send_contexts_.end()) send_contexts_.erase(it);
      e.send_ctx = nullptr;
      if (had_error) {
        throw std::runtime_error("RDMA send completed with CQ error for nrt_va=" +
                                 std::to_string(reinterpret_cast<uintptr_t>(e.nrt_va)));
      }
    } else {
      // Recv: a previous drain call may already have consumed this entry's
      // transfer_id from completed_recv_ids_. Mark the entry once and skip on
      // subsequent drains; otherwise erase() of the now-empty set returns 0
      // and the entry is reported pending forever.
      if (e.recv_consumed) continue;
      if (completed_recv_ids_.erase(e.transfer_id) == 0) {
        pending = true;
      } else {
        e.recv_consumed = true;
      }
    }
  }
  return pending;
}

}  // namespace neuron

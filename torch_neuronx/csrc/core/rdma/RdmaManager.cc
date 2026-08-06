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
  domain_.reset();
  fabric_.reset();
  if (info_) fi_freeinfo(info_);
  if (hints_) fi_freeinfo(hints_);
  info_ = nullptr;
  hints_ = nullptr;
  // Safe to clear: at process teardown the engine has already joined the host
  // worker (OperationExecutionEngine::Shutdown), so no PrepareBatch/
  // DrainAndPending can be touching these containers.
  send_contexts_.clear();
  completed_recv_ids_.clear();
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
  fi_info* hints = fi_allocinfo();
  if (!hints) return RdmaStatus::Error("fi_allocinfo failed");
  hints->fabric_attr->prov_name = strdup("efa");

  fi_info* info = nullptr;
  int ret = fi_getinfo(FI_VERSION(1, 9), nullptr, nullptr, 0, hints, &info);
  fi_freeinfo(hints);
  if (ret) return RdmaStatus::Error(std::string("fi_getinfo failed: ") + fi_strerror(-ret));

  unsigned int domain, bus, device, function;
  if (sscanf(efa_bdf_.c_str(), "%x:%x:%x.%x", &domain, &bus, &device, &function) != 4) {
    fi_freeinfo(info);
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
  fi_freeinfo(info);

  if (!found) return RdmaStatus::Error(std::string("No EFA device found for BDF: ") + efa_bdf_);
  TORCH_NEURONX_RDMA_INFO("Discovered EFA domain: ", efa_interface_, " for BDF: ", efa_bdf_);
  return RdmaStatus::Ok();
}

RdmaStatus RdmaManager::InitFabric() {
  hints_ = fi_allocinfo();
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

  int ret = fi_getinfo(FI_VERSION(1, 9), nullptr, nullptr, 0, hints_, &info_);
  if (ret) return RdmaStatus::Error(std::string("fi_getinfo failed: ") + fi_strerror(-ret));

  if (info_->domain_attr->cq_data_size < 4)
    return RdmaStatus::Error("EFA provider does not support 4-byte immediate data");

  fid_fabric* fabric_raw = nullptr;
  ret = fi_fabric(info_->fabric_attr, &fabric_raw, nullptr);
  if (ret) return RdmaStatus::Error(std::string("fi_fabric failed: ") + fi_strerror(-ret));
  fabric_.reset(fabric_raw);

  fid_domain* domain_raw = nullptr;
  ret = fi_domain(fabric_.get(), info_, &domain_raw, nullptr);
  if (ret) return RdmaStatus::Error(std::string("fi_domain failed: ") + fi_strerror(-ret));
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
  if (ret) return RdmaStatus::Error(std::string("fi_cq_open failed: ") + fi_strerror(-ret));
  cq_.reset(cq_raw);

  fi_av_attr av_attr = {};
  av_attr.type = FI_AV_MAP;
  fid_av* av_raw = nullptr;
  ret = fi_av_open(domain_.get(), &av_attr, &av_raw, nullptr);
  if (ret) return RdmaStatus::Error(std::string("fi_av_open failed: ") + fi_strerror(-ret));
  av_.reset(av_raw);

  fid_ep* ep_raw = nullptr;
  ret = fi_endpoint(domain_.get(), info_, &ep_raw, nullptr);
  if (ret) return RdmaStatus::Error(std::string("fi_endpoint failed: ") + fi_strerror(-ret));
  endpoint_.reset(ep_raw);

  ret = fi_ep_bind(endpoint_.get(), &cq_->fid, FI_TRANSMIT | FI_RECV);
  if (ret) return RdmaStatus::Error(std::string("fi_ep_bind(cq) failed: ") + fi_strerror(-ret));
  ret = fi_ep_bind(endpoint_.get(), &av_->fid, 0);
  if (ret) return RdmaStatus::Error(std::string("fi_ep_bind(av) failed: ") + fi_strerror(-ret));

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
                              fi_strerror(-sret), " (continuing — SHM may be active)");
    } else {
      TORCH_NEURONX_RDMA_INFO("Disabled FI_OPT_SHARED_MEMORY_PERMITTED on endpoint");
    }
  }

  ret = fi_enable(endpoint_.get());
  if (ret) return RdmaStatus::Error(std::string("fi_enable failed: ") + fi_strerror(-ret));

  char ep_addr[256];
  size_t ep_addr_len = sizeof(ep_addr);
  ret = fi_getname(&endpoint_->fid, ep_addr, &ep_addr_len);
  if (ret) return RdmaStatus::Error(std::string("fi_getname failed: ") + fi_strerror(-ret));
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
  if (ret) return RdmaStatus::Error(std::string("fi_mr_regattr failed: ") + fi_strerror(-ret));

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
  if (ret) return RdmaStatus::Error(std::string("fi_mr_regattr failed: ") + fi_strerror(-ret));

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
  // key carrying the EFA BDF so GetOrInsertPeer can log same- vs cross-NIC peers.
  const std::string ep_key = "rdma_ep_" + std::to_string(rank);
  store_->set(ep_key,
              std::vector<uint8_t>(local_endpoint_addr_.begin(), local_endpoint_addr_.end()));
  const std::string bdf_key = "rdma_ep_bdf_" + std::to_string(rank);
  store_->set(bdf_key, std::vector<uint8_t>(efa_bdf_.begin(), efa_bdf_.end()));
  TORCH_NEURONX_RDMA_INFO("EnsureInitialized: published ", ep_key, " (",
                          local_endpoint_addr_.size(), " bytes), ", bdf_key, "=", efa_bdf_);

  initialized_.store(true);
  TORCH_NEURONX_RDMA_INFO("EnsureInitialized: done; rank=", rank, " efa_bdf=", efa_bdf_);
}

fi_addr_t RdmaManager::GetOrInsertPeer(int peer) {
  auto it = av_cache_.find(peer);
  if (it != av_cache_.end()) {
    TORCH_NEURONX_RDMA_DEBUG("GetOrInsertPeer: cache hit peer=", peer, " fi_addr=", it->second);
    return it->second;
  }
  TORCH_NEURONX_RDMA_INFO("GetOrInsertPeer: cache miss peer=", peer, ", fetching rdma_ep_", peer,
                          " from store");
  auto bytes = store_->get("rdma_ep_" + std::to_string(peer));
  TORCH_NEURONX_RDMA_DEBUG("GetOrInsertPeer: got ", bytes.size(), " bytes for peer=", peer);
  fi_addr_t addr;
  // Synchronous AV (FI_AV_MAP, no FI_EVENT): fi_av_insert returns the count of
  // addresses inserted, not a negative errno. On failure it is < 1 and addr is
  // set to FI_ADDR_NOTAVAIL — do not run it through fi_strerror.
  int ret = fi_av_insert(av_.get(), bytes.data(), 1, &addr, 0, nullptr);
  TORCH_CHECK(ret == 1, "fi_av_insert failed for peer ", peer, ": inserted ", ret,
              " of 1 addresses (", bytes.size(), "-byte endpoint)");
  av_cache_[peer] = addr;
  // Log whether the peer shares our NIC. The bdf key is published alongside
  // rdma_ep_<peer> in EnsureInitialized, so it is already present here.
  auto bdf_bytes = store_->get("rdma_ep_bdf_" + std::to_string(peer));
  std::string peer_bdf(bdf_bytes.begin(), bdf_bytes.end());
  TORCH_NEURONX_RDMA_INFO("GetOrInsertPeer: inserted peer=", peer, " fi_addr=", addr,
                          " peer_bdf=", peer_bdf, " self_bdf=", efa_bdf_,
                          peer_bdf != efa_bdf_ ? " (REMOTE NIC)" : " (same NIC)");
  return addr;
}

int RdmaManager::NextSeq(int src, int dst, int tag) {
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

std::string RdmaManager::MakeStoreKey(int src, int dst, int tag, int seq) {
  return "rdma_" + std::to_string(src) + "_" + std::to_string(dst) + "_" + std::to_string(tag) +
         "_" + std::to_string(seq);
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
// Three phases per batch:
//   1. resolve VAs and build local recv descriptors;
//   2. publish all recv descriptors (multiSet) and fetch send descriptors (multiGet);
//   3. post fi_writedata for each send entry.
//
// Same-rank pairs (peer == self) are filtered before reaching the engine —
// NeuronBackend::endCoalescing pairs them and resolves via at::Tensor::copy_.
// Non-coalesced same-rank send/recv is rejected at the user-thread entry.
//
// Deadlock-freedom is structural, not barrier-based: within a single batch
// every recv descriptor is multiSet (phase 1) before any send descriptor is
// multiGet (phase 2). Since multiGet blocks on a missing key, a peer's needed
// recv descriptor is always published before that peer can block waiting on
// it. There is no inter-rank barrier between phases — a fast rank may reach
// phase 2 while its peer is still in phase 1, and that is safe.
void RdmaManager::PrepareBatch(std::vector<P2PEntry>& entries) {
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
  for (auto& e : entries) e.nrt_va = ResolveHbmVA(e.user_va);
  EnsureInitialized(entries.front().nrt_va);

  // Phase 1: build recv descriptors. Receiver also AV-inserts the sender's
  // endpoint — EFA RDM requires bilateral AV registration. Without this,
  // the sender's fi_writedata fails with prov_errno=14 REMOTE_ERROR_UNKNOWN_PEER
  // because the receiver's NIC has no address handle for the incoming write.
  std::vector<std::string> recv_keys;
  std::vector<std::vector<uint8_t>> recv_vals;
  for (auto& e : entries) {
    if (e.kind == P2PEntry::Kind::kRecv) {
      // Bilateral AV: receiver inserts sender's endpoint.
      GetOrInsertPeer(e.peer);

      auto reg = EnsureBufferRegistered(e.nrt_va, e.size);
      TORCH_CHECK(reg.ok(), "RdmaManager: register recv buffer failed: ", reg.error);
      fid_mr* mr = FindMR(e.nrt_va, e.size);
      TORCH_CHECK(mr, "RdmaManager: recv VA not in registered memory: ", e.nrt_va);

      uint32_t tid = next_transfer_id_.fetch_add(1);
      if (tid == 0) tid = next_transfer_id_.fetch_add(1);
      e.transfer_id = tid;

      neuron::rdma::TransferDescriptor desc;
      desc.set_receiver_endpoint(local_endpoint_addr_);
      desc.set_transfer_id(tid);
      if (rank >= 0) desc.set_rank(rank);
      auto* region = desc.add_regions();
      region->set_remote_addr(reinterpret_cast<uint64_t>(e.nrt_va));
      region->set_size(e.size);
      region->set_rkey(fi_mr_key(mr));

      std::string serialized;
      TORCH_CHECK(desc.SerializeToString(&serialized),
                  "RdmaManager: failed to serialize recv descriptor");
      int seq = NextSeq(e.peer, rank, e.tag);
      recv_keys.push_back(MakeStoreKey(e.peer, rank, e.tag, seq));
      recv_vals.push_back(std::vector<uint8_t>(serialized.begin(), serialized.end()));
    }
  }
  if (!recv_keys.empty()) {
    TORCH_NEURONX_RDMA_INFO("PrepareBatch: phase1 multiSet ", recv_keys.size(),
                            " recv descriptor keys");
    for (size_t i = 0; i < recv_keys.size(); ++i) {
      TORCH_NEURONX_RDMA_DEBUG("PrepareBatch:   set[", i, "] key=", recv_keys[i],
                               " val_bytes=", recv_vals[i].size());
    }
    store_->multiSet(recv_keys, recv_vals);
  }

  // Phase 2: fetch send descriptors.
  std::vector<std::string> send_keys;
  std::vector<size_t> send_indices;  // entries[] index for each send_keys[] entry
  for (size_t i = 0; i < entries.size(); ++i) {
    if (entries[i].kind != P2PEntry::Kind::kSend) continue;
    int seq = NextSeq(rank, entries[i].peer, entries[i].tag);
    send_keys.push_back(MakeStoreKey(rank, entries[i].peer, entries[i].tag, seq));
    send_indices.push_back(i);
  }
  std::vector<std::vector<uint8_t>> send_vals;
  if (!send_keys.empty()) {
    TORCH_NEURONX_RDMA_INFO("PrepareBatch: phase2 multiGet ", send_keys.size(),
                            " send descriptor keys (may block until peers publish)");
    for (size_t i = 0; i < send_keys.size(); ++i) {
      TORCH_NEURONX_RDMA_DEBUG("PrepareBatch:   get[", i, "] key=", send_keys[i]);
    }
    send_vals = store_->multiGet(send_keys);
    TORCH_CHECK(send_vals.size() == send_keys.size(), "RdmaManager: multiGet returned wrong size");
    TORCH_NEURONX_RDMA_INFO("PrepareBatch: phase2 multiGet complete (", send_vals.size(),
                            " values returned)");
  }

  // Phase 3: post sends. NeuronBackend::endCoalescing handles same-rank
  // (peer == self) pairs on the user thread via at::Tensor::copy_, so by the
  // time we reach the engine, every entry's peer is a remote rank.
  for (size_t k = 0; k < send_indices.size(); ++k) {
    auto& e = entries[send_indices[k]];
    std::string descriptor(send_vals[k].begin(), send_vals[k].end());
    neuron::rdma::TransferDescriptor desc;
    TORCH_CHECK(desc.ParseFromString(descriptor), "RdmaManager: failed to parse send descriptor");
    TORCH_CHECK(desc.regions_size() > 0, "RdmaManager: empty send descriptor");
    TORCH_CHECK(desc.rank() != rank,
                "RdmaManager: same-rank P2P should be handled before reaching the engine "
                "(NeuronBackend::endCoalescing). Got peer rank == self (",
                rank, ") for entry user_va=", e.user_va);

    fi_addr_t remote_addr = GetOrInsertPeer(e.peer);

    auto reg = EnsureBufferRegistered(e.nrt_va, e.size);
    TORCH_CHECK(reg.ok(), "RdmaManager: register send buffer failed: ", reg.error);
    fid_mr* mr = FindMR(e.nrt_va, e.size);
    TORCH_CHECK(mr, "RdmaManager: send VA not in registered memory: ", e.nrt_va);

    // Per-send completion context. Multiple in-flight sends may share the
    // same nrt_va (looped/interleaved schedules); we use the ctx pointer (the
    // op_context returned in CQ events) rather than nrt_va as the
    // identification key, so concurrent sends from the same buffer don't
    // collide.
    auto ctx = std::make_unique<SendContext>();
    ctx->addr = e.nrt_va;
    ctx->complete = false;
    ctx->error = false;
    SendContext* ctx_raw = ctx.get();

    const auto& region = desc.regions(0);
    TORCH_NEURONX_RDMA_DEBUG("PrepareBatch: phase3 fi_writedata local nrt_va=", e.nrt_va,
                             " size=", e.size, " peer=", e.peer, " peer_fi_addr=", remote_addr,
                             " remote_addr=0x", std::hex, region.remote_addr(), std::dec,
                             " rkey=0x", std::hex, region.rkey(), std::dec,
                             " imm(transfer_id)=", desc.transfer_id());
    int ret = fi_writedata(endpoint_.get(), e.nrt_va, e.size, fi_mr_desc(mr), desc.transfer_id(),
                           remote_addr, region.remote_addr(), region.rkey(), ctx_raw);
    if (ret != 0) {
      TORCH_NEURONX_RDMA_ERROR("PrepareBatch: fi_writedata failed: ", fi_strerror(-ret),
                               " (nrt_va=", e.nrt_va, " peer=", e.peer, ")");
    }
    TORCH_CHECK(ret == 0, "fi_writedata failed: ", fi_strerror(-ret));
    send_contexts_.insert(std::move(ctx));
    e.send_ctx = ctx_raw;
  }

  for (auto& e : entries) e.prepared = true;
  TORCH_NEURONX_RDMA_INFO("PrepareBatch: prepared ", entries.size(),
                          " entries (sends posted, recvs awaiting CQ)");
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
            if (ctx) ctx->complete = true;
          } else if (cq_entries[i].flags & (FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA)) {
            uint32_t tid = static_cast<uint32_t>(cq_entries[i].data);
            TORCH_NEURONX_RDMA_DEBUG("CQ: FI_REMOTE_CQ_DATA arrived transfer_id=", tid);
            completed_recv_ids_.insert(tid);
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
        TORCH_NEURONX_RDMA_ERROR("fi_cq_read returned ", ret, " (", fi_strerror(-ret),
                                 ") — calling fi_cq_readerr for details");
        fi_cq_err_entry err = {};
        int err_ret = fi_cq_readerr(cq_.get(), &err, 0);
        if (err_ret > 0) {
          char prov_buf[256] = {};
          fi_cq_strerror(cq_.get(), err.prov_errno, err.err_data, prov_buf, sizeof(prov_buf));
          TORCH_NEURONX_RDMA_ERROR("CQ err_entry: err=", err.err, " (", fi_strerror(err.err),
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

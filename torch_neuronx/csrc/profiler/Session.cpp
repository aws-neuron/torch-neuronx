#include "Session.h"

#include <ThreadUtil.h>
#include <libkineto.h>
#include <nrt/nrt_profile.h>
#include <output_base.h>

#include <algorithm>
#include <atomic>
#include <chrono>

#include "CorrelationTracker.h"
#include "LogWrapper.h"
#include "NConfigParser.h"

// Forward declaration for Neuron runtime lazy initialization
namespace torch_neuronx {
void maybe_lazy_init();
}

namespace at::neuron {

namespace {

using namespace torch_neuronx::profiler;

// Helper to cast context to NRT config
inline nrt_inspect_config_t* toNrtConfig(void* ctx) {
  return static_cast<nrt_inspect_config_t*>(ctx);
}

ConfigRegistry createNeuronConfigRegistry() {
  return {
      {"neuron_profile", NConfigParser::parseString,
       [](void* ctx, const ConfigValue& v) {
         const auto& modes = std::get<std::string>(v);
         auto parts = torch_neuronx::profiler::detail::splitAndTrim(modes, ',');
         for (const auto& mode : parts) {
           if (mode == "device") {
             NRT_STATUS s =
                 nrt_inspect_config_set_activity(toNrtConfig(ctx), "device_profile", true);
             if (s != NRT_SUCCESS) {
               NPROF_WARN("  Failed to enable device_profile (status:", s, ")");
             }
             s = nrt_inspect_config_set_inspect_device_profile_mode(
                 toNrtConfig(ctx), NRT_INSPECT_DEVICE_PROFILE_MODE_SESSION);
             if (s != NRT_SUCCESS) {
               NPROF_WARN("  Failed to set device profile mode (status:", s, ")");
             }
             NPROF_INFO("  Enabled NRT activity: device_profile (from neuron_profile config)");
           } else if (mode == "runtime") {
             NRT_STATUS s =
                 nrt_inspect_config_set_activity(toNrtConfig(ctx), "system_profile", true);
             if (s != NRT_SUCCESS) {
               NPROF_WARN("  Failed to enable system_profile (status:", s, ")");
             }
             NPROF_INFO("  Enabled NRT activity: system_profile (from neuron_profile config)");
           } else if (mode == "cpu_util") {
             NRT_STATUS s = nrt_inspect_config_set_activity(toNrtConfig(ctx), "cpu_util", true);
             if (s != NRT_SUCCESS) {
               NPROF_WARN("  Failed to enable cpu_util (status:", s, ")");
             }
             NPROF_INFO("  Enabled NRT activity: cpu_util (from neuron_profile config)");
           } else if (mode == "host_memory") {
             NRT_STATUS s = nrt_inspect_config_set_activity(toNrtConfig(ctx), "host_memory", true);
             if (s != NRT_SUCCESS) {
               NPROF_WARN("  Failed to enable host_memory (status:", s, ")");
             }
             NPROF_INFO("  Enabled NRT activity: host_memory (from neuron_profile config)");
           } else {
             NPROF_WARN("  Unknown neuron_profile mode:", mode);
           }
         }
         return true;
       }},

      {"max_events_per_nc", NConfigParser::parseUint64,
       [](void* ctx, const ConfigValue& v) {
         NRT_STATUS status = nrt_inspect_config_set_sys_trace_max_events_per_nc(
             toNrtConfig(ctx), std::get<uint64_t>(v));
         if (status != NRT_SUCCESS) {
           NPROF_WARN("  Failed to set max_events_per_nc (status:", status, ")");
           return false;
         }
         NPROF_INFO("  Set max_events_per_nc:", std::get<uint64_t>(v));
         return true;
       }},

      {"capture_enabled_for_nc", NConfigParser::parseNcIndexList,
       [](void* ctx, const ConfigValue& v) {
         bool allOk = true;
         for (uint32_t ncIdx : std::get<std::vector<uint32_t>>(v)) {
           NRT_STATUS status =
               nrt_inspect_config_set_capture_enabled_for_nc(toNrtConfig(ctx), ncIdx, true);
           if (status != NRT_SUCCESS) {
             NPROF_WARN("  Failed to enable capture for NC:", ncIdx);
             allOk = false;
           } else {
             NPROF_INFO("  Enabled capture for NeuronCore:", ncIdx);
           }
         }
         return allOk;
       }},

      {"host_memory", NConfigParser::parseBool,
       [](void* ctx, const ConfigValue& v) {
         bool enabled = std::get<bool>(v);
         NRT_STATUS status =
             nrt_inspect_config_set_activity(toNrtConfig(ctx), "host_memory", enabled);
         if (status != NRT_SUCCESS) {
           NPROF_WARN("  Failed to set host_memory (status:", status, ")");
           return false;
         }
         NPROF_INFO("  Set host_memory:", enabled ? "enabled" : "disabled");
         return true;
       }},

      {"profile_output_dir", NConfigParser::parseString,
       [](void* ctx, const ConfigValue& v) {
         NRT_STATUS status =
             nrt_inspect_config_set_output_dir(toNrtConfig(ctx), std::get<std::string>(v).c_str());
         if (status != NRT_SUCCESS) {
           NPROF_WARN("  Failed to set profile_output_dir (status:", status, ")");
           return false;
         }
         NPROF_INFO("  Set profile_output_dir:", std::get<std::string>(v));
         return true;
       }},

      {"cpu_util", NConfigParser::parseBool,
       [](void* ctx, const ConfigValue& v) {
         bool enabled = std::get<bool>(v);
         NRT_STATUS status = nrt_inspect_config_set_activity(toNrtConfig(ctx), "cpu_util", enabled);
         if (status != NRT_SUCCESS) {
           NPROF_WARN("  Failed to set cpu_util (status:", status, ")");
           return false;
         }
         NPROF_INFO("  Set cpu_util:", enabled ? "enabled" : "disabled");
         return true;
       }},

      {"neff_cache_dir", NConfigParser::parseString,
       [](void* ctx, const ConfigValue& v) {
         NRT_STATUS status = nrt_inspect_config_set_neff_cache_dir(
             toNrtConfig(ctx), std::get<std::string>(v).c_str());
         if (status != NRT_SUCCESS) {
           NPROF_WARN("  Failed to set neff_cache_dir (status:", status, ")");
           return false;
         }
         NPROF_INFO("  Set neff_cache_dir:", std::get<std::string>(v));
         return true;
       }},
  };
}

// Guard to prevent multiple concurrent profiler sessions across all instances.
static std::atomic<bool> s_sessionActive{false};

}  // namespace

NeuronActivityProfilerSession::NeuronActivityProfilerSession(
    const std::set<libkineto::ActivityType>& activityTypes, const libkineto::Config& config)
    : activityTypes_(activityTypes), config_(config.clone()) {
  NPROF_INFO("Profiler session created.");
  NPROF_INFO("  Log file:", config_->activitiesLogFile());
}

void NeuronActivityProfilerSession::initNrtConfig() {
  NRT_STATUS status = nrt_inspect_config_allocate(&nrtConfig_);

  if (status != NRT_SUCCESS) {
    NPROF_ERROR(errors_, "Failed to allocate NRT config:", std::to_string(status));
    nrtConfig_ = nullptr;
    return;
  }

  NRT_STATUS defaultsStatus = nrt_inspect_config_set_defaults(nrtConfig_);
  if (defaultsStatus != NRT_SUCCESS) {
    NPROF_ERROR(errors_, "nrt_inspect_config_set_defaults failed:", std::to_string(defaultsStatus));
    cleanupNrtConfig();
    return;
  }
  nrt_inspect_config_set_enable_inspect(nrtConfig_, true);
  nrt_inspect_config_set_activity(nrtConfig_, "all", false);  // baseline: all off

  // neuron_profile config key selectively enables requested modes
  applyCustomConfig();
}

void NeuronActivityProfilerSession::applyCustomConfig() {
  if (!config_ || !nrtConfig_) {
    return;
  }

  const std::string& customConfig = config_->getCustomConfig();
  if (customConfig.empty()) {
    NPROF_INFO("  No custom config provided");
    return;
  }

  NPROF_INFO("  Custom config:", customConfig);

  static const auto registry = createNeuronConfigRegistry();
  torch_neuronx::profiler::NConfigParser::parseAndApply(customConfig, registry, nrtConfig_);
}

NeuronActivityProfilerSession::~NeuronActivityProfilerSession() {
  NPROF_INFO("Session DESTROYED");
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (isActive_) {
      setActiveProfilerBuffer(nullptr);
      eventBuffer_.deactivate();
      nrt_inspect_stop();
      isActive_ = false;
      status_ = libkineto::TraceStatus::PROCESSING;
      s_sessionActive.store(false);
    }
  }
  cleanupNrtConfig();
}

void NeuronActivityProfilerSession::cleanupNrtConfig() {
  if (nrtConfig_) {
    nrt_inspect_config_free(nrtConfig_);
    nrtConfig_ = nullptr;
  }
}

void NeuronActivityProfilerSession::start() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (isActive_) {
    NPROF_INFO("Already started, ignoring duplicate start()");
    return;
  }

  if (s_sessionActive.exchange(true)) {
    NPROF_ERROR(errors_, "Another profiler session is already active");
    status_ = libkineto::TraceStatus::ERROR;
    return;
  }

  NPROF_INFO("Starting profiler session..");

  // Neuron runtime will be initialized at the first device access, say
  // but the profiler session can be started before that for example:
  // ```cpp
  //.     with pytorch.profiler.profile(activities=..) as prof:
  //        torch.randn(10, 5, device="neuron")
  //        ...
  // ```
  // But profiling does not work for neuron unless NRT is initialized.
  // So we initialize runtime if it is not initialized.
  torch_neuronx::maybe_lazy_init();

  if (!nrtConfig_) {
    initNrtConfig();
  }

  if (!nrtConfig_) {
    NPROF_ERROR(errors_, "Cannot start: NRT config not initialized");
    status_ = libkineto::TraceStatus::ERROR;
    s_sessionActive.store(false);
    return;
  }

  // Compute clock domain offset: system_clock - steady_clock.
  // Kineto passes capture window in system_clock; our timestamps use steady_clock.
  // Offset is computed once and applied to all steady_clock timestamps.
  auto steadyNow = toNs(std::chrono::steady_clock::now());
  auto sysNow = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count();
  clockDomainOffset_ = sysNow - steadyNow;
  profileStartTime_ = sysNow;  // Store in system_clock domain

  NRT_STATUS status = nrt_inspect_begin_with_options(nrtConfig_);

  if (status != NRT_SUCCESS) {
    NPROF_ERROR(errors_, "Failed to start NRT profiling:", std::to_string(status));
    status_ = libkineto::TraceStatus::ERROR;
    s_sessionActive.store(false);
    return;
  }

  eventBuffer_.activate();
  setActiveProfilerBuffer(&eventBuffer_);

  isActive_ = true;
  status_ = libkineto::TraceStatus::RECORDING;
  NPROF_INFO("Session started.");
}

void NeuronActivityProfilerSession::stop() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!isActive_) {
    NPROF_INFO("Session not active; nothing to stop");
    return;
  }

  setActiveProfilerBuffer(nullptr);
  eventBuffer_.deactivate();

  // Convert to system_clock domain to match profileStartTime_ and Kineto's capture window.
  profileEndTime_ = toNs(std::chrono::steady_clock::now()) + clockDomainOffset_;

  // Stop profiling first - this writes trace data to the output directory
  NRT_STATUS status = nrt_inspect_stop();
  if (status != NRT_SUCCESS) {
    NPROF_ERROR(errors_, "Failed to stop NRT profiling:", std::to_string(status));
  }

  isActive_ = false;
  status_ = libkineto::TraceStatus::PROCESSING;
  s_sessionActive.store(false);
  NPROF_INFO("NRT profiling STOPPED");
}

std::vector<std::string> NeuronActivityProfilerSession::errors() {
  std::lock_guard<std::mutex> lock(mutex_);
  return errors_;
}

void NeuronActivityProfilerSession::processTrace(libkineto::ActivityLogger& logger) {
  NPROF_INFO("processTrace (sync) called - using full capture window");
  processTrace(logger, nullptr, profileStartTime_, profileEndTime_);
}

// Converts profiled operations into Kineto GenericTraceActivity events and logs them.
//
// This is the async overload called by Kineto's GenericActivityProfiler after stop().
// The capture window timestamps are in system_clock (nanoseconds since Unix epoch),
// as set by Kineto via libkineto::timeSinceEpoch(system_clock::now()).
//
// Clock domain handling:
//   ProfiledOperation timestamps (submit_start_ns, submit_time_ns, execute_start_ns,
//   execute_end_ns) originate from steady_clock via OperationContext/CompleteOperation.
//   We convert them to system_clock domain here using clockDomainOffset_ (computed once
//   in start()) so they align with Kineto's capture window. Without this conversion,
//   all events would be rejected by the capture window filter because steady_clock
//   values (~10^14) are far smaller than system_clock values (~1.7×10^18).
//
// Activity placement:
//   Each ProfiledOperation produces two GenericTraceActivity events:
//   1. PRIVATEUSE1_RUNTIME — submission on PyTorch thread (submit_start → submit_time),
//      placed on the CPU process track under the PyTorch calling thread's OS tid.
//      This is the cudaLaunchKernel equivalent: the synchronous enqueue call.
//   2. PRIVATEUSE1_DRIVER — device execution (execute_start → execute_end), placed
//      on the NeuronDevice track under the logical stream.
//   A flow arrow (kLinkAsyncCpuGpu) connects RUNTIME → DRIVER via kineto_correlation_id.
void NeuronActivityProfilerSession::processTrace(
    libkineto::ActivityLogger& logger, libkineto::getLinkedActivityCallback getLinkedActivity,
    int64_t captureWindowStartTime, int64_t captureWindowEndTime) {
  NPROF_INFO("processTrace: capture window", captureWindowStartTime, "to", captureWindowEndTime);

  cpuActivity_ = std::move(getLinkedActivity);
  captureWindowStart_ = captureWindowStartTime;
  captureWindowEnd_ = captureWindowEndTime;

  // Heap-allocate trace buffer so the TraceSpan address is stable
  // (GenericTraceActivity stores a raw pointer to the span).
  traceBuffer_ = std::make_unique<libkineto::CpuTraceBuffer>();
  traceBuffer_->span =
      libkineto::TraceSpan(profileStartTime_, profileEndTime_, "__neuron_profiler__");

  // Drain profiled operations collected between start() and stop()
  auto profiled_ops = eventBuffer_.drain();
  NPROF_INFO("processTrace: drained", profiled_ops.size(), "profiled operations");

  // Convert steady_clock timestamps to system_clock domain for Kineto compatibility.
  // ProfiledOperation timestamps are recorded in steady_clock (nanoseconds since boot).
  // Kineto's capture window uses system_clock (nanoseconds since Unix epoch).
  // The offset bridges the two domains so the filter below works correctly.
  for (auto& op : profiled_ops) {
    op.submit_start_ns += clockDomainOffset_;
    op.submit_time_ns += clockDomainOffset_;
    op.execute_start_ns += clockDomainOffset_;
    op.execute_end_ns += clockDomainOffset_;
  }

  // Filter to capture window — skip operations entirely outside the window
  profiled_ops.erase(std::remove_if(profiled_ops.begin(), profiled_ops.end(),
                                    [this](const ProfiledOperation& op) {
                                      return op.execute_end_ns < captureWindowStart_ ||
                                             op.submit_start_ns > captureWindowEnd_;
                                    }),
                     profiled_ops.end());
  NPROF_INFO("processTrace:", profiled_ops.size(), "operations after capture window filter");

  // Capture primary device index from first operation
  if (!profiled_ops.empty()) {
    primaryDevice_ = profiled_ops.front().device_index;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Activity Linking Architecture
  // ═══════════════════════════════════════════════════════════════════════════
  //
  // Three execution contexts participate in a Neuron profiled operation:
  //
  //   1. PyTorch thread    — runs the framework op (e.g., aten::mm)
  //   2. Eager worker      — NeuronStream worker thread dispatches to NRT
  //   3. Neuron Device     — Trainium/Inferentia executes the compute
  //
  // Two linking mechanisms connect activities across these contexts:
  //
  //   linked_activity_ (parent-child data link, CPU → RUNTIME):
  //     RUNTIME.linked = cpuActivity_(kineto_correlation_id)
  //     Sets RUNTIME.parent → CPU op in collection.cpp:1156.
  //     Multiple NRT ops may share the same kineto_correlation_id when a
  //     single PyTorch op (e.g., aten::mm) decomposes into several NRT
  //     dispatches within the same RecordFunction scope.
  //
  //   Flow arrows (visual link, RUNTIME → DRIVER):
  //     RUNTIME.flow = {id=N, type=kLinkAsyncCpuGpu, start=1}
  //     DRIVER.flow  = {id=N, type=kLinkAsyncCpuGpu, start=0}
  //     Sets DRIVER.parent → RUNTIME in collection.cpp:1166-1169.
  //     Draws a visual arrow in chrome://tracing from the worker thread
  //     track to the device track.
  //
  // Combined parent chain: DRIVER → RUNTIME → CPU op (aten::mm)
  //
  // Design constraints:
  //   - GenericTraceActivity has exactly ONE flow struct per activity, so an
  //     activity can participate in only one flow connection (start or end).
  //   - Flow IDs must be unique across all flow starts (enforced by
  //     collection.cpp:1152). We use a monotonic counter (nextFlowId_)
  //     instead of kineto_correlation_id because multiple ops can share a
  //     correlation ID.
  //   - The RUNTIME → DRIVER flow follows the CUDA/CUPTI pattern where the
  //     visual arrow crosses from host runtime to device execution.
  // ═══════════════════════════════════════════════════════════════════════════

  // Activity placement — maps Neuron concepts to Chrome Trace Viewer rows:
  //
  // PRIVATEUSE1_RUNTIME = PyTorch thread submission (submit_start → submit_time):
  //   .device   = process PID              — CPU process row
  //   .resource = pytorch_os_tid            — OS thread of the PyTorch caller
  //   .threadId = pytorch_os_tid
  //   This places RUNTIME on the same track as the CPU op (e.g., aten::mm),
  //   so the flow arrow goes diagonally from PyTorch thread → device.
  //
  // PRIVATEUSE1_DRIVER = Neuron Device execution (execute_start → execute_end):
  //   .device   = device_index              — Neuron device (0, 1, ...)
  //   .resource = stream_id                 — logical NeuronStream (0, 1, 2)
  //
  // Flow arrow connects PyTorch thread → Neuron Device via unique monotonic flow ID.
  // pytorch_thread_id (PyTorch logical counter, NOT an OS tid) is emitted as metadata.
  // neuron_worker_tid preserves the worker thread OS tid for debugging.
  for (auto& op : profiled_ops) {
    // Activity 1: PRIVATEUSE1_RUNTIME — submission on PyTorch thread (submit_start → submit_time)
    traceBuffer_->emplace_activity(traceBuffer_->span, libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
                                   op.op_name);
    auto& dispatch_act = *traceBuffer_->activities.back();

    dispatch_act.startTime = op.submit_start_ns;
    dispatch_act.endTime = op.submit_time_ns;
    dispatch_act.id = static_cast<int32_t>(op.kineto_correlation_id);
    dispatch_act.device = libkineto::processId();  // CPU process PID
    dispatch_act.resource = op.pytorch_os_tid;     // PyTorch calling thread OS tid
    dispatch_act.threadId = op.pytorch_os_tid;

    // Assign unique flow ID for this RUNTIME→DRIVER pair.
    // Must be unique across all flow starts (enforced by collection.cpp:1152).
    uint32_t flowId = nextFlowId_++;

    // Flow arrow: RUNTIME → DRIVER (start)
    dispatch_act.flow.id = flowId;
    dispatch_act.flow.type = libkineto::kLinkAsyncCpuGpu;
    dispatch_act.flow.start = 1;

    // Link to the originating CPU activity via Kineto correlation
    if (op.kineto_correlation_id != 0 && cpuActivity_) {
      dispatch_act.linked = cpuActivity_(static_cast<int32_t>(op.kineto_correlation_id));
    }

    dispatch_act.addMetadata("nrt_sequence_id", op.nrt_sequence_id);
    dispatch_act.addMetadata("stream_id", op.stream_id);
    dispatch_act.addMetadata("pytorch_logical_tid", op.pytorch_thread_id);
    dispatch_act.addMetadata("neuron_worker_tid", op.neuron_stream_os_tid);
    if (!op.hierarchy_string.empty()) {
      dispatch_act.addMetadataQuoted("hierarchy", op.hierarchy_string);
    }
    if (!op.neff_cache_key.empty()) {
      dispatch_act.addMetadataQuoted("neff_cache_key", op.neff_cache_key);
    }
    if (!op.persistent_cache_key.empty()) {
      dispatch_act.addMetadataQuoted("persistent_cache_key", op.persistent_cache_key);
    }

    dispatch_act.log(logger);

    // Activity 2: PRIVATEUSE1_DRIVER — device execution window (execute_start → execute_end)
    traceBuffer_->emplace_activity(traceBuffer_->span, libkineto::ActivityType::PRIVATEUSE1_DRIVER,
                                   op.op_name);
    auto& exec_act = *traceBuffer_->activities.back();

    exec_act.startTime = op.execute_start_ns;
    exec_act.endTime = op.execute_end_ns;
    exec_act.id = static_cast<int32_t>(op.kineto_correlation_id);
    exec_act.device = static_cast<int32_t>(op.device_index);
    exec_act.resource = static_cast<int32_t>(op.stream_id);

    // Flow arrow endpoint: RUNTIME → DRIVER (end)
    exec_act.flow.id = flowId;
    exec_act.flow.type = libkineto::kLinkAsyncCpuGpu;
    exec_act.flow.start = 0;  // flow END (destination)

    // Link to the originating CPU activity
    if (op.kineto_correlation_id != 0 && cpuActivity_) {
      exec_act.linked = cpuActivity_(static_cast<int32_t>(op.kineto_correlation_id));
    }

    exec_act.addMetadata("nrt_sequence_id", op.nrt_sequence_id);
    if (!op.hierarchy_string.empty()) {
      exec_act.addMetadataQuoted("hierarchy", op.hierarchy_string);
    }
    if (!op.neff_cache_key.empty()) {
      exec_act.addMetadataQuoted("neff_cache_key", op.neff_cache_key);
    }
    if (!op.persistent_cache_key.empty()) {
      exec_act.addMetadataQuoted("persistent_cache_key", op.persistent_cache_key);
    }

    exec_act.log(logger);

    // PyTorch thread registered under CPU process (for RUNTIME activity row label)
    recordResource(libkineto::processId(), op.pytorch_os_tid,
                   "PyTorch Thread " + std::to_string(op.pytorch_os_tid));
    // Stream registered under device (for DRIVER activity row label)
    recordResource(op.device_index, static_cast<int32_t>(op.stream_id),
                   "Stream " + std::to_string(op.stream_id));
  }

  NPROF_INFO("processTrace: emitted", traceBuffer_->activities.size(), "activities");
}

std::unique_ptr<libkineto::DeviceInfo> NeuronActivityProfilerSession::getDeviceInfo() {
  int32_t pid = libkineto::processId();
  std::string name = libkineto::processName(pid);
  return std::make_unique<libkineto::DeviceInfo>(primaryDevice_,
                                                 primaryDevice_ + libkineto::kExceedMaxPid, name,
                                                 "NeuronDevice " + std::to_string(primaryDevice_));
}

std::vector<libkineto::ResourceInfo> NeuronActivityProfilerSession::getResourceInfos() {
  std::vector<libkineto::ResourceInfo> result;
  result.reserve(resourceInfo_.size());
  for (const auto& entry : resourceInfo_) {
    result.push_back(entry.second);
  }
  return result;
}

std::unique_ptr<libkineto::CpuTraceBuffer> NeuronActivityProfilerSession::getTraceBuffer() {
  if (!traceBuffer_) {
    return std::make_unique<libkineto::CpuTraceBuffer>();
  }
  return std::move(traceBuffer_);
}

void NeuronActivityProfilerSession::pushCorrelationId(uint64_t id) { CorrelationTracker::push(id); }

void NeuronActivityProfilerSession::popCorrelationId() { CorrelationTracker::pop(); }

void NeuronActivityProfilerSession::pushUserCorrelationId(uint64_t id) {
  CorrelationTracker::pushUser(id);
}

void NeuronActivityProfilerSession::popUserCorrelationId() { CorrelationTracker::popUser(); }

bool NeuronActivityProfilerSession::hasDeviceResource(int32_t device, int32_t id) const {
  return resourceInfo_.find({device, id}) != resourceInfo_.end();
}

void NeuronActivityProfilerSession::recordResource(int32_t device, int32_t id,
                                                   const std::string& name) {
  if (!hasDeviceResource(device, id)) {
    resourceInfo_.emplace(std::make_pair(device, id),
                          libkineto::ResourceInfo(device, id, id, name));
  }
}

}  // namespace at::neuron

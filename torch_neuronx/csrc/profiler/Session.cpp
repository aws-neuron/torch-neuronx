#include "Session.h"

#include <ThreadUtil.h>
#include <libkineto.h>
#include <nrt/nrt_async.h>
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

inline const char* GetXUTypeName(int32_t xu_type) {
  switch (xu_type) {
    case NRTA_XU_TENSOR_OP:
      return "TensorOp";
    case NRTA_XU_COMPUTE:
      return "Compute";
    case NRTA_XU_COLLECTIVES:
      return "Collectives";
    default:
      return "Unknown";
  }
}

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

  // NRT config is initialized eagerly here (during prepare_trace / warmup phase)
  // so that the config allocation cost is not paid during the profiled iteration.
  // Pre-caching of node info is enabled from NeuronConfig.__init__ in Python,
  // which fires before the profiler context is entered — covering both wait
  // and warmup iterations.
  torch_neuronx::maybe_lazy_init();
  initNrtConfig();
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
  // Ensure precaching is disabled even if stop() was never called
  nrt_inspect_precache_disable();
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
//   Each ProfiledOperation produces up to three GenericTraceActivity events:
//   1. PRIVATEUSE1_RUNTIME (SUBMIT) — submission on PyTorch thread (submit_start → submit_time),
//      placed on the CPU process track under the PyTorch calling thread's OS tid.
//   2. PRIVATEUSE1_RUNTIME (DISPATCH) — execution engine scheduling (dispatch_start → execute_end),
//      placed on a per-slot virtual tid for parallel row visualization (8 slots per XU type).
//      Only emitted for device ops with dispatch_slot >= 0.
//   3. PRIVATEUSE1_DRIVER — device execution (execute_start → execute_end), placed
//      on the NeuronDevice track under the per-XU type resource.
//   A flow arrow (kLinkAsyncCpuGpu) connects DISPATCH → DRIVER via unique monotonic flow ID.
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
    if (op.dispatch_start_ns > 0) op.dispatch_start_ns += clockDomainOffset_;
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

  // Virtual tid layout for dispatch slots (above Linux pid_max to avoid collisions):
  //   Base + stream_id * kSlotsPerStream + xu_type * 8 + slot
  // Sleep tid is placed after all possible stream slots.
  constexpr int32_t kDispatchSlotTidBase = 0x500000;
  constexpr int32_t kSlotsPerStream = NRTA_XU_TYPE_NUM * 8;  // 24 device slots per stream
  constexpr int32_t kMaxStreams = 16;                        // Safe upper bound for stream count
  constexpr int32_t kSleepTid = kDispatchSlotTidBase + kMaxStreams * kSlotsPerStream;

  for (auto& op : profiled_ops) {
    // Activity 1: PRIVATEUSE1_RUNTIME — submission on PyTorch thread (submit_start → submit_time)
    traceBuffer_->emplace_activity(traceBuffer_->span, libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
                                   op.op_name);
    auto& submit_act = *traceBuffer_->activities.back();

    submit_act.startTime = op.submit_start_ns;
    submit_act.endTime = op.submit_time_ns;
    submit_act.id = static_cast<int32_t>(op.kineto_correlation_id);
    submit_act.device = libkineto::processId();
    submit_act.resource = op.pytorch_os_tid;
    submit_act.threadId = op.pytorch_os_tid;

    if (op.kineto_correlation_id != 0 && cpuActivity_) {
      submit_act.linked = cpuActivity_(static_cast<int32_t>(op.kineto_correlation_id));
    }

    submit_act.addMetadata("nrt_sequence_id", op.nrt_sequence_id);
    submit_act.addMetadata("stream_id", op.stream_id);
    submit_act.addMetadata("pytorch_seq_nr", op.pytorch_sequence_nr);
    submit_act.addMetadata("pytorch_logical_tid", op.pytorch_thread_id);
    submit_act.addMetadata("neuron_worker_tid", op.neuron_stream_os_tid);
    if (!op.hierarchy_string.empty()) {
      submit_act.addMetadataQuoted("hierarchy", op.hierarchy_string);
    }
    if (!op.neff_cache_key.empty()) {
      submit_act.addMetadataQuoted("neff_cache_key", op.neff_cache_key);
    }
    if (!op.persistent_cache_key.empty()) {
      submit_act.addMetadataQuoted("persistent_cache_key", op.persistent_cache_key);
    }

    submit_act.log(logger);

    // Activity 2: PRIVATEUSE1_RUNTIME — dispatch on RuntimeWorker thread
    // Shows when the execution engine picks up the op and schedules it to NRT.
    // DISPATCH encapsulates DRIVER: dispatch_start → execute_end.
    // Only emit for device ops that were assigned a dispatch slot.
    uint32_t flowId = nextFlowId_++;

    if (op.dispatch_slot >= 0 && op.dispatch_start_ns > 0) {
      TORCH_INTERNAL_ASSERT(op.xu_type >= 0, "dispatch_slot set but xu_type is negative");
      // Device op — per-slot virtual tid
      traceBuffer_->emplace_activity(traceBuffer_->span,
                                     libkineto::ActivityType::PRIVATEUSE1_RUNTIME, op.op_name);
      auto& dispatch_act = *traceBuffer_->activities.back();

      dispatch_act.startTime = op.dispatch_start_ns;
      dispatch_act.endTime = op.execute_end_ns;
      dispatch_act.id = static_cast<int32_t>(op.kineto_correlation_id);
      dispatch_act.device = libkineto::processId();

      int32_t dispatch_tid =
          kDispatchSlotTidBase + op.stream_id * kSlotsPerStream + op.xu_type * 8 + op.dispatch_slot;
      dispatch_act.resource = dispatch_tid;
      dispatch_act.threadId = dispatch_tid;

      // Flow START: visual arrow from DISPATCH → DRIVER
      dispatch_act.flow.id = flowId;
      dispatch_act.flow.type = libkineto::kLinkAsyncCpuGpu;
      dispatch_act.flow.start = 1;

      if (op.kineto_correlation_id != 0 && cpuActivity_) {
        dispatch_act.linked = cpuActivity_(static_cast<int32_t>(op.kineto_correlation_id));
      }

      dispatch_act.addMetadata("nrt_sequence_id", op.nrt_sequence_id);
      dispatch_act.addMetadata("stream_id", op.stream_id);
      dispatch_act.addMetadata("pytorch_seq_nr", op.pytorch_sequence_nr);
      dispatch_act.addMetadata("dispatch_slot", op.dispatch_slot);
      dispatch_act.addMetadataQuoted("xu_type", GetXUTypeName(op.xu_type));

      dispatch_act.log(logger);

      std::string slot_label = "Stream " + std::to_string(op.stream_id) + " / " +
                               GetXUTypeName(op.xu_type) + " Slot " +
                               std::to_string(op.dispatch_slot);
      recordResource(libkineto::processId(), dispatch_tid, slot_label);
    }

    // Activity 3: PRIVATEUSE1_DRIVER — device execution per XU type
    traceBuffer_->emplace_activity(traceBuffer_->span, libkineto::ActivityType::PRIVATEUSE1_DRIVER,
                                   op.op_name);
    auto& exec_act = *traceBuffer_->activities.back();

    exec_act.startTime = op.execute_start_ns;
    exec_act.endTime = op.execute_end_ns;
    exec_act.id = static_cast<int32_t>(op.kineto_correlation_id);

    // kHost ops run on a HostWorkerPool CPU thread, so render them on the CPU
    // process track under the worker's real OS tid (like the SUBMIT activity).
    // Device-XU ops render on the NeuronDevice track, resource encoding
    // stream + XU type (3 rows per stream).
    int32_t xu_resource;
    if (op.host_worker_idx >= 0) {
      exec_act.device = libkineto::processId();
      xu_resource = op.neuron_stream_os_tid;
      exec_act.threadId = op.neuron_stream_os_tid;
    } else {
      exec_act.device = static_cast<int32_t>(op.device_index);
      xu_resource = (op.xu_type >= 0)
                        ? static_cast<int32_t>(op.stream_id * NRTA_XU_TYPE_NUM + op.xu_type)
                        : static_cast<int32_t>(op.stream_id);
    }
    exec_act.resource = xu_resource;

    // Flow END: receives arrow from DISPATCH
    if (op.dispatch_slot >= 0 && op.dispatch_start_ns > 0) {
      exec_act.flow.id = flowId;
      exec_act.flow.type = libkineto::kLinkAsyncCpuGpu;
      exec_act.flow.start = 0;
    }

    if (op.kineto_correlation_id != 0 && cpuActivity_) {
      exec_act.linked = cpuActivity_(static_cast<int32_t>(op.kineto_correlation_id));
    }

    exec_act.addMetadata("nrt_sequence_id", op.nrt_sequence_id);
    exec_act.addMetadata("pytorch_seq_nr", op.pytorch_sequence_nr);
    if (op.xu_type >= 0) {
      exec_act.addMetadataQuoted("xu_type", GetXUTypeName(op.xu_type));
    }
    if (op.host_worker_idx >= 0) {
      exec_act.addMetadata("host_worker_idx", op.host_worker_idx);
    }
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

    // Register resources
    recordResource(libkineto::processId(), op.pytorch_os_tid,
                   "PyTorch Thread " + std::to_string(op.pytorch_os_tid));
    if (op.host_worker_idx >= 0) {
      // kHost op on the CPU process track under the host worker's OS tid.
      recordResource(libkineto::processId(), op.neuron_stream_os_tid,
                     "host_compute-w" + std::to_string(op.host_worker_idx));
    } else if (op.xu_type >= 0) {
      std::string xu_label =
          "Stream " + std::to_string(op.stream_id) + " / " + GetXUTypeName(op.xu_type);
      recordResource(op.device_index, xu_resource, xu_label);
    } else {
      recordResource(op.device_index, xu_resource, "Stream " + std::to_string(op.stream_id));
    }
  }

  // Emit RuntimeWorker sleep events on a dedicated virtual tid
  auto sleep_events = eventBuffer_.drainSleepEvents();
  if (!sleep_events.empty()) {
    for (auto& evt : sleep_events) {
      int64_t start = evt.start_ns + clockDomainOffset_;
      int64_t end = evt.end_ns + clockDomainOffset_;
      if (end < captureWindowStart_ || start > captureWindowEnd_) continue;

      traceBuffer_->emplace_activity(
          traceBuffer_->span, libkineto::ActivityType::PRIVATEUSE1_RUNTIME, "RuntimeWorker::sleep");
      auto& sleep_act = *traceBuffer_->activities.back();
      sleep_act.startTime = start;
      sleep_act.endTime = end;
      sleep_act.device = libkineto::processId();
      sleep_act.resource = kSleepTid;
      sleep_act.threadId = kSleepTid;
      sleep_act.log(logger);
    }
    recordResource(libkineto::processId(), kSleepTid, "RuntimeWorker / Idle");
    NPROF_INFO("processTrace: emitted", sleep_events.size(), "sleep events");
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

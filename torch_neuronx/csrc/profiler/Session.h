#pragma once

#include <ActivityType.h>
#include <Config.h>
#include <IActivityProfiler.h>
#include <libkineto.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "ContinuousExporter.h"
#include "ProfilerEventBuffer.h"

struct nrt_inspect_config;
typedef struct nrt_inspect_config nrt_inspect_config_t;

namespace at::neuron {
class NeuronActivityProfilerSession : public libkineto::IActivityProfilerSession {
 public:
  explicit NeuronActivityProfilerSession(const std::set<libkineto::ActivityType>& activityTypes,
                                         const libkineto::Config& config);

  ~NeuronActivityProfilerSession() override;

  void start() override;
  void stop() override;
  [[nodiscard]] std::vector<std::string> errors() override;

  void processTrace(libkineto::ActivityLogger& logger) override;

  void processTrace(libkineto::ActivityLogger& logger,
                    libkineto::getLinkedActivityCallback getLinkedActivity,
                    int64_t captureWindowStartTime, int64_t captureWindowEndTime) override;

  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override;
  std::vector<libkineto::ResourceInfo> getResourceInfos() override;
  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override;

  void pushCorrelationId(uint64_t id) override;
  void popCorrelationId() override;
  void pushUserCorrelationId(uint64_t id) override;
  void popUserCorrelationId() override;

 private:
  std::set<libkineto::ActivityType> activityTypes_;

  std::unique_ptr<const libkineto::Config> config_;

  std::vector<std::string> errors_;

  nrt_inspect_config_t* nrtConfig_{nullptr};
  // Host-side export settings parsed from custom_profiler_config (continuous_export key).
  ExportConfig exportConfig_{};
  // Async export worker; constructed in start() only when exportConfig_.enabled, published
  // to the bridge, and stopped/reset before nrt_inspect_stop() in stop() and the dtor.
  // Null for sessions that did not opt in.
  std::unique_ptr<ContinuousExporter> continuousExporter_;
  int64_t profileStartTime_{0};
  int64_t profileEndTime_{0};
  int64_t clockDomainOffset_{0};  // system_clock_ns - steady_clock_ns (computed once at start)
  bool isActive_{false};

  mutable std::mutex mutex_;

  // Profiler event buffer for collecting operation snapshots
  ProfilerEventBuffer eventBuffer_;

  // Trace buffer populated by processTrace(), returned by getTraceBuffer()
  // Heap-allocated so the TraceSpan address is stable (GenericTraceActivity stores raw pointers).
  std::unique_ptr<libkineto::CpuTraceBuffer> traceBuffer_;

  // Callback to look up the CPU activity linked to a given correlation ID
  libkineto::getLinkedActivityCallback cpuActivity_;

  // Capture window timestamps (set by processTrace async overload)
  int64_t captureWindowStart_{0};
  int64_t captureWindowEnd_{0};

  // Primary device index captured from the first profiled operation (default 0).
  int32_t primaryDevice_{0};

  // Monotonic counter for unique flow IDs (0 = no flow).
  // Each RUNTIME→DRIVER pair gets a unique flow ID to avoid duplicate flow starts
  // in PyTorch's setParents() flow_map (collection.cpp:1152).
  uint32_t nextFlowId_{1};

  // Resource info collected during processTrace for getResourceInfos()
  std::map<std::pair<int32_t, int32_t>, libkineto::ResourceInfo> resourceInfo_;

  void initNrtConfig();
  void cleanupNrtConfig();
  void applyCustomConfig();
  // Export teardown, shared by stop() and the destructor. Must run BEFORE
  // nrt_inspect_stop(): unpublishes the bridge, then joins and destroys the exporter,
  // so no async drain is in flight when the final flush runs. (Sync export_now drains
  // are ordered against nrt_inspect_stop by NRT's own global_lock.)
  void teardownContinuousExport();

  bool hasDeviceResource(int32_t device, int32_t id) const;
  void recordResource(int32_t device, int32_t id, const std::string& name);
};
}  // namespace at::neuron

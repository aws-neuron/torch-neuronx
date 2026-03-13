#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"

namespace at::neuron {

// Forward declarations
class CompilationCache;
class ModelHandleCache;
class OperationExecutionEngine;

// Enhanced semaphore implementation for resource management with priority support.
//
// This semaphore provides thread-safe resource allocation with:
// - Priority-based acquisition (higher priority operations get slots first)
// - Blocking and timeout-based acquisition
// - Resource usage statistics
//
// Can be used for thread pools and other resource management.
class ResourceSemaphore {
 public:
  explicit ResourceSemaphore(int initial_count);

  // Resource acquisition (legacy interface - uses default priority)
  void Acquire();
  bool TryAcquireFor(std::chrono::milliseconds timeout);

  // Priority-aware resource acquisition
  void AcquireWithPriority(int priority);
  bool TryAcquireForWithPriority(std::chrono::milliseconds timeout, int priority);

  void Release();

  // Thread-safe statistics accessors
  int AvailableCount() const;
  int WaitingThreads() const;
  int AcquiredSlots() const;
  int MaxSlots() const;

 private:
  struct WaitingRequest {
    int priority;
    std::condition_variable cv;
    bool granted = false;
    std::chrono::steady_clock::time_point request_time;

    explicit WaitingRequest(int prio)
        : priority(prio), request_time(std::chrono::steady_clock::now()) {}
  };

  mutable std::mutex mutex_;
  int count_;
  const int max_count_;
  std::atomic<int> waiting_count_{0};
  std::atomic<int> acquired_count_{0};

  // Priority queue for waiting requests (higher priority first, then FIFO within priority)
  std::vector<std::shared_ptr<WaitingRequest>> waiting_requests_;

  // NOTE: Helper functions, in which all must be called with mutex_ held

  // Grants the next available slot to the highest priority waiting request.
  void GrantNextSlot();

  // Attempts immediate slot acquisition if available, returns true if successful.
  bool TryAcquireImmediate();

  // Adds a request to the waiting queue and sorts by priority.
  void AddToWaitingQueue(std::shared_ptr<WaitingRequest> request);

  // Removes a specific request from the waiting queue.
  void RemoveFromWaitingQueue(std::shared_ptr<WaitingRequest> request);

  // Sorts the waiting queue by priority (higher first) then request time (FIFO).
  void SortWaitingQueue();
};

// RAII wrapper for resource semaphore management with priority support.
//
// Supports both blocking and timeout-based acquisition patterns with priority.
// Automatically releases the semaphore on destruction.
//
// Thread Safety: This class is NOT thread-safe. Each instance should be used
// by a single thread only.
class ResourceGuard {
 public:
  explicit ResourceGuard(ResourceSemaphore* semaphore, int priority = 0);
  ~ResourceGuard();

  // Acquisition methods (uses priority set in constructor)
  bool TryAcquireFor(std::chrono::milliseconds timeout);
  void Acquire();

  // Explicit release (for early release patterns)
  void Release();

  // Get the priority of this guard
  int GetPriority() const { return priority_; }

  bool IsAcquired() const { return acquired_; }

  // Non-copyable, non-movable for safety
  ResourceGuard(const ResourceGuard&) = delete;
  ResourceGuard& operator=(const ResourceGuard&) = delete;
  ResourceGuard(ResourceGuard&&) = delete;
  ResourceGuard& operator=(ResourceGuard&&) = delete;

 private:
  ResourceSemaphore* semaphore_;
  int priority_;
  bool acquired_{false};
};

// NeuronResourceManager manages shared resources across all Neuron streams and devices.
//
// This singleton provides centralized access to resources that need to be shared
// across the entire Neuron execution system:
// - CompilationCache: Shared NEFF compilation and caching
// - ModelHandleCache: Shared NRT model handle caching
// - OperationExecutionEngine: Centralized execution coordination across all streams
// - ResourceSemaphore: General resource management (for future thread pools)
class NeuronResourceManager {
 public:
  // Get the singleton instance
  static NeuronResourceManager& Instance();

  // Get the shared compilation cache
  // Returns: Reference to the global compilation cache
  CompilationCache& GetCompilationCache();

  // Get the shared model handle cache
  // Returns: Reference to the global model handle cache
  ModelHandleCache& GetModelHandleCache();

  // Get the global execution engine for centralized execution coordination
  // Returns: Reference to the execution engine
  OperationExecutionEngine& GetOperationExecutionEngine();

  // Get the shared resource semaphore for general resource management
  // Returns: Reference to the global resource semaphore
  ResourceSemaphore& GetResourceSemaphore();

  // Get the tensor pool for a specific device
  // Returns: Reference to the device's tensor pool
  c10_neuron::NrtTensorPool& GetTensorPool(int device);

 private:
  // Private constructor for singleton pattern
  NeuronResourceManager();
  ~NeuronResourceManager();

  // Non-copyable, non-movable
  NeuronResourceManager(const NeuronResourceManager&) = delete;
  NeuronResourceManager& operator=(const NeuronResourceManager&) = delete;
  NeuronResourceManager(NeuronResourceManager&&) = delete;
  NeuronResourceManager& operator=(NeuronResourceManager&&) = delete;

  // Helper function to get max resource slots from environment
  static int GetMaxResourceSlots();

  mutable std::mutex mutex_;

  std::vector<std::unique_ptr<c10_neuron::NrtTensorPool>> tensor_pools_;
  std::unique_ptr<CompilationCache> compilation_cache_;
  std::unique_ptr<ModelHandleCache> model_handle_cache_;
  std::unique_ptr<OperationExecutionEngine> operation_execution_engine_;
  std::unique_ptr<ResourceSemaphore> resource_semaphore_;
};

}  // namespace at::neuron

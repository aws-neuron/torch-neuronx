#pragma once

#include <c10/core/Device.h>
#include <c10/util/intrusive_ptr.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <vector>

#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"

namespace torch_neuronx {
namespace distributed {

/**
 * TensorShelf: Thread-safe container for stashed tensors.
 *
 * TensorShelf pattern for safe tensor lifecycle management
 * across main thread and watchdog thread.
 *
 * The mutex protects the tensor vector because it can be accessed from:
 * - Main thread: via synchronize() -> unstash()
 * - Watchdog thread: via finishWork() -> empty() check + transfer to shelves
 */
class TensorShelf {
 public:
  /**
   * Stash tensors to keep them alive until work completes.
   * Prevents CachingAllocator from recycling memory prematurely.
   */
  void stash(const std::vector<at::Tensor>& tensors);

  /**
   * Unstash tensors - allows CachingAllocator to recycle them.
   * Same as clear().
   */
  void unstash();

  /**
   * Check if shelf is empty (thread-safe).
   */
  bool empty() const;

  /**
   * Clear the shelf (thread-safe).
   */
  void clear();

 private:
  std::vector<at::Tensor> tensors_;
  mutable std::mutex mutex_;
};

/**
 * NeuronWork: Async work tracking for Neuron distributed collectives.
 *
 * This class tracks the lifecycle of a collective operation using Neuron
 * streams and events to provide proper async semantics. It inherits from
 * c10d::Work and implements the required methods for FSDP and other
 * distributed frameworks.
 *
 * The work lifecycle:
 * 1. Collective operation submits work to a stream
 * 2. NeuronWork is created, recording an end event on the stream
 * 3. The event is queued after the collective work
 * 4. When the collective completes, the event signals
 * 5. wait()/isCompleted() check/wait for the event signal
 *
 * Design mirrors PyTorch's NCCL WorkNCCL implementation.
 * https://github.com/pytorch/pytorch/blob/0e3e52f5907d215230934aaf36cb18c3f717e6f3/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L509
 */
class NeuronWork : public c10d::Work {
 public:
  /**
   * Construct a NeuronWork instance.
   *
   * @param pg_uid Unique identifier for the process group
   * @param pg_desc Description of the process group
   * @param device The Neuron device this work is executing on
   * @param rank The rank of this process
   * @param op_type Type of collective operation (for debugging)
   * @param seq_num Sequence number for this operation (for debugging)
   * @param outputs Optional list of output tensors to keep alive
   * @param enable_timing Whether to enable timing measurements
   * @param timeout_ms Operation timeout in milliseconds
   * @param stream The stream executing the collective (nullptr = current stream)
   */
  NeuronWork(const std::string& pg_uid, const std::string& pg_desc, at::Device device, int rank,
             const std::string& op_type, uint64_t seq_num, std::vector<at::Tensor> outputs = {},
             bool enable_timing = false, float timeout_ms = 300000.0f,
             at::neuron::NeuronStream* stream = nullptr, bool blocking_wait = false);

  ~NeuronWork() override = default;

  // Disable copy
  NeuronWork(const NeuronWork&) = delete;
  NeuronWork& operator=(const NeuronWork&) = delete;

  // Allow move
  NeuronWork(NeuronWork&&) = default;
  NeuronWork& operator=(NeuronWork&&) = default;

  // ============== c10d::Work interface ==============

  /**
   * Check if the work has completed without blocking.
   *
   * This is a pure query function with no side effects - it simply checks
   * the event status. This matches NCCL's finishedGPUExecutionInternal().
   */
  bool isCompleted() override;

  /**
   * Check if the work completed successfully.
   * @deprecated Use isCompleted() and exception() instead.
   */
  bool isSuccess() const override;
  /**
   * Return the source rank for receive operations.
   * Not applicable for most collective operations.
   */
  int sourceRank() const override { return -1; }

  /**
   * Return the output tensors from this operation.
   */
  std::vector<at::Tensor> result() override;

  /**
   * Block until the work completes.
   *
   * This is the critical method for FSDP - it ensures the collective
   * operation has completed before the caller proceeds to use the
   * output tensors.
   *
   * @param timeout Timeout duration (default: no timeout)
   * @return True if work completed successfully
   */
  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

  /**
   * Return a Future that completes when the work finishes.
   */
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

  // ============== NeuronWork-specific methods ==============

  /**
   * Record the start event (call BEFORE the collective operation).
   * Only has effect if timing is enabled.
   *
   * @param stream Stream to record on (nullptr = use work's stream)
   */
  void recordStartEvent(at::neuron::NeuronStream* stream = nullptr);

  /**
   * Record the end event (call AFTER the collective operation).
   * This must be called for the work to track completion properly.
   *
   * @param stream Stream to record on (nullptr = use work's stream)
   */
  void recordEndEvent(at::neuron::NeuronStream* stream = nullptr);

  /**
   * Register this work with output tensors for wait_tensor() support.
   * Must be called from C++ so the actual NeuronWork pointer is registered,
   * allowing unregister_work(this) in synchronize() to match properly.
   *
   * Only registers when allow_inflight_collective_as_graph_input() is true.
   */
  void registerWithTensors();

  /**
   * Make the current stream wait for this work's completion.
   *
   * This is used for stream synchronization patterns where you want
   * the current stream to wait for a collective that ran on a different
   * stream, without blocking the CPU.
   */
  void synchronize();

  /**
   * Check if the work has started execution on the device.
   */
  bool isStarted() const;

  /**
   * Check if this work has exceeded its timeout.
   *
   * This is a non-blocking check that can be called periodically by a
   * watchdog or monitoring system. If timed out, sets exception internally
   * following NCCL pattern.
   *
   * @param timeout Optional timeout override
   * @return true if timed out, false otherwise
   */
  bool checkTimeout(std::optional<std::chrono::milliseconds> timeout = std::nullopt);

  /**
   * Stash tensors to keep them alive until work completes.
   *
   * stash both inputs and outputs to prevent caching allocator
   * from recycling memory while collective is in-flight.
   *
   * @param tensors Tensors to stash
   */
  void stash(const std::vector<at::Tensor>& tensors);

  /**
   * Release references to stashed tensors for allocator recycling.
   *
   * Called after synchronization to allow the caching allocator to
   * recycle tensor memory. Safe because stream ordering guarantees
   * any subsequent work will wait for the collective to complete.
   */
  void unstashTensors();

  // Get and clear stashed tensors, transferring ownership to caller.
  /**
   * Detach and return the TensorShelf, transferring ownership to the caller.
   * After this call, work no longer owns the shelf (pointer is null).
   * This enables clear ownership transfer from work -> watchdog.
   */
  std::unique_ptr<TensorShelf> detachStashedTensorShelf() {
    std::lock_guard<std::mutex> lock(shelf_detach_mutex_);
    if (stashed_for_allocator_safety_ && !outputs_.empty()) {
      stashed_for_allocator_safety_->stash(outputs_);
    }
    outputs_.clear();
    future_.reset();
    return std::move(stashed_for_allocator_safety_);
  }

  /**
   * Get the sequence number for debugging.
   */
  uint64_t getSequenceNumber() const { return seq_num_; }

  /**
   * Get the operation type for debugging.
   */
  const std::string& getOpType() const { return op_type_; }

  /**
   * Get the device this work is running on.
   */
  at::Device getDevice() const { return device_; }

  /**
   * Get the duration of the collective operation in milliseconds.
   * Requires timing to be enabled when the work was created.
   */
  float getDuration() const;

  /**
   * Set an exception on this work.
   */
  void setException(std::exception_ptr exception);

  /**
   * Abort this work, unblocking any waiters.
   *
   * This is called by the watchdog when a timeout is detected.
   * It force-completes the event to unblock any threads waiting on
   * this work, following the NCCL WorkNCCL::abort() pattern.
   *
   * Note: This is a "soft" abort that unblocks waiters. The underlying
   * collective operation on the device may continue running.
   */
  void abort();

  /**
   * Get the end event for external synchronization.
   */
  const at::neuron::NeuronEvent& getEndEvent() const { return end_event_; }

  /**
   * Get the stream this work is associated with.
   */
  const at::neuron::NeuronStream& getStream() const { return stream_; }

  // For watchdog access
  friend class NeuronWatchdog;

 private:
  // Process group info
  std::string pg_uid_;
  std::string pg_desc_;

  // Device and rank info
  at::Device device_;
  int rank_;

  // Operation metadata
  std::string op_type_;
  uint64_t seq_num_;

  // Timing and timeout
  bool timing_enabled_;
  float timeout_ms_;
  std::chrono::steady_clock::time_point work_start_time_;

  // Events for tracking completion
  at::neuron::NeuronEvent start_event_;
  at::neuron::NeuronEvent end_event_;

  // Stream this work runs on
  at::neuron::NeuronStream stream_;

  // Output tensors for result()
  std::vector<at::Tensor> outputs_;

  // Stashed tensors for memory safety (TensorShelf pattern).
  // Uses unique_ptr - ownership transfers from work to watchdog via detachStashedTensorShelf().
  // After detach, work's pointer is null and watchdog owns the shelf exclusively.
  std::unique_ptr<TensorShelf> stashed_for_allocator_safety_;
  mutable std::mutex shelf_detach_mutex_;

  // Note: exception_ and mutex_ are inherited from c10d::Work base class

  // Future for async completion
  c10::intrusive_ptr<c10::ivalue::Future> future_;

  // Success flag
  std::atomic<bool> success_{true};

  // Whether wait() should block CPU thread until completion (NCCL pattern)
  bool blockingWait_{false};

  // Helper methods
  void checkAndSetException();
  std::string logPrefix() const;
  void markFutureWithError();
  void markFutureComplete();

 public:
  // Public method for watchdog to mark future complete
  // This matches NCCL behavior
  void markFutureCompleteIfNeeded();
  void handleException();
};

}  // namespace distributed
}  // namespace torch_neuronx

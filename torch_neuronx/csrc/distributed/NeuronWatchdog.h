#pragma once

#include <c10/util/intrusive_ptr.h>

#include <atomic>
#include <condition_variable>
#include <list>
#include <mutex>
#include <thread>

#include "torch_neuronx/csrc/distributed/NeuronWork.h"

namespace torch_neuronx {
namespace distributed {

/**
 * Watchdog thread for monitoring async collective operations.
 *
 * The watchdog periodically checks all enqueued work items to detect:
 * - Completion: When collectives finish, unstash tensors to free memory
 * - Timeout: When collectives exceed their timeout, mark as failed
 *
 * This is essential for async operations where the user may not immediately
 * call work.wait(), as it ensures tensors are cleaned up automatically.
 *
 * Design mirrors NCCL's ProcessGroupNCCL::Watchdog implementation.
 */
class NeuronWatchdog {
 public:
  // Sleep interval matches NCCL's kWatchdogThreadSleepMillis
  static constexpr int SLEEP_MILLIS = 100;

  // Max wait time for shelves to populate during cleanup
  static constexpr int CLEANUP_WAIT_MILLIS = 1;

  /**
   * Initialize watchdog.
   * The watchdog manages its own internal work list.
   */
  NeuronWatchdog();

  ~NeuronWatchdog();

  // Disable copy and move
  NeuronWatchdog(const NeuronWatchdog&) = delete;
  NeuronWatchdog& operator=(const NeuronWatchdog&) = delete;
  NeuronWatchdog(NeuronWatchdog&&) = delete;
  NeuronWatchdog& operator=(NeuronWatchdog&&) = delete;

  /**
   * Start the watchdog thread.
   */
  void start();

  /**
   * Stop the watchdog thread gracefully.
   */
  void stop();

  /**
   * Notify watchdog that new work was added.
   *
   * Wakes up the watchdog thread immediately instead of waiting for
   * the next polling interval. This improves responsiveness.
   */
  void notify();

  /**
   * Add work to the monitoring queue.
   *
   * @param work The work to monitor
   */
  void enqueueWork(c10::intrusive_ptr<NeuronWork> work);

  /**
   * Clean up tensor shelves from completed work.
   *
   * Called automatically at the start of enqueueWork().
   */
  void cleanupCompletedShelves();

 private:
  /**
   * Main watchdog loop.
   *
   * Waits on condition variable with timeout (wakes on notify OR timeout),
   * then checks all pending work for completion/timeout.
   */
  void runLoop();

  /**
   * Handle completed work.
   * Unstashes tensors and unregisters from PyTorch's WorkRegistry.
   *
   * CRITICAL: Must unregister work from WorkRegistry to prevent wait_tensor()
   * from finding stale work when caching allocator reuses tensor storage.
   *
   * @param work Completed NeuronWork instance
   */
  void handleCompletion(c10::intrusive_ptr<NeuronWork>& work);

  /**
   * Handle timed out work.
   * Sets exception and unstashes tensors.
   *
   * @param work Timed out NeuronWork instance
   */
  void handleTimeout(c10::intrusive_ptr<NeuronWork>& work);

  // Watchdog thread
  std::thread neuronCommWatchdogThread_;

  // Termination flag
  std::atomic<bool> terminate_{false};

  // Thread exit flag - set by runLoop() just before exiting.
  // Used by stop() to implement timed join with fallback to detach,
  // matching Python's thread.join(timeout=1.0) pattern.
  std::atomic<bool> thread_exited_{false};

  // Work list and synchronization
  std::list<c10::intrusive_ptr<NeuronWork>> work_list_;
  std::mutex work_mutex_;
  std::condition_variable work_cv_;

  // Tensor shelves from completed work, to be cleaned up on main thread.
  // Shelves for deferred tensor cleanup
  // Stores unique_ptr<TensorShelf> moved from completed work.
  // Ownership transfers from work -> watchdog via detachStashedTensorShelf().
  std::vector<std::unique_ptr<TensorShelf>> shelvesToUnstash_;
  std::mutex shelves_mutex_;
  std::condition_variable shelves_cv_;
};

}  // namespace distributed
}  // namespace torch_neuronx

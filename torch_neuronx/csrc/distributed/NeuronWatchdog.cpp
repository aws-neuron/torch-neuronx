#include "torch_neuronx/csrc/distributed/NeuronWatchdog.h"

#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/utils/pybind.h>

#include <chrono>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/allocator/ShelvesCleanup.h"

namespace torch_neuronx {
namespace distributed {

NeuronWatchdog::NeuronWatchdog() {
  ShelvesCleanupRegistry::instance().registerCleanup(this, [this]() { cleanupCompletedShelves(); });
}

NeuronWatchdog::~NeuronWatchdog() {
  // Unregister FIRST to prevent calls to destroyed object
  ShelvesCleanupRegistry::instance().unregisterCleanup(this);
  stop();
}

void NeuronWatchdog::start() {
  TORCH_CHECK(!neuronCommWatchdogThread_.joinable(), "Watchdog thread already started");

  neuronCommWatchdogThread_ = std::thread(&NeuronWatchdog::runLoop, this);
  LOG(INFO) << "Watchdog thread started";
}

void NeuronWatchdog::stop() {
  if (!neuronCommWatchdogThread_.joinable()) {
    return;
  }

  terminate_.store(true, std::memory_order_release);

  notify();  // Wake up thread immediately

  // PyObject refs decref (stashed tensors).
  const bool gil_available = Py_IsInitialized() && PyGILState_Check();
  if (gil_available) {
    pybind11::gil_scoped_release release;
    neuronCommWatchdogThread_.join();
  } else {
    neuronCommWatchdogThread_.join();
  }

  // Clean up any remaining shelves that were queued during final loop iterations
  // but never cleaned up (since cleanupCompletedShelves is only called from
  // enqueueWork). Only safe under GIL because at::Tensor decref on
  // Python-owned tensors can reenter Python.
  //
  // Non-GIL stop() (e.g., reached during C++ static teardown) deliberately
  // defers both shelf cleanup and stream drain to ~NeuronResourceManager,
  // which runs last and holds the only view of the still-live stream pool.
  if (gil_available) {
    cleanupCompletedShelves();
    // Unstashed tensors drop their last ref, and the lazy-alloc path submits
    // a kDealloc op per recorded stream. Drain those now while the engine
    // workers are still alive — otherwise they survive past the cleanup
    // capsule's sync and deadlock ~StreamImpl::Synchronize later.
    at::neuron::drain_visible_device_streams("NeuronWatchdog::stop");
  }

  LOG(INFO) << "Watchdog thread stopped";
}

void NeuronWatchdog::notify() {
  // We don't hold the mutex when calling notify_one().
  work_cv_.notify_one();
}

void NeuronWatchdog::enqueueWork(c10::intrusive_ptr<NeuronWork> work) {
  // Ensure we clean up the tensor shelves from the prior work, while
  // the host still holds GIL.
  cleanupCompletedShelves();
  {
    std::lock_guard<std::mutex> lock(work_mutex_);
    work_list_.push_back(std::move(work));
  }
  notify();  // Wake up watchdog immediately
}

void NeuronWatchdog::cleanupCompletedShelves() {
  // Wake watchdog to process any completed work and populate shelves
  notify();

  // Move shelves to local vector while holding lock
  std::vector<std::unique_ptr<TensorShelf>> localShelves;
  {
    std::unique_lock<std::mutex> lock(shelves_mutex_);
    localShelves = std::move(shelvesToUnstash_);
    shelvesToUnstash_.clear();
  }

  // Unstash outside lock to avoid deadlock
  for (auto& shelf : localShelves) {
    if (shelf) {
      shelf->unstash();
    }
  }
}

void NeuronWatchdog::runLoop() {
  try {
    // Loop until terminate is set AND work_list is empty.
    // On terminate, abort any remaining work to avoid hanging.
    while (true) {
      bool terminating = terminate_.load(std::memory_order_acquire);
      // Collect completed/timed-out work while holding the lock,
      // then process them OUTSIDE the lock to minimize contention.
      std::vector<c10::intrusive_ptr<NeuronWork>> completed_work;
      std::vector<c10::intrusive_ptr<NeuronWork>> timedout_work;

      {
        // hold lock from wait_for through work iteration
        std::unique_lock<std::mutex> lock(work_mutex_);

        // Wait with timeout: wake on notify() OR after SLEEP_MILLIS
        work_cv_.wait_for(lock, std::chrono::milliseconds(SLEEP_MILLIS),
                          [this]() { return terminate_.load(std::memory_order_acquire); });

        // Check work list while still holding the lock
        for (auto it = work_list_.begin(); it != work_list_.end();) {
          auto& work = *it;

          try {
            if (work->isCompleted()) {
              completed_work.push_back(work);
              it = work_list_.erase(it);
              continue;
            }

            bool timed_out = work->checkTimeout();
            if (timed_out) {
              timedout_work.push_back(work);
              it = work_list_.erase(it);
              continue;
            }

            // Work still pending, move to next
            ++it;
          } catch (const std::exception& e) {
            LOG(ERROR) << "Error checking work (seq=" << work->getSequenceNumber()
                       << "): " << e.what();
            // Remove problematic work to prevent infinite errors
            it = work_list_.erase(it);
          }
        }

        // When terminating, abort any remaining work to avoid hanging
        if (terminating && !work_list_.empty()) {
          for (auto& work : work_list_) {
            timedout_work.push_back(work);
          }
          work_list_.clear();
        }
      }  // Lock released here

      // Handle completed/timed-out work outside the lock to minimize contention
      for (auto& work : completed_work) {
        handleCompletion(work);
      }
      for (auto& work : timedout_work) {
        handleTimeout(work);
      }

      // Exit after processing remaining work when terminate is set
      if (terminating) {
        break;
      }
    }

    // TODO: Implement abort() functionality to safely drain pending work.

  } catch (const std::exception& e) {
    LOG(ERROR) << "Watchdog thread crashed: " << e.what();
  }
}

void NeuronWatchdog::handleCompletion(c10::intrusive_ptr<NeuronWork>& work) {
  VLOG(2) << "Work completed: op=" << work->getOpType() << ", seq=" << work->getSequenceNumber();

  // CRITICAL: Unregister work from PyTorch's WorkRegistry.
  // This prevents wait_tensor() from finding stale work when the caching
  // allocator reuses tensor storage. Without this, subsequent tests that
  // reuse the same storage would hang forever waiting on already-completed work.
  if (c10d::allow_inflight_collective_as_graph_input()) {
    c10d::unregister_work(work);
  }

  // Mark future complete

  // Move ownership of TensorShelf from work to watchdog.
  // After detach, work no longer holds reference to the shelf.
  auto shelf = work->detachStashedTensorShelf();

  // Only shelve if valid and not empty
  if (shelf && !shelf->empty()) {
    std::lock_guard<std::mutex> lock(shelves_mutex_);
    shelvesToUnstash_.push_back(std::move(shelf));
    shelves_cv_.notify_all();
  }
}

void NeuronWatchdog::handleTimeout(c10::intrusive_ptr<NeuronWork>& work) {
  LOG(ERROR) << "Work timed out: op=" << work->getOpType() << ", seq=" << work->getSequenceNumber();

  // CRITICAL: Unregister work from PyTorch's WorkRegistry (same as handleCompletion)
  // Timed-out work must also be unregistered to prevent stale work references.
  if (c10d::allow_inflight_collective_as_graph_input()) {
    c10d::unregister_work(work);
  }

  // Abort first to set exception and force-complete event
  work->abort();

  // Mark future with error

  // Move ownership of TensorShelf from work to watchdog (same as completion path)
  auto shelf = work->detachStashedTensorShelf();

  if (shelf && !shelf->empty()) {
    std::lock_guard<std::mutex> lock(shelves_mutex_);
    shelvesToUnstash_.push_back(std::move(shelf));
    shelves_cv_.notify_all();
  }
}

}  // namespace distributed
}  // namespace torch_neuronx

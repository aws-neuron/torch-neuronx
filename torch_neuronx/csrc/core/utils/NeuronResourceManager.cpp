#include "NeuronResourceManager.h"

#include <algorithm>
#include <cstdlib>
#include <memory>

#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"
#include "torch_neuronx/csrc/core/runtime/ModelHandleCache.h"

namespace {
// Resource configuration constants
constexpr int kDefaultMaxResourceSlots = 4;
}  // namespace

namespace at::neuron {

ResourceSemaphore::ResourceSemaphore(int initial_count)
    : count_(initial_count), max_count_(initial_count) {}

void ResourceSemaphore::Acquire() { AcquireWithPriority(0 /* default priority */); }

bool ResourceSemaphore::TryAcquireFor(std::chrono::milliseconds timeout) {
  return TryAcquireForWithPriority(timeout, 0 /* default priority */);
}

void ResourceSemaphore::AcquireWithPriority(int priority) {
  std::unique_lock<std::mutex> lock(mutex_);
  ++waiting_count_;

  if (TryAcquireImmediate()) {
    --waiting_count_;
    return;
  }

  // Create a waiting request if there is pending work
  auto request = std::make_shared<WaitingRequest>(priority);
  AddToWaitingQueue(request);

  // Wait on this slot's turn
  request->cv.wait(lock, [request] { return request->granted; });

  // Done, dequeue from the waiting list
  RemoveFromWaitingQueue(request);

  --waiting_count_;
  ++acquired_count_;
}

bool ResourceSemaphore::TryAcquireForWithPriority(std::chrono::milliseconds timeout, int priority) {
  std::unique_lock<std::mutex> lock(mutex_);
  ++waiting_count_;

  if (TryAcquireImmediate()) {
    --waiting_count_;
    return true;
  }

  // Need to wait - create a waiting request
  auto request = std::make_shared<WaitingRequest>(priority);
  AddToWaitingQueue(request);

  // Wait on this slot's turn
  bool acquired = request->cv.wait_for(lock, timeout, [request] { return request->granted; });

  if (acquired) {
    ++acquired_count_;
  }

  // Done, dequeue from the waiting list
  RemoveFromWaitingQueue(request);

  --waiting_count_;
  return acquired;
}

void ResourceSemaphore::Release() {
  std::lock_guard<std::mutex> lock(mutex_);
  ++count_;
  --acquired_count_;
  GrantNextSlot();
}

void ResourceSemaphore::GrantNextSlot() {
  // This method should be called with mutex_ already held

  if (count_ <= 0) {
    return;  // No slots available
  }

  if (waiting_requests_.empty()) {
    // No one waiting, slot will remain available for next acquire
    return;
  }

  // Grant to highest priority waiter (already sorted)
  auto& next_request = waiting_requests_.front();
  next_request->granted = true;
  next_request->cv.notify_one();
  --count_;  // Consume the slot
}

int ResourceSemaphore::AvailableCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return count_;
}

int ResourceSemaphore::WaitingThreads() const { return waiting_count_; }

int ResourceSemaphore::AcquiredSlots() const { return acquired_count_; }

int ResourceSemaphore::MaxSlots() const { return max_count_; }

ResourceGuard::ResourceGuard(ResourceSemaphore* semaphore, int priority)
    : semaphore_(semaphore), priority_(priority) {
  if (!semaphore_) {
    throw std::runtime_error("Cannot create ResourceGuard with null semaphore");
  }
}

ResourceGuard::~ResourceGuard() { Release(); }

bool ResourceGuard::TryAcquireFor(std::chrono::milliseconds timeout) {
  if (!acquired_ && semaphore_->TryAcquireForWithPriority(timeout, priority_)) {
    acquired_ = true;
    return true;
  }
  return acquired_;
}

void ResourceGuard::Acquire() {
  if (!acquired_) {
    semaphore_->AcquireWithPriority(priority_);
    acquired_ = true;
  }
}

void ResourceGuard::Release() {
  if (acquired_) {
    semaphore_->Release();
    acquired_ = false;
  }
}

NeuronResourceManager& NeuronResourceManager::Instance() {
  static NeuronResourceManager manager;
  return manager;
}

CompilationCache& NeuronResourceManager::GetCompilationCache() {
  std::lock_guard<std::mutex> lock(mutex_);
  return *compilation_cache_;
}

ModelHandleCache& NeuronResourceManager::GetModelHandleCache() {
  std::lock_guard<std::mutex> lock(mutex_);
  return *model_handle_cache_;
}

OperationExecutionEngine& NeuronResourceManager::GetOperationExecutionEngine() {
  std::lock_guard<std::mutex> lock(mutex_);
  return *operation_execution_engine_;
}

ResourceSemaphore& NeuronResourceManager::GetResourceSemaphore() {
  std::lock_guard<std::mutex> lock(mutex_);
  return *resource_semaphore_;
}

int NeuronResourceManager::GetMaxResourceSlots() {
  const char* env_value = std::getenv("NEURON_MAX_RESOURCE_SLOTS");
  if (env_value) {
    try {
      return std::stoi(env_value);
    } catch (const std::exception& e) {
      TORCH_NEURONX_WARN("Invalid NEURON_MAX_RESOURCE_SLOTS value", "value=", env_value,
                         "error=", e.what(), "using_default=", kDefaultMaxResourceSlots);
    }
  }
  return kDefaultMaxResourceSlots;
}

NeuronResourceManager::NeuronResourceManager() {
  TORCH_NEURONX_DEBUG("Initializing NeuronResourceManager singleton");

  try {
    // Initialize compilation cache
    compilation_cache_ = std::make_unique<CompilationCache>();
    TORCH_NEURONX_DEBUG("CompilationCache initialized");

    // Initialize model handle cache
    model_handle_cache_ = std::make_unique<ModelHandleCache>();
    TORCH_NEURONX_DEBUG("ModelHandleCache initialized");

    // Initialize global execution engine
    operation_execution_engine_ = std::make_unique<OperationExecutionEngine>(
        compilation_cache_.get(), model_handle_cache_.get());
    TORCH_NEURONX_DEBUG("OperationExecutionEngine initialized");

    // Initialize resource semaphore for future thread pool usage
    int max_slots = GetMaxResourceSlots();
    resource_semaphore_ = std::make_unique<ResourceSemaphore>(max_slots);

  } catch (const std::exception& e) {
    TORCH_NEURONX_ERROR("Failed to initialize NeuronResourceManager", "error=", e.what());
    throw;
  }
}

NeuronResourceManager::~NeuronResourceManager() {
  // Shutdown execution engine first (stops worker threads)
  if (operation_execution_engine_) {
    operation_execution_engine_->Shutdown();
  }
  // Now safe to cleanup stream pools
  CleanupStreamPools();
}

c10_neuron::NrtTensorPool& NeuronResourceManager::GetTensorPool(int device) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (tensor_pools_.empty()) {
    int count = c10_neuron::device_count();
    if (count <= 0) {
      throw std::runtime_error("No Neuron devices available");
    }
    tensor_pools_.resize(count);
  }

  if (device < 0 || device >= static_cast<int>(tensor_pools_.size())) {
    throw std::runtime_error("Invalid device: " + std::to_string(device));
  }

  auto& pool = tensor_pools_[device];
  if (!pool) {
    pool = std::make_unique<c10_neuron::NrtTensorPool>(device);
  }

  return *pool;
}

bool ResourceSemaphore::TryAcquireImmediate() {
  // Precondition: mutex_ is held by calling thread
  if (count_ > 0) {
    --count_;
    ++acquired_count_;
    return true;
  }
  return false;
}

void ResourceSemaphore::AddToWaitingQueue(std::shared_ptr<WaitingRequest> request) {
  // Precondition: mutex_ is held by calling thread
  waiting_requests_.push_back(request);
  SortWaitingQueue();
}

void ResourceSemaphore::RemoveFromWaitingQueue(std::shared_ptr<WaitingRequest> request) {
  // Precondition: mutex_ is held by calling thread
  waiting_requests_.erase(std::remove(waiting_requests_.begin(), waiting_requests_.end(), request),
                          waiting_requests_.end());
}

void ResourceSemaphore::SortWaitingQueue() {
  // Precondition: mutex_ is held by calling thread
  std::sort(waiting_requests_.begin(), waiting_requests_.end(),
            [](const std::shared_ptr<WaitingRequest>& a, const std::shared_ptr<WaitingRequest>& b) {
              if (a->priority != b->priority) {
                return a->priority > b->priority;  // Higher priority first
              }
              return a->request_time < b->request_time;  // FIFO within same priority
            });
}

}  // namespace at::neuron

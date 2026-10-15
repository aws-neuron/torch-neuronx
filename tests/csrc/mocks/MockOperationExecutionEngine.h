#pragma once

#include <gmock/gmock.h>

#include <algorithm>
#include <list>
#include <memory>

#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"
#include "torch_neuronx/csrc/core/runtime/ModelHandleCache.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"

namespace at::neuron::testing {

// Mock OperationExecutionEngine that skips actual device execution
// This allows testing stream management without needing real device tensors or NRT.
class MockOperationExecutionEngine : public OperationExecutionEngine {
 public:
  MockOperationExecutionEngine(CompilationCache* compilation_cache,
                               ModelHandleCache* model_handle_cache)
      : OperationExecutionEngine(compilation_cache, model_handle_cache) {}

  using OperationExecutionEngine::DecrementExecutionQueueDepth;
  using OperationExecutionEngine::dispatch_mutex_;
  using OperationExecutionEngine::dispatch_queue_;
  using OperationExecutionEngine::DispatchAsyncExecutionTask;
  using OperationExecutionEngine::execution_queue_depth_;
  using OperationExecutionEngine::HandleCompilationComplete;
  using OperationExecutionEngine::max_inflight_per_xu_;
  using OperationExecutionEngine::MoveToRuntimeQueueInOrder;
  using OperationExecutionEngine::pending_async_ops_count_;
  using OperationExecutionEngine::ProcessCompletions;
  using OperationExecutionEngine::ProcessPrepareTask;
  using OperationExecutionEngine::RouteCompilationFailure;
  using OperationExecutionEngine::ScheduleOperation;
  using OperationExecutionEngine::SnapshotConcatenationQueueDepth;
  using OperationExecutionEngine::xu_inflight_counts_;

  // Test-only: push directly onto stream->runtime_queue_, bypassing the worker.
  void InjectIntoRuntimeQueue(StreamImpl* stream, std::shared_ptr<OperationContext> op) {
    TestInjectIntoRuntimeQueue(stream, std::move(op));
  }

  void ClearRuntimeQueue(StreamImpl* stream) { TestClearRuntimeQueue(stream); }

  size_t RuntimeQueueSize(StreamImpl* stream) const { return TestRuntimeQueueSize(stream); }

  // True iff the per-stream cursor has walked past the last queued op.
  bool RuntimeCursorAtEnd(StreamImpl* stream) const { return TestRuntimeCursorAtEnd(stream); }

  // True iff the cursor is currently pointing at the given op.
  bool RuntimeCursorAt(StreamImpl* stream, const OperationContext* op) const {
    return TestRuntimeCursorAt(stream, op);
  }

  // Test-only: position the cursor at the given op. Returns false if op is not
  // in the runtime queue.
  bool SetRuntimeCursorTo(StreamImpl* stream, const OperationContext* op) {
    return TestSetRuntimeCursorTo(stream, op);
  }

  // Front op of the runtime queue (or nullptr).
  OperationContext* RuntimeQueueFront(StreamImpl* stream) const {
    return TestRuntimeQueueFront(stream);
  }

  // Snapshot the queue as a vector of raw pointers (in order). Useful for
  // assertions on partition/cursor state.
  std::vector<OperationContext*> RuntimeQueueSnapshot(StreamImpl* stream) const {
    return TestRuntimeQueueSnapshot(stream);
  }
};

// RAII: swap in a MockOperationExecutionEngine as the global engine for the
// scope of a test. Some StreamImpl paths (e.g. CompleteOperation calling
// engine.DecrementExecutionQueueDepth) route through the global engine, so
// tests need to install the one they inspect.
class ScopedTestEngine {
 public:
  ScopedTestEngine(CompilationCache* compilation_cache, ModelHandleCache* model_handle_cache) {
    auto mock =
        std::make_unique<MockOperationExecutionEngine>(compilation_cache, model_handle_cache);
    mock_ptr_ = mock.get();
    previous_ = NeuronResourceManager::Instance().SetExecutionEngineForTesting(std::move(mock));
  }
  ~ScopedTestEngine() {
    auto installed =
        NeuronResourceManager::Instance().SetExecutionEngineForTesting(std::move(previous_));
    // Shut down the test engine before destroying it.
    if (installed) installed->Shutdown();
  }

  ScopedTestEngine(const ScopedTestEngine&) = delete;
  ScopedTestEngine& operator=(const ScopedTestEngine&) = delete;

  MockOperationExecutionEngine& engine() { return *mock_ptr_; }
  MockOperationExecutionEngine* operator->() { return mock_ptr_; }

 private:
  MockOperationExecutionEngine* mock_ptr_{nullptr};
  std::unique_ptr<OperationExecutionEngine> previous_;
};

// Wrapper for backward compatibility
class MockOperationExecutionEngineWrapper {
 public:
  MockOperationExecutionEngineWrapper()
      : compilation_cache_(),
        model_handle_cache_(),
        engine_(&compilation_cache_, &model_handle_cache_) {}

  ~MockOperationExecutionEngineWrapper() { engine_.Shutdown(); }

  OperationExecutionEngine* GetEngine() { return &engine_; }

  // Manually complete an operation (called by test after submission)
  void CompleteOperation(OperationContext* op, StreamImpl* stream) {
    if (op && stream) {
      stream->CompleteOperation(op);
    }
  }

 private:
  CompilationCache compilation_cache_;
  ModelHandleCache model_handle_cache_;
  MockOperationExecutionEngine engine_;
};

}  // namespace at::neuron::testing

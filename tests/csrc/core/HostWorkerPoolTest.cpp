#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"

using namespace at::neuron;
using namespace std::chrono_literals;

namespace {

// Build a kHost OperationContext running an arbitrary `void()` body, wrapped in
// the HostKernelExecution closure signature (unused refs ignored).
using Body = std::function<void()>;

std::unique_ptr<OperationContext> MakeHostOp(const std::string& name, Body body = {}) {
  HostKernelExecution::HostFn fn = [body = std::move(body)](const std::vector<TensorDataRef>&,
                                                            const std::vector<TensorDataRef>&) {
    if (body) body();
  };
  auto kernel = std::make_unique<HostKernelExecution>(
      name, /*src_refs=*/std::vector<TensorDataRef>{},
      /*dst_refs=*/std::vector<TensorDataRef>{}, std::move(fn), /*device_id=*/0);
  return std::make_unique<OperationContext>(std::move(kernel));
}

// Per-op "done" barrier on the pool's on_complete_ callback (fires once per op
// after RunHost + AdvanceHostCompletion). Register every op before submitting;
// the pool's on_complete_ lambda calls Complete(op).
class CompletionLatch {
 public:
  // Register an op so its future exists before the worker can complete it.
  void Register(OperationContext* op) {
    std::lock_guard<std::mutex> g(mu_);
    promises_[op] = std::make_shared<std::promise<void>>();
    futures_[op] = promises_[op]->get_future().share();
  }

  // Called from the pool's on_complete_ (possibly a worker thread). Idempotent:
  // a second completion for the same op (shutdown drain racing a worker) is
  // swallowed, matching the old promise_already_satisfied tolerance.
  void Complete(OperationContext* op) {
    std::shared_ptr<std::promise<void>> p;
    {
      std::lock_guard<std::mutex> g(mu_);
      auto it = promises_.find(op);
      if (it == promises_.end() || !it->second) return;
      p = std::move(it->second);
      it->second = nullptr;  // mark consumed
    }
    try {
      p->set_value();
    } catch (const std::future_error&) {
      // already satisfied — tolerate (idempotent)
    }
  }

  // Wait for a specific op's completion, failing the test on timeout.
  void ExpectDone(OperationContext* op, std::chrono::milliseconds timeout,
                  const std::string& label) {
    std::shared_future<void> f;
    {
      std::lock_guard<std::mutex> g(mu_);
      auto it = futures_.find(op);
      ASSERT_NE(it, futures_.end()) << "op '" << label << "' was not Register()ed";
      f = it->second;
    }
    ASSERT_EQ(f.wait_for(timeout), std::future_status::ready)
        << "op '" << label << "' did not complete within " << timeout.count() << "ms";
  }

  bool IsDone(OperationContext* op) {
    std::shared_future<void> f;
    {
      std::lock_guard<std::mutex> g(mu_);
      auto it = futures_.find(op);
      if (it == futures_.end()) return false;
      f = it->second;
    }
    return f.wait_for(0s) == std::future_status::ready;
  }

 private:
  std::mutex mu_;
  std::map<OperationContext*, std::shared_ptr<std::promise<void>>> promises_;
  std::map<OperationContext*, std::shared_future<void>> futures_;
};

}  // namespace

// ── Submit + run: success path ───────────────────────────────────────────────
TEST(HostWorkerPoolTest, SubmitAndRunSucceeds) {
  std::atomic<int> completion_count{0};
  std::atomic<bool> ran{false};
  CompletionLatch latch;

  HostWorkerPool pool(
      /*num_threads=*/1, [&](OperationContext* op, std::exception_ptr eptr) {
        EXPECT_NE(op, nullptr);
        EXPECT_EQ(eptr, nullptr);
        completion_count.fetch_add(1);
        latch.Complete(op);
      });

  auto op = MakeHostOp("ok", [&ran] { ran.store(true); });
  latch.Register(op.get());
  pool.Submit(op.get());

  latch.ExpectDone(op.get(), 2s, "ok");
  EXPECT_TRUE(ran.load());
  EXPECT_EQ(completion_count.load(), 1);
  pool.Shutdown();
}

// ── Submit + RunHost throws: error path ──────────────────────────────────────
TEST(HostWorkerPoolTest, RunHostThrowsRoutesEptrThroughOnComplete) {
  std::atomic<int> success_count{0};
  std::atomic<int> error_count{0};
  std::string captured_msg;
  std::mutex captured_msg_mutex;
  CompletionLatch latch;

  HostWorkerPool pool(
      /*num_threads=*/1, [&](OperationContext* op, std::exception_ptr eptr) {
        EXPECT_NE(op, nullptr);
        if (eptr) {
          try {
            std::rethrow_exception(eptr);
          } catch (const std::exception& e) {
            std::lock_guard<std::mutex> g(captured_msg_mutex);
            captured_msg = e.what();
          }
          error_count.fetch_add(1);
        } else {
          success_count.fetch_add(1);
        }
        latch.Complete(op);
      });

  auto op = MakeHostOp("boom", [] { throw std::runtime_error("kaboom"); });
  latch.Register(op.get());
  pool.Submit(op.get());

  // on_complete_ fires even on error — the error flows through eptr.
  latch.ExpectDone(op.get(), 2s, "boom");
  EXPECT_EQ(error_count.load(), 1);
  EXPECT_EQ(success_count.load(), 0);
  {
    std::lock_guard<std::mutex> g(captured_msg_mutex);
    EXPECT_NE(captured_msg.find("kaboom"), std::string::npos)
        << "error message was: '" << captured_msg << "'";
  }
  pool.Shutdown();
}

// ── FIFO ordering with N=1 worker: ops run in submission order ──────────────
TEST(HostWorkerPoolTest, SingleWorkerFifoOrder) {
  std::vector<int> run_order;
  std::mutex run_order_mutex;
  CompletionLatch latch;

  HostWorkerPool pool(
      /*num_threads=*/1, [&](OperationContext* op, std::exception_ptr eptr) {
        EXPECT_EQ(eptr, nullptr);
        latch.Complete(op);
      });

  constexpr int N = 8;
  std::vector<std::unique_ptr<OperationContext>> ops;
  ops.reserve(N);
  for (int i = 0; i < N; ++i) {
    ops.push_back(MakeHostOp("op" + std::to_string(i), [&run_order, &run_order_mutex, i] {
      std::lock_guard<std::mutex> g(run_order_mutex);
      run_order.push_back(i);
    }));
  }
  for (auto& op : ops) latch.Register(op.get());
  for (auto& op : ops) pool.Submit(op.get());

  for (auto& op : ops) {
    latch.ExpectDone(op.get(), 2s, op->GetOpName());
  }

  pool.Shutdown();

  std::lock_guard<std::mutex> g(run_order_mutex);
  ASSERT_EQ(run_order.size(), static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(run_order[i], i) << "ordering violated at idx " << i;
  }
}

// ── Multi-worker pool: all jobs complete (ordering is per-worker, not global) ─
// We only assert all jobs run and complete, not global ordering across workers.
TEST(HostWorkerPoolTest, MultiWorkerPoolDrainsAllJobs) {
  std::atomic<int> completion_count{0};
  CompletionLatch latch;
  HostWorkerPool pool(
      /*num_threads=*/4, [&](OperationContext* op, std::exception_ptr) {
        completion_count.fetch_add(1);
        latch.Complete(op);
      });

  constexpr int N = 64;
  std::vector<std::unique_ptr<OperationContext>> ops;
  ops.reserve(N);
  for (int i = 0; i < N; ++i) {
    ops.push_back(MakeHostOp("op" + std::to_string(i), [] {}));
  }
  for (auto& op : ops) latch.Register(op.get());
  for (auto& op : ops) pool.Submit(op.get());

  for (auto& op : ops) {
    latch.ExpectDone(op.get(), 5s, op->GetOpName());
  }
  EXPECT_EQ(completion_count.load(), N);
  EXPECT_EQ(pool.NumThreads(), 4u);
  pool.Shutdown();
}

// ── WorkerLoop stamps host_worker_idx + execute_start ───────────────────────
// Profiler depends on host_worker_idx >= 0 to route DRIVER activity to the
// right per-worker pftrace track. With a single-thread pool, we expect
// host_worker_idx = 0 deterministically.
TEST(HostWorkerPoolTest, StampsHostWorkerIdxAndExecuteStart) {
  CompletionLatch latch;
  HostWorkerPool pool(
      /*num_threads=*/1, [&](OperationContext* op, std::exception_ptr) { latch.Complete(op); });
  auto op = MakeHostOp("stamp", [] {});

  ASSERT_EQ(op->host_worker_idx, -1);
  ASSERT_EQ(op->execute_start.time_since_epoch().count(), 0);

  latch.Register(op.get());
  pool.Submit(op.get());
  latch.ExpectDone(op.get(), 2s, "stamp");

  // The worker writes host_worker_idx and execute_start before on_complete_
  // fires, so by the time the latch is ready, the stamps are visible.
  EXPECT_EQ(op->host_worker_idx, 0);
  EXPECT_GT(op->execute_start.time_since_epoch().count(), 0);
  // neuron_stream_os_tid is also stamped in the same site.
  EXPECT_GT(op->neuron_stream_os_tid, 0);
  pool.Shutdown();
}

// ── Multi-worker stamping uses distinct worker indices ──────────────────────
// With more workers than queued jobs serialised one-at-a-time, the indices
// stamped will vary; we just check the index is in [0, num_threads).
TEST(HostWorkerPoolTest, MultiWorkerStampedIndicesInRange) {
  constexpr size_t kThreads = 4;
  CompletionLatch latch;
  HostWorkerPool pool(kThreads,
                      [&](OperationContext* op, std::exception_ptr) { latch.Complete(op); });

  constexpr int N = 32;
  std::vector<std::unique_ptr<OperationContext>> ops;
  for (int i = 0; i < N; ++i) {
    ops.push_back(
        MakeHostOp("stamp" + std::to_string(i), [] { std::this_thread::sleep_for(1ms); }));
  }
  for (auto& op : ops) latch.Register(op.get());
  for (auto& op : ops) pool.Submit(op.get());

  for (auto& op : ops) {
    latch.ExpectDone(op.get(), 5s, op->GetOpName());
  }
  pool.Shutdown();

  for (auto& op : ops) {
    EXPECT_GE(op->host_worker_idx, 0);
    EXPECT_LT(op->host_worker_idx, static_cast<int32_t>(kThreads));
  }
}

// ── Submit after Shutdown is rejected; on_complete_ still fires ─────────────
// An op submitted post-shutdown must still fire on_complete_ (with error) and
// advance the seq (no-op here with null stream; real-stream unblock is in
// HostCCSchedulingTest), else a gated device op would hang.
TEST(HostWorkerPoolTest, SubmitAfterShutdownRejectedAndCompletes) {
  std::atomic<int> error_count{0};
  CompletionLatch latch;
  HostWorkerPool pool(
      /*num_threads=*/1, [&](OperationContext* op, std::exception_ptr eptr) {
        EXPECT_NE(eptr, nullptr);
        error_count.fetch_add(1);
        latch.Complete(op);
      });
  pool.Shutdown();

  auto op = MakeHostOp("post_shutdown", [] {});
  latch.Register(op.get());
  pool.Submit(op.get());

  // Reject path runs on_complete_ synchronously inside Submit, so it is already
  // done here, but assert via the latch for uniformity.
  latch.ExpectDone(op.get(), 2s, "post_shutdown");
  EXPECT_EQ(error_count.load(), 1);
}

// ── Pending jobs at shutdown: leftover loop runs on_complete ────────────────
// Pre-fill the queue with more work than the worker can drain immediately
// (the worker runs them serially); shutdown will join workers (which may run
// some) and then walk the leftover queue, firing on_complete_ on the rest.
// In all cases, on_complete_ fires exactly once per submitted op.
TEST(HostWorkerPoolTest, ShutdownDrainsPendingJobsAndCompletesAll) {
  std::atomic<int> success_count{0};
  std::atomic<int> error_count{0};
  CompletionLatch latch;

  // Holds the workers up so jobs queue.
  std::promise<void> gate_promise;
  std::shared_future<void> gate_future = gate_promise.get_future().share();

  auto pool = std::make_unique<HostWorkerPool>(
      /*num_threads=*/1, [&](OperationContext* op, std::exception_ptr eptr) {
        if (eptr) {
          error_count.fetch_add(1);
        } else {
          success_count.fetch_add(1);
        }
        latch.Complete(op);
      });

  constexpr int N = 16;
  std::vector<std::unique_ptr<OperationContext>> ops;
  for (int i = 0; i < N; ++i) {
    ops.push_back(MakeHostOp("op" + std::to_string(i),
                             // First op blocks until gate is released; subsequent
                             // ops are queued behind it.
                             [i, gate_future] {
                               if (i == 0) {
                                 gate_future.wait();
                               }
                             }));
  }
  for (auto& op : ops) latch.Register(op.get());
  for (auto& op : ops) pool->Submit(op.get());

  // Release the gate at the same time as initiating shutdown — the leftover
  // queue is highly likely to still have entries when the shutdown's
  // compare_exchange wins. The exact split is timing-dependent; we only
  // assert the totals.
  gate_promise.set_value();
  pool->Shutdown();

  for (auto& op : ops) {
    latch.ExpectDone(op.get(), 2s, op->GetOpName());
  }
  EXPECT_EQ(success_count.load() + error_count.load(), N);
}

// ── Idempotent shutdown ─────────────────────────────────────────────────────
TEST(HostWorkerPoolTest, ShutdownIsIdempotent) {
  HostWorkerPool pool(
      /*num_threads=*/2, [](OperationContext*, std::exception_ptr) {});
  EXPECT_NO_THROW(pool.Shutdown());
  EXPECT_NO_THROW(pool.Shutdown());
  EXPECT_NO_THROW(pool.Shutdown());
}

// ── Constructor argument validation ─────────────────────────────────────────
TEST(HostWorkerPoolTest, ConstructorRejectsZeroThreads) {
  EXPECT_THROW(HostWorkerPool(0, [](OperationContext*, std::exception_ptr) {}), std::exception);
}

TEST(HostWorkerPoolTest, ConstructorRejectsNullCallback) {
  EXPECT_THROW(HostWorkerPool(1, nullptr), std::exception);
}

// ── Submit(nullptr) is hard-rejected ────────────────────────────────────────
TEST(HostWorkerPoolTest, SubmitNullThrows) {
  HostWorkerPool pool(
      /*num_threads=*/1, [](OperationContext*, std::exception_ptr) {});
  EXPECT_THROW(pool.Submit(nullptr), std::exception);
  pool.Shutdown();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

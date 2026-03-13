#include "CorrelationTracker.h"

#include <vector>

#include "LogWrapper.h"

namespace at::neuron {

namespace {
// Stack of Kineto external correlation IDs for the current thread.
// Kineto's profiler pushes an ID before dispatching each CPU operator and
// pops it after. We mirror that stack here so that when the Neuron backend
// submits device work, it can snapshot the current correlation ID and later
// attach it to the resulting device-side trace activity. This links each
// device activity back to the CPU operator that launched it.
thread_local std::vector<uint64_t> tls_correlation_stack;

// Stack of user-supplied correlation IDs for the current thread.
// These originate from user annotations (e.g., record_function ranges)
// rather than Kineto's automatic per-operator IDs. They follow the same
// push/pop/current pattern and allow user-defined regions to be correlated
// with the device activities they produce.
thread_local std::vector<uint64_t> tls_user_correlation_stack;
}  // namespace

void CorrelationTracker::push(uint64_t id) { tls_correlation_stack.push_back(id); }

void CorrelationTracker::pop() {
  if (tls_correlation_stack.empty()) {
    NPROF_WARN("CorrelationTracker::pop() called on empty stack - mismatched push/pop?");
    return;
  }
  tls_correlation_stack.pop_back();
}

uint64_t CorrelationTracker::current() {
  if (tls_correlation_stack.empty()) {
    // 0 is a safe sentinel: PyTorch correlation IDs start at 1
    // (collection.cpp id_start_ = 1 + ...), and all consumers
    // (Session::processTrace, Kineto JSON output) treat 0 as "no correlation"
    // and skip linking. This is the normal case for operations submitted
    // outside a RecordFunction scope (e.g., background NRT work).
    // Push/pop mismatches are caught by pop() which warns on empty stack.
    return 0;
  }
  return tls_correlation_stack.back();
}

void CorrelationTracker::pushUser(uint64_t id) { tls_user_correlation_stack.push_back(id); }

void CorrelationTracker::popUser() {
  if (tls_user_correlation_stack.empty()) {
    NPROF_WARN("CorrelationTracker::popUser() called on empty stack - mismatched push/pop?");
    return;
  }
  tls_user_correlation_stack.pop_back();
}

uint64_t CorrelationTracker::currentUser() {
  if (tls_user_correlation_stack.empty()) {
    // 0 is a safe sentinel: PyTorch correlation IDs start at 1
    // (collection.cpp id_start_ = 1 + ...), and all consumers
    // (Session::processTrace, Kineto JSON output) treat 0 as "no correlation"
    // and skip linking. This is the normal case for operations submitted
    // outside a RecordFunction scope (e.g., background NRT work).
    // Push/pop mismatches are caught by popUser() which warns on empty stack.
    return 0;
  }
  return tls_user_correlation_stack.back();
}

}  // namespace at::neuron

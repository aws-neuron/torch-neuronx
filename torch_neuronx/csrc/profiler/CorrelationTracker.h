#pragma once

#include <cstdint>

namespace at::neuron {

// Thread-local stack of Kineto correlation IDs.
// Kineto pushes a correlation ID before each CPU op and pops it after.
// We capture the current ID at operation submission time to link
// device activities back to their originating CPU ops.
class CorrelationTracker {
 public:
  // Push a correlation ID onto the current thread's stack.
  static void push(uint64_t id);

  // Pop the top correlation ID from the current thread's stack.
  static void pop();

  // Return the top correlation ID, or 0 if the stack is empty.
  static uint64_t current();

  // Push a user correlation ID onto the current thread's stack.
  static void pushUser(uint64_t id);

  // Pop the top user correlation ID from the current thread's stack.
  static void popUser();

  // Return the top user correlation ID, or 0 if the stack is empty.
  static uint64_t currentUser();
};

}  // namespace at::neuron

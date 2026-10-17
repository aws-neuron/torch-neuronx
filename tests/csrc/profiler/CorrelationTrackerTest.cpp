#include <gtest/gtest.h>

#include <thread>

#include "torch_neuronx/csrc/profiler/CorrelationTracker.h"

namespace at::neuron {
namespace {

TEST(CorrelationTrackerTest, EmptyStackReturnsZero) { EXPECT_EQ(CorrelationTracker::current(), 0); }

TEST(CorrelationTrackerTest, PushAndCurrent) {
  CorrelationTracker::push(42);
  EXPECT_EQ(CorrelationTracker::current(), 42);
  CorrelationTracker::pop();
  EXPECT_EQ(CorrelationTracker::current(), 0);
}

TEST(CorrelationTrackerTest, NestedPushPop) {
  CorrelationTracker::push(1);
  CorrelationTracker::push(2);
  CorrelationTracker::push(3);

  EXPECT_EQ(CorrelationTracker::current(), 3);
  CorrelationTracker::pop();
  EXPECT_EQ(CorrelationTracker::current(), 2);
  CorrelationTracker::pop();
  EXPECT_EQ(CorrelationTracker::current(), 1);
  CorrelationTracker::pop();
  EXPECT_EQ(CorrelationTracker::current(), 0);
}

TEST(CorrelationTrackerTest, PopOnEmptyIsNoOp) {
  CorrelationTracker::pop();
  EXPECT_EQ(CorrelationTracker::current(), 0);
}

TEST(CorrelationTrackerTest, PerThreadIsolation) {
  // Push a value on the main thread
  CorrelationTracker::push(100);

  uint64_t other_thread_value = 999;
  std::thread t([&other_thread_value]() {
    // Other thread should see empty stack
    other_thread_value = CorrelationTracker::current();

    // Push on other thread should not affect main
    CorrelationTracker::push(200);
    CorrelationTracker::pop();
  });
  t.join();

  EXPECT_EQ(other_thread_value, 0);
  EXPECT_EQ(CorrelationTracker::current(), 100);

  CorrelationTracker::pop();
}

TEST(CorrelationTrackerTest, UserEmptyStackReturnsZero) {
  EXPECT_EQ(CorrelationTracker::currentUser(), 0);
}

TEST(CorrelationTrackerTest, UserPushAndCurrent) {
  CorrelationTracker::pushUser(42);
  EXPECT_EQ(CorrelationTracker::currentUser(), 42);
  CorrelationTracker::popUser();
  EXPECT_EQ(CorrelationTracker::currentUser(), 0);
}

TEST(CorrelationTrackerTest, UserNestedPushPop) {
  CorrelationTracker::pushUser(10);
  CorrelationTracker::pushUser(20);
  CorrelationTracker::pushUser(30);

  EXPECT_EQ(CorrelationTracker::currentUser(), 30);
  CorrelationTracker::popUser();
  EXPECT_EQ(CorrelationTracker::currentUser(), 20);
  CorrelationTracker::popUser();
  EXPECT_EQ(CorrelationTracker::currentUser(), 10);
  CorrelationTracker::popUser();
  EXPECT_EQ(CorrelationTracker::currentUser(), 0);
}

TEST(CorrelationTrackerTest, UserPopOnEmptyIsNoOp) {
  CorrelationTracker::popUser();
  EXPECT_EQ(CorrelationTracker::currentUser(), 0);
}

TEST(CorrelationTrackerTest, UserAndKinetoStacksAreIndependent) {
  CorrelationTracker::push(100);
  CorrelationTracker::pushUser(200);

  EXPECT_EQ(CorrelationTracker::current(), 100);
  EXPECT_EQ(CorrelationTracker::currentUser(), 200);

  CorrelationTracker::pop();
  EXPECT_EQ(CorrelationTracker::current(), 0);
  EXPECT_EQ(CorrelationTracker::currentUser(), 200);

  CorrelationTracker::popUser();
  EXPECT_EQ(CorrelationTracker::currentUser(), 0);
}

TEST(CorrelationTrackerTest, UserPerThreadIsolation) {
  CorrelationTracker::pushUser(100);

  uint64_t other_thread_value = 999;
  std::thread t([&other_thread_value]() {
    other_thread_value = CorrelationTracker::currentUser();
    CorrelationTracker::pushUser(200);
    CorrelationTracker::popUser();
  });
  t.join();

  EXPECT_EQ(other_thread_value, 0);
  EXPECT_EQ(CorrelationTracker::currentUser(), 100);

  CorrelationTracker::popUser();
}

}  // namespace
}  // namespace at::neuron

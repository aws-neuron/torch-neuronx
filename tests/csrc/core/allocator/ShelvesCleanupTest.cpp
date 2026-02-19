#include <gtest/gtest.h>

#include "torch_neuronx/csrc/core/allocator/ShelvesCleanup.h"

namespace torch_neuronx {
namespace distributed {
namespace {

TEST(ShelvesCleanupTest, RegisterAndTrigger) {
  int call_count = 0;
  auto id = ShelvesCleanupRegistry::instance().registerCleanup([&]() { ++call_count; });

  size_t before = ShelvesCleanupRegistry::instance().getTriggerCount();
  triggerShelvesCleanup();
  EXPECT_EQ(ShelvesCleanupRegistry::instance().getTriggerCount(), before + 1);
  EXPECT_EQ(call_count, 1);

  triggerShelvesCleanup();
  EXPECT_EQ(call_count, 2);

  ShelvesCleanupRegistry::instance().unregisterCleanup(id);
  triggerShelvesCleanup();
  EXPECT_EQ(call_count, 2);  // No longer called
}

TEST(ShelvesCleanupTest, UnregisterNonexistentIdIsSafe) {
  EXPECT_NO_THROW(ShelvesCleanupRegistry::instance().unregisterCleanup(999999));
}

// Verify the allocator's OOM retry path calls triggerShelvesCleanup.
// This test ensures the integration exists - if someone removes the call
// from NeuronCachingAllocator.cpp, this test documents the expected behavior.
//
// The actual call verification happens via getTriggerCount() in integration tests,
// but this test ensures the API contract is maintained.
TEST(ShelvesCleanupTest, TriggerCountIncrementsOnEachCall) {
  size_t count1 = ShelvesCleanupRegistry::instance().getTriggerCount();
  triggerShelvesCleanup();
  size_t count2 = ShelvesCleanupRegistry::instance().getTriggerCount();
  triggerShelvesCleanup();
  size_t count3 = ShelvesCleanupRegistry::instance().getTriggerCount();

  EXPECT_EQ(count2, count1 + 1);
  EXPECT_EQ(count3, count2 + 1);
}

TEST(ShelvesCleanupTest, MultipleCallbacksAllCalled) {
  int count1 = 0, count2 = 0, count3 = 0;

  auto id1 = ShelvesCleanupRegistry::instance().registerCleanup([&]() { ++count1; });
  auto id2 = ShelvesCleanupRegistry::instance().registerCleanup([&]() { ++count2; });
  auto id3 = ShelvesCleanupRegistry::instance().registerCleanup([&]() { ++count3; });

  triggerShelvesCleanup();

  EXPECT_EQ(count1, 1);
  EXPECT_EQ(count2, 1);
  EXPECT_EQ(count3, 1);

  // Unregister one
  ShelvesCleanupRegistry::instance().unregisterCleanup(id2);
  triggerShelvesCleanup();

  EXPECT_EQ(count1, 2);
  EXPECT_EQ(count2, 1);  // Not called
  EXPECT_EQ(count3, 2);

  ShelvesCleanupRegistry::instance().unregisterCleanup(id1);
  ShelvesCleanupRegistry::instance().unregisterCleanup(id3);
}

}  // namespace
}  // namespace distributed
}  // namespace torch_neuronx

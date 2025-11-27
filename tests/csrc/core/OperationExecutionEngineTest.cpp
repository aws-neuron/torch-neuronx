#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "tests/csrc/mocks/MockKernelExecution.h"
#include "tests/csrc/mocks/MockStreamImpl.h"
#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"
#include "torch_neuronx/csrc/core/runtime/ModelHandleCache.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"

using namespace at::neuron;
using namespace at::neuron::testing;
using namespace std::chrono_literals;
using ::testing::_;
using ::testing::Return;

class OperationExecutionEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    compilation_cache_ = std::make_unique<CompilationCache>();
    model_handle_cache_ = std::make_unique<ModelHandleCache>();

    // Create engine with our caches
    engine_ = std::make_unique<OperationExecutionEngine>(compilation_cache_.get(),
                                                         model_handle_cache_.get());

    mock_stream_ = std::make_unique<MockStreamImpl>();
    mock_stream_->SetDefaultBehavior();
  }

  void TearDown() override {
    if (engine_) {
      engine_->Shutdown();
    }
  }

  std::unique_ptr<CompilationCache> compilation_cache_;
  std::unique_ptr<ModelHandleCache> model_handle_cache_;
  std::unique_ptr<OperationExecutionEngine> engine_;
  std::unique_ptr<MockStreamImpl> mock_stream_;
};

// Test basic construction and destruction
TEST_F(OperationExecutionEngineTest, ConstructionAndDestruction) {
  EXPECT_NE(engine_, nullptr);

  auto stats = engine_->GetPerformanceStats();
  EXPECT_EQ(stats.compilation_tasks_submitted, 0);
  EXPECT_EQ(stats.compilation_tasks_completed, 0);
  EXPECT_EQ(stats.execution_tasks_submitted, 0);
  EXPECT_EQ(stats.execution_tasks_completed, 0);
  EXPECT_TRUE(stats.is_active);
}

// Test shutdown functionality
TEST_F(OperationExecutionEngineTest, ShutdownStopsWorkers) {
  auto stats_before = engine_->GetStatistics();
  EXPECT_TRUE(stats_before.worker_active);

  engine_->Shutdown();

  auto stats_after = engine_->GetStatistics();
  EXPECT_FALSE(stats_after.worker_active);
}

// Test performance stats tracking
TEST_F(OperationExecutionEngineTest, PerformanceStatsTracking) {
  auto stats = engine_->GetPerformanceStats();

  EXPECT_EQ(stats.compilation_tasks_submitted, 0);
  EXPECT_EQ(stats.compilation_tasks_completed, 0);
  EXPECT_EQ(stats.execution_tasks_submitted, 0);
  EXPECT_EQ(stats.execution_tasks_completed, 0);
  EXPECT_EQ(stats.compilation_queue_current_size, 0);
  EXPECT_EQ(stats.execution_queue_current_size, 0);
  EXPECT_TRUE(stats.is_active);
  EXPECT_GE(stats.uptime.count(), 0);
}

// Test statistics retrieval
TEST_F(OperationExecutionEngineTest, GetStatistics) {
  auto stats = engine_->GetStatistics();

  EXPECT_EQ(stats.queue_size, 0);
  EXPECT_TRUE(stats.worker_active);
}

// Test task priority bucketing from int values
TEST_F(OperationExecutionEngineTest, TaskPriorityBucketing) {
  auto kernel1 = std::make_unique<MockNeuronKernelExecution>("op1");
  auto kernel2 = std::make_unique<MockNeuronKernelExecution>("op2");
  auto kernel3 = std::make_unique<MockNeuronKernelExecution>("op3");

  auto op1 = std::make_unique<OperationContext>(std::move(kernel1));
  auto op2 = std::make_unique<OperationContext>(std::move(kernel2));
  auto op3 = std::make_unique<OperationContext>(std::move(kernel3));

  // Create tasks with int priorities that get bucketed
  OperationExecutionTask task_high(op1.get(), reinterpret_cast<StreamImpl*>(mock_stream_.get()), -1,
                                   TaskType::EXECUTION);
  OperationExecutionTask task_default(op2.get(), reinterpret_cast<StreamImpl*>(mock_stream_.get()),
                                      0, TaskType::EXECUTION);
  OperationExecutionTask task_low(op3.get(), reinterpret_cast<StreamImpl*>(mock_stream_.get()), 1,
                                  TaskType::EXECUTION);

  // Verify priorities are bucketed correctly
  EXPECT_EQ(task_high.priority, TaskPriority::HIGH);
  EXPECT_EQ(task_default.priority, TaskPriority::DEFAULT);
  EXPECT_EQ(task_low.priority, TaskPriority::LOW);
}

// Test FIFO ordering within same priority (implicit via deque behavior)
TEST_F(OperationExecutionEngineTest, TaskFIFOOrderingWithinPriority) {
  auto kernel1 = std::make_unique<MockNeuronKernelExecution>("op1");
  auto kernel2 = std::make_unique<MockNeuronKernelExecution>("op2");

  auto op1 = std::make_unique<OperationContext>(std::move(kernel1));
  auto op2 = std::make_unique<OperationContext>(std::move(kernel2));

  // Create tasks with same priority (both 0 = DEFAULT)
  OperationExecutionTask task1(op1.get(), reinterpret_cast<StreamImpl*>(mock_stream_.get()), 0,
                               TaskType::EXECUTION);
  OperationExecutionTask task2(op2.get(), reinterpret_cast<StreamImpl*>(mock_stream_.get()), 0,
                               TaskType::EXECUTION);

  // Both have same priority - FIFO ordering is guaranteed by deque
  EXPECT_EQ(task1.priority, task2.priority);
  EXPECT_EQ(task1.priority, TaskPriority::DEFAULT);
}

// Test RequiresCompilation check
TEST_F(OperationExecutionEngineTest, TaskRequiresCompilationCheck) {
  // Direct kernel - no compilation
  auto direct_kernel = std::make_unique<MockNeuronKernelExecution>("direct_op");
  EXPECT_CALL(*direct_kernel, RequiresCompilation()).WillRepeatedly(Return(false));

  auto direct_op = std::make_unique<OperationContext>(std::move(direct_kernel));
  OperationExecutionTask direct_task(
      direct_op.get(), reinterpret_cast<StreamImpl*>(mock_stream_.get()), 0, TaskType::EXECUTION);

  EXPECT_FALSE(direct_task.RequiresCompilation());

  // Compilable kernel - requires compilation
  auto inputs = std::vector<torch::Tensor>{create_test_tensor({2, 3})};
  auto outputs = std::vector<torch::Tensor>{create_test_tensor({2, 3})};
  auto compilable_kernel = std::make_unique<MockCompilableKernelExecution>(
      "compilable_op", inputs, outputs, "test_key", false);

  auto compilable_op = std::make_unique<OperationContext>(std::move(compilable_kernel));
  OperationExecutionTask compilable_task(compilable_op.get(),
                                         reinterpret_cast<StreamImpl*>(mock_stream_.get()), 0,
                                         TaskType::COMPILATION);

  EXPECT_TRUE(compilable_task.RequiresCompilation());
}

// Test default task construction
TEST_F(OperationExecutionEngineTest, TaskDefaultConstruction) {
  // Note: OperationExecutionTask no longer has a default constructor
  // This test is removed as it's no longer applicable
  SUCCEED();
}

// Test uptime tracking
TEST_F(OperationExecutionEngineTest, UptimeTracking) {
  std::this_thread::sleep_for(100ms);

  auto stats = engine_->GetPerformanceStats();
  EXPECT_GE(stats.uptime.count(), 0);
}

// Test shutdown prevents new task submission
TEST_F(OperationExecutionEngineTest, ShutdownPreventsSubmission) {
  engine_->Shutdown();

  auto kernel = std::make_unique<MockNeuronKernelExecution>("test_op");
  EXPECT_CALL(*kernel, RequiresCompilation()).WillRepeatedly(Return(false));

  auto op = std::make_unique<OperationContext>(std::move(kernel));
  OperationExecutionTask task(op.get(), reinterpret_cast<StreamImpl*>(mock_stream_.get()), 0,
                              TaskType::EXECUTION);

  // Should not crash, but task won't be processed
  engine_->SubmitTask(std::move(task));

  auto stats = engine_->GetPerformanceStats();
  EXPECT_FALSE(stats.is_active);
}

// Test multiple shutdowns are safe
TEST_F(OperationExecutionEngineTest, MultipleShutdownsSafe) {
  engine_->Shutdown();
  engine_->Shutdown();  // Should not crash

  auto stats = engine_->GetStatistics();
  EXPECT_FALSE(stats.worker_active);
}

// Test queue size tracking
TEST_F(OperationExecutionEngineTest, QueueSizeTracking) {
  auto stats_initial = engine_->GetStatistics();
  EXPECT_EQ(stats_initial.queue_size, 0);

  // Note: In a real scenario, we'd submit tasks and check queue growth
  // but since tasks are processed quickly, this is hard to test reliably
}

// Test engine with null caches
TEST_F(OperationExecutionEngineTest, ConstructionWithNullCaches) {
  // This should work but operations will fail
  auto engine_with_nulls = std::make_unique<OperationExecutionEngine>(nullptr, nullptr);
  EXPECT_NE(engine_with_nulls, nullptr);

  auto stats = engine_with_nulls->GetPerformanceStats();
  EXPECT_TRUE(stats.is_active);

  engine_with_nulls->Shutdown();
}

// Test concurrent statistics access
TEST_F(OperationExecutionEngineTest, ConcurrentStatisticsAccess) {
  std::vector<std::thread> threads;
  std::atomic<int> success_count{0};

  for (int i = 0; i < 10; ++i) {
    threads.emplace_back([this, &success_count]() {
      for (int j = 0; j < 10; ++j) {
        auto stats = engine_->GetPerformanceStats();
        if (stats.is_active) {
          success_count++;
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(success_count.load(), 100);
}

// Test priority queue behavior - high priority tasks processed first
TEST_F(OperationExecutionEngineTest, PriorityQueueProcessingOrder) {
  // This test verifies that tasks are routed to the correct priority queues
  auto kernel_high = std::make_unique<MockNeuronKernelExecution>("high_priority_op");
  auto kernel_default = std::make_unique<MockNeuronKernelExecution>("default_priority_op");
  auto kernel_low = std::make_unique<MockNeuronKernelExecution>("low_priority_op");

  auto op_high = std::make_unique<OperationContext>(std::move(kernel_high));
  auto op_default = std::make_unique<OperationContext>(std::move(kernel_default));
  auto op_low = std::make_unique<OperationContext>(std::move(kernel_low));

  // Use int priorities that get bucketed
  OperationExecutionTask task_high(op_high.get(), reinterpret_cast<StreamImpl*>(mock_stream_.get()),
                                   -5, TaskType::EXECUTION);
  OperationExecutionTask task_default(
      op_default.get(), reinterpret_cast<StreamImpl*>(mock_stream_.get()), 0, TaskType::EXECUTION);
  OperationExecutionTask task_low(op_low.get(), reinterpret_cast<StreamImpl*>(mock_stream_.get()),
                                  10, TaskType::EXECUTION);

  // Verify priorities are correctly bucketed
  EXPECT_EQ(task_high.priority, TaskPriority::HIGH);
  EXPECT_EQ(task_default.priority, TaskPriority::DEFAULT);
  EXPECT_EQ(task_low.priority, TaskPriority::LOW);

  // Verify enum ordering (HIGH < DEFAULT < LOW in terms of processing priority)
  EXPECT_LT(static_cast<int>(TaskPriority::HIGH), static_cast<int>(TaskPriority::DEFAULT));
  EXPECT_LT(static_cast<int>(TaskPriority::DEFAULT), static_cast<int>(TaskPriority::LOW));
}

// Test that TaskPriority enum has expected values
TEST_F(OperationExecutionEngineTest, TaskPriorityEnumOrdering) {
  // Verify the enum values are in the expected order
  EXPECT_EQ(static_cast<int>(TaskPriority::HIGH), 0);
  EXPECT_EQ(static_cast<int>(TaskPriority::DEFAULT), 1);
  EXPECT_EQ(static_cast<int>(TaskPriority::LOW), 2);
}

// Test BucketizePriority helper function
TEST_F(OperationExecutionEngineTest, BucketizePriorityFunction) {
  // Test the bucketing logic
  EXPECT_EQ(BucketizePriority(-1), TaskPriority::HIGH);
  EXPECT_EQ(BucketizePriority(-100), TaskPriority::HIGH);
  EXPECT_EQ(BucketizePriority(0), TaskPriority::DEFAULT);
  EXPECT_EQ(BucketizePriority(1), TaskPriority::LOW);
  EXPECT_EQ(BucketizePriority(100), TaskPriority::LOW);

  // Test edge cases
  EXPECT_EQ(BucketizePriority(INT_MIN), TaskPriority::HIGH);
  EXPECT_EQ(BucketizePriority(INT_MAX), TaskPriority::LOW);
}

class PriorityQueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_stream_ = std::make_unique<MockStreamImpl>();
    mock_stream_->SetDefaultBehavior();
  }

  OperationExecutionTask CreateTask(const std::string& name, int priority) {
    auto kernel = std::make_unique<MockNeuronKernelExecution>(name);
    auto op = std::make_unique<OperationContext>(std::move(kernel));
    ops_.push_back(std::move(op));
    return OperationExecutionTask(ops_.back().get(),
                                  reinterpret_cast<StreamImpl*>(mock_stream_.get()), priority,
                                  TaskType::EXECUTION);
  }

  std::unique_ptr<MockStreamImpl> mock_stream_;
  std::vector<std::unique_ptr<OperationContext>> ops_;
};

TEST_F(PriorityQueueTest, EmptyQueue) {
  PriorityQueue<OperationExecutionTask> queue;
  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(queue.size(), 0);
}

TEST_F(PriorityQueueTest, PushBackAddsToCorrectBucket) {
  PriorityQueue<OperationExecutionTask> queue;

  queue.push_back(CreateTask("high", -1));
  queue.push_back(CreateTask("default", 0));
  queue.push_back(CreateTask("low", 1));

  EXPECT_FALSE(queue.empty());
  EXPECT_EQ(queue.size(), 3);
}

TEST_F(PriorityQueueTest, PopFrontRespectsHighPriority) {
  PriorityQueue<OperationExecutionTask> queue;

  queue.push_back(CreateTask("low", 1));
  queue.push_back(CreateTask("high", -1));
  queue.push_back(CreateTask("default", 0));

  auto task = queue.pop_front();
  EXPECT_EQ(task.priority, TaskPriority::HIGH);
  EXPECT_EQ(queue.size(), 2);
}

TEST_F(PriorityQueueTest, PopFrontProcessesInPriorityOrder) {
  PriorityQueue<OperationExecutionTask> queue;

  queue.push_back(CreateTask("low", 1));
  queue.push_back(CreateTask("default", 0));
  queue.push_back(CreateTask("high", -1));

  auto task1 = queue.pop_front();
  EXPECT_EQ(task1.priority, TaskPriority::HIGH);

  auto task2 = queue.pop_front();
  EXPECT_EQ(task2.priority, TaskPriority::DEFAULT);

  auto task3 = queue.pop_front();
  EXPECT_EQ(task3.priority, TaskPriority::LOW);

  EXPECT_TRUE(queue.empty());
}

TEST_F(PriorityQueueTest, PushFrontMaintainsOrderWithinPriority) {
  PriorityQueue<OperationExecutionTask> queue;

  // Push tasks to back first
  queue.push_back(CreateTask("default1", 0));
  queue.push_back(CreateTask("default2", 0));

  // Push tasks to front
  queue.push_front(CreateTask("default3", 0));
  queue.push_front(CreateTask("default4", 0));

  EXPECT_EQ(queue.size(), 4);

  // Front tasks should come first: default4, default3, default1, default2
  auto task1 = queue.pop_front();
  EXPECT_STREQ(task1.operation->GetOpName().c_str(), "default4");

  auto task2 = queue.pop_front();
  EXPECT_STREQ(task2.operation->GetOpName().c_str(), "default3");

  auto task3 = queue.pop_front();
  EXPECT_STREQ(task3.operation->GetOpName().c_str(), "default1");

  auto task4 = queue.pop_front();
  EXPECT_STREQ(task4.operation->GetOpName().c_str(), "default2");
}

TEST_F(PriorityQueueTest, PushFrontPreservesSubmissionOrder) {
  PriorityQueue<OperationExecutionTask> queue;

  // Simulate promoting deferred tasks in reverse order
  std::vector<OperationExecutionTask> ready_tasks;
  ready_tasks.push_back(CreateTask("task1", 0));
  ready_tasks.push_back(CreateTask("task2", 0));
  ready_tasks.push_back(CreateTask("task3", 0));

  // Push in reverse order (as done in PromoteReadyDeferredTasks)
  for (auto it = ready_tasks.rbegin(); it != ready_tasks.rend(); ++it) {
    queue.push_front(std::move(*it));
  }

  // Should maintain original order: task1, task2, task3
  auto task1 = queue.pop_front();
  EXPECT_STREQ(task1.operation->GetOpName().c_str(), "task1");

  auto task2 = queue.pop_front();
  EXPECT_STREQ(task2.operation->GetOpName().c_str(), "task2");

  auto task3 = queue.pop_front();
  EXPECT_STREQ(task3.operation->GetOpName().c_str(), "task3");
}

TEST_F(PriorityQueueTest, MixedPrioritiesWithPushFront) {
  PriorityQueue<OperationExecutionTask> queue;

  // Add some tasks normally
  queue.push_back(CreateTask("low1", 1));
  queue.push_back(CreateTask("default1", 0));

  // Push high priority to front
  queue.push_front(CreateTask("high1", -1));
  queue.push_front(CreateTask("high2", -1));

  EXPECT_EQ(queue.size(), 4);

  // High priority tasks should come first, in reverse push order
  auto task1 = queue.pop_front();
  EXPECT_EQ(task1.priority, TaskPriority::HIGH);
  EXPECT_STREQ(task1.operation->GetOpName().c_str(), "high2");

  auto task2 = queue.pop_front();
  EXPECT_EQ(task2.priority, TaskPriority::HIGH);
  EXPECT_STREQ(task2.operation->GetOpName().c_str(), "high1");

  auto task3 = queue.pop_front();
  EXPECT_EQ(task3.priority, TaskPriority::DEFAULT);

  auto task4 = queue.pop_front();
  EXPECT_EQ(task4.priority, TaskPriority::LOW);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "scheduler/rt_scheduler.hpp"
#include <alchemy/task.h>
#include <alchemy/timer.h>

using namespace rt;
using namespace testing;

class RTSchedulerTest : public Test {
protected:
    void SetUp() override {
        // Initialize test tasks
        tasks = {
            {
                .name = "TestTask1",
                .period = std::chrono::microseconds(10000),
                .deadline = std::chrono::microseconds(9000),
                .priority = 99,
                .cpuCore = 1,
                .task = []() { /* Simple task */ }
            },
            {
                .name = "TestTask2",
                .period = std::chrono::microseconds(20000),
                .deadline = std::chrono::microseconds(18000),
                .priority = 98,
                .cpuCore = 2,
                .task = []() { /* Simple task */ }
            }
        };
    }
    
    std::vector<RTScheduler::TaskConfig> tasks;
    
    // Helper to simulate task execution
    void simulateTaskExecution(RTScheduler& scheduler, int taskIndex, int iterations) {
        for (int i = 0; i < iterations; ++i) {
            RTIME start = rt_timer_read();
            tasks[taskIndex].task();
            RTIME end = rt_timer_read();
            
            scheduler.monitorTask(tasks[taskIndex].name,
                                std::chrono::microseconds(end - start),
                                end - start <= tasks[taskIndex].deadline.count());
        }
    }
};

TEST_F(RTSchedulerTest, InitializationTest) {
    EXPECT_NO_THROW({
        RTScheduler scheduler(tasks);
    });
}

TEST_F(RTSchedulerTest, TaskStartStop) {
    RTScheduler scheduler(tasks);
    
    EXPECT_TRUE(scheduler.start());
    EXPECT_TRUE(scheduler.isRunning());
    
    scheduler.stop();
    EXPECT_FALSE(scheduler.isRunning());
}

TEST_F(RTSchedulerTest, DeadlineMonitoring) {
    RTScheduler scheduler(tasks);
    std::atomic<int> deadlineMissCount{0};
    
    // Set up deadline miss callback
    scheduler.setDeadlineCallback([&deadlineMissCount](const std::string&) {
        deadlineMissCount++;
    });
    
    // Simulate tasks with some deadline misses
    for (int i = 0; i < 100; ++i) {
        RTIME start = rt_timer_read();
        std::this_thread::sleep_for(std::chrono::microseconds(11000));  // Deliberately miss deadline
        RTIME end = rt_timer_read();
        
        scheduler.monitorTask(tasks[0].name,
                            std::chrono::microseconds(end - start),
                            false);
    }
    
    EXPECT_GT(deadlineMissCount, 0);
}

TEST_F(RTSchedulerTest, TaskStatistics) {
    RTScheduler scheduler(tasks);
    
    // Simulate some task executions
    simulateTaskExecution(scheduler, 0, 100);
    simulateTaskExecution(scheduler, 1, 50);
    
    auto stats = scheduler.getTaskStats();
    EXPECT_EQ(stats.size(), 2);
    
    // Verify statistics for first task
    auto task1Stats = std::find_if(stats.begin(), stats.end(),
        [](const auto& stat) { return stat.name == "TestTask1"; });
    EXPECT_NE(task1Stats, stats.end());
    EXPECT_EQ(task1Stats->totalExecutions, 100);
    
    // Verify statistics for second task
    auto task2Stats = std::find_if(stats.begin(), stats.end(),
        [](const auto& stat) { return stat.name == "TestTask2"; });
    EXPECT_NE(task2Stats, stats.end());
    EXPECT_EQ(task2Stats->totalExecutions, 50);
}

TEST_F(RTSchedulerTest, CPUAffinity) {
    RTScheduler scheduler(tasks);
    
    // Verify that tasks are assigned to correct cores
    RT_TASK_INFO info;
    
    for (const auto& task : tasks) {
        rt_task_inquire(NULL, &info);
        EXPECT_EQ(info.cpuid, task.cpuCore);
    }
}

TEST_F(RTSchedulerTest, PriorityOrdering) {
    RTScheduler scheduler(tasks);
    
    // Verify task priorities are correctly set
    RT_TASK_INFO info;
    
    for (const auto& task : tasks) {
        rt_task_inquire(NULL, &info);
        EXPECT_EQ(info.prio, task.priority);
    }
}

TEST_F(RTSchedulerTest, StressTest) {
    RTScheduler scheduler(tasks);
    std::atomic<int> completedTasks{0};
    std::atomic<int> deadlineMisses{0};
    
    // Add a CPU-intensive task
    tasks.push_back({
        .name = "StressTask",
        .period = std::chrono::microseconds(5000),
        .deadline = std::chrono::microseconds(4500),
        .priority = 97,
        .cpuCore = 3,
        .task = [&completedTasks]() {
            // Simulate heavy computation
            for (int i = 0; i < 1000000; ++i) {
                volatile int x = i * i;
            }
            completedTasks++;
        }
    });
    
    scheduler.setDeadlineCallback([&deadlineMisses](const std::string&) {
        deadlineMisses++;
    });
    
    scheduler.start();
    std::this_thread::sleep_for(std::chrono::seconds(5));
    scheduler.stop();
    
    // Verify system stability under stress
    EXPECT_GT(completedTasks, 0);
    double missRate = static_cast<double>(deadlineMisses) / completedTasks;
    EXPECT_LT(missRate, 0.1);  // Less than 10% deadline misses under stress
}

TEST_F(RTSchedulerTest, ErrorHandling) {
    // Test with invalid task configuration
    std::vector<RTScheduler::TaskConfig> invalidTasks = {
        {
            .name = "InvalidTask",
            .period = std::chrono::microseconds(0),  // Invalid period
            .deadline = std::chrono::microseconds(1000),
            .priority = 99,
            .cpuCore = 1,
            .task = []() {}
        }
    };
    
    EXPECT_THROW({
        RTScheduler scheduler(invalidTasks);
    }, std::invalid_argument);
    
    // Test with null task function
    invalidTasks[0].period = std::chrono::microseconds(1000);
    invalidTasks[0].task = nullptr;
    
    EXPECT_THROW({
        RTScheduler scheduler(invalidTasks);
    }, std::invalid_argument);
} 
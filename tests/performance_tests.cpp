#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "utils/performance_monitor.hpp"
#include <thread>
#include <chrono>

using namespace rt;
using namespace testing;

class PerformanceMonitorTest : public Test {
protected:
    void SetUp() override {
        monitor = std::make_unique<PerformanceMonitor>();
    }
    
    std::unique_ptr<PerformanceMonitor> monitor;
    
    // Helper to simulate task execution with known duration
    void simulateTask(const std::string& taskName, 
                     std::chrono::microseconds duration,
                     int iterations) {
        for (int i = 0; i < iterations; ++i) {
            auto start = monitor->startMeasurement(taskName);
            std::this_thread::sleep_for(duration);
            monitor->endMeasurement(taskName, start);
        }
    }
};

TEST_F(PerformanceMonitorTest, BasicMeasurement) {
    const std::string taskName = "TestTask";
    const auto duration = std::chrono::microseconds(1000);
    
    auto start = monitor->startMeasurement(taskName);
    std::this_thread::sleep_for(duration);
    auto result = monitor->endMeasurement(taskName, start);
    
    EXPECT_GE(result.executionTime, duration);
}

TEST_F(PerformanceMonitorTest, AverageExecutionTime) {
    const std::string taskName = "TestTask";
    const auto duration = std::chrono::microseconds(1000);
    const int iterations = 100;
    
    simulateTask(taskName, duration, iterations);
    
    auto stats = monitor->getTaskStats(taskName);
    EXPECT_GE(stats.averageExecutionTime, duration.count());
    EXPECT_EQ(stats.totalExecutions, iterations);
}

TEST_F(PerformanceMonitorTest, MultipleTaskTracking) {
    const std::vector<std::string> taskNames = {"Task1", "Task2", "Task3"};
    const auto duration = std::chrono::microseconds(1000);
    const int iterations = 50;
    
    for (const auto& task : taskNames) {
        simulateTask(task, duration, iterations);
    }
    
    auto allStats = monitor->getAllTaskStats();
    EXPECT_EQ(allStats.size(), taskNames.size());
    
    for (const auto& task : taskNames) {
        EXPECT_TRUE(monitor->hasTask(task));
        auto stats = monitor->getTaskStats(task);
        EXPECT_EQ(stats.totalExecutions, iterations);
    }
}

TEST_F(PerformanceMonitorTest, DeadlineTracking) {
    const std::string taskName = "DeadlineTask";
    const auto deadline = std::chrono::microseconds(1000);
    const auto duration = std::chrono::microseconds(2000);  // Deliberately miss deadline
    const int iterations = 10;
    
    for (int i = 0; i < iterations; ++i) {
        auto start = monitor->startMeasurement(taskName);
        std::this_thread::sleep_for(duration);
        monitor->endMeasurement(taskName, start, deadline);
    }
    
    auto stats = monitor->getTaskStats(taskName);
    EXPECT_GT(stats.missedDeadlines, 0);
    EXPECT_LT(stats.deadlineMeetRate, 1.0);
}

TEST_F(PerformanceMonitorTest, JitterCalculation) {
    const std::string taskName = "JitterTask";
    const auto baseTime = std::chrono::microseconds(1000);
    const int iterations = 100;
    
    // Simulate varying execution times
    for (int i = 0; i < iterations; ++i) {
        auto start = monitor->startMeasurement(taskName);
        // Add some random variation to execution time
        std::this_thread::sleep_for(baseTime + 
            std::chrono::microseconds(rand() % 500));
        monitor->endMeasurement(taskName, start);
    }
    
    auto stats = monitor->getTaskStats(taskName);
    EXPECT_GT(stats.jitter, 0.0);
}

TEST_F(PerformanceMonitorTest, ThreadSafety) {
    const std::string taskName = "ThreadSafetyTask";
    const int numThreads = 10;
    const int iterationsPerThread = 100;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, taskName, iterationsPerThread]() {
            simulateTask(taskName, std::chrono::microseconds(100), 
                        iterationsPerThread);
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto stats = monitor->getTaskStats(taskName);
    EXPECT_EQ(stats.totalExecutions, numThreads * iterationsPerThread);
}

TEST_F(PerformanceMonitorTest, HistogramGeneration) {
    const std::string taskName = "HistogramTask";
    const int iterations = 1000;
    
    // Generate a distribution of execution times
    for (int i = 0; i < iterations; ++i) {
        auto start = monitor->startMeasurement(taskName);
        std::this_thread::sleep_for(std::chrono::microseconds(
            500 + (rand() % 1000)));
        monitor->endMeasurement(taskName, start);
    }
    
    auto histogram = monitor->getExecutionTimeHistogram(taskName);
    EXPECT_FALSE(histogram.empty());
    
    // Verify histogram properties
    int totalSamples = 0;
    for (const auto& bin : histogram) {
        totalSamples += bin.second;
    }
    EXPECT_EQ(totalSamples, iterations);
}

TEST_F(PerformanceMonitorTest, ResetStatistics) {
    const std::string taskName = "ResetTask";
    simulateTask(taskName, std::chrono::microseconds(1000), 100);
    
    auto beforeReset = monitor->getTaskStats(taskName);
    EXPECT_GT(beforeReset.totalExecutions, 0);
    
    monitor->resetStatistics(taskName);
    
    auto afterReset = monitor->getTaskStats(taskName);
    EXPECT_EQ(afterReset.totalExecutions, 0);
    EXPECT_EQ(afterReset.missedDeadlines, 0);
    EXPECT_DOUBLE_EQ(afterReset.averageExecutionTime, 0.0);
}

TEST_F(PerformanceMonitorTest, ErrorHandling) {
    const std::string taskName = "ErrorTask";
    
    // Test invalid task name
    EXPECT_THROW({
        monitor->getTaskStats("NonexistentTask");
    }, std::out_of_range);
    
    // Test invalid measurement end (no corresponding start)
    EXPECT_THROW({
        monitor->endMeasurement(taskName, 
            std::chrono::high_resolution_clock::now());
    }, std::runtime_error);
} 
#pragma once

#include <chrono>
#include <functional>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "../utils/performance_monitor.hpp"
#include "../utils/logger.hpp"

namespace rt {

class RTScheduler {
public:
    struct TaskConfig {
        std::string name;
        std::chrono::microseconds period;
        std::chrono::microseconds deadline;
        int priority;
        int cpuCore;
        std::function<void()> task;
    };

    struct TaskStats {
        std::string name;
        uint64_t totalExecutions{0};
        uint64_t missedDeadlines{0};
        double averageExecutionTime{0.0};
        double maxExecutionTime{0.0};
        double jitter{0.0};
    };

    explicit RTScheduler(std::vector<TaskConfig> tasks);
    ~RTScheduler();

    void start();
    void stop();
    
    // Monitoring and statistics
    std::vector<TaskStats> getTaskStats() const;
    void setDeadlineCallback(std::function<void(const std::string&)> callback);
    bool isRunning() const { return running_; }

private:
    void taskWrapper(const TaskConfig& config);
    void monitorTask(const std::string& taskName,
                    const std::chrono::microseconds& executionTime,
                    bool deadlineMissed);
    void setCPUAffinity(int cpuCore);

    std::vector<TaskConfig> tasks_;
    std::vector<std::unique_ptr<std::thread>> taskThreads_;
    std::atomic<bool> running_{false};
    
    // Statistics tracking
    mutable std::mutex statsMutex_;
    std::unordered_map<std::string, TaskStats> taskStats_;
    
    std::function<void(const std::string&)> deadlineCallback_;
    
    PerformanceMonitor perfMonitor_;
    Logger logger_;
}; 
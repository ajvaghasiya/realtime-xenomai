#include <iostream>
#include <signal.h>
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <alchemy/task.h>
#include <alchemy/timer.h>
#include <alchemy/mutex.h>
#include <alchemy/sem.h>
#include "camera/stereo_capture.hpp"
#include "detection/yolo_detector.hpp"
#include "scheduler/rt_scheduler.hpp"
#include "utils/performance_monitor.hpp"

namespace {
    volatile std::sig_atomic_t gSignalStatus;
    RT_MUTEX frameMutex;
    RT_SEM frameSync;
    RT_SEM preprocessSync;
    RT_SEM detectionSync;
    
    // Shared buffers
    cv::Mat mergedFrame;
    cv::Mat preprocessedFrame;
    std::vector<rt::YOLODetector::DetectionResult> detectionResults;
    
    // Timing constants (in nanoseconds)
    const RTIME CYCLE_TIME_NS = 660000000;  // 0.66 seconds total cycle
    const RTIME CAPTURE_PERIOD_NS = 110000000;  // ~0.11s per capture (1/9 of cycle)
    const RTIME PREPROCESS_PERIOD_NS = 110000000;  // ~0.11s for preprocessing
    const RTIME DETECTION_PERIOD_NS = 220000000;  // ~0.22s for detection
    const RTIME MONITOR_PERIOD_NS = 110000000;  // ~0.11s for monitoring
    const RTIME DISPLAY_PERIOD_NS = 110000000;  // ~0.11s for display
}

void signal_handler(int signal) {
    gSignalStatus = signal;
}

// Xenomai task wrapper
struct XenomaiTask {
    RT_TASK task;
    bool running;
    RTIME lastWakeupTime;
    RTIME period;
    int cpuCore;
    
    void init(const char* name, int prio, int cpu) {
        cpuCore = cpu;
        int ret = rt_task_create(&task, name, 0, prio, T_JOINABLE);
        if (ret < 0) {
            throw std::runtime_error(fmt::format("Failed to create task {}: {}", name, ret));
        }
        
        ret = rt_task_set_affinity(&task, CPU_MASK_CPU(cpu));
        if (ret < 0) {
            throw std::runtime_error(fmt::format("Failed to set CPU affinity for task {}: {}", name, ret));
        }
    }
    
    void start(void (*entry)(void *), void* cookie) {
        running = true;
        int ret = rt_task_start(&task, entry, cookie);
        if (ret < 0) {
            throw std::runtime_error(fmt::format("Failed to start task: {}", ret));
        }
    }
    
    void stop() {
        running = false;
        rt_task_join(&task);
        rt_task_delete(&task);
    }
    
    bool checkDeadline(RTIME now) {
        RTIME deadline = lastWakeupTime + period;
        return now <= deadline;
    }
};

// Task entry points
void leftCameraTask(void* cookie) {
    auto* system = static_cast<rt::StereoCaptureSystem*>(cookie);
    RT_TASK_INFO info;
    rt_task_inquire(NULL, &info);
    
    rt_task_set_periodic(NULL, TM_NOW, CAPTURE_PERIOD_NS);
    spdlog::info("Started left camera task on CPU {}", info.cpuid);
    
    while (!gSignalStatus) {
        rt_task_wait_period(NULL);
        RTIME start = rt_timer_read();
        
        cv::Mat leftFrame;
        if (system->captureLeftFrame(leftFrame)) {
            rt_mutex_acquire(&frameMutex, TM_INFINITE);
            system->updateMergedView(leftFrame, true);
            rt_mutex_release(&frameMutex);
        }
        
        RTIME end = rt_timer_read();
        if (end - start > CAPTURE_PERIOD_NS) {
            spdlog::warn("Left camera capture missed deadline");
        }
    }
}

void rightCameraTask(void* cookie) {
    auto* system = static_cast<rt::StereoCaptureSystem*>(cookie);
    RT_TASK_INFO info;
    rt_task_inquire(NULL, &info);
    
    rt_task_set_periodic(NULL, TM_NOW, CAPTURE_PERIOD_NS);
    spdlog::info("Started right camera task on CPU {}", info.cpuid);
    
    while (!gSignalStatus) {
        rt_task_wait_period(NULL);
        RTIME start = rt_timer_read();
        
        cv::Mat rightFrame;
        if (system->captureRightFrame(rightFrame)) {
            rt_mutex_acquire(&frameMutex, TM_INFINITE);
            system->updateMergedView(rightFrame, false);
            rt_mutex_release(&frameMutex);
            rt_sem_broadcast(&preprocessSync);
        }
        
        RTIME end = rt_timer_read();
        if (end - start > CAPTURE_PERIOD_NS) {
            spdlog::warn("Right camera capture missed deadline");
        }
    }
}

void preprocessTask(void* cookie) {
    RT_TASK_INFO info;
    rt_task_inquire(NULL, &info);
    
    rt_task_set_periodic(NULL, TM_NOW, PREPROCESS_PERIOD_NS);
    spdlog::info("Started preprocess task on CPU {}", info.cpuid);
    
    while (!gSignalStatus) {
        rt_task_wait_period(NULL);
        RTIME start = rt_timer_read();
        
        if (rt_sem_p(&preprocessSync, TM_INFINITE) == 0) {
            rt_mutex_acquire(&frameMutex, TM_INFINITE);
            cv::Mat localFrame = mergedFrame.clone();
            rt_mutex_release(&frameMutex);
            
            // Preprocess frame
            cv::Mat processed;
            cv::resize(localFrame, processed, cv::Size(416, 416));
            cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
            processed.convertTo(processed, CV_32F, 1.0/255);
            
            rt_mutex_acquire(&frameMutex, TM_INFINITE);
            preprocessedFrame = processed;
            rt_mutex_release(&frameMutex);
            rt_sem_broadcast(&detectionSync);
        }
        
        RTIME end = rt_timer_read();
        if (end - start > PREPROCESS_PERIOD_NS) {
            spdlog::warn("Preprocess task missed deadline");
        }
    }
}

void detectionTask(void* cookie) {
    auto* detector = static_cast<rt::YOLODetector*>(cookie);
    RT_TASK_INFO info;
    rt_task_inquire(NULL, &info);
    
    rt_task_set_periodic(NULL, TM_NOW, DETECTION_PERIOD_NS);
    spdlog::info("Started detection task on CPU {}", info.cpuid);
    
    while (!gSignalStatus) {
        rt_task_wait_period(NULL);
        RTIME start = rt_timer_read();
        
        if (rt_sem_p(&detectionSync, TM_INFINITE) == 0) {
            rt_mutex_acquire(&frameMutex, TM_INFINITE);
            cv::Mat frameCopy = preprocessedFrame.clone();
            rt_mutex_release(&frameMutex);
            
            auto results = detector->detect(frameCopy);
            
            rt_mutex_acquire(&frameMutex, TM_INFINITE);
            detectionResults = results;
            rt_mutex_release(&frameMutex);
        }
        
        RTIME end = rt_timer_read();
        if (end - start > DETECTION_PERIOD_NS) {
            spdlog::warn("Detection task missed deadline");
        }
    }
}

void monitorTask(void* cookie) {
    RT_TASK_INFO info;
    rt_task_inquire(NULL, &info);
    
    rt_task_set_periodic(NULL, TM_NOW, MONITOR_PERIOD_NS);
    spdlog::info("Started monitor task on CPU {}", info.cpuid);
    
    uint64_t totalCycles = 0;
    uint64_t missedDeadlines = 0;
    
    while (!gSignalStatus) {
        rt_task_wait_period(NULL);
        RTIME start = rt_timer_read();
        
        totalCycles++;
        
        // Check if we're meeting the overall cycle time
        RTIME cycleTime = rt_timer_read() % CYCLE_TIME_NS;
        if (cycleTime > CYCLE_TIME_NS) {
            missedDeadlines++;
            spdlog::warn("System cycle missed deadline: {:.2f}ms", 
                        cycleTime/1000000.0);
        }
        
        // Log statistics
        if (totalCycles % 100 == 0) {
            double missRate = (double)missedDeadlines / totalCycles * 100.0;
            spdlog::info("Performance: Cycles={}, Missed={}, Rate={:.2f}%",
                        totalCycles, missedDeadlines, missRate);
        }
        
        RTIME end = rt_timer_read();
        if (end - start > MONITOR_PERIOD_NS) {
            spdlog::warn("Monitor task missed deadline");
        }
    }
}

void displayTask(void* cookie) {
    RT_TASK_INFO info;
    rt_task_inquire(NULL, &info);
    
    rt_task_set_periodic(NULL, TM_NOW, DISPLAY_PERIOD_NS);
    spdlog::info("Started display task on CPU {}", info.cpuid);
    
    while (!gSignalStatus) {
        rt_task_wait_period(NULL);
        
        rt_mutex_acquire(&frameMutex, TM_INFINITE);
        auto localResults = detectionResults;
        rt_mutex_release(&frameMutex);
        
        // Display results in terminal
        std::cout << "\033[2J\033[1;1H";  // Clear screen
        std::cout << "Detection Results:\n";
        std::cout << "================\n";
        for (const auto& det : localResults) {
            std::cout << fmt::format("Object: {}, Confidence: {:.2f}, Box: ({}, {}, {}, {})\n",
                det.className, det.confidence,
                det.box.x, det.box.y, det.box.width, det.box.height);
        }
    }
}

int main() {
    // Initialize Xenomai real-time services
    rt_print_auto_init(1);
    
    // Initialize synchronization primitives
    rt_mutex_create(&frameMutex, "FrameMutex");
    rt_sem_create(&frameSync, "FrameSync", 0, S_PRIO);
    rt_sem_create(&preprocessSync, "PreprocessSync", 0, S_PRIO);
    rt_sem_create(&detectionSync, "DetectionSync", 0, S_PRIO);
    
    // Create RT tasks
    RT_TASK t1, t2, t3, t4, t5, t6;
    
    try {
        // Create and configure tasks
        rt_task_create(&t1, "LeftCamera", 0, 99, T_JOINABLE);
        rt_task_create(&t2, "RightCamera", 0, 99, T_JOINABLE);
        rt_task_create(&t3, "Preprocess", 0, 98, T_JOINABLE);
        rt_task_create(&t4, "Detection", 0, 97, T_JOINABLE);
        rt_task_create(&t5, "Monitor", 0, 96, T_JOINABLE);
        rt_task_create(&t6, "Display", 0, 95, T_JOINABLE);
        
        // Set CPU affinity
        rt_task_set_affinity(&t1, CPU_MASK_CPU(2));  // Core 2
        rt_task_set_affinity(&t2, CPU_MASK_CPU(3));  // Core 3
        rt_task_set_affinity(&t3, CPU_MASK_CPU(1));  // Core 1
        rt_task_set_affinity(&t4, CPU_MASK_CPU(3));  // Core 3
        // t5 and t6 can run on any core
        
        // Initialize camera system and detector
        rt::StereoCaptureSystem stereoSystem(/* config */);
        rt::YOLODetector detector(/* config */);
        
        // Start tasks
        rt_task_start(&t1, &leftCameraTask, &stereoSystem);
        rt_task_start(&t2, &rightCameraTask, &stereoSystem);
        rt_task_start(&t3, &preprocessTask, nullptr);
        rt_task_start(&t4, &detectionTask, &detector);
        rt_task_start(&t5, &monitorTask, nullptr);
        rt_task_start(&t6, &displayTask, nullptr);
        
        // Wait for termination signal
        pause();
        
        // Cleanup
        gSignalStatus = 1;
        rt_task_join(&t1);
        rt_task_join(&t2);
        rt_task_join(&t3);
        rt_task_join(&t4);
        rt_task_join(&t5);
        rt_task_join(&t6);
        
        rt_mutex_delete(&frameMutex);
        rt_sem_delete(&frameSync);
        rt_sem_delete(&preprocessSync);
        rt_sem_delete(&detectionSync);
        
    } catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        return 1;
    }
    
    return 0;
} 
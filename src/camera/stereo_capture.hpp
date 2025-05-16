#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <alchemy/task.h>
#include "../utils/logger.hpp"
#include "../utils/performance_monitor.hpp"

namespace rt {

// Forward declaration
struct XenomaiTask;

class StereoCaptureSystem {
public:
    struct CameraConfig {
        int deviceId;
        int width;
        int height;
        int fps;
        int cpuCore;
    };

    StereoCaptureSystem(const CameraConfig& leftConfig, 
                       const CameraConfig& rightConfig);
    ~StereoCaptureSystem();

    // Camera operations
    bool captureLeftFrame(cv::Mat& frame);
    bool captureRightFrame(cv::Mat& frame);
    void updateMergedView(const cv::Mat& frame, bool isLeft);
    cv::Mat getMergedFrame() const;
    void stop();

    // Xenomai tasks
    XenomaiTask leftTask;
    XenomaiTask rightTask;

private:
    void captureThread(const CameraConfig& config, bool isLeft);
    void setCPUAffinity(int cpuCore);
    void monitorPerformance(const std::string& cameraId);

    std::unique_ptr<cv::VideoCapture> leftCam_;
    std::unique_ptr<cv::VideoCapture> rightCam_;
    
    cv::Mat mergedFrame_;  // Stores the side-by-side view
    
    CameraConfig leftConfig_;
    CameraConfig rightConfig_;
    
    PerformanceMonitor perfMonitor_;
    Logger logger_;
}; 
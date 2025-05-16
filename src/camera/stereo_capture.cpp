#include "stereo_capture.hpp"
#include <spdlog/spdlog.h>

namespace rt {

StereoCaptureSystem::StereoCaptureSystem(const CameraConfig& leftConfig, 
                                       const CameraConfig& rightConfig)
    : leftConfig_(leftConfig)
    , rightConfig_(rightConfig)
    , leftCam_(std::make_unique<cv::VideoCapture>())
    , rightCam_(std::make_unique<cv::VideoCapture>()) {
    
    // Open cameras
    if (!leftCam_->open(leftConfig.deviceId)) {
        throw std::runtime_error("Failed to open left camera");
    }
    
    if (!rightCam_->open(rightConfig.deviceId)) {
        throw std::runtime_error("Failed to open right camera");
    }
    
    // Configure cameras
    leftCam_->set(cv::CAP_PROP_FRAME_WIDTH, leftConfig.width);
    leftCam_->set(cv::CAP_PROP_FRAME_HEIGHT, leftConfig.height);
    leftCam_->set(cv::CAP_PROP_FPS, leftConfig.fps);
    
    rightCam_->set(cv::CAP_PROP_FRAME_WIDTH, rightConfig.width);
    rightCam_->set(cv::CAP_PROP_FRAME_HEIGHT, rightConfig.height);
    rightCam_->set(cv::CAP_PROP_FPS, rightConfig.fps);
    
    // Initialize merged frame
    mergedFrame_ = cv::Mat(leftConfig.height, leftConfig.width * 2, CV_8UC3);
}

StereoCaptureSystem::~StereoCaptureSystem() {
    stop();
}

bool StereoCaptureSystem::captureLeftFrame(cv::Mat& frame) {
    if (!leftCam_->read(frame)) {
        spdlog::error("Failed to capture left frame");
        return false;
    }
    return true;
}

bool StereoCaptureSystem::captureRightFrame(cv::Mat& frame) {
    if (!rightCam_->read(frame)) {
        spdlog::error("Failed to capture right frame");
        return false;
    }
    return true;
}

void StereoCaptureSystem::updateMergedView(const cv::Mat& frame, bool isLeft) {
    cv::Rect roi;
    if (isLeft) {
        roi = cv::Rect(0, 0, frame.cols, frame.rows);
    } else {
        roi = cv::Rect(frame.cols, 0, frame.cols, frame.rows);
    }
    
    // Copy frame to the appropriate side of the merged view
    frame.copyTo(mergedFrame_(roi));
    
    // Optional: Add vertical separator line
    cv::line(mergedFrame_, 
             cv::Point(frame.cols, 0),
             cv::Point(frame.cols, frame.rows),
             cv::Scalar(0, 255, 0), 2);
    
    // Optional: Add labels
    std::string label = isLeft ? "Left Camera" : "Right Camera";
    cv::putText(mergedFrame_, label,
                cv::Point(roi.x + 10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(0, 255, 0), 2);
}

cv::Mat StereoCaptureSystem::getMergedFrame() const {
    return mergedFrame_.clone();
}

void StereoCaptureSystem::stop() {
    if (leftCam_) {
        leftCam_->release();
    }
    if (rightCam_) {
        rightCam_->release();
    }
}

} // namespace rt 
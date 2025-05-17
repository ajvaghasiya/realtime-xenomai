#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "camera/stereo_capture.hpp"

using namespace rt;
using namespace testing;

class MockVideoCapture : public cv::VideoCapture {
public:
    MOCK_METHOD(bool, open, (int index), (override));
    MOCK_METHOD(bool, read, (cv::Mat& image), (override));
    MOCK_METHOD(bool, set, (int propId, double value), (override));
};

class StereoCameraTest : public Test {
protected:
    void SetUp() override {
        leftConfig = StereoCaptureSystem::CameraConfig{
            .deviceId = 0,
            .width = 640,
            .height = 480,
            .fps = 30,
            .cpuCore = 2
        };
        
        rightConfig = StereoCaptureSystem::CameraConfig{
            .deviceId = 2,
            .width = 640,
            .height = 480,
            .fps = 30,
            .cpuCore = 3
        };
    }
    
    StereoCaptureSystem::CameraConfig leftConfig;
    StereoCaptureSystem::CameraConfig rightConfig;
};

TEST_F(StereoCameraTest, InitializationSuccess) {
    EXPECT_NO_THROW({
        StereoCaptureSystem system(leftConfig, rightConfig);
    });
}

TEST_F(StereoCameraTest, FrameCaptureAndMerge) {
    StereoCaptureSystem system(leftConfig, rightConfig);
    
    cv::Mat leftFrame(480, 640, CV_8UC3, cv::Scalar(255, 0, 0));  // Blue
    cv::Mat rightFrame(480, 640, CV_8UC3, cv::Scalar(0, 255, 0)); // Green
    
    // Capture frames
    EXPECT_TRUE(system.captureLeftFrame(leftFrame));
    EXPECT_TRUE(system.captureRightFrame(rightFrame));
    
    // Update merged view
    system.updateMergedView(leftFrame, true);
    system.updateMergedView(rightFrame, false);
    
    // Get merged result
    cv::Mat merged = system.getMergedFrame();
    
    // Verify dimensions
    EXPECT_EQ(merged.rows, 480);
    EXPECT_EQ(merged.cols, 1280);  // 2 * 640
    
    // Check colors in left and right regions
    cv::Vec3b leftColor = merged.at<cv::Vec3b>(240, 320);   // Center of left image
    cv::Vec3b rightColor = merged.at<cv::Vec3b>(240, 960);  // Center of right image
    
    EXPECT_EQ(leftColor[0], 255);   // Blue in left
    EXPECT_EQ(rightColor[1], 255);  // Green in right
}

TEST_F(StereoCameraTest, ErrorHandling) {
    StereoCaptureSystem system(leftConfig, rightConfig);
    cv::Mat frame;
    
    // Test with invalid frame
    frame = cv::Mat();
    EXPECT_FALSE(system.updateMergedView(frame, true));
    
    // Test with mismatched dimensions
    frame = cv::Mat(100, 100, CV_8UC3);
    EXPECT_FALSE(system.updateMergedView(frame, true));
}

TEST_F(StereoCameraTest, ThreadSafety) {
    StereoCaptureSystem system(leftConfig, rightConfig);
    
    // Simulate concurrent access
    std::vector<std::thread> threads;
    const int numThreads = 10;
    std::atomic<int> successCount{0};
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&system, &successCount]() {
            cv::Mat frame(480, 640, CV_8UC3);
            if (system.captureLeftFrame(frame)) {
                system.updateMergedView(frame, true);
                successCount++;
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Check if all operations completed successfully
    EXPECT_GT(successCount, 0);
}

TEST_F(StereoCameraTest, PerformanceTest) {
    StereoCaptureSystem system(leftConfig, rightConfig);
    cv::Mat frame(480, 640, CV_8UC3);
    
    // Measure frame capture time
    auto start = std::chrono::high_resolution_clock::now();
    
    const int numFrames = 100;
    int successfulCaptures = 0;
    
    for (int i = 0; i < numFrames; ++i) {
        if (system.captureLeftFrame(frame)) {
            successfulCaptures++;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Check performance metrics
    double fps = 1000.0 * successfulCaptures / duration.count();
    EXPECT_GT(fps, 25.0);  // Should achieve at least 25 FPS
    EXPECT_GT(successfulCaptures, numFrames * 0.9);  // 90% success rate
} 
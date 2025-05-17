#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "detection/yolo_detector.hpp"

using namespace rt;
using namespace testing;

class YOLODetectorTest : public Test {
protected:
    void SetUp() override {
        config = YOLODetector::Config{
            .modelPath = "models/yolov4-tiny.weights",
            .configPath = "models/yolov4-tiny.cfg",
            .classesPath = "models/coco.names",
            .confThreshold = 0.5f,
            .nmsThreshold = 0.4f,
            .inputWidth = 416,
            .inputHeight = 416,
            .useGPU = false
        };
    }
    
    YOLODetector::Config config;
    
    // Helper to create a test image with a simple pattern
    cv::Mat createTestImage() {
        cv::Mat img(416, 416, CV_8UC3, cv::Scalar(0, 0, 0));
        // Draw a rectangle that resembles a person
        cv::rectangle(img, cv::Rect(100, 50, 200, 300), cv::Scalar(255, 255, 255), -1);
        return img;
    }
};

TEST_F(YOLODetectorTest, InitializationTest) {
    EXPECT_NO_THROW({
        YOLODetector detector(config);
    });
}

TEST_F(YOLODetectorTest, WarmupTest) {
    YOLODetector detector(config);
    EXPECT_NO_THROW({
        detector.warmup();
    });
}

TEST_F(YOLODetectorTest, DetectionWithEmptyImage) {
    YOLODetector detector(config);
    cv::Mat emptyImg;
    
    EXPECT_THROW({
        detector.detect(emptyImg);
    }, std::runtime_error);
}

TEST_F(YOLODetectorTest, DetectionWithValidImage) {
    YOLODetector detector(config);
    cv::Mat testImg = createTestImage();
    
    auto results = detector.detect(testImg);
    EXPECT_FALSE(results.empty());
    
    // Verify detection results format
    for (const auto& det : results) {
        EXPECT_GE(det.confidence, config.confThreshold);
        EXPECT_GT(det.box.width, 0);
        EXPECT_GT(det.box.height, 0);
        EXPECT_FALSE(det.className.empty());
    }
}

TEST_F(YOLODetectorTest, PerformanceTest) {
    YOLODetector detector(config);
    cv::Mat testImg = createTestImage();
    
    const int numIterations = 10;
    std::vector<double> inferenceTimings;
    
    for (int i = 0; i < numIterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        detector.detect(testImg);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        inferenceTimings.push_back(duration.count());
    }
    
    // Calculate average inference time
    double avgTime = std::accumulate(inferenceTimings.begin(), inferenceTimings.end(), 0.0) / numIterations;
    
    // Check if inference time is within acceptable range (adjust threshold as needed)
    EXPECT_LT(avgTime, 100.0);  // Should be less than 100ms per frame
}

TEST_F(YOLODetectorTest, ConfigValidation) {
    // Test with invalid confidence threshold
    YOLODetector::Config invalidConfig = config;
    invalidConfig.confThreshold = 1.5f;  // Should be between 0 and 1
    
    EXPECT_THROW({
        YOLODetector detector(invalidConfig);
    }, std::invalid_argument);
    
    // Test with invalid NMS threshold
    invalidConfig = config;
    invalidConfig.nmsThreshold = -0.1f;  // Should be between 0 and 1
    
    EXPECT_THROW({
        YOLODetector detector(invalidConfig);
    }, std::invalid_argument);
}

TEST_F(YOLODetectorTest, ThreadSafetyTest) {
    YOLODetector detector(config);
    cv::Mat testImg = createTestImage();
    
    std::vector<std::thread> threads;
    std::atomic<int> successCount{0};
    const int numThreads = 4;
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&detector, &testImg, &successCount]() {
            try {
                auto results = detector.detect(testImg);
                if (!results.empty()) {
                    successCount++;
                }
            } catch (...) {
                // Count failed
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(successCount, numThreads);
}

TEST_F(YOLODetectorTest, PreprocessingTest) {
    YOLODetector detector(config);
    
    // Test with different input image sizes
    std::vector<cv::Size> testSizes = {
        cv::Size(640, 480),
        cv::Size(1920, 1080),
        cv::Size(320, 240)
    };
    
    for (const auto& size : testSizes) {
        cv::Mat testImg(size, CV_8UC3);
        auto results = detector.detect(testImg);
        
        // Verify that detection still works with different input sizes
        EXPECT_NO_THROW({
            detector.detect(testImg);
        });
    }
} 
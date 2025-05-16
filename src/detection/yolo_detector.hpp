#pragma once

#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <memory>
#include "../utils/performance_monitor.hpp"

namespace rt {

class YOLODetector {
public:
    struct DetectionResult {
        int classId;
        float confidence;
        cv::Rect box;
        std::string className;
    };

    struct Config {
        std::string modelPath;
        std::string configPath;
        std::string classesPath;
        float confThreshold;
        float nmsThreshold;
        int inputWidth;
        int inputHeight;
        bool useGPU;
    };

    explicit YOLODetector(const Config& config);

    std::vector<DetectionResult> detect(const cv::Mat& frame);
    void warmup();  // Run inference on dummy data to initialize
    
    // Performance metrics
    double getInferenceTime() const;
    double getPreprocessTime() const;
    double getPostprocessTime() const;

private:
    cv::Mat preprocess(const cv::Mat& frame);
    std::vector<DetectionResult> postprocess(
        const cv::Mat& frame,
        const std::vector<cv::Mat>& outs
    );

    cv::dnn::Net net_;
    std::vector<std::string> classes_;
    Config config_;
    
    std::vector<std::string> getOutputsNames();
    void drawPredictions(cv::Mat& frame, 
                        const std::vector<DetectionResult>& results);

    PerformanceMonitor perfMonitor_;
    
    // Cache for performance
    std::vector<cv::Mat> outs_;
    std::vector<std::string> outLayerNames_;
}; 
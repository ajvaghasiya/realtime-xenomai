# Real-time Stereo Object Detection System

A high-performance C++ application demonstrating real-time object detection using stereo cameras with YOLOv4-Tiny, featuring sophisticated task scheduling and performance monitoring.

## Key Features

- **Real-time Performance**
  - Hard real-time task scheduling with deadline monitoring
  - CPU core affinity for predictable timing
  - Performance statistics and monitoring
  - Deadline violation detection and handling

- **Advanced Vision Processing**
  - Stereo camera synchronization
  - YOLOv4-Tiny object detection
  - Efficient frame preprocessing
  - GPU acceleration support (optional)

- **Modern C++ Design**
  - RAII-based resource management
  - Lock-free data structures where possible
  - Exception-safe design
  - Modern CMake build system

- **Monitoring and Debugging**
  - Comprehensive performance metrics
  - Real-time statistics logging
  - Task execution timing analysis
  - Deadline violation tracking

## System Architecture

### Thread Structure
```
T1 Left Camera Capture  | Priority: 99 | Core 2 | Period: 33.3ms
T2 Right Camera Capture | Priority: 99 | Core 3 | Period: 33.3ms
T3 Object Detection     | Priority: 97 | Core 3 | Period: 100ms
```

### Components

1. **StereoCaptureSystem**
   - Manages synchronized stereo camera capture
   - Thread-safe frame buffer management
   - Camera configuration and control

2. **YOLODetector**
   - YOLOv4-Tiny inference engine
   - Frame preprocessing and post-processing
   - Detection result management

3. **RTScheduler**
   - Real-time task scheduling
   - Deadline monitoring
   - Performance statistics collection

4. **PerformanceMonitor**
   - Execution time tracking
   - Deadline violation detection
   - Statistical analysis

## Building the Project

### Prerequisites
- CMake 3.12 or higher
- C++17 compliant compiler
- OpenCV 4.x
- spdlog
- fmt

### Build Instructions
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

1. Configure camera settings in `main.cpp`
2. Place YOLO model files in the `models` directory:
   - `yolov4-tiny.weights`
   - `yolov4-tiny.cfg`
   - `coco.names`
3. Run the application:
   ```bash
   ./realtime_object_detection
   ```

## Performance Optimization

The system is optimized for real-time performance through:

1. **Memory Management**
   - Pre-allocated buffers
   - Zero-copy data passing where possible
   - Cache-friendly data structures

2. **Thread Management**
   - CPU core affinity
   - Priority-based scheduling
   - Minimal context switching

3. **Real-time Considerations**
   - Deadline monitoring
   - Predictable execution times
   - Resource isolation

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
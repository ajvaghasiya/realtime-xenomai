add_library(rt_detection_lib
    camera/stereo_capture.cpp
    detection/yolo_detector.cpp
    processing/frame_processor.cpp
    scheduler/rt_scheduler.cpp
    utils/performance_monitor.cpp
    utils/logger.cpp
)

target_include_directories(rt_detection_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(rt_detection_lib
    PRIVATE
        ${OpenCV_LIBS}
        spdlog::spdlog
        fmt::fmt
) 
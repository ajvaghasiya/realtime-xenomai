# Download and configure Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Add test executables
add_executable(rt_system_tests
    camera_tests.cpp
    detector_tests.cpp
    scheduler_tests.cpp
    performance_tests.cpp
)

target_link_libraries(rt_system_tests
    PRIVATE
        rt_detection_lib
        GTest::gtest_main
        GTest::gmock_main
        ${OpenCV_LIBS}
        spdlog::spdlog
        fmt::fmt
)

# Include test directories
target_include_directories(rt_system_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

# Register tests
include(GoogleTest)
gtest_discover_tests(rt_system_tests) 
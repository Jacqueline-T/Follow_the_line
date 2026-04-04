cmake_minimum_required(VERSION 3.8)
project(lane_detection)

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
set(ncnn_DIR "/home/dalw/ros2_ws/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)
include_directories(/home/dalw/ros2_ws/ncnn/build/install/include)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(lane_detection src/main.cpp)
target_link_libraries(lane_detection ${OpenCV_LIBS} ncnn)
install(TARGETS lane_detection
  DESTINATION lib/${PROJECT_NAME})

ament_package()
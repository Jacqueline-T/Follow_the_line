cmake_minimum_required(VERSION 3.8)
project(lane_detection)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)

set(ncnn_DIR "/home/dalw/ros2_ws/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

include_directories(/home/dalw/ros2_ws/ncnn/build/install/include)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cal_node src/calibracion.cpp)
target_link_libraries(cal_node ${OpenCV_LIBS} ncnn)
ament_target_dependencies(cal_node rclcpp)

add_executable(lane_node src/lane_detect.cpp)
target_link_libraries(lane_node ${OpenCV_LIBS} ncnn)
ament_target_dependencies(lane_node rclcpp std_msgs geometry_msgs)

add_executable(sign_node src/sign_detect.cpp)
target_link_libraries(sign_node ${OpenCV_LIBS} ncnn)
ament_target_dependencies(sign_node rclcpp std_msgs)

add_executable(visual_node src/visual_tmr.cpp)
target_link_libraries(visual_node ${OpenCV_LIBS} ncnn)
ament_target_dependencies(visual_node rclcpp std_msgs geometry_msgs)

install(TARGETS
  cal_node
  lane_node
  sign_node
  visual_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
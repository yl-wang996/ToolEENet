cmake_minimum_required(VERSION 3.0.2)
project(dataset_generation)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  message_generation
  rospy
  sensor_msgs
  std_msgs
)

generate_messages(
DEPENDENCIES
  std_msgs
)

catkin_package(
CATKIN_DEPENDS
  message_runtime
  sensor_msgs
  std_msgs
)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
PROGRAMS
DESTINATION
  ${CATKIN_PACKAGE_BIN_DESTINATION}
)
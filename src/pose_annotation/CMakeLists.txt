cmake_minimum_required(VERSION 3.0.2)
project(pose_annotation)

find_package(catkin REQUIRED COMPONENTS
    std_msgs
    message_generation
    interactive_markers
    rospy
    visualization_msgs
    tf
)

generate_messages(
DEPENDENCIES
  std_msgs
)
###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  CATKIN_DEPENDS interactive_markers rospy visualization_msgs tf
)

###########
## Build ##
###########

include_directories(include
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

catkin_install_python(PROGRAMS
  scripts/basic_controls.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
cmake_minimum_required(VERSION 3.0.2)
project(tool_ee)

find_package(catkin REQUIRED COMPONENTS
    std_msgs
    message_generation
    rospy
    visualization_msgs
    tf
    geometry_msgs
    sensor_msgs
)

#add_message_files(FILES
#  )

## Generate services in the 'srv' folder
add_service_files(FILES
    PosePred.srv
    SegPred.srv
 )

generate_messages(DEPENDENCIES
    std_msgs
    geometry_msgs
    sensor_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(CATKIN_DEPENDS
    rospy
    visualization_msgs
    tf
    geometry_msgs
    sensor_msgs
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
  scripts/pose_pred_service.py
  scripts/seg_pred_service.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
cmake_minimum_required(VERSION 3.0.2)
project(sigma50_driver)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package()

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

## Install Python nodes
catkin_install_python(
  PROGRAMS
    scripts/sigma50_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

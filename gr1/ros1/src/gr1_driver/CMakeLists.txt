cmake_minimum_required(VERSION 3.0.2)
project(gr1_driver)

## Find catkin and required components
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  #  INCLUDE_DIRS include
  #  LIBRARIES gr1_driver
  #  CATKIN_DEPENDS rospy std_msgs
  #  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

# Install the Python nodes
catkin_install_python(
  PROGRAMS
    scripts/gr1_node.py
    scripts/gr1_dose_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

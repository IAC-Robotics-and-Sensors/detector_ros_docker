#!/bin/bash
set -e

# Source ROS + workspace
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

echo "Starting hamamatsu_with_dose.launch..."

exec roslaunch hamamatsu_driver hamamatsu_with_dose.launch

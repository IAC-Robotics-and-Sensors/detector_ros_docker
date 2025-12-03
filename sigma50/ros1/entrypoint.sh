#!/bin/bash
set -e

source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

roscore &

sleep 3

echo "Starting sigma50_with_dose launch..."
roslaunch sigma50_driver sigma50_with_dose.launch

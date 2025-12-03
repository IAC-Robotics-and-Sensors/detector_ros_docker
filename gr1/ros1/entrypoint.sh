#!/bin/bash
set -e

echo "Starting GR1 container entrypoint..."

# Source ROS
source /opt/ros/noetic/setup.bash

# Go to workspace and source it
cd /catkin_ws
if [ -f devel/setup.bash ]; then
  source devel/setup.bash
fi

echo "Starting gr1_with_dose launch..."
roslaunch gr1_driver gr1_with_dose.launch

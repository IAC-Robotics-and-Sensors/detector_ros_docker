#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

if [ -f /ros2_ws/install/local_setup.bash ]; then
  source /ros2_ws/install/local_setup.bash
else
  echo "WARNING: /ros2_ws/install/local_setup.bash not found"
fi

echo "Available ROS 2 packages (filtered):"
ros2 pkg list | grep hamamatsu || true

echo "Starting hamamatsu_with_dose ROS2 launch..."
exec ros2 launch hamamatsu_driver hamamatsu_with_dose.launch.py

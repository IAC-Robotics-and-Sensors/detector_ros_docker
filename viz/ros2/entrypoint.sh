#!/bin/bash
set -e

# Source ROS2
source /opt/ros/$ROS_DISTRO/setup.bash

# Source workspace
if [ -f /ros2_ws/install/setup.bash ]; then
  source /ros2_ws/install/setup.bash
fi

echo "Starting ROS2 spectrum visualiser node..."
exec "$@"

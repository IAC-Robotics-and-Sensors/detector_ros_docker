#!/bin/bash
set -e

source /opt/ros/noetic/setup.bash
if [ -f /catkin_ws/devel/setup.bash ]; then
  source /catkin_ws/devel/setup.bash
fi

# For Python to see any source packages if needed
export PYTHONPATH=/catkin_ws/src:$PYTHONPATH

echo "Starting spectrum visualiser node..."
exec "$@"

FROM ros:noetic-ros-core

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-numpy \
    python3-catkin-tools \
    python3-hid \
    && rm -rf /var/lib/apt/lists/*

# Create and enter catkin workspace
WORKDIR /catkin_ws

# Copy your catkin src tree (workspace-level CMakeLists.txt + packages)
COPY src ./src

# Build catkin workspace
RUN /bin/bash -c ". /opt/ros/noetic/setup.bash && cd /catkin_ws && catkin_make"

# Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["rosrun", "sigma50_driver", "sigma50_node.py"]

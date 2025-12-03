FROM ros:noetic-ros-core

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-numpy \
    python3-catkin-tools \
    python3-hid \
    python3-usb \
    netcat \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /catkin_ws

# Copy catkin src tree
COPY src ./src

# Build catkin workspace
RUN /bin/bash -c ". /opt/ros/noetic/setup.bash && cd /catkin_ws && catkin_make"

# Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["rosrun", "hamamatsu_driver", "hamamatsu_node.py"]

FROM ros:humble-ros-core

# System deps (add colcon here)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-numpy \
    python3-matplotlib \
    python3-tk \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws

# Copy source
COPY src ./src

# Build the workspace
RUN /bin/bash -c ". /opt/ros/$ROS_DISTRO/setup.bash && \
    cd /ros2_ws && \
    colcon build"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "run", "spectrum_viz", "spectrum_gui"]

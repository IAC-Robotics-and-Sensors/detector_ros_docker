FROM ros:humble-ros-core

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-numpy \
    python3-hid \
    python3-colcon-common-extensions \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws

COPY src ./src

RUN /bin/bash -c " \
  source /opt/ros/humble/setup.bash && \
  colcon build --symlink-install \
"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

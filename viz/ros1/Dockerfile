FROM ros:noetic-ros-core

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-numpy \
    python3-matplotlib \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /catkin_ws

COPY src ./src

RUN /bin/bash -c ". /opt/ros/noetic/setup.bash && cd /catkin_ws && catkin_make"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["rosrun", "spectrum_viz", "spectrum_plot.py"]

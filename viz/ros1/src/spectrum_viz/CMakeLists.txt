cmake_minimum_required(VERSION 3.0.2)
project(spectrum_viz)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package()

# Install our Python scripts so rosrun can find them
catkin_install_python(
  PROGRAMS
    scripts/spectrum_plot.py
    scripts/spectrum_gui.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

cmake_minimum_required(VERSION 3.5)
project(hamamatsu_driver)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)

# Install Python package 'hamamatsu_driver/'
ament_python_install_package(${PROJECT_NAME})

# Install scripts, config, and launch files into share
install(
  DIRECTORY
    scripts
    config
    launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

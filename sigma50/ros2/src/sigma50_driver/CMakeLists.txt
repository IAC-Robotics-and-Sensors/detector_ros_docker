cmake_minimum_required(VERSION 3.5)
project(sigma50_driver)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)

# Install the Python package directory 'sigma50_driver/'
ament_python_install_package(${PROJECT_NAME})

# Install scripts (NOT as libexec nodes, just as data)
install(
  DIRECTORY
    scripts
  DESTINATION share/${PROJECT_NAME}
)

# Install config and launch files
install(
  DIRECTORY
    config
    launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

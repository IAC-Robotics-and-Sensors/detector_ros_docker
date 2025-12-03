# Detector ROS Docker Suite

This repository contains full ROS1 + ROS2 Docker-based drivers, dose calculators,
and visualisation tools for:

- **Hamamatsu spectrometers**
- **Kromek Sigma50**
- **Kromek GR1**
- A cross-sensor **spectrum + CPS + dose GUI visualiser**

## Features

- ROS1 and ROS2 support
- Docker images for each detector
- Unified visualisation node (Tkinter + Matplotlib)
- Dose calculators for Sigma50, Hamamatsu & GR1
- Multi-sensor compatible spectrum plotting
- Network-transparent (DDS cross-container support)

## Setting up

# Update and certificates
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker’s official key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker.gpg

# Add Docker repo
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list

# Install
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Test
sudo docker run hello-world

# Add user to group
sudo usermod -aG docker $USER

## Relog
LOGIN & OUT


# Udev rules
/etc/udev/rules.d/99-radiation-detectors.rules

###############################################
# Radiation Detectors USB Access Rules
# Applies to:
#   - Kromek Sigma50 (HID)
#   - Kromek GR1 (HID)
#   - Hamamatsu scintillation detectors (bulk USB)
#   - Kromek D3S (serial ACM)
#
# This file ensures the detectors are accessible
# to Docker containers and normal users.
###############################################

### --------------------------
### Kromek Sigma50
### Vendor 04D8, Product 0023
### HID device → /dev/hidraw*
### --------------------------
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="04d8", ATTRS{idProduct}=="0023", MODE="0666", GROUP="plugdev", TAG+="uaccess"

### --------------------------
### Kromek GR1
### Vendor 04D8, Product 0000
### HID device → /dev/hidraw*
### --------------------------
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="04d8", ATTRS{idProduct}=="0000", MODE="0666", GROUP="plugdev", TAG+="uaccess"

### --------------------------
### Hamamatsu C12137 / C12138 / C12139 / etc.
### Vendor 0661, Product 2917
### Bulk USB → /dev/bus/usb/xxx/yyy
### --------------------------
SUBSYSTEM=="usb", ATTRS{idVendor}=="0661", ATTRS{idProduct}=="2917", MODE="0666", GROUP="plugdev", TAG+="uaccess"

# Reload and trigger udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

#GR1_ROS1
cd ~/rad_detector_docker/gr1/ros1
docker build --no-cache -t gr1_ros1 .
docker run -it \
  --net=host \
  --privileged \
  --device=/dev/hidraw1 \
  -v ~/rad_logs/gr1:/data \
  gr1_ros1

#GR1_ROS2
cd ~/rad_detector_docker/gr1/ros2
docker build --no-cache -t gr1_ros2 .
docker run -it \
  --net=host \
  --ipc=host \
  --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  -v ~/rad_logs/gr2:/data \
  gr1_ros2



#Hamamatsu_ROS1
cd ~/rad_detector_docker/hammamatsu/ros1
docker build --no-cache -t hamamatsu_ros1 .
docker run -it \
  --net=host \
  --privileged \
  --device=/dev/hidraw1 \
  -v ~/rad_logs/hamamatsu:/data \
  hamamatsu_ros1

#Hamamatsu_ROS2
cd ~/rad_detector_docker/hammamatsu1/ros2
docker build --no-cache -t hammamatsu_ros2 .
docker run -it \
  --net=host \
  --ipc=host \
  --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  -v ~/rad_logs/hammamatsu:/data \
  hamamatsu_ros2

#sigma50_ROS1
cd ~/rad_detector_docker/sigma50/ros1
docker build --no-cache -t sigma50_ros1 .
docker run -it \
  --net=host \
  --privileged \
  --device=/dev/hidraw1 \
  -v ~/rad_logs/sigma50:/data \
  sigma50_ros1

#sigma50_ROS2
cd ~/rad_detector_docker/sigma50/ros2
docker build --no-cache -t sigma50_ros2 .
docker run -it \
  --net=host \
  --ipc=host \
  --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  -v ~/rad_logs/sigma50:/data \
  sigma50_ros2

#Visualiser_ROS1

cd ~/rad_detector_docker/viz/ros1
docker build --no-cache -t spectrum_viz_ros1 .

xhost +local:docker

docker run -it \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -e ROS_MASTER_URI=http://127.0.0.1:11311 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  spectrum_viz_ros1 \
  rosrun spectrum_viz spectrum_gui.py


#Visualiser_ROS2

cd ~/rad_detector_docker/viz/ros2
docker build --no-cache -t spectrum_viz_ros2 .

xhost +local:docker

docker run --rm -it \
  --net=host \
  --ipc=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  spectrum_viz_ros2 \
  ros2 run spectrum_viz spectrum_gui

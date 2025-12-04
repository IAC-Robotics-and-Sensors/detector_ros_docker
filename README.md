# Detector ROS Docker Suite

This repository contains full **ROS1 + ROS2 Docker-based drivers**, **dose calculators**,  
and **visualisation tools** for:

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
- Saved spectrum (.csv)
---

## Installing Docker

### Update and install certificates

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
```

### Add Docker’s official key

```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /usr/share/keyrings/docker.gpg
```

### Add Docker repository

```bash
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list
```

### Install Docker Engine

```bash
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
```

### Test installation

```bash
sudo docker run hello-world
```

### Add your user to the docker group

```bash
sudo usermod -aG docker $USER
```

**Log out and back in to apply.**

---

## Udev Rules

Create:

```
sudo nano /etc/udev/rules.d/99-radiation-detectors.rules
```

with:

```bash
# Kromek Sigma50  (Vendor: 04D8, Product: 0023)
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="04d8", ATTRS{idProduct}=="0023", \
  MODE="0666", GROUP="plugdev", TAG+="uaccess"

# Kromek GR1  (Vendor: 04D8, Product: 0000)
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="04d8", ATTRS{idProduct}=="0000", \
  MODE="0666", GROUP="plugdev", TAG+="uaccess"

# Hamamatsu C12137 / C12138 / C12139  (Vendor: 0661, Product: 2917)
SUBSYSTEM=="usb", ATTRS{idVendor}=="0661", ATTRS{idProduct}=="2917", \
  MODE="0666", GROUP="plugdev", TAG+="uaccess"
```

Reload rules:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

---

# Running Detector Containers

---

## GR1 — ROS1

```bash
cd ~/detector_ros_docker/gr1/ros1
docker build --no-cache -t gr1_ros1 .

docker run -it \
  --net=host \
  --privileged \
  --device=/dev/hidraw1 \
  -v ~/rad_logs/gr1:/data \
  gr1_ros1
```

---

## GR1 — ROS2

```bash
cd ~/detector_ros_docker/gr1/ros2
docker build --no-cache -t gr1_ros2 .

docker run -it \
  --net=host \
  --ipc=host \
  --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  -v ~/rad_logs/gr1:/data \
  gr1_ros2
```

---

## Hamamatsu — ROS1

```bash
cd ~/detector_ros_docker/hammamatsu/ros1
docker build --no-cache -t hamamatsu_ros1 .

docker run -it \
  --net=host \
  --privileged \
  --device=/dev/hidraw1 \
  -v ~/rad_logs/hamamatsu:/data \
  hamamatsu_ros1
```

---

## Hamamatsu — ROS2

```bash
cd ~/detector_ros_docker/hammamatsu/ros2
docker build --no-cache -t hamamatsu_ros2 .

docker run -it \
  --net=host \
  --ipc=host \
  --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  -v ~/rad_logs/hamamatsu:/data \
  hamamatsu_ros2
```

---

## Sigma50 — ROS1

```bash
cd ~/detector_ros_docker/sigma50/ros1
docker build --no-cache -t sigma50_ros1 .

docker run -it \
  --net=host \
  --privileged \
  --device=/dev/hidraw1 \
  -v ~/rad_logs/sigma50:/data \
  sigma50_ros1
```

---

## Sigma50 — ROS2

```bash
cd ~/detector_ros_docker/sigma50/ros2
docker build --no-cache -t sigma50_ros2 .

docker run -it \
  --net=host \
  --ipc=host \
  --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  -v ~/rad_logs/sigma50:/data \
  sigma50_ros2
```

---

# Visualiser

---
Run the visualiser container in a seprate terminal once the detector container has been built, it will allow you to select topics for CPS spectrum and dose.
## Visualiser — ROS1

```bash
cd ~/detector_ros_docker/viz/ros1
docker build --no-cache -t spectrum_viz_ros1 .

xhost +local:docker

docker run -it \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -e ROS_MASTER_URI=http://127.0.0.1:11311 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  spectrum_viz_ros1 \
  rosrun spectrum_viz spectrum_gui.py
```

---

## Visualiser — ROS2

```bash
cd ~/detector_ros_docker/viz/ros2
docker build --no-cache -t spectrum_viz_ros2 .

xhost +local:docker

docker run --rm -it \
  --net=host \
  --ipc=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  spectrum_viz_ros2 \
  ros2 run spectrum_viz spectrum_gui
```
# Notes
## Spectrums
Spectrums of the different detectors can be found:
```bash
cd ~/rad_logs
```
## View messages 
To view the incoming ros messages first enter the container (tabbing will autofill the containers name)
```bash
docker exec -it <tab> bash 
```
Source the environments for the different ROS distros rostopic list should confirm topics are visible.

## ROS1 
```bash
source /opt/ros/noetic/setup.bash
rostopic list
```
Example for viewing the gr1 cps messages

```bash
rostopic echo /gr1/cps
```
## ROS2
```bash
source /opt/ros/humble/setup.bash
ros2 topic list
```
Example for viewing the gr1 cps messages

```bash
ros2 topic echo /gr1/cps
```
## Dose Rates
Dose rates are calculated using dose lookup tables within the 
```bash
~/detector_ros_docker/<detector>/<ros_version>/src/<detector_driver>/config
```
folder each unique detector should be calibrated before reliable use of the dose function.


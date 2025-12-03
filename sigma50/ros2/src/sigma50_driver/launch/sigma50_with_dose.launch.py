import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("sigma50_driver")

    scripts_dir = os.path.join(pkg_share, "scripts")
    sigma50_node_script = os.path.join(scripts_dir, "sigma50_node.py")
    sigma50_dose_node_script = os.path.join(scripts_dir, "sigma50_dose_node.py")

    # Command for the main detector node
    sigma50_cmd = [
        "python3",
        sigma50_node_script,
        "--ros-args",
        "-r",
        "__node:=sigma50_node",
        "-p",
        "log_dir:=/data",
        "-p",
        "log_enabled:=true",
    ]

    # Command for the dose node
    dose_cmd = [
        "python3",
        sigma50_dose_node_script,
        "--ros-args",
        "-r",
        "__node:=sigma50_dose_node",
        "-p",
        "spectrum_topic:=/sigma50/spectrum",
        "-p",
        "dose_topic:=/sigma50/dose_rate",
        "-p",
        "n_channels:=4096",
    ]

    return LaunchDescription(
        [
            ExecuteProcess(
                cmd=sigma50_cmd,
                output="screen",
            ),
            ExecuteProcess(
                cmd=dose_cmd,
                output="screen",
            ),
        ]
    )

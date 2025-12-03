import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("gr1_driver")

    scripts_dir = os.path.join(pkg_share, "scripts")
    gr1_node_script = os.path.join(scripts_dir, "gr1_node.py")
    gr1_dose_node_script = os.path.join(scripts_dir, "gr1_dose_node.py")

    gr1_cmd = [
        "python3",
        gr1_node_script,
        "--ros-args",
        "-r",
        "__node:=gr1_node",
        "-p",
        "log_dir:=/data",
        "-p",
        "log_enabled:=true",
    ]

    dose_cmd = [
        "python3",
        gr1_dose_node_script,
        "--ros-args",
        "-r",
        "__node:=gr1_dose_node",
        "-p",
        "spectrum_topic:=/gr1/spectrum",
        "-p",
        "dose_topic:=/gr1/dose_rate",
        "-p",
        "n_channels:=4096",
        # lookup_path left to default from package share
    ]

    return LaunchDescription(
        [
            ExecuteProcess(cmd=gr1_cmd, output="screen"),
            ExecuteProcess(cmd=dose_cmd, output="screen"),
        ]
    )

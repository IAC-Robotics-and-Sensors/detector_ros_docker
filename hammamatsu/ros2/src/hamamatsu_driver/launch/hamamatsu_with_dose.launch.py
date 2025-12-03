import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("hamamatsu_driver")

    scripts_dir = os.path.join(pkg_share, "scripts")
    ham_node_script = os.path.join(scripts_dir, "hamamatsu_node.py")
    ham_dose_script = os.path.join(scripts_dir, "hamamatsu_dose_node.py")

    ham_cmd = [
        "python3",
        ham_node_script,
        "--ros-args",
        "-r",
        "__node:=hamamatsu_node",
        "-p",
        "log_dir:=/data",
        "-p",
        "log_enabled:=true",
    ]

    dose_cmd = [
        "python3",
        ham_dose_script,
        "--ros-args",
        "-r",
        "__node:=hamamatsu_dose_node",
        "-p",
        "spectrum_topic:=/hamamatsu/spectrum",
        "-p",
        "dose_topic:=/hamamatsu/dose_rate",
        "-p",
        "n_channels:=4096",
        # lookup_path left to default, resolved from package share
    ]

    return LaunchDescription(
        [
            ExecuteProcess(cmd=ham_cmd, output="screen"),
            ExecuteProcess(cmd=dose_cmd, output="screen"),
        ]
    )

#!/usr/bin/env python3

import os
import csv
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import UInt32MultiArray, Float64


class Sigma50DoseCalculator(Node):
    """
    ROS 2 version of Sigma50DoseCalculator.
    Subscribes to /sigma50/spectrum and publishes dose rate in uSv/h.
    """

    def __init__(self):
        super().__init__("sigma50_dose_node")

        # Declare parameters with defaults
        self.declare_parameter("spectrum_topic", "/sigma50/spectrum")
        self.declare_parameter("dose_topic", "/sigma50/dose_rate")
        self.declare_parameter("n_channels", 4096)
        self.declare_parameter("lookup_path", "")

        self.spectrum_topic = (
            self.get_parameter("spectrum_topic").get_parameter_value().string_value
        )
        self.dose_topic = (
            self.get_parameter("dose_topic").get_parameter_value().string_value
        )
        self.n_channels = (
            self.get_parameter("n_channels").get_parameter_value().integer_value
        )

        # Resolve lookup path: explicit param or package share/config
        explicit_lookup = (
            self.get_parameter("lookup_path").get_parameter_value().string_value
        )
        if explicit_lookup:
            self.lookup_path = explicit_lookup
        else:
            self.lookup_path = self._default_lookup_path()

        self.get_logger().info(
            f"Sigma50DoseCalculator: using lookup CSV: {self.lookup_path}"
        )

        # Load Chi (nSv/h per cps) per bin
        self.chi = self._load_lookup(self.lookup_path, self.n_channels)

        # Publisher / subscriber
        self.dose_pub = self.create_publisher(Float64, self.dose_topic, 10)
        self.spectrum_sub = self.create_subscription(
            UInt32MultiArray,
            self.spectrum_topic,
            self.spectrum_callback,
            10,
        )

        self.get_logger().info(
            f"Sigma50DoseCalculator node started. "
            f"Subscribing to {self.spectrum_topic}, publishing dose rate to {self.dose_topic}"
        )

    def _default_lookup_path(self) -> str:
        """
        Resolve <pkg share>/config/bin_energy_dose_lookup.csv using ament_index.
        """
        try:
            pkg_share = get_package_share_directory("sigma50_driver")
            return os.path.join(pkg_share, "config", "bin_energy_dose_lookup.csv")
        except Exception as e:
            self.get_logger().warn(
                "Could not resolve sigma50_driver share directory, "
                f"falling back to relative path: {e}"
            )
            script_dir = os.path.dirname(os.path.abspath(__file__))
            return os.path.join(
                script_dir, "..", "config", "bin_energy_dose_lookup.csv"
            )

    def _load_lookup(self, path: str, n_channels: int) -> np.ndarray:
        """
        Load Chi_for_bin_nSv_h_per_cps into a numpy array of length n_channels.

        CSV columns:
          bin, E_bin_MeV, Chi_for_bin_nSv_h_per_cps

        We map bin N -> channel index N-1 (0-based).
        Missing bins and NaNs get Chi = 0.
        """
        chi = np.zeros(n_channels, dtype=np.float64)

        if not os.path.exists(path):
            self.get_logger().warn(
                f"Sigma50DoseCalculator: lookup CSV {path} not found. "
                "All Chi set to 0; dose rate will always be 0."
            )
            return chi

        try:
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        b_str = row.get("bin", "")
                        if b_str is None:
                            continue
                        b = int(b_str.strip())
                    except Exception:
                        continue

                    idx = b - 1
                    if idx < 0 or idx >= n_channels:
                        continue

                    chi_val_str: Optional[str] = row.get(
                        "Chi_for_bin_nSv_h_per_cps", ""
                    )
                    if chi_val_str is None or chi_val_str.strip() == "":
                        c = 0.0
                    else:
                        try:
                            c = float(chi_val_str)
                        except Exception:
                            c = 0.0

                    if c != c:  # NaN
                        c = 0.0

                    chi[idx] = c

            non_zero = int(np.count_nonzero(chi))
            self.get_logger().info(
                f"Sigma50DoseCalculator: loaded Chi for {non_zero} / {n_channels} channels"
            )
        except Exception as e:
            self.get_logger().error(
                f"Sigma50DoseCalculator: error loading lookup CSV: {e}"
            )

        return chi

    def spectrum_callback(self, msg: UInt32MultiArray) -> None:
        spec = np.array(msg.data, dtype=np.float64)

        # Align lengths
        if spec.size > self.chi.size:
            spec = spec[: self.chi.size]
        elif spec.size < self.chi.size:
            padded = np.zeros(self.chi.size, dtype=np.float64)
            padded[: spec.size] = spec
            spec = padded

        # Chi is nSv/h per cps, spec is counts in 1 s (i.e. cps)
        dose_uSv_h = float(np.dot(spec, self.chi)) / 1000.0

        out = Float64()
        out.data = dose_uSv_h
        self.dose_pub.publish(out)

        self.get_logger().debug(f"Dose rate: {dose_uSv_h:.6f} uSv/h")


def main(args=None):
    rclpy.init(args=args)
    node = Sigma50DoseCalculator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

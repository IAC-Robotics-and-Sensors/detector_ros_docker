#!/usr/bin/env python3

import os
import csv
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import UInt32MultiArray, Float64


def load_dose_lookup(path: str, n_channels: int, logger: Optional[Node]) -> np.ndarray:
    """
    Load per-channel dose coefficients from CSV.

    Expected format: at least 2 columns; we assume:
        col0 = channel index (int)
        last column = dose_per_count (µSv/h per count-per-second)

    Any channels not present in the file get dose 0.0.
    """
    coeffs = np.zeros(n_channels, dtype=np.float64)
    log_info = logger.get_logger().info if isinstance(logger, Node) else print
    log_warn = logger.get_logger().warn if isinstance(logger, Node) else print
    log_err = logger.get_logger().error if isinstance(logger, Node) else print

    if not os.path.isfile(path):
        log_warn(f"GR1DoseCalculator: lookup CSV not found: {path}")
        return coeffs

    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                try:
                    ch = int(row[0])
                    dose_val = float(row[-1])
                except ValueError:
                    continue
                if 0 <= ch < n_channels:
                    coeffs[ch] = dose_val
    except Exception as e:
        log_err(f"GR1DoseCalculator: error reading lookup CSV: {e}")
        return coeffs

    nonzero = int(np.count_nonzero(coeffs))
    log_info(
        f"GR1DoseCalculator: loaded dose coefficients for {nonzero} / {n_channels} channels"
    )
    return coeffs


class GR1DoseCalculator(Node):
    def __init__(self):
        super().__init__("gr1_dose_node")

        # Parameters
        self.declare_parameter("spectrum_topic", "/gr1/spectrum")
        self.declare_parameter("dose_topic", "/gr1/dose_rate")
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

        explicit_lookup = (
            self.get_parameter("lookup_path").get_parameter_value().string_value
        )

        if explicit_lookup:
            self.lookup_path = explicit_lookup
        else:
            try:
                pkg_share = get_package_share_directory("gr1_driver")
                self.lookup_path = os.path.join(
                    pkg_share, "config", "bin_energy_dose_lookup.csv"
                )
            except Exception as e:
                self.get_logger().warn(
                    f"GR1DoseCalculator: could not resolve package share directory: {e}"
                )
                self.lookup_path = "bin_energy_dose_lookup.csv"

        self.get_logger().info(
            f"GR1DoseCalculator: using lookup CSV: {self.lookup_path}"
        )

        # Load lookup table
        self.dose_coeffs = load_dose_lookup(
            self.lookup_path, self.n_channels, logger=self
        )

        # Publisher & subscriber
        self.dose_pub = self.create_publisher(Float64, self.dose_topic, 10)
        self.spec_sub = self.create_subscription(
            UInt32MultiArray,
            self.spectrum_topic,
            self.spectrum_cb,
            10,
        )

        self.get_logger().info(
            f"GR1DoseCalculator node started. "
            f"Subscribing to {self.spectrum_topic}, publishing dose rate to {self.dose_topic} (µSv/h)."
        )

    def spectrum_cb(self, msg: UInt32MultiArray):
        data = np.array(msg.data, dtype=np.float64)
        if data.size == 0:
            return

        n = min(data.size, self.dose_coeffs.size)
        dose_rate = float(np.sum(data[:n] * self.dose_coeffs[:n]))/1000

        out = Float64()
        out.data = dose_rate
        self.dose_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = GR1DoseCalculator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

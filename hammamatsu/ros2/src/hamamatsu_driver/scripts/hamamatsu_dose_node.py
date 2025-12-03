#!/usr/bin/env python3

import os
import csv
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import UInt32MultiArray, Float32


def load_dose_lookup(path: str, n_channels: int, logger: Optional[Node]) -> np.ndarray:
    coeff = np.zeros(n_channels, dtype=np.float64)
    log_info = logger.get_logger().info if isinstance(logger, Node) else print
    log_err = logger.get_logger().error if isinstance(logger, Node) else print

    if not os.path.exists(path):
        log_err(f"HamamatsuDoseCalculator: dose lookup CSV not found: {path}")
        return coeff

    count = 0
    try:
        with open(path, "r") as f:
            r = csv.reader(f)
            for row in r:
                if len(row) < 2:
                    continue
                try:
                    ch = int(row[0])
                    dose = float(row[1])
                except ValueError:
                    continue
                if 0 <= ch < n_channels:
                    coeff[ch] = dose
                    count += 1
    except Exception as e:
        log_err(f"HamamatsuDoseCalculator: error reading lookup CSV: {e}")
        return coeff

    log_info(
        f"HamamatsuDoseCalculator: loaded dose coefficients for {count} / {n_channels} channels"
    )
    return coeff


class HamamatsuDoseCalculator(Node):
    def __init__(self):
        super().__init__("hamamatsu_dose_node")

        # Parameters
        self.declare_parameter("spectrum_topic", "/hamamatsu/spectrum")
        self.declare_parameter("dose_topic", "/hamamatsu/dose_rate")
        self.declare_parameter("lookup_path", "")
        self.declare_parameter("n_channels", 4096)

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
                pkg_share = get_package_share_directory("hamamatsu_driver")
                self.lookup_path = os.path.join(
                    pkg_share, "config", "bin_energy_dose_lookup.csv"
                )
            except Exception as e:
                self.get_logger().warn(
                    f"HamamatsuDoseCalculator: could not resolve package share: {e}"
                )
                self.lookup_path = "bin_energy_dose_lookup.csv"

        self.get_logger().info(
            f"HamamatsuDoseCalculator: using lookup CSV: {self.lookup_path}"
        )

        # Load coefficients
        self.coeff = load_dose_lookup(
            self.lookup_path, self.n_channels, logger=self
        )

        # ROS I/O
        self.sub = self.create_subscription(
            UInt32MultiArray,
            self.spectrum_topic,
            self.cb_spectrum,
            10,
        )
        self.pub = self.create_publisher(Float32, self.dose_topic, 10)

        self.get_logger().info(
            f"HamamatsuDoseCalculator: started, listening to {self.spectrum_topic} → {self.dose_topic}"
        )

    def cb_spectrum(self, msg: UInt32MultiArray):
        spec = np.array(msg.data, dtype=np.float64)
        if spec.size > self.n_channels:
            spec = spec[: self.n_channels]

        dose_rate = float(np.sum(spec * self.coeff))/1000
        out = Float32()
        out.data = dose_rate
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = HamamatsuDoseCalculator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import UInt32MultiArray, UInt32, Float64

import matplotlib.pyplot as plt


class SpectrumPlotNode(Node):
    def __init__(self):
        super().__init__('spectrum_plot')

        # Params (roughly match ROS1 defaults)
        default_spec_topic = '/sigma50/spectrum'
        self.declare_parameter('spectrum_topic', default_spec_topic)
        self.declare_parameter('topic', default_spec_topic)  # backwards-ish
        self.declare_parameter('cps_topic', '/sigma50/cps')
        self.declare_parameter('dose_topic', '/sigma50/dose_rate')

        # Backwards compatible param handling
        spec_topic_param = self.get_parameter('spectrum_topic').get_parameter_value().string_value
        topic_param = self.get_parameter('topic').get_parameter_value().string_value
        spec_topic = spec_topic_param or topic_param or default_spec_topic

        cps_topic = self.get_parameter('cps_topic').get_parameter_value().string_value
        dose_topic = self.get_parameter('dose_topic').get_parameter_value().string_value

        self.get_logger().info(f"SpectrumPlot: spectrum_topic = {spec_topic}")
        self.get_logger().info(f"SpectrumPlot: cps_topic      = {cps_topic}")
        self.get_logger().info(f"SpectrumPlot: dose_topic     = {dose_topic}")

        # State
        self.latest_spectrum = None
        self.latest_cps = None
        self.latest_dose = None
        self.data_dirty = False

        # Subscriptions
        self.create_subscription(
            UInt32MultiArray,
            spec_topic,
            self.spectrum_callback,
            10
        )
        self.create_subscription(
            UInt32,
            cps_topic,
            self.cps_callback,
            10
        )
        self.create_subscription(
            Float64,
            dose_topic,
            self.dose_callback,
            10
        )

        # Matplotlib setup
        plt.ion()
        self.fig, self.ax = plt.subplots()

        self.line, = self.ax.plot([], [], lw=1)
        self.ax.set_xlabel("Energy channel")
        self.ax.set_ylabel("Counts")
        self.ax.set_title("Spectrum")

        self.text_cps = self.ax.text(
            0.02,
            0.95,
            "CPS: --",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
        )
        self.text_dose = self.ax.text(
            0.02,
            0.90,
            "Dose rate: -- µSv/h",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
        )

        # Timer to update plot at ~10 Hz
        self.create_timer(0.1, self.update_plot)

    # Callbacks
    def spectrum_callback(self, msg: UInt32MultiArray):
        self.latest_spectrum = np.array(msg.data, dtype=np.uint32)
        self.data_dirty = True

    def cps_callback(self, msg: UInt32):
        self.latest_cps = int(msg.data)

    def dose_callback(self, msg: Float64):
        self.latest_dose = float(msg.data)

    def update_plot(self):
        if not self.data_dirty or self.latest_spectrum is None:
            plt.pause(0.01)
            return

        self.data_dirty = False

        x = np.arange(len(self.latest_spectrum))
        y = self.latest_spectrum

        self.line.set_xdata(x)
        self.line.set_ydata(y)

        self.ax.set_xlim(0, max(1, len(y)))

        y_max = int(y.max()) if y.size > 0 else 1
        if y_max < 1:
            y_max = 1
        self.ax.set_ylim(0, y_max * 1.1)

        if self.latest_cps is not None:
            self.text_cps.set_text(f"CPS: {self.latest_cps}")
        else:
            self.text_cps.set_text("CPS: --")

        if self.latest_dose is not None:
            self.text_dose.set_text(f"Dose rate: {self.latest_dose:.3f} µSv/h")
        else:
            self.text_dose.set_text("Dose rate: -- µSv/h")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)


def main():
    rclpy.init()
    node = SpectrumPlotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

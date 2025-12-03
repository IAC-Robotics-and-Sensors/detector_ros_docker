#!/usr/bin/env python3

import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from std_msgs.msg import UInt32MultiArray, UInt32, Float32, Float64

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SpectrumGUI:
    """
    ROS2 port of the original ROS1 accumulated spectrum GUI.

    - Discovers topics dynamically
    - Lets you choose spectrum / CPS / dose topics from dropdowns
    - Accumulates spectra over time
    - Displays CPS and dose in the header
    """

    def __init__(self, master: tk.Tk, node: Node):
        self.master = master
        self.node = node

        self.master.title("ROS2 Spectrum Visualiser (Accumulated)")

        # Current selected topics
        self.current_spectrum_topic = None
        self.current_cps_topic = None
        self.current_dose_topic = None

        # Subscriptions
        self.spectrum_sub = None
        self.cps_sub = None
        self.dose_sub = None

        # topic -> type string (e.g. 'std_msgs/msg/Float64')
        self.topic_type_map = {}

        # Accumulated spectrum (running sum)
        self.spectrum_total = np.zeros(4096, dtype=np.uint64)
        self.spectrum_lock = threading.Lock()

        # Latest CPS and dose values
        self.cps_value = 0
        self.dose_value = 0.0
        self.cps_lock = threading.Lock()
        self.dose_lock = threading.Lock()

        # ------------------- UI layout -------------------

        # Top frame: spectrum controls
        top_frame = ttk.Frame(master)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(top_frame, text="Spectrum topic:").pack(side=tk.LEFT)

        self.spectrum_topic_var = tk.StringVar()
        self.spectrum_topic_combo = ttk.Combobox(
            top_frame,
            textvariable=self.spectrum_topic_var,
            width=45,
            state="readonly",
        )
        self.spectrum_topic_combo.pack(side=tk.LEFT, padx=5)

        self.refresh_button = ttk.Button(
            top_frame, text="Refresh topics", command=self.refresh_topics
        )
        self.refresh_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = ttk.Button(
            top_frame, text="Reset spectrum", command=self.reset_spectrum
        )
        self.reset_button.pack(side=tk.LEFT, padx=5)

        # Second row: CPS controls
        cps_frame = ttk.Frame(master)
        cps_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        ttk.Label(cps_frame, text="CPS topic:").pack(side=tk.LEFT)

        self.cps_topic_var = tk.StringVar()
        self.cps_topic_combo = ttk.Combobox(
            cps_frame,
            textvariable=self.cps_topic_var,
            width=45,
            state="readonly",
        )
        self.cps_topic_combo.pack(side=tk.LEFT, padx=5)

        self.cps_display_var = tk.StringVar(value="CPS: 0")
        cps_label = ttk.Label(
            cps_frame,
            textvariable=self.cps_display_var,
            font=("TkDefaultFont", 12, "bold"),
        )
        cps_label.pack(side=tk.RIGHT, padx=5)

        # Third row: Dose controls
        dose_frame = ttk.Frame(master)
        dose_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        ttk.Label(dose_frame, text="Dose topic:").pack(side=tk.LEFT)

        self.dose_topic_var = tk.StringVar()
        self.dose_topic_combo = ttk.Combobox(
            dose_frame,
            textvariable=self.dose_topic_var,
            width=45,
            state="readonly",
        )
        self.dose_topic_combo.pack(side=tk.LEFT, padx=5)

        self.dose_display_var = tk.StringVar(value="Dose: 0.000 µSv/h")
        dose_label = ttk.Label(
            dose_frame,
            textvariable=self.dose_display_var,
            font=("TkDefaultFont", 12, "bold"),
        )
        dose_label.pack(side=tk.RIGHT, padx=5)

        # Status line
        self.status_var = tk.StringVar(value="No topic selected")
        ttk.Label(master, textvariable=self.status_var).pack(
            side=tk.TOP, anchor="w", padx=5
        )

        # ------------------- Matplotlib figure -------------------

        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Channel")
        self.ax.set_ylabel("Counts (accumulated)")
        self.ax.set_title("Accumulated Spectrum")

        x = np.arange(len(self.spectrum_total))
        (self.line,) = self.ax.plot(x, self.spectrum_total, drawstyle="steps-mid")
        self.ax.set_xlim(0, len(self.spectrum_total))
        self.ax.set_ylim(0, 1)

        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Bind combobox events
        self.spectrum_topic_combo.bind(
            "<<ComboboxSelected>>", self.on_spectrum_topic_selected
        )
        self.cps_topic_combo.bind(
            "<<ComboboxSelected>>", self.on_cps_topic_selected
        )
        self.dose_topic_combo.bind(
            "<<ComboboxSelected>>", self.on_dose_topic_selected
        )

        # Initial topic scan
        self.refresh_topics()

        # Periodic redraw + label updates
        self.update_plot()

        self.node.get_logger().info("Spectrum GUI initialised")

    # ------------------- Topic discovery -------------------

    def refresh_topics(self):
        """Find spectrum, CPS and dose topics and populate dropdowns."""
        try:
            topics = self.node.get_topic_names_and_types()
        except Exception:
            topics = []

        self.topic_type_map = {}

        spectrum_topics = []
        cps_topics = []
        dose_topics = []

        for name, types in topics:
            for ttype in types:
                # Remember most recent type for this topic
                self.topic_type_map[name] = ttype

                if ttype.endswith("/UInt32MultiArray"):
                    spectrum_topics.append(name)
                elif ttype.endswith("/UInt32"):
                    cps_topics.append(name)
                elif ttype.endswith("/Float32") or ttype.endswith("/Float64"):
                    dose_topics.append(name)

        self.spectrum_topic_combo["values"] = spectrum_topics
        self.cps_topic_combo["values"] = cps_topics
        self.dose_topic_combo["values"] = dose_topics

        if not spectrum_topics and not cps_topics and not dose_topics:
            self.status_var.set("No relevant topics found (spectrum/CPS/dose).")
        else:
            self.status_var.set(
                f"Found {len(spectrum_topics)} spectrum, "
                f"{len(cps_topics)} CPS, {len(dose_topics)} dose topics."
            )

        # Clear selections if old topics disappeared
        if self.current_spectrum_topic not in spectrum_topics:
            self.current_spectrum_topic = None
            self.spectrum_topic_var.set("")

        if self.current_cps_topic not in cps_topics:
            self.current_cps_topic = None
            self.cps_topic_var.set("")

        if self.current_dose_topic not in dose_topics:
            self.current_dose_topic = None
            self.dose_topic_var.set("")

    # ------------------- Spectrum handling -------------------

    def on_spectrum_topic_selected(self, event=None):
        topic = self.spectrum_topic_var.get().strip()
        if not topic:
            return

        if self.spectrum_sub is not None:
            self.node.destroy_subscription(self.spectrum_sub)
            self.spectrum_sub = None

        self.current_spectrum_topic = topic
        self.status_var.set(f"Subscribed to spectrum: {topic} (accumulating)")
        self.node.get_logger().info(f"Subscribed to spectrum topic: {topic}")

        # Reset accumulation when changing topic
        self.reset_spectrum()

        self.spectrum_sub = self.node.create_subscription(
            UInt32MultiArray,
            topic,
            self.spectrum_cb,
            10,
        )

    def spectrum_cb(self, msg: UInt32MultiArray):
        data = np.array(msg.data, dtype=np.uint64)
        self.node.get_logger().debug(f"Received spectrum with {data.size} bins")

        with self.spectrum_lock:
            if data.shape[0] != self.spectrum_total.shape[0]:
                # Resize accumulated spectrum to match incoming data
                self.node.get_logger().info(
                    f"Resizing accumulated spectrum from "
                    f"{self.spectrum_total.shape[0]} to {data.shape[0]} bins"
                )
                self.spectrum_total = np.zeros_like(data, dtype=np.uint64)
                x = np.arange(len(data))
                self.line.set_xdata(x)
                self.ax.set_xlim(0, len(data))

            # Accumulate
            self.spectrum_total += data

    def reset_spectrum(self):
        with self.spectrum_lock:
            self.spectrum_total = np.zeros_like(
                self.spectrum_total, dtype=np.uint64
            )
        self.status_var.set(
            f"Accumulated spectrum reset for "
            f"{self.current_spectrum_topic or 'no topic'}"
        )

    # ------------------- CPS handling -------------------

    def on_cps_topic_selected(self, event=None):
        topic = self.cps_topic_var.get().strip()
        if not topic:
            return

        if self.cps_sub is not None:
            self.node.destroy_subscription(self.cps_sub)
            self.cps_sub = None

        self.current_cps_topic = topic
        self.status_var.set(
            f"Spectrum: {self.current_spectrum_topic or 'none'}; "
            f"CPS: {topic}; "
            f"Dose: {self.current_dose_topic or 'none'}"
        )
        self.node.get_logger().info(f"Subscribed to CPS topic: {topic}")

        self.cps_sub = self.node.create_subscription(
            UInt32,
            topic,
            self.cps_cb,
            10,
        )

    def cps_cb(self, msg: UInt32):
        val = int(msg.data)
        self.node.get_logger().debug(f"Received CPS: {val}")
        with self.cps_lock:
            self.cps_value = val

    # ------------------- Dose handling -------------------

    def on_dose_topic_selected(self, event=None):
        topic = self.dose_topic_var.get().strip()
        if not topic:
            return

        if self.dose_sub is not None:
            self.node.destroy_subscription(self.dose_sub)
            self.dose_sub = None

        self.current_dose_topic = topic

        ttype = self.topic_type_map.get(topic, "")
        self.status_var.set(
            f"Spectrum: {self.current_spectrum_topic or 'none'}; "
            f"CPS: {self.current_cps_topic or 'none'}; "
            f"Dose: {topic} ({ttype})"
        )
        self.node.get_logger().info(
            f"Subscribed to dose topic: {topic} ({ttype})"
        )

        # Choose correct msg type based on discovered type
        if ttype.endswith("/Float64"):
            msg_type = Float64
        else:
            msg_type = Float32

        self.dose_sub = self.node.create_subscription(
            msg_type,
            topic,
            self.dose_cb,
            10,
        )

    def dose_cb(self, msg):
        val = float(msg.data)
        self.node.get_logger().debug(f"Received dose: {val}")
        with self.dose_lock:
            self.dose_value = val

    # ------------------- Plot + labels update -------------------

    def update_plot(self):
        # Spectrum
        with self.spectrum_lock:
            y = self.spectrum_total.copy()

        if y.size > 0:
            self.line.set_ydata(y)
            y_max = int(y.max())
            if y_max < 1:
                y_max = 1
            self.ax.set_ylim(0, y_max * 1.1)

        # CPS / dose
        with self.cps_lock:
            cps_val = self.cps_value
        with self.dose_lock:
            dose_val = self.dose_value

        self.cps_display_var.set(f"CPS: {cps_val}")
        self.dose_display_var.set(f"Dose: {dose_val:.3f} µSv/h")

        self.canvas.draw_idle()
        self.master.after(100, self.update_plot)  # ~10 Hz


def main():
    rclpy.init()
    node = rclpy.create_node("spectrum_gui")
    node.get_logger().info("Spectrum GUI node started")

    # ROS executor in background thread
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def ros_spin():
        try:
            executor.spin()
        finally:
            executor.shutdown()
            node.get_logger().info("Executor stopped")

    spin_thread = threading.Thread(target=ros_spin, daemon=True)
    spin_thread.start()

    # Tk GUI in main thread
    root = tk.Tk()
    gui = SpectrumGUI(root, node)

    try:
        root.mainloop()
    finally:
        node.get_logger().info("Tk mainloop exiting, shutting down")
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

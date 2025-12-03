#!/usr/bin/env python3

import threading
import numpy as np
import rospy

from std_msgs.msg import UInt32MultiArray, UInt32, Float32, Float64

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SpectrumGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("ROS Spectrum Visualiser (Accumulated)")

        # ROS bits
        self.current_spectrum_topic = None
        self.current_cps_topic = None
        self.current_dose_topic = None

        self.spectrum_sub = None
        self.cps_sub = None
        self.dose_sub = None

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
            top_frame, textvariable=self.spectrum_topic_var,
            width=45, state="readonly"
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
            cps_frame, textvariable=self.cps_topic_var,
            width=45, state="readonly"
        )
        self.cps_topic_combo.pack(side=tk.LEFT, padx=5)

        self.cps_display_var = tk.StringVar(value="CPS: 0")
        cps_label = ttk.Label(
            cps_frame,
            textvariable=self.cps_display_var,
            font=("TkDefaultFont", 12, "bold")
        )
        cps_label.pack(side=tk.RIGHT, padx=5)

        # Third row: Dose controls
        dose_frame = ttk.Frame(master)
        dose_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        ttk.Label(dose_frame, text="Dose topic:").pack(side=tk.LEFT)

        self.dose_topic_var = tk.StringVar()
        self.dose_topic_combo = ttk.Combobox(
            dose_frame, textvariable=self.dose_topic_var,
            width=45, state="readonly"
        )
        self.dose_topic_combo.pack(side=tk.LEFT, padx=5)

        self.dose_display_var = tk.StringVar(value="Dose: 0.000 µSv/h")
        dose_label = ttk.Label(
            dose_frame,
            textvariable=self.dose_display_var,
            font=("TkDefaultFont", 12, "bold")
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
        self.line, = self.ax.plot(x, self.spectrum_total, drawstyle="steps-mid")
        self.ax.set_xlim(0, len(self.spectrum_total))
        self.ax.set_ylim(0, 1)

        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Bind combobox events
        self.spectrum_topic_combo.bind("<<ComboboxSelected>>", self.on_spectrum_topic_selected)
        self.cps_topic_combo.bind("<<ComboboxSelected>>", self.on_cps_topic_selected)
        self.dose_topic_combo.bind("<<ComboboxSelected>>", self.on_dose_topic_selected)

        # Periodic redraw + label updates
        self.update_plot()

        # Initial topic scan
        self.refresh_topics()

    # ------------------- Topic discovery -------------------

    def refresh_topics(self):
        """Find spectrum, CPS and dose topics and populate dropdowns."""
        try:
            topics = rospy.get_published_topics()
        except rospy.ROSException:
            topics = []

        # Be robust to slight variations, match by suffix
        spectrum_topics = [
            name for (name, ttype) in topics
            if ttype is not None and ttype.endswith("UInt32MultiArray")
        ]
        cps_topics = [
            name for (name, ttype) in topics
            if ttype is not None and ttype.endswith("UInt32")
        ]
        dose_topics = [
            name for (name, ttype) in topics
            if ttype.endswith("Float32") or ttype.endswith("Float64")

        ]

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

        # Clear selections if topics disappeared
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
            try:
                self.spectrum_sub.unregister()
            except Exception:
                pass
            self.spectrum_sub = None

        self.current_spectrum_topic = topic
        self.status_var.set(f"Subscribed to spectrum: {topic} (accumulating)")
        self.reset_spectrum()

        self.spectrum_sub = rospy.Subscriber(
            topic, UInt32MultiArray, self.spectrum_cb, queue_size=10
        )

    def spectrum_cb(self, msg):
        data = np.array(msg.data, dtype=np.uint64)

        with self.spectrum_lock:
            if data.shape[0] != self.spectrum_total.shape[0]:
                self.spectrum_total = np.zeros_like(data, dtype=np.uint64)
                x = np.arange(len(data))
                self.line.set_xdata(x)
                self.ax.set_xlim(0, len(data))

            self.spectrum_total += data

    def reset_spectrum(self):
        with self.spectrum_lock:
            self.spectrum_total = np.zeros_like(self.spectrum_total, dtype=np.uint64)
        self.status_var.set(
            f"Accumulated spectrum reset for {self.current_spectrum_topic or 'no topic'}"
        )

    # ------------------- CPS handling -------------------

    def on_cps_topic_selected(self, event=None):
        topic = self.cps_topic_var.get().strip()
        if not topic:
            return

        if self.cps_sub is not None:
            try:
                self.cps_sub.unregister()
            except Exception:
                pass
            self.cps_sub = None

        self.current_cps_topic = topic
        self.status_var.set(
            f"Spectrum: {self.current_spectrum_topic or 'none'}; "
            f"CPS: {topic}; "
            f"Dose: {self.current_dose_topic or 'none'}"
        )

        self.cps_sub = rospy.Subscriber(
            topic, UInt32, self.cps_cb, queue_size=10
        )

    def cps_cb(self, msg):
        with self.cps_lock:
            self.cps_value = int(msg.data)

    # ------------------- Dose handling -------------------

    def on_dose_topic_selected(self, event=None):
        topic = self.dose_topic_var.get().strip()
        if not topic:
            return

        if self.dose_sub is not None:
            try:
                self.dose_sub.unregister()
            except Exception:
                pass
            self.dose_sub = None

        self.current_dose_topic = topic
        self.status_var.set(
            f"Spectrum: {self.current_spectrum_topic or 'none'}; "
            f"CPS: {self.current_cps_topic or 'none'}; "
            f"Dose: {topic}"
        )

        self.dose_sub = rospy.Subscriber(
            topic, Float32, self.dose_cb, queue_size=10
        )

    def dose_cb(self, msg):
        with self.dose_lock:
            self.dose_value = float(msg.data)

    # ------------------- Plot + labels update -------------------

    def update_plot(self):
        with self.spectrum_lock:
            y = self.spectrum_total.copy()

        if y.size > 0:
            self.line.set_ydata(y)
            y_max = int(y.max())
            if y_max < 1:
                y_max = 1
            self.ax.set_ylim(0, y_max * 1.1)

        with self.cps_lock:
            cps_val = self.cps_value
        with self.dose_lock:
            dose_val = self.dose_value

        self.cps_display_var.set(f"CPS: {cps_val}")
        self.dose_display_var.set(f"Dose: {dose_val:.3f} µSv/h")

        self.canvas.draw_idle()
        self.master.after(100, self.update_plot)  # ~10 Hz


def main():
    rospy.init_node("spectrum_gui", disable_signals=True)

    root = tk.Tk()
    gui = SpectrumGUI(root)

    try:
        root.mainloop()
    finally:
        rospy.signal_shutdown("GUI closed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import numpy as np
import rospy

from std_msgs.msg import UInt32MultiArray, UInt32, Float64

import matplotlib.pyplot as plt


# Global state updated by callbacks
latest_spectrum = None
latest_cps = None
latest_dose = None
data_dirty = False


def spectrum_callback(msg: UInt32MultiArray):
    global latest_spectrum, data_dirty
    latest_spectrum = np.array(msg.data, dtype=np.uint32)
    data_dirty = True


def cps_callback(msg: UInt32):
    global latest_cps
    latest_cps = int(msg.data)


def dose_callback(msg: Float64):
    global latest_dose
    latest_dose = float(msg.data)


def main():
    rospy.init_node("spectrum_plot")

    # Backwards compatible: support both ~topic and ~spectrum_topic
    default_spec_topic = "/sigma50/spectrum"
    spec_topic = rospy.get_param(
        "~spectrum_topic",
        rospy.get_param("~topic", default_spec_topic),
    )

    cps_topic = rospy.get_param("~cps_topic", "/sigma50/cps")
    dose_topic = rospy.get_param("~dose_topic", "/sigma50/dose_rate")

    rospy.loginfo("SpectrumPlot: spectrum_topic = %s", spec_topic)
    rospy.loginfo("SpectrumPlot: cps_topic      = %s", cps_topic)
    rospy.loginfo("SpectrumPlot: dose_topic     = %s", dose_topic)

    # Subscriptions
    rospy.Subscriber(spec_topic, UInt32MultiArray, spectrum_callback, queue_size=10)
    rospy.Subscriber(cps_topic, UInt32, cps_callback, queue_size=10)
    rospy.Subscriber(dose_topic, Float64, dose_callback, queue_size=10)

    # Matplotlib setup
    plt.ion()
    fig, ax = plt.subplots()

    # initialise with empty data
    line, = ax.plot([], [], lw=1)
    ax.set_xlabel("Energy channel")
    ax.set_ylabel("Counts")
    ax.set_title("Spectrum")

    # Text overlays in top-left
    text_cps = ax.text(
        0.02,
        0.95,
        "CPS: --",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )
    text_dose = ax.text(
        0.02,
        0.90,
        "Dose rate: -- µSv/h",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )

    global data_dirty, latest_spectrum, latest_cps, latest_dose

    rate = rospy.Rate(10)  # update up to 10 Hz

    while not rospy.is_shutdown():
        if data_dirty and latest_spectrum is not None:
            data_dirty = False

            x = np.arange(len(latest_spectrum))
            y = latest_spectrum

            line.set_xdata(x)
            line.set_ydata(y)

            # X axis: channels
            ax.set_xlim(0, max(1, len(y)))

            # Y axis: counts
            y_max = int(y.max()) if y.size > 0 else 1
            if y_max < 1:
                y_max = 1
            ax.set_ylim(0, y_max * 1.1)

            # Update CPS text
            if latest_cps is not None:
                text_cps.set_text(f"CPS: {latest_cps}")
            else:
                text_cps.set_text("CPS: --")

            # Update Dose text (µSv/h)
            if latest_dose is not None:
                text_dose.set_text(f"Dose rate: {latest_dose:.3f} µSv/h")
            else:
                text_dose.set_text("Dose rate: -- µSv/h")

            fig.canvas.draw()
            fig.canvas.flush_events()

        plt.pause(0.01)
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

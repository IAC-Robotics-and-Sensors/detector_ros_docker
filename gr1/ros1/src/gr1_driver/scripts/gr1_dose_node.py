#!/usr/bin/env python3

import os
import csv
import numpy as np
import rospy

from std_msgs.msg import UInt32MultiArray, Float32


def load_dose_lookup(path: str, n_channels: int) -> np.ndarray:
    """
    Load per-channel dose coefficients from CSV.

    Expected format: at least 2 columns; we assume:
        col0 = channel index (int)
        last column = dose_per_count (µSv/h per count-per-second)

    Any channels not present in the file get dose 0.0.
    """
    coeffs = np.zeros(n_channels, dtype=np.float64)

    if not os.path.isfile(path):
        rospy.logwarn(f"GR1DoseCalculator: lookup CSV not found: {path}")
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
        rospy.logerr(f"GR1DoseCalculator: error reading lookup CSV: {e}")

    nonzero = np.count_nonzero(coeffs)
    rospy.loginfo(
        f"GR1DoseCalculator: loaded dose coefficients for {nonzero} / {n_channels} channels"
    )
    return coeffs


class GR1DoseCalculator:
    def __init__(self):
        # Parameters
        self.spectrum_topic = rospy.get_param("~spectrum_topic", "/gr1/spectrum")
        self.dose_topic = rospy.get_param("~dose_topic", "/gr1/dose_rate")
        self.n_channels = int(rospy.get_param("~n_channels", 4096))

        default_lookup = os.path.join(
            rospy.get_param("~package_path", "/catkin_ws/src/gr1_driver"),
            "config",
            "bin_energy_dose_lookup.csv",
        )
        self.lookup_path = rospy.get_param("~lookup_path", default_lookup)

        rospy.loginfo(
            f"GR1DoseCalculator: using lookup CSV: {self.lookup_path}"
        )

        # Load lookup table
        self.dose_coeffs = load_dose_lookup(self.lookup_path, self.n_channels)

        # Publisher & subscriber
        # 👉 Advertise Float32, not Float64
        self.dose_pub = rospy.Publisher(
            self.dose_topic, Float32, queue_size=10
        )
        self.spec_sub = rospy.Subscriber(
            self.spectrum_topic, UInt32MultiArray, self.spectrum_cb, queue_size=10
        )

        rospy.loginfo(
            f"GR1DoseCalculator node started. "
            f"Subscribing to {self.spectrum_topic}, publishing dose rate to {self.dose_topic} (µSv/h, Float32)."
        )

    def spectrum_cb(self, msg: UInt32MultiArray):
        data = np.array(msg.data, dtype=np.float64)
        if data.size == 0:
            return

        n = min(data.size, self.dose_coeffs.size)
        # dose_coeffs units: µSv/h per count-per-second; counts here are per 1 s,
        # so counts ≈ cps → dose rate directly in µSv/h
        dose_rate = float(np.sum(data[:n] * self.dose_coeffs[:n]))/1000

        out = Float32()
        out.data = dose_rate
        self.dose_pub.publish(out)


def main():
    rospy.init_node("gr1_dose_node")
    _node = GR1DoseCalculator()
    rospy.spin()


if __name__ == "__main__":
    main()

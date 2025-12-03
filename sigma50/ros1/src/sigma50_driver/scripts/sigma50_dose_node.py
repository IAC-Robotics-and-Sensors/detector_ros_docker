#!/usr/bin/env python3

import os
import csv
from datetime import datetime

import numpy as np
import rospy
from std_msgs.msg import UInt32MultiArray, Float32


class Sigma50DoseCalculator:
    """
    Node that subscribes to /sigma50/spectrum (per-second spectrum),
    applies a bin→dose conversion lookup, and publishes the dose rate
    for that second as uSv/h (Float32).
    """

    def __init__(self):
        rospy.init_node("sigma50_dose_node")

        # Parameters
        self.spectrum_topic = rospy.get_param("~spectrum_topic", "/sigma50/spectrum")
        self.dose_topic = rospy.get_param("~dose_topic", "/sigma50/dose_rate")
        self.n_channels = rospy.get_param("~n_channels", 4096)

        # Path to lookup CSV
        default_lookup = self._default_lookup_path()
        self.lookup_path = rospy.get_param(
            "~lookup_path",
            default_lookup,
        )

        rospy.loginfo("Sigma50DoseCalculator: using lookup CSV: %s", self.lookup_path)

        # Load Chi (nSv/h per cps) per bin
        self.chi = self._load_lookup(self.lookup_path, self.n_channels)

        # Subscriber and publisher
        self.dose_pub = rospy.Publisher(self.dose_topic, Float32, queue_size=10)
        self.spectrum_sub = rospy.Subscriber(
            self.spectrum_topic, UInt32MultiArray, self.spectrum_callback, queue_size=10
        )

        rospy.loginfo(
            "Sigma50DoseCalculator node started. "
            "Subscribing to %s, publishing dose rate to %s (Float32 uSv/h)",
            self.spectrum_topic,
            self.dose_topic,
        )

    @staticmethod
    def _default_lookup_path():
        """
        Try to resolve $(find sigma50_driver)/config/bin_energy_dose_lookup.csv.
        If rospkg is missing or resolution fails, fall back to relative path
        next to this file.
        """
        try:
            import rospkg

            rp = rospkg.RosPack()
            pkg_path = rp.get_path("sigma50_driver")
            return os.path.join(pkg_path, "config", "bin_energy_dose_lookup.csv")
        except Exception:
            # Fallback: relative to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            return os.path.join(
                script_dir, "..", "config", "bin_energy_dose_lookup.csv"
            )

    @staticmethod
    def _load_lookup(path, n_channels):
        """
        Load Chi_for_bin_nSv_h_per_cps into a numpy array of length n_channels.

        CSV columns:
          bin, E_bin_MeV, Chi_for_bin_nSv_h_per_cps

        We map bin N -> channel index N-1 (0-based).
        Missing bins and NaNs get Chi = 0.
        """
        chi = np.zeros(n_channels, dtype=np.float64)

        if not os.path.exists(path):
            rospy.logwarn(
                "Sigma50DoseCalculator: lookup CSV %s not found. "
                "All Chi set to 0; dose rate will always be 0.",
                path,
            )
            return chi

        try:
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        b = int(row.get("bin", "").strip())
                    except Exception:
                        continue

                    idx = b - 1  # bin index to 0-based channel index
                    if idx < 0 or idx >= n_channels:
                        continue

                    chi_val_str = row.get("Chi_for_bin_nSv_h_per_cps", "")
                    if chi_val_str is None or chi_val_str.strip() == "":
                        c = 0.0
                    else:
                        try:
                            c = float(chi_val_str)
                        except Exception:
                            c = 0.0

                    # Replace NaN with 0
                    if c != c:  # NaN check
                        c = 0.0

                    chi[idx] = c

            non_zero = int(np.count_nonzero(chi))
            rospy.loginfo(
                "Sigma50DoseCalculator: loaded Chi for %d / %d channels",
                non_zero,
                n_channels,
            )
        except Exception as e:
            rospy.logerr("Sigma50DoseCalculator: error loading lookup CSV: %s", e)

        return chi

    def spectrum_callback(self, msg: UInt32MultiArray):
        # Convert incoming spectrum to numpy
        spec = np.array(msg.data, dtype=np.float64)

        # Align lengths
        if spec.size > self.chi.size:
            spec = spec[: self.chi.size]
        elif spec.size < self.chi.size:
            padded = np.zeros(self.chi.size, dtype=np.float64)
            padded[: spec.size] = spec
            spec = padded

        # ---------------------------------------------------------
        # DOSE CALCULATION
        # Chi is nSv/h per cps
        # spec[] is counts in one second -> cps = counts
        # ---------------------------------------------------------
        dose_uSv_h = float(np.dot(spec, self.chi)) / 1000.0

        # ---------------------------------------------------------
        # Publish as Float32
        # ---------------------------------------------------------
        msg_out = Float32()
        msg_out.data = dose_uSv_h
        self.dose_pub.publish(msg_out)

        rospy.logdebug("Dose rate: %.6f uSv/h", dose_uSv_h)


def main():
    node = Sigma50DoseCalculator()
    rospy.spin()


if __name__ == "__main__":
    main()

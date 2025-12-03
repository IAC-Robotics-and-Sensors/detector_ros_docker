#!/usr/bin/env python3
import os
import csv
import numpy as np
import rospy
from std_msgs.msg import UInt32MultiArray, Float32


class HamamatsuDoseCalculator:
    def __init__(self):
        rospy.init_node("hamamatsu_dose_node")

        # Parameters
        self.spectrum_topic = rospy.get_param("~spectrum_topic", "/hamamatsu/spectrum")
        self.dose_topic = rospy.get_param("~dose_topic", "/hamamatsu/dose_rate")
        self.lookup_path = rospy.get_param(
            "~lookup_path",
            "/catkin_ws/src/hamamatsu_driver/config/bin_energy_dose_lookup.csv"
        )
        self.n_channels = rospy.get_param("~n_channels", 4096)

        # Load dose coefficients
        self.coeff = np.zeros(self.n_channels, dtype=np.float64)
        self.load_lookup()

        # ROS I/O
        rospy.Subscriber(self.spectrum_topic, UInt32MultiArray, self.cb_spectrum)
        self.pub = rospy.Publisher(self.dose_topic, Float32, queue_size=10)

        rospy.loginfo(
            "HamamatsuDoseCalculator: started, listening to %s → publishing to %s",
            self.spectrum_topic, self.dose_topic
        )

        rospy.spin()

    # ----------------------------------------------------
    # Load dose lookup CSV
    # ----------------------------------------------------
    def load_lookup(self):
        if not os.path.exists(self.lookup_path):
            rospy.logerr("Dose lookup CSV not found: %s", self.lookup_path)
            return

        count = 0
        with open(self.lookup_path, "r") as f:
            r = csv.reader(f)
            for row in r:
                if len(row) < 2:
                    continue
                try:
                    ch = int(row[0])
                    dose = float(row[1])
                except ValueError:
                    continue
                if 0 <= ch < self.n_channels:
                    self.coeff[ch] = dose
                    count += 1

        rospy.loginfo(
            "HamamatsuDoseCalculator: loaded dose coefficients for %d / %d channels",
            count, self.n_channels
        )

    # ----------------------------------------------------
    # Spectrum callback → compute dose rate
    # ----------------------------------------------------
    def cb_spectrum(self, msg):
        spec = np.array(msg.data, dtype=np.float64)
        if spec.size > self.n_channels:
            spec = spec[:self.n_channels]

        # Dose = Σ (counts × dose_per_count)
        dose_rate = float(np.sum(spec * self.coeff))/1000

        self.pub.publish(Float32(dose_rate))


if __name__ == "__main__":
    HamamatsuDoseCalculator()

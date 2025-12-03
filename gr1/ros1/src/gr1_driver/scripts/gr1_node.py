#!/usr/bin/env python3

import os
import sys
import time
from datetime import datetime

import numpy as np
import rospy
from std_msgs.msg import UInt32MultiArray, UInt32

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


from gr1_driver.gr1_detector import GR1


class SpectrumCSVLogger:
    """
    Collects per-second spectra and writes them to a CSV on shutdown.

    CSV layout:

        first row:      bin_energy,<t0>,<t1>,...,<tN-1>
        rows 1..Nch:    <channel_index>, counts_at_t0, counts_at_t1, ...
        last row:       cps, cps_at_t0, cps_at_t1, ...

    Where <tK> is a wall-clock timestamp string, e.g. "2025-02-24 15:30:01".
    """

    def __init__(self, detector_name: str, log_dir: str = "/data", enabled: bool = True):
        self.enabled = enabled
        self.detector_name = detector_name
        self.log_dir = log_dir

        self.spectra = []   # list of np.ndarray[channels]
        self.cps = []       # list of ints
        self.times = []     # list of timestamp strings
        self.n_channels = None

        if not self.enabled:
            return

        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(
            self.log_dir, f"{ts}_{self.detector_name}_spectrum.csv"
        )
        rospy.loginfo("Spectrum CSV logger initialised: %s", self.filename)

    def add_sample(self, spectrum: np.ndarray, timestamp=None):
        """
        Add one spectrum sample (per-second spectrum) with its timestamp.
        """
        if not self.enabled:
            return

        spec = np.array(spectrum, dtype=np.uint64).flatten()

        # Track channel count and reset if it ever changes
        if self.n_channels is None:
            self.n_channels = spec.shape[0]
        elif spec.shape[0] != self.n_channels:
            rospy.logwarn_once(
                "SpectrumCSVLogger: spectrum length changed (%d -> %d), "
                "resetting stored data",
                self.n_channels, spec.shape[0],
            )
            self.spectra = []
            self.cps = []
            self.times = []
            self.n_channels = spec.shape[0]

        self.spectra.append(spec)
        self.cps.append(int(spec.sum()))

        if timestamp is None:
            timestamp = datetime.now()

        # Store as a human-readable string
        if hasattr(timestamp, "strftime"):
            t_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            t_str = str(timestamp)
        self.times.append(t_str)

    def write_csv(self):
        if not self.enabled:
            return

        # If no spectra -> still create a marker CSV
        if not self.spectra or self.n_channels is None:
            try:
                with open(self.filename, "w") as f:
                    f.write("bin_energy\n")
                    f.write("# SpectrumCSVLogger: no spectra recorded\n")
                rospy.loginfo(
                    "SpectrumCSVLogger: no data; wrote marker CSV %s",
                    self.filename,
                )
            except Exception as e:
                rospy.logerr("SpectrumCSVLogger: failed to create marker file: %s", e)
            return

        n_steps = len(self.spectra)

        # Safety: if times length mismatches, rebuild simple indices
        if len(self.times) != n_steps:
            rospy.logwarn(
                "SpectrumCSVLogger: times length (%d) != spectra length (%d); "
                "rebuilding times as simple indices.",
                len(self.times), n_steps,
            )
            self.times = [f"{i}" for i in range(n_steps)]

        # Build data array [channels+1 x n_steps]
        data = np.zeros((self.n_channels + 1, n_steps), dtype=np.uint64)
        for j, spec in enumerate(self.spectra):
            data[: self.n_channels, j] = spec
            data[self.n_channels, j] = self.cps[j]

        # Header: first cell is bin_energy, then timestamps across the top
        header = ",".join(["bin_energy"] + list(self.times))

        try:
            with open(self.filename, "w") as f:
                # Header row
                f.write(header + "\n")

                # Channel rows: channel index in first column
                for ch in range(self.n_channels):
                    row_vals = data[ch]
                    line = ",".join([str(ch)] + [str(v) for v in row_vals])
                    f.write(line + "\n")

                # Last row: CPS
                cps_row = data[self.n_channels]
                cps_line = ",".join(["cps"] + [str(v) for v in cps_row])
                f.write(cps_line + "\n")

            rospy.loginfo(
                "SpectrumCSVLogger: saved %d spectra (%d channels) to %s",
                n_steps, self.n_channels, self.filename,
            )

        except Exception as e:
            rospy.logerr("SpectrumCSVLogger: failed to save CSV: %s", e)


def wait_for_spectrum(data_q, timeout=2.0):
    """
    Block until we get a 'spectrum' message from data_q or timeout.

    Expected format from detector worker:
      ["spectrum", [spectrum_array, elapsed_seconds]]
    """
    start = time.time()
    while (time.time() - start) < timeout and not rospy.is_shutdown():
        if not data_q.empty():
            msg = data_q.get()
            if (
                isinstance(msg, list)
                and len(msg) == 2
                and msg[0] == "spectrum"
                and isinstance(msg[1], (list, tuple))
                and len(msg[1]) == 2
            ):
                spectrum, elapsed = msg[1]
                spectrum = np.array(spectrum, dtype=np.uint32)
                return spectrum, float(elapsed)
        rospy.sleep(0.01)
    return None, None


def main():
    rospy.init_node("gr1_node")

    spectrum_pub = rospy.Publisher("/gr1/spectrum", UInt32MultiArray, queue_size=10)
    cps_pub = rospy.Publisher("/gr1/cps", UInt32, queue_size=10)

    verbose = rospy.get_param("~verbose", 0)
    log_dir = rospy.get_param("~log_dir", "/data")
    log_enabled = rospy.get_param("~log_enabled", True)

    logger = SpectrumCSVLogger("gr1", log_dir=log_dir, enabled=log_enabled)

    # Start detector worker process
    worker, q_list = GR1(verbose=verbose)
    cmd_q, data_q = q_list

    cmd_q.put(["reset", None])

    prev_spectrum = np.zeros(4096, dtype=np.uint32)
    rate = rospy.Rate(1.0)  # 1 Hz

    rospy.loginfo(
        "GR1 ROS node started, publishing /gr1/spectrum and /gr1/cps (integers); "
        "logging to %s",
        log_dir,
    )

    try:
        while not rospy.is_shutdown():
            cmd_q.put(["spectrum", None])

            spectrum, elapsed = wait_for_spectrum(data_q, timeout=2.0)
            if spectrum is None:
                rospy.logwarn_throttle(5.0, "GR1: no spectrum received within timeout")
                rate.sleep()
                continue

            # per-second spectrum = difference from previous integrated
            delta_spectrum = spectrum - prev_spectrum
            prev_spectrum = spectrum

            # total counts (CPS for that second)
            total_counts = int(np.sum(delta_spectrum))

            # Timestamp for this spectrum
            ts = datetime.now()
            logger.add_sample(delta_spectrum, timestamp=ts)

            # Publish spectrum
            spec_msg = UInt32MultiArray()
            spec_msg.data = delta_spectrum.astype(np.uint32).tolist()
            spectrum_pub.publish(spec_msg)

            # Publish CPS
            cps_msg = UInt32()
            cps_msg.data = total_counts
            cps_pub.publish(cps_msg)

            rate.sleep()

    finally:
        rospy.loginfo("GR1 ROS node shutting down, sending quit to worker")
        try:
            cmd_q.put(["quit", None])
        except Exception:
            pass
        # Write CSV on shutdown
        logger.write_csv()


if __name__ == "__main__":
    main()

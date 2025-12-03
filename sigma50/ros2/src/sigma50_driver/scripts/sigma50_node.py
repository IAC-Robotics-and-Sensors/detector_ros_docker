#!/usr/bin/env python3

import os
import sys
import time
from datetime import datetime

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt32MultiArray, UInt32

# Make sure we can import the local Python package:
#   <package root>/src/sigma50_driver/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sigma50_driver.sigma_detector import Sigma50  # type: ignore[import]


class SpectrumCSVLogger:
    """
    Collects per-second spectra and writes them to a CSV on shutdown.

    CSV layout:

        first row:      bin_energy,<t0>,<t1>,...,<tN-1>
        rows 1..Nch:    <channel_index>, counts_at_t0, counts_at_t1, ...
        last row:       cps, cps_at_t0, cps_at_t1, ...

    Where <tK> is a wall-clock timestamp string, e.g. "2025-02-24 15:30:01".
    """

    def __init__(
        self,
        detector_name: str,
        log_dir: str = "/data",
        enabled: bool = True,
        logger: Node = None,
    ):
        self.enabled = enabled
        self.detector_name = detector_name
        self.log_dir = log_dir
        self._logger = logger.get_logger() if isinstance(logger, Node) else None

        self.spectra = []   # list of np.ndarray[channels]
        self.cps = []       # list of ints
        self.times = []     # list of datetime or string
        self.n_channels = None

        if not self.enabled:
            return

        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(
            self.log_dir, f"{ts}_{self.detector_name}_spectrum.csv"
        )
        self._log_info(f"Spectrum CSV logger initialised: {self.filename}")

    # Small helpers so we can log even if no ROS logger is passed
    def _log_info(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.info(msg)
        else:
            print(msg)

    def _log_warn(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.warn(msg)
        else:
            print(f"[WARN] {msg}")

    def _log_err(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.error(msg)
        else:
            print(f"[ERROR] {msg}")

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
            self._log_warn(
                f"SpectrumCSVLogger: spectrum length changed "
                f"({self.n_channels} -> {spec.shape[0]}), resetting stored data"
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
                self._log_info(
                    f"SpectrumCSVLogger: no data; wrote marker CSV {self.filename}"
                )
            except Exception as e:
                self._log_err(
                    f"SpectrumCSVLogger: failed to create marker file: {e}"
                )
            return

        n_steps = len(self.spectra)

        # Safety: if times length mismatches, rebuild simple indices
        if len(self.times) != n_steps:
            self._log_warn(
                "SpectrumCSVLogger: times length (%d) != spectra length (%d); "
                "rebuilding times as simple indices."
                % (len(self.times), n_steps)
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

            self._log_info(
                "SpectrumCSVLogger: saved %d spectra (%d channels) to %s"
                % (n_steps, self.n_channels, self.filename)
            )

        except Exception as e:
            self._log_err(f"SpectrumCSVLogger: failed to save CSV: {e}")


def wait_for_spectrum(data_q, timeout=2.0):
    """
    Block until we get a 'spectrum' message from data_q or timeout.

    Expected format from detector worker:
      ["spectrum", [spectrum_array, elapsed_seconds]]
    """
    start = time.time()
    while (time.time() - start) < timeout:
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
        time.sleep(0.01)
        if not rclpy.ok():
            break
    return None, None


class Sigma50Node(Node):
    def __init__(self):
        super().__init__("sigma50_node")

        # Declare parameters
        self.declare_parameter("verbose", 0)
        self.declare_parameter("log_dir", "/data")
        self.declare_parameter("log_enabled", True)

        verbose = self.get_parameter("verbose").get_parameter_value().integer_value
        log_dir = self.get_parameter("log_dir").get_parameter_value().string_value
        log_enabled = (
            self.get_parameter("log_enabled").get_parameter_value().bool_value
        )

        self.spectrum_pub = self.create_publisher(
            UInt32MultiArray, "/sigma50/spectrum", 10
        )
        self.cps_pub = self.create_publisher(UInt32, "/sigma50/cps", 10)

        self.logger = SpectrumCSVLogger(
            "sigma50", log_dir=log_dir, enabled=log_enabled, logger=self
        )

        self.worker, q_list = Sigma50(verbose=verbose)
        self.cmd_q, self.data_q = q_list

        # Reset detector spectrum at start
        self.cmd_q.put(["reset", None])

        self.prev_spectrum = np.zeros(4096, dtype=np.uint32)

        self.get_logger().info(
            "Sigma50 ROS2 node started, publishing /sigma50/spectrum and /sigma50/cps; "
            f"logging to {log_dir}"
        )

    def loop_once(self):
        # Request a spectrum
        self.cmd_q.put(["spectrum", None])

        spectrum, elapsed = wait_for_spectrum(self.data_q, timeout=2.0)
        if spectrum is None:
            self.get_logger().warn("Sigma50: no spectrum received within timeout")
            time.sleep(1.0)
            return

        # per-second spectrum = difference from previous integrated
        delta_spectrum = spectrum - self.prev_spectrum
        self.prev_spectrum = spectrum

        # total counts (CPS for that second)
        total_counts = int(np.sum(delta_spectrum))

        # Log to CSV (one column per second)
        self.logger.add_sample(delta_spectrum)

        # Publish spectrum
        spec_msg = UInt32MultiArray()
        spec_msg.data = delta_spectrum.astype(np.uint32).tolist()
        self.spectrum_pub.publish(spec_msg)

        # Publish CPS
        cps_msg = UInt32()
        cps_msg.data = total_counts
        self.cps_pub.publish(cps_msg)

        # 1 Hz loop
        time.sleep(1.0)

    def shutdown(self):
        self.get_logger().info("Sigma50 ROS2 node shutting down, sending quit to worker")
        try:
            self.cmd_q.put(["quit", None])
        except Exception:
            pass
        self.logger.write_csv()


def main(args=None):
    rclpy.init(args=args)
    node = Sigma50Node()
    try:
        while rclpy.ok():
            node.loop_once()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

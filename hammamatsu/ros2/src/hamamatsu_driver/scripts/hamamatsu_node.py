#!/usr/bin/env python3

import os
import sys
import time
from datetime import datetime

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt32MultiArray, UInt32

# Make sure we can import hamamatsu_driver.hamamatsu_detector
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from hamamatsu_driver.hamamatsu_detector import Hamamatsu  # type: ignore[import]


class SpectrumCSVLogger:
    def __init__(
        self,
        detector_name: str,
        log_dir: str = "/data",
        enabled: bool = True,
        logger: Node | None = None,
    ):
        self.enabled = enabled
        self.detector_name = detector_name
        self.log_dir = log_dir
        self._logger = logger.get_logger() if isinstance(logger, Node) else None

        self.spectra = []
        self.cps = []
        self.times = []
        self.n_channels = None

        if not self.enabled:
            return

        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(
            self.log_dir, f"{ts}_{self.detector_name}_spectrum.csv"
        )
        self._log_info(f"Spectrum CSV logger initialised: {self.filename}")

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
        if not self.enabled:
            return

        spec = np.array(spectrum, dtype=np.uint64).flatten()

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

        if hasattr(timestamp, "strftime"):
            t_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            t_str = str(timestamp)
        self.times.append(t_str)

    def write_csv(self):
        if not self.enabled:
            return

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

        if len(self.times) != n_steps:
            self._log_warn(
                "SpectrumCSVLogger: times length (%d) != spectra length (%d); "
                "rebuilding times as simple indices." % (len(self.times), n_steps)
            )
            self.times = [f"{i}" for i in range(n_steps)]

        data = np.zeros((self.n_channels + 1, n_steps), dtype=np.uint64)
        for j, spec in enumerate(self.spectra):
            data[: self.n_channels, j] = spec
            data[self.n_channels, j] = self.cps[j]

        header = ",".join(["bin_energy"] + list(self.times))

        try:
            with open(self.filename, "w") as f:
                f.write(header + "\n")

                for ch in range(self.n_channels):
                    row_vals = data[ch]
                    line = ",".join([str(ch)] + [str(v) for v in row_vals])
                    f.write(line + "\n")

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
    start = time.time()
    while (time.time() - start) < timeout and rclpy.ok():
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
    return None, None


class HamamatsuNode(Node):
    def __init__(self):
        super().__init__("hamamatsu_node")

        self.declare_parameter("verbose", 0)
        self.declare_parameter("log_dir", "/data")
        self.declare_parameter("log_enabled", True)

        verbose = self.get_parameter("verbose").get_parameter_value().integer_value
        log_dir = self.get_parameter("log_dir").get_parameter_value().string_value
        log_enabled = (
            self.get_parameter("log_enabled").get_parameter_value().bool_value
        )

        self.spectrum_pub = self.create_publisher(
            UInt32MultiArray, "/hamamatsu/spectrum", 10
        )
        self.cps_pub = self.create_publisher(UInt32, "/hamamatsu/cps", 10)

        self.logger = SpectrumCSVLogger(
            "hamamatsu", log_dir=log_dir, enabled=log_enabled, logger=self
        )

        self.worker, q_list = Hamamatsu(verbose=verbose)
        self.cmd_q, self.data_q = q_list

        self.cmd_q.put(["reset", None])

        self.prev_spectrum = np.zeros(4096, dtype=np.uint32)

        self.get_logger().info(
            "Hamamatsu ROS2 node started, publishing /hamamatsu/spectrum and /hamamatsu/cps; "
            f"logging to {log_dir}"
        )

    def loop_once(self):
        self.cmd_q.put(["spectrum", None])

        spectrum, elapsed = wait_for_spectrum(self.data_q, timeout=2.0)
        if spectrum is None:
            self.get_logger().warn(
                "Hamamatsu: no spectrum received within timeout"
            )
            time.sleep(1.0)
            return

        delta_spectrum = spectrum - self.prev_spectrum
        self.prev_spectrum = spectrum

        total_counts = int(np.sum(delta_spectrum))

        ts = datetime.now()
        self.logger.add_sample(delta_spectrum, timestamp=ts)

        spec_msg = UInt32MultiArray()
        spec_msg.data = delta_spectrum.astype(np.uint32).tolist()
        self.spectrum_pub.publish(spec_msg)

        cps_msg = UInt32()
        cps_msg.data = total_counts
        self.cps_pub.publish(cps_msg)

        time.sleep(1.0)

    def shutdown(self):
        self.get_logger().info(
            "Hamamatsu ROS2 node shutting down, sending quit to worker"
        )
        try:
            self.cmd_q.put(["quit", None])
        except Exception:
            pass
        self.logger.write_csv()


def main(args=None):
    rclpy.init(args=args)
    node = HamamatsuNode()
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

#!/usr/bin/env python3

import time
from multiprocessing import Queue, Process
from typing import Tuple, List, Any

import hid
import numpy as np


def GR1Loop(cmd_q: Queue, data_q: Queue, verbose: int = 1) -> None:
    """
    Connects to the GR1 detector, accumulates a 4096-bin spectrum, and responds to commands.

    Commands (from cmd_q):
      "quit"
      "reset"
      "spectrum"
      ["timedSpectrum", duration_seconds]
      ["raw_start", min_interval_seconds]
      "raw_stop"

    Messages (to data_q):
      ["connectionStatus", [True/False, None]]
      ["spectrum", [spectrum, elapsed_seconds]]
      ["timedSpectrum", [timed_spectrum, elapsed_seconds]]
      ["raw", [frame_bytes_list, elapsed_seconds_since_tick]]
    """
    if verbose > 0:
        print("GR1: Opening")

    h = hid.device()

    timedSpectrumRequested = False
    timedSpectrumDuration = 0.0
    refSpectrum = None  # type: ignore[assignment]

    raw_enabled = False
    raw_last_send = 0.0
    raw_min_interval = 0.0

    try:
        # Kromek GR1 VID/PID
        h.open(0x04D8, 0x0000)
        h.set_nonblocking(1)

        data_q.put(["connectionStatus", [True, None]])

        if verbose > 0:
            try:
                print(f"GR1: Manufacturer - {h.get_manufacturer_string()}")
                print(f"GR1: Product      - {h.get_product_string()}")
            except Exception:
                pass

        tick = time.time()
        spectrum = np.zeros(4096, dtype=np.uint32)

        try:
            while True:
                GR1_data = h.read(62)
                now = time.time()

                if GR1_data:
                    b1 = GR1_data[1]
                    b2 = GR1_data[2]
                    channel = ((b1 & 0xFF) << 4) | ((b2 & 0xFF) >> 4)
                    if 0 <= channel < 4096:
                        spectrum[channel] += 1

                    if verbose > 1:
                        print("GR1_data:", GR1_data)
                        print("Channel:", channel)

                    if raw_enabled and (now - raw_last_send) >= raw_min_interval:
                        data_q.put(["raw", [list(GR1_data), now - tick]])
                        raw_last_send = now

                elapsed_since_tick = now - tick
                if (
                    timedSpectrumRequested
                    and elapsed_since_tick > timedSpectrumDuration
                    and refSpectrum is not None
                ):
                    timedSpectrum = spectrum - refSpectrum
                    data_q.put(["timedSpectrum", [timedSpectrum, elapsed_since_tick]])
                    timedSpectrumRequested = False

                if not cmd_q.empty():
                    msg: Any = cmd_q.get()
                    if isinstance(msg, (list, tuple)) and len(msg) == 2:
                        command, data = msg
                    else:
                        command, data = msg, None

                    if verbose > 0:
                        print("GR1: Command:", command)

                    if command == "quit":
                        break

                    elif command == "reset":
                        spectrum = np.zeros(4096, dtype=np.uint32)
                        tick = time.time()

                    elif command == "spectrum":
                        tock = time.time()
                        data_q.put(["spectrum", [spectrum, tock - tick]])

                    elif command == "timedSpectrum":
                        refSpectrum = np.copy(spectrum)
                        timedSpectrumRequested = True
                        timedSpectrumDuration = float(data) if data is not None else 0.0
                        tick = time.time()

                    elif command == "raw_start":
                        raw_min_interval = float(data) if data is not None else 0.0
                        raw_last_send = 0.0
                        raw_enabled = True
                        if verbose > 0:
                            print(
                                f"GR1: raw streaming enabled "
                                f"(min interval {raw_min_interval:.6f}s)"
                            )

                    elif command == "raw_stop":
                        raw_enabled = False
                        if verbose > 0:
                            print("GR1: raw streaming disabled")

                time.sleep(0.001)

        except KeyboardInterrupt:
            if verbose > 0:
                print("GR1: Interrupted by user")

    except OSError:
        if verbose > 0:
            print("GR1: Device cannot be opened")
        try:
            data_q.put(["connectionStatus", [False, None]])
        except Exception:
            pass

    finally:
        try:
            h.close()
        except Exception:
            pass
        try:
            data_q.put(["connectionStatus", [False, None]])
        except Exception:
            pass


def GR1(
    port: None = None, uhubctl: None = None, verbose: int = 1
) -> Tuple[Process, List[Queue]]:
    """
    Start GR1 acquisition in a separate process.

    Returns:
      (process, [cmd_q, data_q])
    """
    cmd_q: Queue = Queue()
    data_q: Queue = Queue()

    worker = Process(
        target=GR1Loop,
        args=(cmd_q, data_q, verbose),
        daemon=True,
    )
    worker.start()

    return worker, [cmd_q, data_q]

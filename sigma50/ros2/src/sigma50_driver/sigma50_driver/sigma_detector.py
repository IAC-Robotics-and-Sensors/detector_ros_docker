# sigma50_detector.py

import time
from multiprocessing import Queue, Process
from typing import Tuple, List, Any

import hid
import numpy as np


def Sigma50loop(cmd_q: Queue, data_q: Queue, verbose: int = 1) -> None:
    """
    Main acquisition loop for the Kromek Sigma 50 detector.

    Queues:
      cmd_q  : commands from main process
      data_q : messages back to main process

    Commands (from cmd_q):
      ["quit", None]
      ["reset", None]
      ["spectrum", None]
      ["timedSpectrum", duration_seconds]

    Messages (to data_q):
      ["connectionStatus", True/False, None]
      ["spectrum", [spectrum (np.uint32[4096]), elapsed_seconds]]
      ["timedSpectrum", [spectrum_diff, elapsed_seconds]]
    """
    h = hid.device()
    timedSpectrumRequested = False
    timedSpectrumDuration = 0.0
    refSpectrum = None  # type: ignore[assignment]

    try:
        # Open Sigma50 (VID/PID from original code)
        h.open(0x04D8, 0x0023)
        h.set_nonblocking(1)
        data_q.put(["connectionStatus", True, None])

        if verbose > 0:
            try:
                print(f"Sigma50: Manufacturer - {h.get_manufacturer_string()}")
                print(f"Sigma50: Product      - {h.get_product_string()}")
            except Exception:
                pass

        tick = time.time()
        spectrum = np.zeros(4096, dtype=np.uint32)
        BINCALLS = 0
        BINCALLSNOW = 0
        BINCALLSBEFORE = 0

        try:
            while True:
                Sigma50_data = h.read(63)  # 63 from Kromek spec

                if Sigma50_data:
                    # First byte 4 means good frame
                    if Sigma50_data[0] == 4:
                        i_sigma = 1

                        while i_sigma + 1 in range(1, 63):
                            byte1 = Sigma50_data[i_sigma]
                            byte2 = Sigma50_data[i_sigma + 1]

                            word2 = format(byte2, "08b")
                            valid = int(word2[7])

                            if valid == 0:
                                if verbose > 0:
                                    print("Sigma50: BIN NOT VALID")
                                break

                            if valid == 1:
                                if verbose > 0:
                                    print("Sigma50: BIN VALID")
                                BINCALLS += 1

                            channel = ((byte1 << 4)) | ((byte2 & 0xFF) >> 4)
                            if 0 <= channel < 4096:
                                spectrum[int(channel)] += 1

                            i_sigma += 2

                            if verbose > 1:
                                print("Sigma50_data[1]: ", Sigma50_data[1])
                                print("Sigma50_data[2]: ", Sigma50_data[2])
                                print("Channel: ", channel)

                # timedSpectrum handling
                elapsedT2 = time.time() - tick
                if (
                    timedSpectrumRequested
                    and elapsedT2 > timedSpectrumDuration
                    and refSpectrum is not None
                ):
                    timedSpectrum = spectrum - refSpectrum
                    data_q.put(["timedSpectrum", [timedSpectrum, elapsedT2]])
                    timedSpectrumRequested = False

                # Handle commands from cmd_q
                if not cmd_q.empty():
                    msg: Any = cmd_q.get()
                    if isinstance(msg, (list, tuple)) and len(msg) == 2:
                        command, data = msg
                    else:
                        command, data = msg, None

                    if verbose > 0:
                        print("Sigma50: Command:", command)

                    if command == "quit":
                        break

                    elif command == "reset":
                        if verbose > 0:
                            print("Sigma50: reset – total counts before reset:",
                                  int(np.sum(spectrum)))
                        spectrum = np.zeros(4096, dtype=np.uint32)
                        tick = time.time()

                    elif command == "spectrum":
                        BINCALLSNOW = BINCALLS
                        bincallsdiff = BINCALLSNOW - BINCALLSBEFORE
                        if verbose > 0:
                            print("Sigma50: BINS CALLED", bincallsdiff)
                        BINCALLSBEFORE = BINCALLSNOW

                        tock = time.time()
                        if verbose > 0:
                            print("Sigma50: counts:", int(np.sum(spectrum)))
                        data_q.put(["spectrum", [spectrum, tock - tick]])

                    elif command == "timedSpectrum":
                        refSpectrum = np.copy(spectrum)
                        timedSpectrumRequested = True
                        timedSpectrumDuration = float(data) if data is not None else 0.0
                        tick = time.time()

                time.sleep(0.0005)

        except KeyboardInterrupt:
            if verbose > 0:
                print("Sigma50: Interrupted by user")

    except OSError:
        if verbose > 0:
            print("Sigma50: Device cannot be opened")
        try:
            data_q.put(["connectionStatus", False, None])
        except Exception:
            pass

    finally:
        try:
            h.close()
        except Exception:
            pass
        try:
            data_q.put(["connectionStatus", False, None])
        except Exception:
            pass


def Sigma50(
    port: None = None, uhubctl: None = None, verbose: int = 0
) -> Tuple[Process, List[Queue]]:
    """
    Start Sigma50 acquisition in a separate process.

    Returns:
      (process, [cmd_q, data_q])
    """
    cmd_q: Queue = Queue()
    data_q: Queue = Queue()

    worker = Process(
        target=Sigma50loop,
        args=(cmd_q, data_q, verbose),
        daemon=True,
    )
    worker.start()

    # Wait for first status message (if any)
    timeout_start = time.time()
    while data_q.empty() and (time.time() - timeout_start) < 5.0:
        time.sleep(0.001)
    if not data_q.empty():
        print(data_q.get())

    return worker, [cmd_q, data_q]

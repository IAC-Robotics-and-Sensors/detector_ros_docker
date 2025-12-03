#!/usr/bin/env python3

import os
import time
from multiprocessing import Queue, Process
from typing import Tuple, List, Any

import usb.core
import usb.util
import numpy as np
from struct import unpack


class HamamatsuDetector:
    """
    Minimal Hamamatsu USB interface.

    - Finds device (VID=0x0661, PID=0x2917)
    - Reads frames, extracts detectorEvents, ADC channels, temperature, etc.
    - Exposes:
        - detectorEvents
        - channels (np.uint16[1048])
        - binnedChannels (after binChannels)
        - temperature
        - deviceTime
    """

    def __init__(self, port=None, uhubctl=None, verbose=0, virtual=False):
        self.port = port
        self.uhubctl = uhubctl
        self.verbose = verbose
        self.virtualDevice = virtual

        self.timeOverflows = 0
        self.previousTimeIndex = 0
        self.timeIndex = 0
        self.status = "pre init"

        self.detectorEvents = 0
        self.temperature = 0.0
        self.deviceTime = 0.0

        if self.virtualDevice:
            self.status = "OK"
            self.bootDuration = 0.0
            return

        tic = time.time()

        # Find device
        self.device = usb.core.find(idVendor=0x0661, idProduct=0x2917)
        if self.device is None:
            print("ERROR: Hamamatsu device not found")
            self.status = "device not found"
            return

        # Setup device, configuration, endpoint
        try:
            self.device.set_configuration()
            self.configuration = self.device.get_active_configuration()
            self.interface = self.configuration[(0, 0)]
            self.endpoint = self.interface[0]
            self.ep = self.endpoint
            self.maxPacketSize8 = self.ep.wMaxPacketSize
            self.maxPacketSize16 = self.maxPacketSize8 // 2
        except Exception as e:
            print("ERROR: Hamamatsu setup failed:", e)
            self.status = "setup failed"
            return

        self.bootDuration = time.time() - tic
        self.status = "OK"

        if self.verbose > 0:
            print("Hamamatsu: setup complete, bootDuration=%.3fs" % self.bootDuration)

    def processHeader(self) -> bool:
        """
        Reads one dataframe header from the device and updates:
           - detectorEvents
           - timeIndex
           - tempADC -> temperature
           - timeOverflows, deviceTime

        This version is more tolerant: USB read *timeouts* will just retry
        instead of causing an immediate failure. That makes it more patient
        for the very first spectrum.
        """
        if self.virtualDevice:
            self.detectorEvents = 1000
            self.timeIndex = (self.timeIndex + 1) % 65536
            self.tempADC = 50000
        else:
            # Loop until we see a valid header start
            while True:
                try:
                    self.data = self.device.read(
                        self.ep.bEndpointAddress,
                        self.ep.wMaxPacketSize,
                        timeout=100,  # 100 ms per read
                    )
                    (
                        self.headerStart,
                        self.detectorEvents,
                        self.timeIndex,
                        self.tempADC,
                    ) = unpack(">LHxxHHxxxx", self.data[0:16])

                except usb.core.USBError as e:
                    # On timeout, just keep trying instead of bailing out
                    msg = str(e).lower()
                    if "timed out" in msg or "timeout" in msg:
                        if self.verbose > 1:
                            print("Hamamatsu: USB timeout waiting for header, retrying...")
                        continue
                    # Any other USB error -> real failure
                    if self.verbose > 0:
                        print("Hamamatsu: USB error in processHeader:", e)
                    return False

                except Exception as e:
                    # Non-USB errors: treat as fatal
                    if self.verbose > 0:
                        print("Hamamatsu: unexpected error in processHeader:", e)
                    return False

                if self.headerStart == 1515870810:
                    break
                if self.verbose > 0:
                    print(
                        "Hamamatsu: bad header start value %d - resyncing"
                        % self.headerStart
                    )

            # remaining bytes from header packet are first 24 channels
            self.headerData = self.data[16:]

        # hama time
        if self.previousTimeIndex - self.timeIndex > 65000:
            self.timeOverflows += 1
        self.deviceTime = (65536 * self.timeOverflows + self.timeIndex) / 10.0
        self.temperature = 188.686 - 0.00348 * self.tempADC
        return True

    def processReadings(self) -> bool:
        """
        Reads remaining channel data for this dataframe.
        Fills self.channels (np.uint16[1048]).
        """
        self.channels = np.zeros(1048, dtype=np.uint16)

        if self.virtualDevice:
            self.channels[:1000] = np.random.randint(
                0, 65535, 1000, dtype=np.uint16
            )
            return True

        # first 24 channels came from headerData
        try:
            self.channels[:24] = unpack("<" + "H" * 24, self.headerData)
        except Exception as e:
            if self.verbose > 0:
                print("Hamamatsu: error unpacking headerData:", e)
            return False

        self.headerData = None

        # read remaining channels in packets
        for address in range(24, 1048, self.maxPacketSize16):
            try:
                self.data = self.device.read(
                    self.ep.bEndpointAddress,
                    self.ep.wMaxPacketSize,
                    timeout=100,
                )
                self.channels[address: address + self.maxPacketSize16] = unpack(
                    "<" + "H" * self.maxPacketSize16, self.data
                )
            except usb.core.USBError as e:
                msg = str(e).lower()
                if "timed out" in msg or "timeout" in msg:
                    if self.verbose > 0:
                        print(
                            "Hamamatsu: USB timeout reading channels at %d, aborting frame"
                            % address
                        )
                    return False
                if self.verbose > 0:
                    print(
                        "Hamamatsu: USB error in processReadings at %d: %s"
                        % (address, e)
                    )
                return False
            except Exception as e:
                if self.verbose > 0:
                    print("Hamamatsu: unexpected error in processReadings:", e)
                return False

        return True

    def binChannels(self, binning: int):
        """
        Apply simple binning to channels.
        """
        self.channelBinning = binning
        self.binnedChannels = np.floor_divide(
            self.channels, self.channelBinning
        ).astype(np.int32)


def HamamatsuLoop(
    cmd_q: Queue,
    data_q: Queue,
    port=None,
    uhubctl=None,
    verbose: int = 0,
    virtual: bool = False,
) -> None:
    """
    Simple Hamamatsu acquisition loop.

    Commands on cmd_q:
      - "quit"
      - "reset"
      - "spectrum"

    Messages on data_q:
      - ["connectionStatus", [True/False, reason]]
      - ["spectrum", [spectrum (np.uint32[4096]), elapsed_seconds]]
    """

    spectrum = np.zeros(4096, dtype=np.uint32)
    tic = time.time()
    finished = False

    # Try to set up detector
    hama = HamamatsuDetector(
        port=port, uhubctl=uhubctl, verbose=verbose, virtual=virtual
    )
    if hama.status != "OK":
        data_q.put(["connectionStatus", [False, hama.status]])
        return

    data_q.put(["connectionStatus", [True, None]])

    if verbose > 0:
        print("Hamamatsu: entering acquisition loop")

    try:
        while not finished:
            # Process one dataframe
            if not hama.processHeader():
                if verbose > 0:
                    print("Hamamatsu: processHeader failed")
                break
            if not hama.processReadings():
                if verbose > 0:
                    print("Hamamatsu: processReadings failed")
                break

            # Adapted from your original: bin channels and accumulate spectrum
            hama.binChannels(16)
            for eventID in range(hama.detectorEvents):
                ch = hama.binnedChannels[eventID]
                if 0 <= ch < 4096:
                    spectrum[ch] += 1

            # Handle commands
            while not cmd_q.empty():
                msg: Any = cmd_q.get()
                if isinstance(msg, (list, tuple)) and len(msg) == 2:
                    command, data = msg
                else:
                    command, data = msg, None

                if verbose > 1:
                    print("Hamamatsu: Command:", command)

                if command == "quit":
                    finished = True
                    break

                elif command == "reset":
                    spectrum = np.zeros(4096, dtype=np.uint32)
                    tic = time.time()

                elif command == "spectrum":
                    tock = time.time()
                    data_q.put(["spectrum", [spectrum, tock - tic]])

            time.sleep(0.01)

    except KeyboardInterrupt:
        if verbose > 0:
            print("Hamamatsu: interrupted by user")

    finally:
        try:
            if getattr(hama, "device", None) is not None and not hama.virtualDevice:
                usb.util.dispose_resources(hama.device)
        except Exception:
            pass

        try:
            data_q.put(["connectionStatus", [False, "finished"]])
        except Exception:
            pass


def Hamamatsu(
    port=None, uhubctl=None, verbose: int = 0, virtual: bool = False
) -> Tuple[Process, List[Queue]]:
    """
    Start Hamamatsu acquisition in a separate process.

    Returns:
      (process, [cmd_q, data_q])
    """
    cmd_q: Queue = Queue()
    data_q: Queue = Queue()

    worker = Process(
        target=HamamatsuLoop,
        args=(cmd_q, data_q, port, uhubctl, verbose, virtual),
        daemon=True,
    )
    worker.start()
    return worker, [cmd_q, data_q]

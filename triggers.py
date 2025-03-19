#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from psychopy import parallel
import platform

PLATFORM = platform.platform()
if 'Linux' in PLATFORM:
    port = parallel.ParallelPort(address='/dev/parport0')  # on Linux
elif 'Windows' in PLATFORM:
    port = parallel.ParallelPort(address=0xDFF8)  # Standard EEG address on Windows
else:  # on Mac, simulation mode
    port = None

# Figure out whether to flip pins or fake it
try:
    if port:
        port.setData(128)  # Test if port works
except (NotImplementedError, AttributeError):
    def setParallelData(code=1):
        """Simulated trigger function when parallel port is unavailable"""
        print(f"SIMULATED TRIGGER: {code}")
else:
    if port:
        port.setData(0)  # Reset port
        setParallelData = port.setData
    else:
        def setParallelData(code=1):
            """Simulated trigger function for Mac systems"""
            print(f"SIMULATED TRIGGER: {code}")
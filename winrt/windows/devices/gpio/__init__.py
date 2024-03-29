# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Devices.Gpio")

try:
    import winrt.windows.devices.gpio.provider
except Exception:
    pass

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

class GpioChangePolarity(enum.IntEnum):
    FALLING = 0
    RISING = 1
    BOTH = 2

class GpioOpenStatus(enum.IntEnum):
    PIN_OPENED = 0
    PIN_UNAVAILABLE = 1
    SHARING_VIOLATION = 2
    MUXING_CONFLICT = 3
    UNKNOWN_ERROR = 4

class GpioPinDriveMode(enum.IntEnum):
    INPUT = 0
    OUTPUT = 1
    INPUT_PULL_UP = 2
    INPUT_PULL_DOWN = 3
    OUTPUT_OPEN_DRAIN = 4
    OUTPUT_OPEN_DRAIN_PULL_UP = 5
    OUTPUT_OPEN_SOURCE = 6
    OUTPUT_OPEN_SOURCE_PULL_DOWN = 7

class GpioPinEdge(enum.IntEnum):
    FALLING_EDGE = 0
    RISING_EDGE = 1

class GpioPinValue(enum.IntEnum):
    LOW = 0
    HIGH = 1

class GpioSharingMode(enum.IntEnum):
    EXCLUSIVE = 0
    SHARED_READ_ONLY = 1

GpioChangeCount = _ns_module.GpioChangeCount
GpioChangeRecord = _ns_module.GpioChangeRecord
GpioChangeCounter = _ns_module.GpioChangeCounter
GpioChangeReader = _ns_module.GpioChangeReader
GpioController = _ns_module.GpioController
GpioPin = _ns_module.GpioPin
GpioPinValueChangedEventArgs = _ns_module.GpioPinValueChangedEventArgs

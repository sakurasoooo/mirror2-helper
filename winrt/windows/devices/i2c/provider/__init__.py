# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Devices.I2c.Provider")

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

class ProviderI2cBusSpeed(enum.IntEnum):
    STANDARD_MODE = 0
    FAST_MODE = 1

class ProviderI2cSharingMode(enum.IntEnum):
    EXCLUSIVE = 0
    SHARED = 1

class ProviderI2cTransferStatus(enum.IntEnum):
    FULL_TRANSFER = 0
    PARTIAL_TRANSFER = 1
    SLAVE_ADDRESS_NOT_ACKNOWLEDGED = 2

ProviderI2cTransferResult = _ns_module.ProviderI2cTransferResult
ProviderI2cConnectionSettings = _ns_module.ProviderI2cConnectionSettings
II2cControllerProvider = _ns_module.II2cControllerProvider
II2cDeviceProvider = _ns_module.II2cDeviceProvider
II2cProvider = _ns_module.II2cProvider

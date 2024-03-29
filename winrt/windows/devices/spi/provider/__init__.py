# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Devices.Spi.Provider")

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

class ProviderSpiMode(enum.IntEnum):
    MODE0 = 0
    MODE1 = 1
    MODE2 = 2
    MODE3 = 3

class ProviderSpiSharingMode(enum.IntEnum):
    EXCLUSIVE = 0
    SHARED = 1

ProviderSpiConnectionSettings = _ns_module.ProviderSpiConnectionSettings
ISpiControllerProvider = _ns_module.ISpiControllerProvider
ISpiDeviceProvider = _ns_module.ISpiDeviceProvider
ISpiProvider = _ns_module.ISpiProvider

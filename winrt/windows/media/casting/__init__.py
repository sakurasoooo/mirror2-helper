# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Media.Casting")

try:
    import winrt.windows.devices.enumeration
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

try:
    import winrt.windows.storage.streams
except Exception:
    pass

try:
    import winrt.windows.ui.popups
except Exception:
    pass

class CastingConnectionErrorStatus(enum.IntEnum):
    SUCCEEDED = 0
    DEVICE_DID_NOT_RESPOND = 1
    DEVICE_ERROR = 2
    DEVICE_LOCKED = 3
    PROTECTED_PLAYBACK_FAILED = 4
    INVALID_CASTING_SOURCE = 5
    UNKNOWN = 6

class CastingConnectionState(enum.IntEnum):
    DISCONNECTED = 0
    CONNECTED = 1
    RENDERING = 2
    DISCONNECTING = 3
    CONNECTING = 4

class CastingPlaybackTypes(enum.IntFlag):
    NONE = 0
    AUDIO = 0x1
    VIDEO = 0x2
    PICTURE = 0x4

CastingConnection = _ns_module.CastingConnection
CastingConnectionErrorOccurredEventArgs = _ns_module.CastingConnectionErrorOccurredEventArgs
CastingDevice = _ns_module.CastingDevice
CastingDevicePicker = _ns_module.CastingDevicePicker
CastingDevicePickerFilter = _ns_module.CastingDevicePickerFilter
CastingDeviceSelectedEventArgs = _ns_module.CastingDeviceSelectedEventArgs
CastingSource = _ns_module.CastingSource